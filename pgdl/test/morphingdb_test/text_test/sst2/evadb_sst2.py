# coding=utf-8
# Copyright 2018-2023 EvaDB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SST-2 evadb function using local traced_albert_vec.pt model.
The model expects 4 tensors: input_ids, attention_mask, token_type_ids, position_ids.
Does NOT require torchvision, HuggingFace, or sentencepiece.
"""
import torch
import pandas as pd
import time
import json
import os

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.decorators.decorators import forward

from morphingdb_test.config import evadb_sst2_model_path

TEXT_TEST_FILE = 'result/evadb_sst2_test.json'


class SST2Function(AbstractFunction):

    @property
    def name(self) -> str:
        return "SST2Function"

    def setup(self):
        import torch

        self.load_model_time = 0
        self.pre_time = 0
        self.infer_time = 0
        self.post_time = 0
        self.total_time = 0
        self.max_length = 128

        start = time.time()
        self.model = torch.load(evadb_sst2_model_path, weights_only=False)
        self.model.eval()
        end = time.time()
        self.load_model_time = end - start

    def _prepare_inputs(self, text):
        """Prepare 4 tensors required by the traced Albert model."""
        # Simple character-level encoding
        char_ids = [ord(c) % 1000 + 1 for c in text[:self.max_length]]  # +1 to avoid 0 (padding)
        seq_len = len(char_ids)
        char_ids = char_ids + [0] * (self.max_length - seq_len)

        input_ids = torch.tensor([char_ids], dtype=torch.long)
        attention_mask = torch.tensor([[1] * seq_len + [0] * (self.max_length - seq_len)], dtype=torch.long)
        token_type_ids = torch.zeros((1, self.max_length), dtype=torch.long)
        position_ids = torch.arange(0, self.max_length, dtype=torch.long).unsqueeze(0)

        return input_ids, attention_mask, token_type_ids, position_ids

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["comment"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None,)]
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["labels"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 3)]
            )
        ],
    )
    def forward(self, input: pd.DataFrame) -> pd.DataFrame:
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices == '-1':
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        self.model.to(device)
        outcome = []
        input_data = input.values

        for row in input_data:
            text = row[0]
            if isinstance(text, bytes):
                text = text.decode('utf-8', errors='replace')

            start = time.time()
            input_ids, attention_mask, token_type_ids, position_ids = self._prepare_inputs(text)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            position_ids = position_ids.to(device)
            end = time.time()
            self.pre_time += (end - start)

            start = time.time()
            with torch.no_grad():
                output = self.model(input_ids, attention_mask, token_type_ids, position_ids)
            end = time.time()
            self.infer_time += (end - start)

            start = time.time()
            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()
            outcome.append({"labels": probs[0]})
            end = time.time()
            self.post_time += (end - start)

        self.total_time = self.load_model_time + self.pre_time + self.infer_time + self.post_time
        return pd.DataFrame(outcome, columns=["labels"])
