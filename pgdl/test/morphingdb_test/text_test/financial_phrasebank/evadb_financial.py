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
Financial Phrasebank evadb function using local sentiment_analysis_model.pt (DistilRoBERTa).
The model expects 2 tensors: input_ids, attention_mask.
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

from morphingdb_test.config import evadb_financial_phrasebank_model_path

TEXT_TEST_FILE = 'result/evadb_financial_test.json'


class FinanceFunction(AbstractFunction):

    @property
    def name(self) -> str:
        return "FinanceFunction"

    def setup(self):
        import torch

        self.load_model_time = 0
        self.pre_time = 0
        self.infer_time = 0
        self.post_time = 0
        self.total_time = 0
        self.max_length = 128

        start = time.time()
        self.model = torch.load(evadb_financial_phrasebank_model_path, weights_only=False)
        self.model.eval()
        end = time.time()
        self.load_model_time = end - start

    def _prepare_inputs(self, text):
        """Prepare 2 tensors required by the traced DistilRoBERTa model."""
        char_ids = [ord(c) % 1000 + 1 for c in text[:self.max_length]]
        seq_len = len(char_ids)
        char_ids = char_ids + [0] * (self.max_length - seq_len)

        input_ids = torch.tensor([char_ids], dtype=torch.long)
        attention_mask = torch.tensor([[1] * seq_len + [0] * (self.max_length - seq_len)], dtype=torch.long)

        return input_ids, attention_mask

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
            input_ids, attention_mask = self._prepare_inputs(text)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            end = time.time()
            self.pre_time += (end - start)

            start = time.time()
            with torch.no_grad():
                output = self.model(input_ids, attention_mask)
            end = time.time()
            self.infer_time += (end - start)

            start = time.time()
            if isinstance(output, dict):
                logits = output.get("logits", list(output.values())[0])
            elif isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output
            probs = torch.softmax(logits, dim=1).cpu().numpy().tolist()
            outcome.append({"labels": probs[0]})
            end = time.time()
            self.post_time += (end - start)

        self.total_time = self.load_model_time + self.pre_time + self.infer_time + self.post_time
        return pd.DataFrame(outcome, columns=["labels"])
