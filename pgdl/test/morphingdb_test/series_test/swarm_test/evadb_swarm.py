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
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from evadb.functions.abstract.abstract_function import (
    AbstractFunction,
)
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.decorators.decorators import forward

import time
import json
import os
from morphingdb_test.config import evadb_swarm_model_path

SWARMTEST_FILE = 'result/evadb_swarm_test.json'

class SwarmClassifier(AbstractFunction):

    @property
    def name(self) -> str:
        return "SwarmClassifier"

    def setup(self):
        import torch
        import torch.nn as nn

        model_urls = {
            "swarm-classifier": evadb_swarm_model_path
        }

        self.load_model_time = 0
        self.pre_time = 0
        self.infer_time = 0
        self.post_time = 0
        self.total_time = 0

        def swarm_model():
            model = torch.load(model_urls["swarm-classifier"], weights_only=False)
            return model
        start = time.time()
        self.model = swarm_model()
        self.model.eval()
        end = time.time()
        self.load_model_time = end - start

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["data"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(614, 1)]
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["labels"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 1)]
            )
        ],
    )
    def forward(self, input: pd.DataFrame) -> pd.DataFrame:
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
        if cuda_devices == '-1':
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")

        outcome = []
        input_data = input.values
        for row in input_data:
            start = time.time()
            # row[0] is the array data
            arr = row[0]
            if hasattr(arr, 'tolist'):
                arr = arr.tolist()
            input_tensor = torch.tensor(arr, dtype=torch.float32).reshape(1, -1)
            end = time.time()
            self.pre_time += (end - start)

            input_tensor = input_tensor.to(device)
            self.model.to(device)
            start = time.time()
            predictions = self.model(input_tensor)
            end = time.time()
            self.infer_time += (end - start)

            start = time.time()
            label_indices = predictions.argmax(dim=1)
            outcome.append([{"labels": str(label_idx.item())} for label_idx in label_indices])
            end = time.time()
            self.post_time += (end - start)

        self.total_time = self.load_model_time + self.pre_time + self.infer_time + self.post_time
        return pd.DataFrame(outcome, columns=["labels"])
