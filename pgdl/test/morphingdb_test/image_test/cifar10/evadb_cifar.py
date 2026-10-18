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
CIFAR-10 evadb function using local googlenet_cifar10.pt.
Does NOT require torchvision. Uses PIL and numpy for preprocessing.
"""
import torch
import pandas as pd
import time
import json
import os
import numpy as np
from PIL import Image

from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe
from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.decorators.decorators import forward

from morphingdb_test.config import evadb_cifar10_model_path

IMAGE_TEST_FILE = 'result/evadb_cifar10_test.json'


class CifarFunction(AbstractFunction):

    @property
    def name(self) -> str:
        return "CifarFunction"

    def setup(self):
        import torch

        self.load_model_time = 0
        self.pre_time = 0
        self.infer_time = 0
        self.post_time = 0
        self.total_time = 0

        start = time.time()
        self.model = torch.load(evadb_cifar10_model_path, weights_only=False)
        self.model.eval()
        end = time.time()
        self.load_model_time = end - start

        # CIFAR-10 normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]
        self.mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(1, 1, 3)
        self.target_size = (32, 32)

    def _preprocess_image(self, image_path):
        """Load and preprocess image using PIL + numpy (no torchvision)."""
        if isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8', errors='replace')

        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.target_size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)
        return tensor

    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["image_path"],
                column_types=[NdArrayType.STR],
                column_shapes=[(None,)]
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["labels"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(None, None, 10)]
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
            image_path = row[0]

            start = time.time()
            input_tensor = self._preprocess_image(image_path).to(device)
            end = time.time()
            self.pre_time += (end - start)

            start = time.time()
            with torch.no_grad():
                output = self.model(input_tensor)
            end = time.time()
            self.infer_time += (end - start)

            start = time.time()
            if isinstance(output, tuple):
                output = output[0]
            probs = torch.softmax(output, dim=1).cpu().numpy().tolist()
            outcome.append({"labels": probs[0]})
            end = time.time()
            self.post_time += (end - start)

        self.total_time = self.load_model_time + self.pre_time + self.infer_time + self.post_time
        return pd.DataFrame(outcome, columns=["labels"])
