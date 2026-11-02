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

import pandas as pd
import numpy as np

from evadb.functions.abstract.pytorch_abstract_function import (
    PytorchAbstractClassifierFunction,
)
from evadb.utils.generic_utils import try_to_import_torch, try_to_import_torchvision

import time
import json

from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, CenterCrop

import ast
import torch

from morphingdb_test.config import evadb_imagenet_model_path

IMAGE_TEST_FILE = 'result/evadb_muti_query_test.json'

class Resnet18Test(PytorchAbstractClassifierFunction):

    # def __del__(self):
    #     try:
    #         with open(IMAGE_TEST_FILE, 'r') as f_image:
    #             # Try to load existing data
    #             timing_data_image = json.load(f_image)
    #     except (FileNotFoundError, json.JSONDecodeError):
    #         # If the file does not exist or is not valid JSON, initialize it as an empty list
    #         timing_data_image = []

    #     self.total_time = self.load_model_time + self.pre_time + self.infer_time + self.post_time
    #     if(self.count == 0):
    #         return
    #     # Append the new records to the list
    #     timing_data_image.append({"count": self.count, 
    #                             "total_time": 0, 
    #                             "scan_time": 0,
    #                             "load_model_time": self.load_model_time, 
    #                             "pre_time": self.pre_time,
    #                             "infer_time": self.infer_time, 
    #                             "post_time": self.post_time})

    #     print({"count": self.count, 
    #                             "total_time": 0, 
    #                             "scan_time": 0,
    #                             "load_model_time": self.load_model_time, 
    #                             "pre_time": self.pre_time,
    #                             "infer_time": self.infer_time, 
    #                             "post_time": self.post_time})
    #     # Write back to the file
    #     with open(IMAGE_TEST_FILE, 'w') as f_image:
    #         json.dump(timing_data_image, f_image, indent=4)
        
    @property
    def name(self) -> str:
        return "Resnet18TestMuti"

    def setup(self):
        try_to_import_torch()
        try_to_import_torchvision()
        import torch
        import torch.nn as nn

        model_urls = {
            "resnet18": evadb_imagenet_model_path  # noqa
        }

        self.load_model_time = 0
        self.pre_time = 0
        self.infer_time = 0
        self.post_time = 0
        self.total_time = 0
        self.count = 0

        def resnet18():
            model = torch.load(model_urls["resnet18"], weights_only=False)
            return model

        start = time.time()
        self.model = resnet18()
        self.model.eval()
        end = time.time()
        self.load_model_time = end - start

    @property
    def labels(self):
        res = list([str(num+1) for num in range(1000)])
        return res

    def transform(self, images):
        start = time.time()
        # image_data = ast.literal_eval(images)
        # np_array = np.array(image_data)
        # frame = torch.from_numpy(np_array)
        # frame = frame.to(torch.float32)
        # frame = frame.reshape(1,3,224,224)
        img = Image.open(images)
        img = img.convert('RGB')
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        start = time.time()
        composed = Compose([
            Resize(256),
            CenterCrop(224),
            ToTensor(),  # convert PIL images or NumPy arrays to FloatTensor
            Normalize(mean=mean, std=std),  # normalization
        ])
        frame = composed(img).unsqueeze(0)
        end = time.time()
        self.pre_time += (end - start)
        return frame
        # print(images.shape[0])
        # print(images.shape[1])
        # print(images.shape[2])
        # img = Image.open("/data/image/image-net/data/"+images)
        # img_rgb = img.convert('RGB')
        # img_resized = img_rgb.resize((500, 375))

        # # Convert PIL images to NumPy arrays
        # img_rgb_np = np.array(img_resized)
        # print("img_rgb_np.shape", img_rgb_np.shape)

        # # normalization params used by the pretrained resnet18 model
        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # start = time.time()
        # composed = Compose([
        #     ToTensor(),  # convert PIL images or NumPy arrays to FloatTensor
        #     Normalize(mean=mean, std=std),  # normalization
        # ])

        # # Ensure the input images are RGB (three channels)
        # if img_rgb_np.shape[2] == 3:
        #     #frame = composed(img_rgb[:, :, ::-1]).unsqueeze(0)
        #     frame = composed(img_resized)  # convert img_rgb directly with composed
        #     frame = frame.unsqueeze(0)  # add a dimension to match the model input requirements
        #     end = time.time()
        #     self.pre_time += (end - start)
        #     return frame
        # else:
        #     # If not three channels, it may be grayscale and must be converted or rejected
        #     raise ValueError("Input image is not a RGB image")
        # normalization params used by the pretrained resnet18 model
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        start = time.time()
        composed = Compose(
            [
                ToTensor(),  # convert PIL images or NumPy arrays to FloatTensor
                Normalize(mean=mean, std=std),  # normalization
            ]
        )
        # Ensure the input images are RGB (three channels)
        if images.shape[2] == 3:
            frame = composed(Image.fromarray(images[:, :, ::-1])).unsqueeze(0)
            end = time.time()
            self.pre_time += (end - start)
            return frame
        else:
            # If not three channels, it may be grayscale and must be converted
            raise ValueError("Input image is not a RGB image")

    def forward(self, frames) -> pd.DataFrame:
        
        self.count += 1
        outcome = []
        start = time.time()
        frames = frames.to(torch.device("cpu"))
        self.model = self.model.to(torch.device("cpu"))
        predictions = self.model(frames)

        end = time.time()
        self.infer_time += (end - start)

        start = time.time()
        for prediction in predictions:
            label = self.as_numpy(prediction.data.argmax())
            print("label", label)
            outcome.append({"labels": str(label)})

        labels = pd.DataFrame(outcome, columns=["labels"])
        end  = time.time()
        self.post_time += (end - start)

        return labels
