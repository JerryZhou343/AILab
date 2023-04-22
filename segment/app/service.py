#coding=utf-8

import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel
import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic



class ApplicationService():
    def initializer(self):
        pass




ServiceInstance = ApplicationService()

