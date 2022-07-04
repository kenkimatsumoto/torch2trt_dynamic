import torch
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter

from .ReLU_ import convert_ReLU_


@tensorrt_converter('torch.relu')
@tensorrt_converter('torch.relu_')
@tensorrt_converter('torch.nn.functional.relu')
@tensorrt_converter('torch.nn.functional.relu_')
def convert_relu(ctx):
    ctx.method_args = (torch.nn.ReLU(), ) + ctx.method_args
    convert_ReLU_(ctx)
