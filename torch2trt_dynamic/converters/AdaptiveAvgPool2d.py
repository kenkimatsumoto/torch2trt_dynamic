import torch
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter

from .adaptive_avg_pool2d import convert_adaptive_avg_pool2d


@tensorrt_converter('torch.nn.AdaptiveAvgPool2d.forward')
def convert_AdaptiveAvgPool2d(ctx):
    ctx.method_args = (ctx.method_args[1], ctx.method_args[0].output_size)
    convert_adaptive_avg_pool2d(ctx)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_1x1():
    return torch.nn.AdaptiveAvgPool2d((1, 1))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_2x2():
    return torch.nn.AdaptiveAvgPool2d((2, 2))


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_AdaptiveAvgPool2d_3x3():
    return torch.nn.AdaptiveAvgPool2d((3, 3))
