import tensorrt as trt
import torch
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter, trt_


@tensorrt_converter('torch.nn.functional.tanh')
@tensorrt_converter('torch.tanh')
def convert_tanh(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    layer = ctx.network.add_activation(input_trt, trt.ActivationType.TANH)
    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_tanh_basic():
    return torch.nn.Tanh()
