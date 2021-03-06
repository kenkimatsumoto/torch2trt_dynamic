import torch
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)


@tensorrt_converter('torch.Tensor.softmax')
@tensorrt_converter('torch.softmax')
@tensorrt_converter('torch.nn.functional.softmax')
def convert_softmax(ctx):

    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    # get dims from args or kwargs
    dim = get_arg(ctx, 'dim', pos=1, default=None)
    if dim is None:
        dim = -1
    if dim < 0:
        dim = len(input.shape) + dim

    # axes = 1 << (dim - 1)
    axes = 1 << dim

    layer = ctx.network.add_softmax(input=input_trt)
    layer.axes = axes

    output._trt = layer.get_output(0)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module():
    return torch.nn.Softmax(1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_softmax_module_dim2():
    return torch.nn.Softmax(2)
