import torch
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_)


@tensorrt_converter('torch.nn.functional.pad')
def convert_pad(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    pad = get_arg(ctx, 'pad', pos=1, default=[0, 0, 0, 0])
    pre_padding = (pad[2], pad[0])
    post_padding = (pad[3], pad[1])

    # mode / value are ignored since not supported by TensorRT

    layer = ctx.network.add_padding(input_trt, pre_padding, post_padding)
    output._trt = layer.get_output(0)


class Pad(torch.nn.Module):

    def __init__(self, pad):
        super(Pad, self).__init__()
        self.pad = pad

    def forward(self, x):
        return torch.nn.functional.pad(x, self.pad)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 224, 224)])
def test_pad_basic():
    return Pad((1, 2, 3, 4))
