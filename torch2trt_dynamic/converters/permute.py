import torch
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter, trt_


@tensorrt_converter('torch.Tensor.permute')
def convert_permute(ctx):
    input = ctx.method_args[0]
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    # permutation -1 because TRT does not include batch dim
    if isinstance(ctx.method_args[1], int):
        permutation = tuple(ctx.method_args[1:])  # handle permute(a, b, c)
    else:
        permutation = tuple(ctx.method_args[1])  # handle permute([a, b, c])

    trt_permutation = permutation

    layer = ctx.network.add_shuffle(input_trt)
    layer.second_transpose = tuple(trt_permutation)

    output._trt = layer.get_output(0)


class Permute(torch.nn.Module):

    def __init__(self, *args):
        super(Permute, self).__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args).contiguous()


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_permute_2d_0123():
    return Permute(0, 1, 2, 3)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5)])
def test_permute_2d_0312():
    return Permute(0, 3, 1, 2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_3d_01234():
    return Permute(0, 1, 2, 3, 4)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_3d_04132():
    return Permute(0, 4, 1, 3, 2)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_list():
    return Permute([0, 4, 1, 3, 2])


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 4, 5, 6)])
def test_permute_tuple():
    return Permute((0, 4, 1, 3, 2))
