import torch
from torch2trt_dynamic.module_test import add_module_test
from torch2trt_dynamic.torch2trt_dynamic import (get_arg, tensorrt_converter,
                                                 trt_, trt_cast)

from .size import IntWarper


@tensorrt_converter('torch.Tensor.reshape')
@tensorrt_converter('torch.Tensor.view')
def convert_view(ctx):

    input = ctx.method_args[0]
    size = get_arg(ctx, 'shape', pos=1, default=[])
    if isinstance(size, int):
        size = tuple(ctx.method_args[1:])
    input_trt = trt_(ctx.network, input)
    output = ctx.method_return

    if input.dtype == torch.bool:
        input_trt = trt_cast(ctx.network, input_trt, torch.int32)

    # check if there are shape tensor
    is_shape_tensor = False
    for s in size:
        if isinstance(s, IntWarper):
            is_shape_tensor = True
            break

    # negative shape might cause overflow, forbid for now
    for s in size:
        if s < 0:
            is_shape_tensor = True
            break

    # compute shape tensor
    if is_shape_tensor:
        shape_trt = []
        for idx, s in enumerate(size):
            if isinstance(s, IntWarper):
                shape_trt.append(s._trt)
            else:
                const_shape_trt = trt_(
                    ctx.network, input.new_tensor([s], dtype=torch.int32))
                shape_trt.append(const_shape_trt)

        shape_trt = ctx.network.add_concatenation(shape_trt).get_output(0)

    layer = ctx.network.add_shuffle(input_trt)
    if is_shape_tensor:
        layer.set_input(1, shape_trt)
    else:
        layer.reshape_dims = output.shape

    output_trt = layer.get_output(0)

    if input.dtype == torch.bool:
        output_trt = trt_cast(ctx.network, output_trt, torch.bool)

    output._trt = output_trt


class View(torch.nn.Module):

    def __init__(self, *dims):
        super(View, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.view(*self.dims)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_1d():
    return View(1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_2d():
    return View(1, 1, -1)


@add_module_test(torch.float32, torch.device('cuda'), [(1, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3)])
@add_module_test(torch.float32, torch.device('cuda'), [(1, 3, 3, 3)])
def test_view_3d():
    return View(1, 1, 1, -1)
