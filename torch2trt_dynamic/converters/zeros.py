from collections.abc import Iterable

import tensorrt as trt
import torch
from torch2trt_dynamic.torch2trt_dynamic import tensorrt_converter, trt_


@tensorrt_converter('torch.zeros')
def convert_zeros(ctx):
    size = ctx.method_args[0]
    if not isinstance(size, Iterable):
        size = ctx.method_args
    dtype = torch.float32
    if 'dtype' in ctx.method_kwargs:
        dtype = ctx.method_kwargs['dtype']
    output = ctx.method_return

    if isinstance(size, int):
        size = (size, )

    # check const
    is_const = True
    for s in size:
        if hasattr(s, '_trt'):
            is_const = False
            break

    if is_const:
        # create const value
        output_trt = trt_(ctx.network, output)

    else:
        # create fill
        trt_size = []
        for s in size:
            if hasattr(s, '_trt'):
                trt_size.append(s._trt)
            else:
                trt_size.append(trt_(ctx.network, s))

        trt_size = ctx.network.add_concatenation(trt_size).get_output(0)

        layer = ctx.network.add_fill(size, trt.FillOperation.RANDOM_UNIFORM)
        layer.set_input(0, trt_size)
        layer.set_input(
            1, trt_(ctx.network,
                    torch.tensor(0., dtype=dtype).cuda()))
        layer.set_input(
            2, trt_(ctx.network,
                    torch.tensor(0., dtype=dtype).cuda()))

        output_trt = layer.get_output(0)

    data_type = None
    if dtype == torch.float32:
        data_type = trt.DataType.FLOAT
    elif dtype == torch.int32 or dtype == torch.long:
        data_type = trt.DataType.INT32
    elif dtype == torch.bool:
        data_type = trt.DataType.BOOL
    else:
        print('unsupported convert type:{}'.format(dtype))

    if data_type is not None:
        layer = ctx.network.add_identity(output_trt)
        layer.set_output_type(0, data_type)
        output_trt = layer.get_output(0)

    output._trt = output_trt
