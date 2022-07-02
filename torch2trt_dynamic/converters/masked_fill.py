from ..torch2trt_dynamic import get_arg, tensorrt_converter


@tensorrt_converter('torch.masked_fill', is_real=False)
@tensorrt_converter('torch.Tensor.masked_fill', is_real=False)
@tensorrt_converter('torch.Tensor.masked_fill_', is_real=False)
def convert_masked_fill(ctx):
    input = ctx.method_args[0]
    mask = get_arg(ctx, 'mask', pos=1, default=None)
    value = get_arg(ctx, 'value', pos=2, default=0)
    output = ctx.method_return

    if value == float('-inf'):
        import sys
        float_info = sys.float_info
        value = -(float_info.min * float_info.epsilon)

    float_mask = mask.type_as(input)
    result = input * (1 - float_mask) + value * float_mask

    output._trt = result._trt
    ctx.method_return = output
