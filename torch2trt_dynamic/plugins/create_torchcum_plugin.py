import numpy as np
import tensorrt as trt


def create_torchcum_plugin(layer_name, dim, cum_type):

    creator = trt.get_plugin_registry().get_plugin_creator(
        'TorchCumPluginDynamic', '1', '')

    pfc = trt.PluginFieldCollection()

    pf_dim = trt.PluginField('dim', np.array([dim], dtype=np.int32),
                             trt.PluginFieldType.INT32)
    pfc.append(pf_dim)

    pf_cum_type = trt.PluginField('cum_type',
                                  np.array([cum_type], dtype=np.int32),
                                  trt.PluginFieldType.INT32)
    pfc.append(pf_cum_type)

    return creator.create_plugin(layer_name, pfc)
