import tensorrt as trt
import torch

if trt.__version__ >= '5.1':
    DEFAULT_CALIBRATION_ALGORITHM = \
        trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2
else:
    DEFAULT_CALIBRATION_ALGORITHM = trt.CalibrationAlgoType.ENTROPY_CALIBRATION


class TensorBatchDataset():

    def __init__(self, tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, idx):
        return [t[idx] for t in self.tensors]


class DatasetCalibrator(trt.IInt8Calibrator):

    def __init__(self,
                 names,
                 profile,
                 inputs,
                 dataset,
                 batch_size=1,
                 algorithm=DEFAULT_CALIBRATION_ALGORITHM):
        super(DatasetCalibrator, self).__init__()

        self.dataset = dataset
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.names = names
        self.device = inputs[0].device

        # create buffers that will hold data batches
        self.buffers = []
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        for name, tensor in zip(names, inputs):
            size = tuple(profile.get_shape(name)[1])
            buf = torch.zeros(
                size=size, dtype=tensor.dtype,
                device=tensor.device).contiguous()
            self.buffers.append(buf)

        self.count = 0

    def get_batch(self, *args, **kwargs):
        if self.count < len(self.dataset):

            for i in range(self.batch_size):

                idx = self.count % len(
                    self.dataset)  # roll around if not multiple of dataset
                inputs = self.dataset[idx]
                if isinstance(inputs, torch.Tensor):
                    inputs = [inputs]

                # copy data for (input_idx, dataset_idx) into buffer
                for buffer, tensor in zip(self.buffers, inputs):
                    tensor = tensor.to(self.device)
                    buffer[i].copy_(tensor)

                self.count += 1

            return [int(buf.data_ptr()) for buf in self.buffers]
        else:
            return []

    def get_algorithm(self):
        return self.algorithm

    def get_batch_size(self):
        return self.batch_size

    def read_calibration_cache(self, *args, **kwargs):
        return None

    def write_calibration_cache(self, cache, *args, **kwargs):
        pass
