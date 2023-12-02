# Debug utility to display variables
import numpy as np
import torch


def debug(name, data):
    # Handle PyTorch tensor
    if isinstance(data, torch.Tensor):
        print("{} = Torch.Tensor{} x {}, size={:,} elements = {:,} bytes, device={}".format(
            name, list(data.size()), data.dtype, data.numel(), data.numel() * data.element_size(), data.device))

    # Handle NumPy array
    elif isinstance(data, np.ndarray):
        print("{} = numpy.ndarray{} x {}, size={:,} elements = {:,} bytes".format(name, data.shape, data.dtype, data.size, data.size * data.itemsize))

    # Handle Python list (or similar)
    else:
        shape = (len(data),)
        print("{} = list{} x {}, size={:,} elements".format(name, shape, type(data[0]) if data else None, len(data)))
