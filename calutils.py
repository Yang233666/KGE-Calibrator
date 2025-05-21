'''
Some utility functions
'''
import torch
from scipy.special import expit, softmax
import pathlib
import numpy as np


def checkFile(filename):
    return pathlib.Path(filename).is_file()

def expit_probs_x(x: torch.Tensor, np_ndarray=False):
    if np_ndarray:
        x = np.array(x.cpu())
        x = softmax(x)
    else:
        x = x.softmax(-1).cpu()
    return x

def expit_probs_binary(x: torch.Tensor, y: torch.Tensor):
    max_indices = torch.argmax(x, dim=1)
    y = torch.argmax(y, dim=1)
    correct = (max_indices == y).int()
    return correct

def normalize(tensor, eps=1e-10):
    """
	Normalize a tensor so that it sums to 1.

	Parameters
	----------
	tensor (torch.Tensor): Input tensor to normalize.
	epsilon (float): Small value to avoid division by zero.

	Returns
	-------
	torch.Tensor: Normalized tensor that sums to 1.
	"""
    # Compute the sum of the tensor elements
    tensor_sum = torch.sum(tensor)

    # Add epsilon to avoid division by zero
    tensor_sum = torch.clamp(tensor_sum, eps)

    # Divide the tensor by its sum to normalize
    return tensor / tensor_sum

def convert_to_tensor(a):
    if isinstance(a, np.ndarray):
        return torch.tensor(a)
    elif isinstance(a, torch.Tensor):
        return a
    else:
        raise TypeError("Input must be a numpy.ndarray or torch.Tensor.")
