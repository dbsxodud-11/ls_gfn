import numpy as np
import torch

from torch_geometric.data import Data, Batch


def tensor_to_np(tensor, reduce_singleton=True):
  """ Casts torch.tensor into np.array.
    Slow - requires GPU/CPU sync. Try to use infrequently. """
  if type(tensor) == np.ndarray:
    return tensor
  x = tensor.to('cpu').detach().numpy()
  if reduce_singleton and x.shape == (1,):
    return float(x)
  return x


def batch(list):
  if type(list[0]) is torch.Tensor:
    batch = torch.stack(list)
  elif type(list[0]) is Data:
    # Handle batching for torch geometric data (graphs)
    batch = Batch.from_data_list(list)
  return batch


def pack(ll):
  """ List of lists -> list, indices. """
  flat = []
  idxs = []
  curr_idx = 0
  for l in ll:
    flat += l
    idxs.append((curr_idx, curr_idx + len(l)))
    curr_idx += len(l)
  return flat, idxs


def unpack(flat, idxs):
  """ list, indices -> list of lists """
  return [flat[start:end] for start, end in idxs]
