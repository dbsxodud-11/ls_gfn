from collections import namedtuple
import torch

""" Experience object: data for single trajectory.
  Stored in ReplayBuffer for offline training.

  Fields
  ------
  traj: List of states
  r: reward; float.
  logr: log reward; tensor on device
  logp_guide: logp of traj from guide. Tensor on device. Can be None
              when guide is not used.

  Generally, Experience is initialized with either:
  1. Minimal init: [traj, x, r, logr]. Minimum necessary for training.
  2. Full init, all. Used for replaybuffer / offline training.
  +/- logp_guide, when needed.
"""

fields = [
  'traj', 
  'x',
  'r',
  'logr', 
  'logp_guide',
]
Experience = namedtuple('Experience', fields, defaults=(None,)*len(fields))


def make_full_exp(min_exp, args):
  """ Construct Experience object from sampled trajectory.

    Parameters: minimal Experience, with fields:
    ----------
    traj: List of States from root to leaf x
    x: State
    r: reward float
    logp_guide: float
      Log probability of trajectory under guide

    Returns: Experience with more fields set.
  """
  full_exp = Experience(
    traj = min_exp.traj,
    x = min_exp.x,
    r = min_exp.r,
    logr = torch.log(torch.tensor(min_exp.r, dtype=torch.float32, device=args.device)),
    logp_guide = torch.tensor(min_exp.logp_guide, device=args.device),
  )
  return full_exp