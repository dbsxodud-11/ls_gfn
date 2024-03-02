"""
  seh as string
"""
import pickle, functools
import numpy as np
from tqdm import tqdm
import torch

import gflownet.trainers as trainers
from gflownet.MDPs import molstrmdp
from gflownet.monitor import TargetRewardDistribution, Monitor
from gflownet.GFNs import models

from datasets.sehstr import gbr_proxy

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity


class SEHstringMDP(molstrmdp.MolStrMDP):
  def __init__(self, args):
    super().__init__(args)
    self.args = args

    assert args.blocks_file == 'datasets/sehstr/block_18.json', 'ERROR - x_to_r and rewards are designed for block_18.json'

    self.proxy_model = gbr_proxy.sEH_GBR_Proxy(args)

    # Read from file
    print(f'Loading data ...')
    x_file = args.x_file
    with open(x_file, 'rb') as f:
      self.x_to_r = pickle.load(f)
    self.xs = list(self.x_to_r.keys())
    self.rewards = list(map(float, self.x_to_r.values()))

    # scale rewards
    py = np.array(list(self.rewards))

    self.SCALE_REWARD_MIN = args.scale_reward_min
    self.SCALE_REWARD_MAX = args.scale_reward_max
    self.REWARD_EXP = args.reward_exp
    self.REWARD_MAX = max(py)

    py = np.maximum(py, self.SCALE_REWARD_MIN)
    py = py ** self.REWARD_EXP
    self.scale = self.SCALE_REWARD_MAX / max(py)
    py = py * self.scale

    self.scaled_rewards = py

    # modes
    mode_file = args.mode_file
    with open(mode_file, 'rb') as f:
      self.modes = pickle.load(f)
    print(f"Found num modes: {len(self.modes)}")

  # Core
  @functools.lru_cache(maxsize=None)
  def reward(self, x):
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
    pred = self.proxy_model.predict_state(x)
    r = np.maximum(pred, self.SCALE_REWARD_MIN)
    r = r ** self.REWARD_EXP
    r = r * self.scale
    return r

  def is_mode(self, x, r):
    return x.content in self.modes
  
  def unnormalize(self, r):
      r = r / self.scale
      r = r ** (1 / self.REWARD_EXP)
      return r

  # Diversity
  def dist_states(self, state1, state2):
    """ Tanimoto similarity on morgan fingerprints """
    fp1 = self.get_morgan_fp(state1)
    fp2 = self.get_morgan_fp(state2)
    return 1 - FingerprintSimilarity(fp1, fp2)

  @functools.lru_cache(maxsize=None)
  def get_morgan_fp(self, state):
    mol = self.state_to_mol(state)
    fp = GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return fp

  """
    Interpretation & visualization
  """
  def make_monitor(self):
    """ Make monitor, called during training. """
    target = TargetRewardDistribution()
    target.init_from_base_rewards(self.rewards)
    return Monitor(self.args, target, dist_func=self.dist_states,
                   is_mode_f=self.is_mode,
                   unnormalize=self.unnormalize)

  def reduce_storage(self):
    del self.x_to_r
    del self.xs
    del self.rewards
    del self.scaled_rewards


def mode_seeking(args):
  print("Online mode seeking in sehstr ...")
  mdp = SEHstringMDP(args)
  actor = molstrmdp.MolStrActor(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()
  
  mdp.reduce_storage()
  
  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return
