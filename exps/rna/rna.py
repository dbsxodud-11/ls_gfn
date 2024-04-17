'''
    GFP
    Transformer Proxy
    Start from scratch
'''

import random
import pickle, functools
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pandas as pd
import torch
from polyleven import levenshtein
from itertools import combinations, product

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqinsertmdp, seqarmdp
from gflownet.monitor import TargetRewardDistribution, Monitor, diversity

import flexs
from flexs import baselines
import flexs.utils.sequence_utils as s_utils

def dynamic_inherit_mdp(base, args):

  class RNAMDP(base):
    def __init__(self, args):
      super().__init__(args,
                       alphabet=["U", "C", "G", "A"],
                       forced_stop_len=args.forced_stop_len)
      self.args = args
      self.rna_task = int(self.args.setting[-1])
      
      print(f'Loading data ...')
      problem = flexs.landscapes.rna.registry()[f'L{self.args.forced_stop_len}_RNA{self.rna_task}']
      print(problem)
      self.proxy_model = flexs.landscapes.RNABinding(**problem['params'])
      
      if self.args.forced_stop_len == 14:
        allpreds_file = args.allpreds_file + f'L{self.args.forced_stop_len}_RNA{self.rna_task}_allpreds.pkl'
        with open(allpreds_file, 'rb') as f:
          self.rewards = pickle.load(f)
      else:
        self.rewards = self.proxy_model.get_fitness(["".join(random.choices(self.alphabet, k=self.args.forced_stop_len)) for _ in range(5000)])
      
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
      if self.args.forced_stop_len == 14:
        mode_file = args.mode_file + f'L{self.args.forced_stop_len}_RNA{self.rna_task}_modes.pkl'
        with open(mode_file, 'rb') as f:
          self.modes = pickle.load(f)
        print(f"Found num modes: {len(self.modes)}")
      else:
        mode_percentile = 0.005
        self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))

    # Core
    def reward(self, x):
      assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
      r = self.proxy_model.get_fitness([x.content]).item()
      
      r = np.maximum(r, self.SCALE_REWARD_MIN)
      r = r ** self.REWARD_EXP
      r = r * self.scale
      return r
    
    def get_neighbors(self, x):
      neighbors = []
      for i in range(self.args.forced_stop_len):
        for j in self.alphabet:
          x_ = list(deepcopy(x))
          if x_[i] != j:
            x_[i] = j
            neighbors.append("".join(x_))
      return neighbors
            
    def is_mode(self, x, r):
      if self.args.forced_stop_len == 14:
        return x.content in self.modes
      else:
        if r >= self.mode_r_threshold:
          for neighbor in self.get_neighbors(x.content):
            if r < self.proxy_model.get_fitness([neighbor]).item():
              return False
          return True
        else:
          return False
    
    def unnormalize(self, r):
      r = r / self.scale
      r = r ** (1 / self.REWARD_EXP)
      return r

    '''
      Interpretation & visualization
    '''
    def dist_func(self, state1, state2):
      """ States are SeqPAState or SeqInsertState objects. """
      return levenshtein(state1.content, state2.content)

    def make_monitor(self):
      target = TargetRewardDistribution()
      target.init_from_base_rewards(self.rewards)
      return Monitor(self.args, target, dist_func=self.dist_func,
                     is_mode_f=self.is_mode, callback=self.add_monitor,
                     unnormalize=self.unnormalize)

    def add_monitor(self, xs, rs, allXtoR):
      """ Reimplement scoring with oracle, not unscaled oracle (used as R). """
      tolog = dict()
      return tolog
    
    def reduce_storage(self):
      del self.rewards
      del self.scaled_rewards

  return RNAMDP(args)

def mode_seeking(args):
  print("Online mode seeking in RNA-Binding...")
  base = seqpamdp.SeqPrependAppendMDP
  actorclass = seqpamdp.SeqPAActor
  mdp = dynamic_inherit_mdp(base, args)
  
  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()
  
  mdp.reduce_storage()
  
  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return


def main(args):
  print('Running experiment RNA ...')

  if args.mdp_style == 'pa':
    base = seqpamdp.SeqPrependAppendMDP
    actorclass = seqpamdp.SeqPAActor
  elif args.mdp_style == 'insert':
    base = seqinsertmdp.SeqInsertMDP
    actorclass = seqinsertmdp.SeqInsertActor
  elif args.mdp_style == 'autoregressive':
    base = seqarmdp.SeqAutoregressiveMDP
    actorclass = seqarmdp.SeqARActor
  mdp = dynamic_inherit_mdp(base, args)

  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  # mdp.reduce_storage()

  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return
