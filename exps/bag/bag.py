"""
  Bag
"""
import numpy as np
import os, itertools, random
from scipy.stats import binom

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import bagmdp
from gflownet.monitor import TargetRewardDistribution, Monitor


class BagMDP(bagmdp.BagMDP):
  def __init__(self, args):
    super().__init__(args, alphabet=args.bag_alphabet)
    self.args = args
    self.alphabet = args.bag_alphabet
    self.substruct_size = args.bag_substruct_size
    self.forced_stop_len = args.bag_force_stop

    self.mapper = {
      'none': 0.01,
      'substructure': 10,
      'mode': 30,
    }
    self.top_frac = 0.25
    self.expr_full = self.compute_expected_reward()

    mode_fn = f'datasets/bag/bagmodes-{self.substruct_size}-{self.forced_stop_len}-{args.bag_alphabet}.txt'
    if not os.path.isfile(mode_fn):
      self.generate_modes(mode_fn)
    with open(mode_fn, 'r') as f:
      lines = f.readlines()
    self.modes = [self.state(line.strip(), is_leaf=True) for line in lines]
    self.modes = set(self.modes)

  def generate_modes(self, mode_fn):
    """
      for each letter c: mode has cccc
      and then 10% of remaining 3-letter combinations
    """
    modes = []
    for char in self.alphabet:
      other_chars = [c for c in self.alphabet if c != char]
      other_size = self.forced_stop_len - self.substruct_size
      fillers = [''.join(s) for s in itertools.product(other_chars, repeat=other_size)]
      random.shuffle(fillers)

      modes += [self.substruct_size*char + extra for extra in fillers[:int(self.top_frac * len(fillers))]]
    
    with open(mode_fn, 'w') as f:
      f.write('\n'.join(modes))
    return

  # Core
  def reward(self, x):
    """ State -> float """
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'

    score = self.mapper['none']
    if x.max_group_size() >= self.substruct_size:
      score = self.mapper['substructure']
    if x in self.modes:
      score = self.mapper['mode']
    return score

  def dist_states(self, state1, state2):
    """ content: Counter (dict) mapping {char: count}. """
    num_shared = 0
    n = sum(state1.content.values())
    for chara, ct1 in state1.content.items():
      ct2 = state2.content.get(chara, 0)
      num_shared += min(ct1, ct2)
    return n - num_shared

  """
    Bag reward statistics
  """
  def get_probs(self):
    """ Probabilities of outcomes under a uniform distribution. """
    a = len(self.alphabet)
    k = self.args.bag_substruct_size
    n = self.args.bag_force_stop
    p_sub = a * binom.sf(k-1, n, 1/a)
    p_nonsub = 1 - p_sub
    p_sub_nomode = (p_sub * (1 - self.top_frac))
    p_mode = (p_sub * self.top_frac)
    return p_nonsub, p_sub_nomode, p_mode

  def compute_expected_reward(self):
    # Expected reward if p(x) = r(x)/Z for bag
    p_nonsub, p_sub_nomode, p_mode = self.get_probs()
    denom = p_nonsub * self.mapper['none'] + \
          p_sub_nomode * self.mapper['substructure'] + \
          p_mode * self.mapper['mode']
    numer = p_nonsub * (self.mapper['none']**2) + \
          p_sub_nomode * (self.mapper['substructure']**2) + \
          p_mode * (self.mapper['mode']**2)
    """
      For 7-13 bag, this is 18.17. This is exp. reward for training set.
      When discovering new modes however, because modes are completely randomly distributed, the expected reward is 15.0. Test reward cannot exceed this. 
    """
    return numer / denom

  def get_ad_samples(self):
    """ Treat uniform probabilities of outcome categories as relative counts - 
        adjust by their reward, then sample.
    """
    p_nonsub, p_sub_nomode, p_mode = self.get_probs()
    ps = {
      self.mapper['none']: self.mapper['none'] * p_nonsub,
      self.mapper['substructure']: self.mapper['substructure'] * p_sub_nomode,
      self.mapper['mode']: self.mapper['mode'] * p_mode,
    }
    norm_ps = np.array(list(ps.values()))
    norm_ps /= sum(norm_ps)
    return np.random.choice(list(ps.keys()), size=int(1e6), p=norm_ps)

  """
    Interpretation & visualization
  """
  def is_mode(self, x, r):
    mode_reward = self.mapper['mode']
    return bool(r == mode_reward)

  def unnormalize(self, r):
      return r

  def make_monitor(self):
    """ Make monitor, called during training.

        For bag, target reward statistics need to be manually specified,
        since bag space is too large (7^13 = 100 billion). 
    """
    target = TargetRewardDistribution()
    target.expected_reward = self.compute_expected_reward()
    target.ad_samples = self.get_ad_samples()

    return Monitor(self.args, target, dist_func=self.dist_states,
                   is_mode_f=self.is_mode, callback=self.add_monitor,
                   unnormalize=self.unnormalize)

  def add_monitor(self, xs, rs, allXtoR):
    """ Additional monitoring. Called in monitor.evaluate()

        Inputs
        ------
        samples: List of x, sampled from current gfn
        tolog: dictionary, to be updated and logged to wandb.
    """
    # Track substructures found so far
    subs_found = set()
    # for char in self.alphabet:
    for x, r in allXtoR.items():
      if r >= self.mapper['substructure']:
        for char in self.alphabet:
          if self.substruct_size*char in str(x):
            subs_found.add(char)

    tolog = {
      'All data num substructures found': len(subs_found),
    }
    return tolog


def mode_seeking(args):
  print('Running experiment bag ...')
  mdp = BagMDP(args)
  actor = bagmdp.BagActor(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()
  
  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return
