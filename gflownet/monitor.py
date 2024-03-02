import itertools
import numpy as np
from scipy.stats import anderson_ksamp
from polyleven import levenshtein

import torch
import wandb

from gflownet.utils import tensor_to_np


"""
  Diversity
"""

def diversity(data, dist_func=levenshtein):
  """ Average pairwise distance between data. """
  dists = [dist_func(*pair) for pair in itertools.combinations(data, 2)]
  n = len(data)
  return sum(dists) / (n*(n-1) + 1e-8)


def novelty(new_data, old_data, dist_func=levenshtein):
  scores = [min([dist_func(d, od) for od in old_data]) for d in new_data]
  return np.mean(scores)


def multi_set_distance(ms1, ms2):
  """ Distance between two multisets. Assumes that all sets are same length. """
  n = len(ms1)
  return n - len(ms1 & ms2)

"""
  Target distribution
"""
class TargetRewardDistribution:
  def __init__(self):
    """ Compute and hold statistics on the target reward distribution.

        The target distribution samples x with probability r(x).
        Given r(x), the expected reward is thus r(x)^2 / Z.

        Key properties, expected to be initialized:
        --------------
        expected_reward: float
          Expected reward under target distribution.
        ad_samples: List of floats
          Rewards sampled from target, used for computing Anderson-Darling
          statistic.
    """
    self.expected_reward = None

  def init_from_base_rewards(self, base_rs):
    """ Compute target reward distribution statistics.

        Ideally, base_rs is a list of all rewards for all unique x.
        This is only feasible for relatively small MDPs with enumerable
        X, e.g., < 100 million unique x.
    """
    z = sum(base_rs)
    expr = sum([r**2 for r in base_rs]) / z
    self.expected_reward = expr
    print(f'Expected reward: {expr}')
    return


class Monitor:
  def __init__(self, 
               args, 
               target, 
               dist_func = None, 
               is_mode_f = None, 
               callback = None,
               unnormalize = None):
    self.args = args
    self.target = target
    self.dist_func = dist_func
    self.is_mode_f = is_mode_f
    self.callback = callback
    self.unnormalize = unnormalize
    self.sample_log = dict()
    self.NUM_ROUNDS_BACK = 1
    self.FAST_EVAL_EVERY = self.args.get('monitor_fast_every', 5)
    self.SLOW_EVAL_EVERY = self.args.get('monitor_slow_every', 200)
    
  def log_samples(self, round_num, samples):
    """ Logs samples. """
    self.sample_log[round_num] = samples
    return
  
  def log_real_samples(self, allXtoR):
    name = "Real"
    tolog = dict()
    if self.is_mode_f is not None:
      unique_modes = set(x for x, r in allXtoR.items() if self.is_mode_f(x, r))
      tolog.update({
        f'{name} - Num modes': len(unique_modes),
      })
    return tolog

  def maybe_eval_samplelog(self, model, round_num, allXtoR):
    """ Evaluate model using sample log:
        - evaluate all recent round samples, compare to target distribution
        - evaluate all samples, compare to target distribution
        - evaluate topk unique over history
    """
    log_fast = round_num % self.FAST_EVAL_EVERY == 0 and round_num > 0
    log_slow = round_num % self.SLOW_EVAL_EVERY == 0 and round_num > 0
    if not (log_fast or log_slow):
      return
    tolog = dict()
    ds = [self.eval_recent_rounds(model, round_num, allXtoR, log_slow=log_slow),
          self.eval_all_rounds(model, round_num, allXtoR),
          self.eval_topk(log_slow=log_slow),
    ]
    for d in ds:
      tolog.update(d)
      
    if self.args.monitor_real_samples:
      tolog.update(self.log_real_samples(allXtoR))

    for k, v in tolog.items():
      print(f'\t{k}:\t{v}')
    wandb.log(tolog)
    return
  
  def eval_samplelog(self, model, round_num, allXtoR):
    log_slow = True
    tolog = dict()
    ds = [self.eval_recent_rounds(model, round_num, allXtoR, log_slow=log_slow),
          self.eval_all_rounds(model, round_num, allXtoR),
          self.eval_topk(log_slow=log_slow),
    ]
    for d in ds:
      tolog.update(d)

    for k, v in tolog.items():
      print(f'\t{k}:\t{v}')
    wandb.log(tolog)
    return tolog

  """
    Recent rounds
  """
  def eval_recent_rounds(self, model, round_num, allXtoR, log_slow):
    """ Evaluates last k rounds of samples. """
    ok_round = lambda r: round_num - r <= self.NUM_ROUNDS_BACK
    recent_rounds = [r for r in self.sample_log.keys() if ok_round(r)]
    chain = lambda ll: list(itertools.chain(*ll))
    recent_samples = chain(self.sample_log[rd] for rd in recent_rounds
                      if rd in self.sample_log)
    tolog = self.__evaluate_samples(recent_samples, round_num, model, allXtoR,
                                    name='recent', log_slow=log_slow)
    return tolog

  def eval_all_rounds(self, model, round_num, allXtoR):
    all_rounds = list(self.sample_log.keys())
    chain = lambda ll: list(itertools.chain(*ll))
    samples = chain(self.sample_log[rd] for rd in all_rounds)
    tolog = self.__evaluate_samples(samples, round_num, model, allXtoR,
                                    name='all')
    return tolog    

  def __evaluate_samples(self, 
                         batch, 
                         round_num, 
                         model, 
                         allXtoR, 
                         name, 
                         log_slow=False):
    """ Evaluates a batch of samples. """
    if len(batch) == 0:
      print(f'ERROR: no samples in batch. {name}')
    xs = [exp.x for exp in batch]
    scaled_rewards = [exp.r for exp in batch]
    rewards = [self.unnormalize(r) for r in scaled_rewards]

    # basic stats
    tolog = {
      'Active round': round_num,
      # 'logZ': tensor_to_np(model.logZ),
    }

    # Stats on sampled rewards alone
    tolog.update({
      f'{name} - number': len(batch),
      f'{name} - unique fraction': len(set(xs)) / len(xs),
      f'{name} - mean': np.mean(rewards),
      f'{name} - std': np.std(rewards),
      f'{name} - median': np.median(rewards),
      f'{name} - 25th percentile': np.percentile(rewards, 25),
      f'{name} - 75th percentile': np.percentile(rewards, 75),
      f'{name} - 90th percentile': np.percentile(rewards, 90),
      f'{name} - 95th percentile': np.percentile(rewards, 95),
    })

    if self.is_mode_f is not None:
      unique_modes = set(x for x, r in zip(xs, scaled_rewards) if self.is_mode_f(x, r))
      tolog.update({
        f'{name} - Num modes': len(unique_modes),
      })
    
    if name == 'recent':
      tolog.update({
        f'{name} - diversity': diversity(xs, self.dist_func),
      })

    # Stats comparing sampled reward to target reward distribution
    if len(scaled_rewards) > 0:
      mean_diff = np.mean(rewards) - self.target.expected_reward
      rel_error = mean_diff / self.target.expected_reward
      tolog[f'{name} - mean error to target'] = mean_diff
      tolog[f'{name} - mean sq error to target'] = mean_diff**2
      tolog[f'{name} - relative mean error to target'] = rel_error
      tolog[f'{name} - target expected reward'] = self.target.expected_reward

    if self.callback:
      tolog.update(self.callback(xs, rewards, allXtoR))
    return tolog

  """
    Top k samples
  """
  def get_topk(self, k):
    """ Retrieves the top k unique samples in sample_log.

        Returns
        -------
        xs: List[State], top k unique x
        rs: List[float], reward of top k unique x
    """
    chain = lambda ll: list(itertools.chain(*ll))
    all_samples = chain(self.sample_log[rd] for rd in self.sample_log)
    x_to_r = {exp.x: exp.r for exp in all_samples}
    sorted_x = sorted(x_to_r, key=x_to_r.get, reverse=True)
    top_x = sorted_x[:k]
    top_rs = [x_to_r[x] for x in top_x]
    return top_x, top_rs

  def eval_topk(self, k=100, log_slow=False):
    """ Following bioseqgfn paper, evaluate the top k x/r
        over entire training history by performance and diversity.
    """
    xs, rs = self.get_topk(k)

    tolog = {
      'TopK performance': np.mean([self.unnormalize(r) for r in rs]),
      'TopK performance (Scaled)': np.mean(rs),
    }
    if log_slow:
      tolog.update({
        'TopK diversity': diversity(xs, self.dist_func),
      })
    return tolog
