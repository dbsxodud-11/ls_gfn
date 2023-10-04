'''
    GFP
    Transformer Proxy
    Start from scratch
'''

import copy, pickle, functools
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
      self.rna_task = args.rna_task
      self.rna_length = args.rna_length
      
      self.mode_info_file = args.mode_info_file + f"L{self.rna_length}_RNA{self.rna_task}/mode_info.pkl"
      self.monitor_info_file = args.monitor_info_file + f"L{self.rna_length}_RNA{self.rna_task}/monitor_info.pkl"
      
      print(f'Loading data ...')
      problem = flexs.landscapes.rna.registry()[f'L{self.rna_length}_RNA{self.rna_task}']
      self.oracle = flexs.landscapes.RNABinding(**problem['params'])
      print(problem)
      
      # define modes as top % of xhashes and distance metrics
      with open(self.mode_info_file, 'rb') as f:
        mode_info = pickle.load(f)
      if args.mode_metric == 'default':
        self.modes = mode_info['modes']
      elif args.mode_metric == 'hamming_ball1':
        self.modes = mode_info['modes_hamming_ball1']
      elif args.mode_metric == 'hamming_ball2':
        self.modes = mode_info['modes_hamming_ball2']
      else:
        raise NotImplementedError
      print(f"Found num modes: {len(self.modes)}")

      py = np.concatenate([self.oracle.get_fitness([x]) for x in tqdm(self.modes)])
      self.SCALE_REWARD_MIN = args.scale_reward_min
      self.SCALE_REWARD_MAX = args.scale_reward_max
      self.REWARD_EXP = args.beta
      self.REWARD_MAX = max(py)
      
      py = np.maximum(py, self.SCALE_REWARD_MIN)
      py = py ** self.REWARD_EXP
      self.scale = self.SCALE_REWARD_MAX / max(py)
      py = py * self.scale

    # Core
    def reward(self, x):
      assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
      r = self.oracle.get_fitness([x.content]).item()
      
      r = np.maximum(r, self.SCALE_REWARD_MIN)
      r = r ** self.REWARD_EXP
      r = r * self.scale
      return r

    def is_mode(self, x, r):
      return x.content in self.modes
    
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
      # target.init_from_base_rewards(self.scaled_rewards)
      target.init_from_file(self.monitor_info_file)
      return Monitor(self.args, target, dist_func=self.dist_func,
                     is_mode_f=self.is_mode, callback=self.add_monitor,
                     unnormalize=self.unnormalize)

    def add_monitor(self, xs, rs, allXtoR):
      """ Reimplement scoring with oracle, not unscaled oracle (used as R). """
      tolog = dict()
      return tolog

  return RNAMDP(args)


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

def eval(args):
  print('Running evaluation RNA ...')
  
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

  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  if args.ckpt == -1: # final
    model.load_for_eval_from_checkpoint(ckpt_path + '/' + 'final.pth')
  else:
    model.load_for_eval_from_checkpoint(ckpt_path + '/' + f'round_{args.ckpt}.pth')
    
  # evaluate
  with torch.no_grad():
    eval_samples = model.batch_fwd_sample(args.eval_num_samples, epsilon=0.0)
    
  allXtoR = dict()
  for exp in eval_samples:
    if exp.x not in allXtoR:
      allXtoR[exp.x] = exp.r 
  
  round_num = 1
  monitor.log_samples(round_num, eval_samples)
  log = monitor.eval_samplelog(model, round_num, allXtoR)

  # save results
  result_path = args.saved_models_dir + args.run_name
  log_path = args.saved_models_dir + args.run_name
  if args.ckpt == -1: # final
    result_path += '/' + 'final_eval_samples.pkl'
    log_path += '/' + 'final_eval_log.pkl'
  else:
    result_path += '/' + f'round_{args.ckpt}_eval_samples.pkl'
    log_path += '/' + f'round_{args.ckpt}_eval_log.pkl'
    
  with open(result_path, "wb") as f:
    pickle.dump(eval_samples, f)
    
  with open(log_path, "wb") as f:
    pickle.dump(log, f)

def number_of_modes(args):
  print('Running evaluation RNA ...')

  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  with open(ckpt_path + '/' + f"final_sample.pkl", "rb") as f:
    generated_samples = pickle.load(f)
    
  mode_info_file = args.mode_info_file + f"L{args.rna_length}_RNA{args.rna_task}/mode_info.pkl"
  with open(mode_info_file, "rb") as f:
    mode_info = pickle.load(f)
  
  unique_samples = set()
  batch_size = args.num_samples_per_online_batch
  number_of_modes = {k: np.zeros((len(generated_samples) // batch_size, )) for k in mode_info}
  with tqdm(total=len(generated_samples)) as pbar:
    for i in range(0, len(generated_samples), batch_size):
      for exp in generated_samples[i: i+batch_size]:
        if exp.x not in unique_samples:      
          if exp.x.content in mode_info["modes"]:
            number_of_modes["modes"][i // batch_size] += 1
          if exp.x.content in mode_info["modes_hamming_ball1"]:
            number_of_modes["modes_hamming_ball1"][i // batch_size] += 1
          if exp.x.content in mode_info["modes_hamming_ball2"]:
            number_of_modes["modes_hamming_ball2"][i // batch_size] += 1
          unique_samples.add(exp.x)
      pbar.update(batch_size)
      pbar.set_postfix(number_of_modes=np.sum(number_of_modes["modes"]))
  print(np.sum(number_of_modes["modes"]))
  np.savez_compressed(ckpt_path + '/' + f'number_of_modes_updated.npz', modes=number_of_modes["modes"],
                                                                        modes_hamming_ball1=number_of_modes["modes_hamming_ball1"],
                                                                        modes_hamming_ball2=number_of_modes["modes_hamming_ball2"])
        
        
