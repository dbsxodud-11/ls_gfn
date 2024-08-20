"""
  seh as string
"""
import os
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
    
    mode_info_file = args.mode_info_file
    assert args.blocks_file == 'datasets/sehstr/block_18.json', 'ERROR - x_to_r and rewards are designed for block_18.json'

    self.proxy_model = gbr_proxy.sEH_GBR_Proxy(args)

    with open('datasets/sehstr/sehstr_gbtr_allpreds.pkl', 'rb') as f:
      self.rewards = pickle.load(f)

    # scale rewards
    py = np.array(list(self.rewards))

    self.SCALE_REWARD_MIN = args.scale_reward_min
    self.SCALE_REWARD_MAX = args.scale_reward_max
    self.REWARD_EXP = args.beta
    self.REWARD_MAX = max(py)

    py = np.maximum(py, self.SCALE_REWARD_MIN)
    py = py ** self.REWARD_EXP
    self.scale = self.SCALE_REWARD_MAX / max(py)
    py = py * self.scale

    self.scaled_rewards = py

    # define modes as top % of xhashes.
    if args.mode_metric != "threshold":
      with open(mode_info_file, 'rb') as f:
        self.modes = pickle.load(f)
      print(f"Found num modes: {len(self.modes)}")
    else:
      mode_percentile = args.mode_percentile
      self.mode_r_threshold = np.percentile(py, 100*(1-mode_percentile))

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
    if self.args.mode_metric == "threshold":
      return r >= self.mode_r_threshold
    else:
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
    target.init_from_base_rewards(self.scaled_rewards)
    return Monitor(self.args, target, dist_func=self.dist_states,
                   is_mode_f=self.is_mode,
                   unnormalize=self.unnormalize)

  def reduce_storage(self):
    del self.rewards
    del self.scaled_rewards


def main(args):
  print('Running experiment sehstr ...')
  mdp = SEHstringMDP(args)
  actor = molstrmdp.MolStrActor(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  mdp.reduce_storage()

  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return

def eval(args):
  print('Running evaluation sehstr ...')
  mdp = SEHstringMDP(args)
  actor = molstrmdp.MolStrActor(args, mdp)
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
  print('Running evaluation sehstr ...')
  mdp = SEHstringMDP(args)
  
  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  with open(ckpt_path + '/' + f"final_sample.pkl", "rb") as f:
    generated_samples = pickle.load(f)
    
  unique_modes = set()
  batch_size = args.num_samples_per_online_batch
  number_of_modes = np.zeros((len(generated_samples) // batch_size, ))
  with tqdm(total=len(generated_samples)) as pbar:
    for i in range(0, len(generated_samples), batch_size):
      for exp in generated_samples[i: i+batch_size]:
        if mdp.is_mode(exp.x, exp.r) and exp.x.content not in unique_modes:
          unique_modes.add(exp.x.content)
          number_of_modes[i // batch_size] += 1
      pbar.update(batch_size)
      pbar.set_postfix(number_of_modes=np.sum(number_of_modes))
  print(np.sum(number_of_modes))
  np.savez_compressed(ckpt_path + '/' + f'number_of_modes.npz', modes=number_of_modes) 

