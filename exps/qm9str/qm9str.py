"""
  qm9 as string
"""
import pickle, functools
import numpy as np
from tqdm import tqdm
import torch

import gflownet.trainers as trainers
from gflownet.MDPs import molstrmdp
from gflownet.monitor import TargetRewardDistribution, Monitor, diversity
from gflownet.GFNs import models

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import FingerprintSimilarity


class QM9stringMDP(molstrmdp.MolStrMDP):
  def __init__(self, args):
    super().__init__(args)
    self.args = args

    x_to_r_file = args.x_to_r_file
    mode_info_file = args.mode_info_file

    # Read from file
    print(f'Loading data ...')
    with open(x_to_r_file, 'rb') as f:
      self.oracle = pickle.load(f)
    
    # scale rewards
    py = np.array(list(self.oracle.values()))

    self.SCALE_REWARD_MIN = args.scale_reward_min
    self.SCALE_REWARD_MAX = args.scale_reward_max
    self.REWARD_EXP = args.beta
    self.REWARD_MAX = max(py)

    py = np.maximum(py, self.SCALE_REWARD_MIN)
    py = py ** self.REWARD_EXP
    self.scale = self.SCALE_REWARD_MAX / max(py)
    py = py * self.scale

    self.scaled_oracle = {x: y for x, y in zip(self.oracle.keys(), py) if y > 0}
    assert min(self.scaled_oracle.values()) > 0

    # define modes as top % of xhashes and diversity metrics
    with open(mode_info_file, 'rb') as f:
      mode_info = pickle.load(f)
    if args.mode_metric == 'default':
      self.modes = mode_info['modes']
    elif args.mode_metric == 'div_threshold_05':
      self.modes = mode_info['modes_div_threshold_05']
    elif args.mode_metric == 'div_threshold_075':
      self.modes = mode_info['modes_div_threshold_075']
    else:
      raise NotImplementedError
    print(f"Found num modes: {len(self.modes)}")

  # Core
  def reward(self, x):
    assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
    return self.scaled_oracle[x.content]

  def is_mode(self, x, r):
    return x.content in self.modes
  
  def unnormalize(self, r):
      r = r / self.scale
      r = r ** (1 / self.REWARD_EXP)
      r = r / self.REWARD_MAX
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
    rs_all = list(self.scaled_oracle.values())
    target.init_from_base_rewards(rs_all)
    return Monitor(self.args, target, dist_func=self.dist_states,
                   is_mode_f=self.is_mode,
                   unnormalize=self.unnormalize)


def main(args):
  print('Running experiment qm9str ...')
  mdp = QM9stringMDP(args)
  actor = molstrmdp.MolStrActor(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()
  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return

def eval(args):
  print('Running evaluation qm9str ...')
  mdp = QM9stringMDP(args)
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
  print('Count number of modes qm9str ...')
  
  # load model checkpoint
  ckpt_path = args.saved_models_dir + args.run_name
  with open(ckpt_path + '/' + f"final_sample.pkl", "rb") as f:
    generated_samples = pickle.load(f)
    
  with open(args.mode_info_file, "rb") as f:
    mode_info = pickle.load(f)
    
  unique_samples = set()
  batch_size = args.num_samples_per_online_batch
  number_of_modes = {k: np.zeros((len(generated_samples) // batch_size, )) for k in mode_info}
  with tqdm(total=len(generated_samples)) as pbar:
    for i in range(0, len(generated_samples), batch_size):
      for exp in generated_samples[i: i+batch_size]:
        if exp.x not in unique_samples:
          if exp.x.content in mode_info["modes_div_threshold_075"]:
            number_of_modes["modes_div_threshold_075"][i // batch_size] += 1
          if exp.x.content in mode_info["modes_div_threshold_05"]:
            number_of_modes["modes_div_threshold_05"][i // batch_size] += 1
          if exp.x.content in mode_info["modes"]:
            number_of_modes["modes"][i // batch_size] += 1
        unique_samples.add(exp.x)
      pbar.update(batch_size)
      pbar.set_postfix(number_of_modes=np.sum(number_of_modes["modes"]))
  print(np.sum(number_of_modes["modes"]))
  np.savez_compressed(ckpt_path + '/' + f'number_of_modes_updated.npz', modes=number_of_modes["modes"],
                                                                        modes_div_threshold_05=number_of_modes["modes_div_threshold_05"],
                                                                        modes_div_threshold_075=number_of_modes["modes_div_threshold_075"],) 
