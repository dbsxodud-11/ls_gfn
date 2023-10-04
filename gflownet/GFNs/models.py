from itertools import chain
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
from torch_scatter import scatter, scatter_sum
import wandb

from .basegfn import BaseTBGFlowNet, tensor_to_np
from .advantage_actor_critic import A2C
from .ppo import PPO
from .mars import MARS
from .soft_q_learning import SoftQLearning


class Empty(BaseTBGFlowNet):
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
  
  def train(self, batch):
    return


class TBGFN(BaseTBGFlowNet):
  """ Trajectory balance GFN. Learns forward and backward policy. """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: TBGFN')

  def train(self, batch):
    return self.train_tb(batch)


class SubTBGFN(BaseTBGFlowNet):
  """ SubTB (lambda) """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: SubTBGFN')
    
  def init_subtb(self):
    r"""Precompute all possible subtrajectory indices that we will use for computing the loss:
    \sum_{m=1}^{T-1} \sum_{n=m+1}^T
        \log( \frac{F(s_m) \prod_{i=m}^{n-1} P_F(s_{i+1}|s_i)}
                    {F(s_n) \prod_{i=m}^{n-1} P_B(s_i|s_{i+1})} )^2
    """
    self.subtb_max_len = self.mdp.forced_stop_len + 2
    ar = torch.arange(self.subtb_max_len, device=self.args.device)
    # This will contain a sequence of repeated ranges, e.g.
    # tidx[4] == tensor([0, 0, 1, 0, 1, 2, 0, 1, 2, 3])
    tidx = [torch.tril_indices(i, i, device=self.args.device)[1] for i in range(self.subtb_max_len)]
    # We need two sets of indices, the first are the source indices, the second the destination
    # indices. We precompute such indices for every possible trajectory length.

    # The source indices indicate where we index P_F and P_B, e.g. for m=3 and n=6 we'd need the
    # sequence [3,4,5]. We'll simply concatenate all sequences, for every m and n (because we're
    # computing \sum_{m=1}^{T-1} \sum_{n=m+1}^T), and get [0, 0,1, 0,1,2, ..., 3,4,5, ...].

    # The destination indices indicate the index of the subsequence the source indices correspond to.
    # This is used in the scatter sum to compute \log\prod_{i=m}^{n-1}. For the above example, we'd get
    # [0, 1,1, 2,2,2, ..., 17,17,17, ...]

    # And so with these indices, for example for m=0, n=3, the forward probability
    # of that subtrajectory gets computed as result[2] = P_F[0] + P_F[1] + P_F[2].

    self.precomp = [
        (
            torch.cat([i + tidx[T - i] for i in range(T)]),
            torch.cat(
                [ar[: T - i].repeat_interleave(ar[: T - i] + 1) + ar[T - i + 1 : T + 1].sum() for i in range(T)]
            ),
        )
        for T in range(1, self.subtb_max_len)
    ]
    self.lamda = self.args.lamda
    
  def train(self, batch):
    self.init_subtb()
    return self.train_subtb(batch)
  
  def train_subtb(self, batch, log = True):
    """ Step on trajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_sub_trajectory_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return
  
  def batch_loss_sub_trajectory_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    log_F_s[:, 0] = self.logZ.repeat(len(batch))
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()
      
    total_loss = torch.zeros(len(batch), device=self.args.device)
    ar = torch.arange(self.subtb_max_len)
    for i in range(len(batch)):
      # Luckily, we have a fixed terminal length
      idces, dests = self.precomp[-1]
      P_F_sums = scatter_sum(log_pf_actions[i, idces], dests)
      P_B_sums = scatter_sum(log_pb_actions[i, idces], dests)
      F_start = scatter_sum(log_F_s[i, idces], dests)
      F_end = scatter_sum(log_F_next_s[i, idces], dests)

      weight = torch.pow(self.lamda, torch.bincount(dests) - 1)
      total_loss[i] = (weight * (F_start - F_end + P_F_sums - P_B_sums).pow(2)).sum() / torch.sum(weight)
    losses = torch.clamp(total_loss, max=5000)
    mean_loss = torch.mean(losses)
    return mean_loss

class DBGFN(BaseTBGFlowNet):
  """ Detailed balance GFN """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: DBGFN')
    
  def train(self, batch):
    return self.train_db(batch)
  
  def train_db(self, batch, log = True):
    """ Step on detailed balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_detailed_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    for opt in self.optimizers:
      opt.step()
      
    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return
  
  def batch_loss_detailed_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    log_F_s, log_pf_actions = self.batch_traj_fwd_logp_unroll(batch)
    log_F_next_s, log_pb_actions = self.batch_traj_back_logp_unroll(batch)
    
    for i, exp in enumerate(batch):
      log_F_next_s[i, -1] = exp.logr.clone().detach()

    losses = (log_F_s + log_pf_actions - log_F_next_s - log_pb_actions).pow(2).sum(axis=1)
    losses = torch.clamp(losses, max=5000)
    mean_loss = torch.mean(losses)
    return mean_loss

class MaxEntGFN(BaseTBGFlowNet):
  """ Maximum Entropy GFlowNet with fixed uniform backward policy. 

      Methods back_logps_unique, back_sample override parent BaseTBGFlowNet
      methods, which simply call the backward policy's functions.    
  """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: MaxEntGFN')

  def train(self, batch):
    return self.train_tb(batch)

  def back_logps_unique(self, batch):
    """ Uniform distribution over parents.

        Other idea - just call parent back_logps_unique, then replace
        predicted logps.
        see policy.py : logps_unique(batch)

        Output logps of unique children/parents.

        Typical logic flow (example for getting children)
        1. Call network on state - returns high-dim actions
        2. Translate actions into list of states - not unique
        3. Filter invalid child states
        4. Reduce states to unique, using hash property of states.
           Need to add predicted probabilities.
        5. Normalize probs to sum to 1

        Input: List of [State], n items
        Returns
        -------
        logps: n-length List of torch.tensor of logp.
            Each tensor can have different length.
        states: List of List of [State]; must be unique.
            Each list can have different length.
    """
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_dicts = []
    for state in batch:
      parents = self.mdp.get_unique_parents(state)
      logps = np.log([1/len(parents) for parent in parents])

      state_to_logp = {parent: logp for parent, logp in zip(parents, logps)}
      batch_dicts.append(state_to_logp)
    return batch_dicts if batched else batch_dicts[0]

  def back_sample(self, batch):
    """ Uniformly samples a parent.

        Typical logic flow skips some steps in logps_unique.
        1. Call network on state - return high-dim actions
        2. Translate actions into list of states - not unique
        3. Filter invalid child states
        4. Skipped - no need to reduce states to unique.
        5. Normalize probs to sum to 1
        Return sample

        Input: batch, List of [State]
        Output: List of [State]
    """
    batched = bool(type(batch) is list)
    if not batched:
      batch = [batch]

    batch_samples = []
    for state in batch:
      sample = np.random.choice(self.mdp.get_unique_parents(state))
      batch_samples.append(sample)
    return batch_samples if batched else batch_samples[0]


class SubstructureGFN(BaseTBGFlowNet):
  """ Substructure GFN. Learns with guided trajectory balance. """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: Substructure GFN')

  def train(self, batch):
    return self.train_substructure(batch)

  def train_substructure(self, batch, log = True):
    """ Guided trajectory balance for substructure GFN.
        1. Update back policy to approximate guide,
        2. Update forward policy to match back policy with TB.
        
        Batch: List of [Experience]

        Uses 1 pass for fwd and back net.
    """
    fwd_chain = self.batch_traj_fwd_logp(batch)
    back_chain = self.batch_traj_back_logp(batch)

    # 1. Obtain back policy loss
    logp_guide = torch.stack([exp.logp_guide for exp in batch])
    back_losses = torch.square(back_chain - logp_guide)
    back_losses = torch.clamp(back_losses, max=10**2)
    mean_back_loss = torch.mean(back_losses)

    # 2. Obtain TB loss with target: mix back_chain with logp_guide
    targets = []
    for i, exp in enumerate(batch):
      if exp.logp_guide is not None:
        w = self.args.target_mix_backpolicy_weight
        target = w * back_chain[i].detach() + (1 - w) * (exp.logp_guide + exp.logr)
      else:
        target = back_chain[i].detach()
      targets.append(target)
    targets = torch.stack(targets)

    tb_losses = torch.square(fwd_chain - targets)
    tb_losses = torch.clamp(tb_losses, max=10**2)
    loss_tb = torch.mean(tb_losses)

    # 1. Update back policy on back loss
    self.optimizer_back.zero_grad()
    loss_step1 = mean_back_loss
    loss_step1.backward()
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    self.optimizer_back.step()
    if log:
      loss_step1 = tensor_to_np(loss_step1)
      print(f'Back training:', loss_step1)

    # 2. Update fwd policy on TB loss
    self.optimizer_fwdZ.zero_grad()
    loss_tb.backward()
    for param_set in self.clip_grad_norm_params:
      # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
    self.optimizer_fwdZ.step()
    self.clamp_logZ()
    if log:
      loss_tb = tensor_to_np(loss_tb)
      print(f'Fwd training:', loss_tb)

    if log:
      logZ = tensor_to_np(self.logZ)
      # print(f'{logZ=}')
      print(f'logZ={logZ}')
      wandb.log({
        'Sub back loss': loss_step1,
        'Sub fwdZ loss': loss_tb,
        'Sub logZ': logZ,
      })
    return


def make_model(args, mdp, actor):
  """ Constructs MaxEnt / TB / Sub GFN. """
  if args.model == 'maxent':
    model = MaxEntGFN(args, mdp, actor)
  elif args.model == 'tb':
    model = TBGFN(args, mdp, actor)
  elif args.model == "subtb":
    model = SubTBGFN(args, mdp, actor)
  elif args.model == 'db':
    model = DBGFN(args, mdp, actor)
  elif args.model == 'gtb':
    model = SubstructureGFN(args, mdp, actor)
  elif args.model == 'random':
    args.explore_epsilon = 1.0
    args.num_offline_batches_per_round = 0
    model = Empty(args, mdp, actor)
  elif args.model == 'a2c':
    model = A2C(args, mdp, actor)
  elif args.model == "ppo":
    model = PPO(args, mdp, actor)
  elif args.model == 'sql':
    model = SoftQLearning(args, mdp, actor)
  elif args.model == 'mars':
    model = MARS(args, mdp, actor)
  return model
