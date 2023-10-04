from itertools import chain
import math
import numpy as np
import torch
from torch.distributions import Categorical
import wandb
from pathlib import Path
from tqdm import tqdm

from .basegfn import unroll_trajs
from ..data import Experience
from ..utils import tensor_to_np, batch, pack, unpack
from ..network import make_mlp, StateFeaturizeWrap


class MARS():
    """
        MARS: MARKOV MOLECULAR SAMPLING FOR MULTI-OBJECTIVE DRUG DISCOVERY
        We only need probability distributions for p_add
    """
    
    def __init__(self, args, mdp, actor):
        self.args = args
        self.mdp = mdp
        self.actor = actor
        
        self.policy_fwd = actor.policy_fwd
        self.policy_back = actor.policy_back # not used
        
        self.nets = [self.policy_fwd, self.policy_back]
        for net in self.nets:
            net.to(args.device)

        self.clip_grad_norm_params = [self.policy_fwd.parameters(),
                                      self.policy_back.parameters()]
        
        self.optimizer_back = torch.optim.Adam([
            {
                'params': self.policy_back.parameters(),
                'lr': args.lr_policy
            }])
        self.optimizer_fwd = torch.optim.Adam([
            {
                'params': self.policy_fwd.parameters(),
                'lr': args.lr_policy
            }])
        self.optimizers = [self.optimizer_fwd, self.optimizer_back]
        self.k = math.ceil(mdp.forced_stop_len // 2)
        pass
    
    def fwd_sample(self, batch, epsilon=0.0):
        return self.policy_fwd.sample(batch, epsilon=epsilon)
    
    def fwd_logps_unique(self, batch):
        return self.policy_fwd.logps_unique(batch)
    
    def back_sample(self, batch, epsilon=0.0):
        return self.policy_back.sample(batch, epsilon=epsilon)
    
    def back_logps_unique(self, batch):
        return self.policy_back.logps_unique(batch)
    
    def batch_fwd_sample(self, n, epsilon=0.0, uniform=False, explore_data=None):
        """ Batch samples dataset with n items.

            Parameters
            ----------
            n: int, size of dataset.
            epsilon: Chance in [0, 1] of uniformly sampling a unique child.
            uniform: If true, overrides epsilon to 1.0
            unique: bool, whether all samples should be unique

            Returns
            -------
            dataset: List of [Experience]
        """
        print('Sampling dataset ...')
        if uniform:
            print('Using uniform forward policy on unique children ...')
            epsilon = 1.0
        if explore_data is None:
            incomplete_trajs = [[self.mdp.root()] for _ in range(n)]
            complete_trajs = []
            while len(incomplete_trajs) > 0:
                inp = [t[-1] for t in incomplete_trajs]
                samples = self.fwd_sample(inp, epsilon=epsilon)
                for i, sample in enumerate(samples):
                    incomplete_trajs[i].append(sample)
            
                # Remove complete trajs that hit leaf
                temp_incomplete = []
                for t in incomplete_trajs:
                    if not t[-1].is_leaf:
                        temp_incomplete.append(t)
                    else:
                        complete_trajs.append(t)
                incomplete_trajs = temp_incomplete

            # convert trajs to exps
            list_exps = []
            for traj in complete_trajs:
                x = traj[-1]
                r = self.mdp.reward(x)
                # prevent NaN
                exp = Experience(traj=traj, x=x, r=r,
                    logr=torch.nan_to_num(torch.log(torch.tensor(r, dtype=torch.float32)).to(device=self.args.device), neginf=-100.0))
                list_exps.append(exp)
            return list_exps
        else:
            k_backward_complete_trajs = [[exp.traj[-1]] for exp in explore_data]
            for _ in range(self.k):
                inp = [t[0] for t in k_backward_complete_trajs]
                samples = self.back_sample(inp)
                for i, sample in enumerate(samples):
                    k_backward_complete_trajs[i].insert(0, sample)
            # Do Forward k steps
            k_forward_complete_trajs = [[t[0]] for t in k_backward_complete_trajs]
            for _ in range(self.k):
                inp = [t[-1] for t in k_forward_complete_trajs]
                samples = self.fwd_sample(inp)
                for i, sample in enumerate(samples):
                    k_forward_complete_trajs[i].append(sample) 
                    
            new_explore_data = []
            accepted_data = []
            for i, traj in enumerate(k_forward_complete_trajs):
                x = traj[-1]
                r = self.mdp.reward(x)
                exp = Experience(traj=traj, x=x, r=r,
                                 logr=torch.log(torch.tensor(r, dtype=torch.float32)).to(device=self.args.device))
                new_explore_data.append(exp)
                if r > explore_data[i].r:
                    accepted_data.append((explore_data[i], exp)) 
            return new_explore_data, accepted_data 
            
    def train(self, batch, log = True):
        batch_loss = self.batch_loss_mars(batch)
        
        for opt in self.optimizers:
            opt.zero_grad()
        batch_loss.backward()
        
        for param_set in self.clip_grad_norm_params:
            # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
        for opt in self.optimizers:
            opt.step()
        
        if log:
            batch_loss = tensor_to_np(batch_loss)
            print(f'MARS training:', batch_loss)
            wandb.log({'MARS loss': batch_loss})
            return
        
    def batch_loss_mars(self, batch):
        x_trajs = [exp[0].traj for exp in batch]
        x_prime_trajs = [exp[1].traj for exp in batch]
        
        fwd_states, back_states, unroll_idxs = unroll_trajs(x_trajs)
        fwd_states_prime, back_states_prime, unroll_idxs_prime = unroll_trajs(x_prime_trajs)
        
        states_to_back_logps = self.back_logps_unique(back_states)
        back_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_back_logps, fwd_states)]
        
        states_to_fwd_logps = self.fwd_logps_unique(fwd_states_prime)
        fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_fwd_logps, back_states_prime)]
        
        log_probs = torch.zeros((len(batch), self.k * 2)).to(self.args.device)
        for i, (start, end) in unroll_idxs.items():
            for j in range(self.k):
                log_probs[i, j] = back_logp_chosen[start + j]
        for i, (start, end) in unroll_idxs_prime.items():
            for j in range(self.k):
                log_probs[i, self.k + j] = fwd_logp_chosen[end - 1 - self.k + j]
                
        log_probs = self.clip_policy_logits(log_probs)
        log_probs = torch.nan_to_num(log_probs, neginf=self.args.clip_policy_logit_min)
        
        loss = -log_probs.sum(dim=1)
        loss = torch.clamp(loss, max=5000)
        mean_loss = torch.mean(loss)
        return mean_loss
    
    def save_params(self, file):
        print('Saving checkpoint model ...')
        Path('/'.join(file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy_fwd': self.policy_fwd.state_dict(),
            'policy_back': self.policy_back.state_dict(),
            }, file)
        return

    def load_for_eval_from_checkpoint(self, file):
        print(f'Loading checkpoint model ...')
        checkpoint = torch.load(file)
        self.policy_fwd.load_state_dict(checkpoint['policy_fwd'])
        self.policy_back.load_state_dict(checkpoint['policy_back'])
        for net in self.nets:
            net.eval()
        return

    def clip_policy_logits(self, scores):
        return torch.clip(scores, min=self.args.clip_policy_logit_min,
                                  max=self.args.clip_policy_logit_max)
        
        

