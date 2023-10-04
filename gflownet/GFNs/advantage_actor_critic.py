from itertools import chain
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


class A2C():
    """
        Advantage Actor Critic
        Actor: SSR style
        Critic: SA style
    """
    
    def __init__(self, args, mdp, actor):
        self.args = args
        self.mdp = mdp
        self.actor = actor
        
        self.policy = actor.policy_fwd
        self.policy_back = actor.policy_back # not used
        
        hid_dim = self.args.sa_hid_dim
        n_layers = self.args.sa_n_layers
        net = make_mlp(
            [self.actor.ft_dim] + \
            [hid_dim] * n_layers + \
            [1]
        )
        self.critic = StateFeaturizeWrap(net, self.actor.featurize)
        self.critic.to(args.device)
        
        self.nets = [self.policy, self.critic]
        for net in self.nets:
            net.to(self.args.device)
            
        self.clip_grad_norm_params = [self.policy.parameters(),
                                      self.critic.parameters()]
        
        self.optimizer = torch.optim.Adam([
            {
                'params': self.policy.parameters(),
                'lr': args.lr_policy
            }, {
                'params': self.critic.parameters(),
                'lr': args.lr_critic
            }
        ])
    
    def fwd_sample(self, batch, epsilon=0.0):
        return self.policy.sample(batch, epsilon=epsilon)
    
    def fwd_logps_unique(self, batch):
        return self.policy.logps_unique(batch)
    
    def batch_fwd_sample(self, n, epsilon=0.0, uniform=False):
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
        
    def train(self, batch, log = True):
        batch_loss = self.batch_loss_advantage_actor_critic(batch)
        
        self.optimizer.zero_grad()
        batch_loss.backward()
        
        for param_set in self.clip_grad_norm_params:
            # torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm)
        self.optimizer.step()
        
        if log:
            batch_loss = tensor_to_np(batch_loss)
            print(f'A2C training:', batch_loss)
            wandb.log({'A2C loss': batch_loss})
            return
        
    def batch_loss_advantage_actor_critic(self, batch):
        trajs = [exp.traj for exp in batch]
        fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)
        
        states_to_logps = self.fwd_logps_unique(fwd_states)
        fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]
        log_probs = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        for i, log_prob in enumerate(fwd_logp_chosen):
            log_probs[i] = log_prob
        log_probs = self.clip_policy_logits(log_probs)
        log_probs = torch.nan_to_num(log_probs, neginf=self.args.clip_policy_logit_min)
        
        V = self.critic(fwd_states)
        # The return is the terminal reward everywhere, we're using gamma==1
        G = torch.FloatTensor([exp.r for exp in batch]).repeat_interleave(self.mdp.forced_stop_len + 1).to(self.args.device)
        A = G - V
        
        V_loss = A.pow(2)
        pol_objective = (log_probs * A.detach())
        entropy = torch.zeros((len(fwd_logp_chosen), )).to(self.args.device)
        for i, s2lp in enumerate(states_to_logps):
            for state, logp in s2lp.items():
                entropy[i] = -torch.sum(torch.exp(logp) * logp)
        pol_objective = pol_objective + self.args.entropy_coef * entropy
        pol_loss = -pol_objective
        
        loss = V_loss + pol_loss
        loss = torch.clamp(loss, max=5000)
        mean_loss = torch.mean(loss)
        return mean_loss
    
    def save_params(self, file):
        print('Saving checkpoint model ...')
        Path('/'.join(file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
        torch.save({
            'policy':   self.policy.state_dict(),
            'critic':  self.critic.state_dict(),
            }, file)
        return

    def load_for_eval_from_checkpoint(self, file):
        print(f'Loading checkpoint model ...')
        checkpoint = torch.load(file)
        self.policy.load_state_dict(checkpoint['policy'])
        self.critic.load_state_dict(checkpoint['critic'])
        for net in self.nets:
            net.eval()
        return

    def clip_policy_logits(self, scores):
        return torch.clip(scores, min=self.args.clip_policy_logit_min,
                                  max=self.args.clip_policy_logit_max)
        
        

