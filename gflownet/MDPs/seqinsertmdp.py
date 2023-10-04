"""
  Build a string by inserting anywhere
"""
import copy
import functools
import numpy as np, pandas as pd
from sklearn.preprocessing import OneHotEncoder

import torch

from .. import network, utils
from ..actor import Actor
from .basemdp import BaseState, BaseMDP

import enum
from dataclasses import dataclass


class SeqInsertState(BaseState):
  """ String state, with insert actions. """
  def __init__(self, content, is_leaf=False):
    self.content = self.canonicalize(content)
    self.is_leaf = is_leaf

  def __repr__(self):
    return f'{self.content}-{self.is_leaf}'

  def __eq__(self, other):
    return self.content_equals(other) and self.is_leaf == other.is_leaf

  def __hash__(self):
    return hash(repr(self))

  def __len__(self):
    if len(self.content) == 0:
      return 0
    return len(self.content)

  def canonicalize(self, content):
    return str(content) if type(content) != str else content

  def content_equals(self, other):
    return self.content == other.content

  def is_member(self, other):
    if self.is_leaf:
      return self.__eq__(other)
    target = other.content
    for c in self.content:
      if c in target:
        target = target[target.index(c) + 1:]
      else:
        return False
    return True

  """
    Modify states
  """
  def _del(self, action):
    """ Delete position in string. pos is 0 indexed.
        Example string: ABCD. Can delete positions 0, 1, 2, 3.
    """
    pos = action.position
    if pos >= len(self.content):
      return None
    new_content = copy.copy(self.content)
    return SeqInsertState(new_content[:pos] + new_content[pos+1:])

  def _insert(self, action):
    """ Insert char at position in string. pos is 0 indexed.
        Insertion position indicates the 0-indexed position
        of the inserted character, after insertion.
        Example string: ABCD. Can insert X at positions 0, 1, 2, 3, 4.
          Position 0: XABCD
          Position 1: AXBCD
          Position 2: ABXCD
          Position 3: ABCXD
          Position 4: ABCDX
    """
    char = action.char
    pos = action.position
    if pos > len(self.content):
      return None
    new_content = copy.copy(self.content)
    return SeqInsertState(new_content[:pos] + char + new_content[pos:])

  def _terminate(self):
    if not self.is_leaf:
      return SeqInsertState(self.content, is_leaf=True)
    else:
      return None
  
  def _unterminate(self):
    if self.is_leaf:
      return SeqInsertState(self.content, is_leaf=False)
    else:
      return None

class SeqInsertActionType(enum.Enum):
  # Forward actions
  Stop = enum.auto()
  InsertChar = enum.auto()
  # Backward actions
  UnStop = enum.auto()
  DelPos = enum.auto()


@dataclass
class SeqInsertAction:
  action: SeqInsertActionType
  char: str = None
  position: int = None


class SeqInsertMDP(BaseMDP):
  """ MDP for building a string by inserting chars.

      Action set is a deterministic function of state.

      Forward actions: [stop, insert A at 0, insert B at 0, ...]
      Reverse actions: [Unstop, del 0, del 1, ..., del N]

      This implementation uses a fixed-size action set, with string
      pushed as leftward as possible.
      Featurization: Denote max sequence length as N. Then we one-hot encode
      for |alphabet|*N features. 
      Inserting or deleting positions beyond current sequence length are 
      invalid actions.

      Cannot contain any CUDA elements: instance is passed
      to ray remote workers for substructure guidance, which need
      access to get_children & is_member.
  """
  def __init__(self, args, alphabet=list('0123'), forced_stop_len=8):
    self.args = args
    self.alphabet = alphabet
    self.alphabet_set = set(self.alphabet)
    self.char_to_idx = {a: i for (i, a) in enumerate(self.alphabet)}
    self.forced_stop_len = forced_stop_len
    
    self.positions = list(range(self.forced_stop_len))
    ins_act = lambda char, position: SeqInsertAction(
        SeqInsertActionType.InsertChar, char=char, position=position
    )
    self.fwd_actions = [SeqInsertAction(SeqInsertActionType.Stop)] + \
                       [ins_act(char=c, position=p)
                        for p in self.positions
                        for c in self.alphabet]
    self.back_actions = [SeqInsertAction(SeqInsertActionType.UnStop)] + \
                        [SeqInsertAction(SeqInsertActionType.DelPos, position=p)
                         for p in self.positions]
    self.state = SeqInsertState
    self.parallelize_policy = True

  def root(self):
    return self.state('')

  @functools.lru_cache(maxsize=None)
  def is_member(self, query, target):
    # Returns true if there is a path from query to target in the MDP
    return query.is_member(target)
  
  """
    Children, parents, and transition.
    Calls BaseMDP functions.
    Uses transition_fwd/back and get_fwd/back_actions.
  """
  @functools.lru_cache(maxsize=None)
  def get_children(self, state):
    return BaseMDP.get_children(self, state)

  @functools.lru_cache(maxsize=None)
  def get_parents(self, state):
    return BaseMDP.get_parents(self, state)

  @functools.lru_cache(maxsize=None)
  def get_unique_children(self, state):
    return BaseMDP.get_unique_children(self, state)

  @functools.lru_cache(maxsize=None)
  def get_unique_parents(self, state):
    return BaseMDP.get_unique_parents(self, state)

  def has_stop(self, state):
    return len(state) == self.forced_stop_len

  def has_forced_stop(self, state):
    return len(state) == self.forced_stop_len

  def transition_fwd(self, state, action):
    """ Applies SeqInsertAction to state.
        Returns State or None (invalid transition). 
    """
    if state.is_leaf:
      return None
    if self.has_forced_stop(state) and action.action != SeqInsertActionType.Stop:
        return None

    if action.action == SeqInsertActionType.Stop:
      if self.has_stop(state):
        return state._terminate()
      else:
        return None

    if action.action == SeqInsertActionType.InsertChar:
      return state._insert(action)

  def transition_back(self, state, action):
    """ Applies SeqInsertAction to state. Returns State or None (invalid transition). 
    """
    if state == self.root():
      return None
    if state.is_leaf and action.action != SeqInsertActionType.UnStop:
      return None

    if action.action == SeqInsertActionType.UnStop:
      if state.is_leaf:
        return state._unterminate()
      else:
        return None

    if action.action == SeqInsertActionType.DelPos:
      return state._del(action)

  """
    Actions
  """
  def get_fwd_actions(self, state):
    """ Gets forward actions from state. Returns List of Actions.

        For many MDPs, this is independent of state. The num actions
        returned must match the policy's output dim. List of actions
        is used to associate policy output scores with states, so it
        must be in a consistent, deterministic order given state.
    """
    return self.fwd_actions

  def get_back_actions(self, state):
    """ Gets backward actions from state. Returns List of Actions.

        For many MDPs, this is independent of state. The num actions
        returned must match the policy's output dim. List of actions
        is used to associate policy output scores with states, so it
        must be in a consistent, deterministic order given state.
    """
    return self.back_actions


"""
  Actor
"""
class SeqInsertActor(Actor):
  """ Holds SeqInsertMDP and GPU elements: featurize & policies. """
  def __init__(self, args, mdp):
    self.args = args
    self.mdp = mdp

    self.alphabet = mdp.alphabet
    self.forced_stop_len = mdp.forced_stop_len

    self.char_to_idx = {a: i for (i, a) in enumerate(self.alphabet)}
    self.onehotencoder = OneHotEncoder(sparse = False)
    self.onehotencoder.fit([[c] for c in self.alphabet])

    self.ft_dim = self.get_feature_dim()

    self.policy_fwd = super().make_policy(self.args.sa_or_ssr, 'forward')
    self.policy_back = super().make_policy(self.args.sa_or_ssr, 'backward')

  # Featurization
  @functools.lru_cache(maxsize=None)
  def featurize(self, state):
    """ fixed dim repr of sequence
        [one hot encoding of variable-length string] + [0 padding]
    """
    if len(state.content) > 0:
      embed = np.concatenate(self.onehotencoder.transform(
          [[c] for c in state.content]
      ))
      num_rem = self.forced_stop_len - len(state.content)
      padding = np.zeros((1, num_rem*len(self.alphabet))).flatten()
      embed = np.concatenate([embed, padding])
    else:
      embed = np.zeros((1, self.forced_stop_len*len(self.alphabet))).flatten()
    return torch.tensor(embed, dtype=torch.float, device = self.args.device)

  def get_feature_dim(self):
    # return self.featurize(state).shape[-1]
    return len(self.alphabet) * self.forced_stop_len

  """
    Networks
  """
  def net_forward_sa(self):
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.ft_dim] + \
        [hid_dim] * n_layers + \
        [len(self.mdp.fwd_actions)]
    )
    return network.StateFeaturizeWrap(net, self.featurize)

  def net_backward_sa(self):
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.ft_dim] + \
        [hid_dim] * n_layers + \
        [len(self.mdp.back_actions)]
    )
    return network.StateFeaturizeWrap(net, self.featurize)
  
  def net_encoder_ssr(self):
    hid_dim = self.args.ssr_encoder_hid_dim
    n_layers = self.args.ssr_encoder_n_layers
    ssr_embed_dim = self.args.ssr_embed_dim
    net =  network.make_mlp(
      [self.ft_dim] + \
      [hid_dim] * n_layers + \
      [ssr_embed_dim]
    )
    return network.StateFeaturizeWrap(net, self.featurize)

  def net_scorer_ssr(self):
    """ [encoding1, encoding2] -> scalar """
    hid_dim = self.args.ssr_scorer_hid_dim
    n_layers = self.args.ssr_scorer_n_layers
    ssr_embed_dim = self.args.ssr_embed_dim
    return network.make_mlp(
        [2*ssr_embed_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )


"""
  Testing
"""
from collections import namedtuple
import random

def randomwalk(mdp):
  node = mdp.root()
  num_children = []
  while not node.is_leaf:
    children = mdp.get_unique_children(node)
    node = random.choice(children)
    num_children.append(len(children))
  return node, num_children


def test():
  Args = namedtuple('Args', ['state_embed_dim', 'device'])
  args = Args(8, 'cpu')
  num_chars = 4
  test_alphabet = [chr(s + 97) for s in range(num_chars)]
  mdp = SeqInsertMDP(args, alphabet = test_alphabet)

  # test_randomwalk(mdp)
  # test_model(mdp)
  # test_parents(mdp)
  from tqdm import tqdm
  ncs = []
  for i in tqdm(range(30)):
    node, num_children = randomwalk(mdp)
    ncs += num_children
  import pandas as pd
  print('Statistics on num. children:')
  print(pd.DataFrame(ncs)[0].describe())
  return

if __name__ == '__main__':
  test()
