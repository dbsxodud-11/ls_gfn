"""

"""
from collections import defaultdict, Counter
import copy
import functools

import torch

from .. import network, utils
from ..actor import Actor
from .basemdp import BaseState, BaseMDP

import enum
from dataclasses import dataclass


class BagState(BaseState):
  """ Bag / Multiset state, as Counter (dict) mapping {char: count}. """
  def __init__(self, content, is_leaf=False):
    """ content: Counter (dict) mapping {char: count}. """
    self.content = self.canonicalize(content)
    self.is_leaf = is_leaf

  def __repr__(self):
    sorted_chars = sorted(list(self.content.keys()))
    sortedbag = ''.join([k*self.content[k] for k in sorted_chars])
    return f'{sortedbag}-{self.is_leaf}'

  def __eq__(self, other):
    return self.content_equals(other) and self.is_leaf == other.is_leaf

  def __hash__(self):
    return hash(repr(self))

  def __len__(self):
    if len(self.content) == 0:
      return 0
    return sum(self.content.values())

  def max_group_size(self):
    if len(self.content) == 0:
      return 0
    return max(self.content.values())

  def canonicalize(self, content):
    return Counter(content) if type(content) != Counter else content

  def content_equals(self, other):
    for k, v in self.content.items():
      if v > 0:
        if k not in other.content:
          return False
        if other.content[k] != v:
          return False
    return True

  def is_member(self, other):
    if self.is_leaf:
      return self.__eq__(other)
    for k, v in self.content.items():
      if v > 0:
        if k not in other.content:
          return False
        if other.content[k] < v:
          return False
    return True

  """
    Modifying state
  """
  def _del(self, action):
    """ Construct new BagState, given BagAction.
        Return None if invalid action.
    """
    if self.content[action.char] <= 0:
      return None
    new_content = copy.copy(self.content)
    new_content[action.char] = max(0, new_content[action.char] - 1)
    return BagState(new_content)

  def _add(self, action):
    """ Construct new BagState, given BagAction.
        Return None if invalid action.
    """
    new_content = copy.copy(self.content)
    new_content[action.char] += 1
    return BagState(new_content)

  def _terminate(self):
    if not self.is_leaf:
      return BagState(self.content, is_leaf=True)
    else:
      return None
  
  def _unterminate(self):
    if self.is_leaf:
      return BagState(self.content, is_leaf=False)
    else:
      return None

class BagActionType(enum.Enum):
  # Forward actions
  Stop = enum.auto()
  AddChar = enum.auto()
  # Backward actions
  UnStop = enum.auto()
  RemoveChar = enum.auto()


@dataclass
class BagAction:
  action: BagActionType
  char: str = None


class BagMDP(BaseMDP):
  """ MDP for building a bag or multiset, comprised of an alphabet 'ABCDEFG'.

      Action set is fixed and not a function of state.

      Forward actions: [stop, add A, add B, ..., add G]
      Reverse actions: [Unstop, remove A, remove B, ..., remove G]

      Cannot contain any CUDA elements: instance is passed
      to ray remote workers for substructure guidance, which need
      access to get_children & is_member.
  """
  def __init__(self, args, alphabet = list('ABCDEFG')):
    self.args = args
    self.alphabet = alphabet
    self.alphabet_set = set(self.alphabet)
    self.substruct_size = 4
    self.forced_stop_len = 7

    self.fwd_actions = [BagAction(BagActionType.Stop)] + \
                       [BagAction(BagActionType.AddChar, c)
                        for c in self.alphabet]
    self.back_actions = [BagAction(BagActionType.UnStop)] + \
                        [BagAction(BagActionType.RemoveChar, c)
                         for c in self.alphabet]
    self.state = BagState
    self.parallelize_policy = False

  def root(self):
    return self.state([])

  @functools.lru_cache(maxsize=None)
  def is_member(self, query, target):
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
    """ Applies BagAction to state. Returns State or None (invalid transition). 
        
        Action Types: Stop, AddChar
    """
    if state.is_leaf:
      return None
    if self.has_forced_stop(state) and action.action != BagActionType.Stop:
      return None

    if action.action == BagActionType.Stop:
      if self.has_stop(state):
        return state._terminate()
      else:
        return None

    if action.action == BagActionType.AddChar:
      return state._add(action)
    
  def transition_back(self, state, action):
    """ Applies BagAction to state. Returns State or None (invalid transition). 

        Action types: UnStop, RemoveChar 
    """
    if state == self.root():
      return None
    if state.is_leaf and action.action != BagActionType.UnStop:
      return None

    if action.action == BagActionType.UnStop:
      if state.is_leaf:
        return state._unterminate()
      else:
        return None

    if action.action == BagActionType.RemoveChar:
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
class BagActor(Actor):
  """ Holds BagMDP and GPU elements: featurize & policies. """
  def __init__(self, args, mdp):
    self.args = args
    self.mdp = mdp

    self.alphabet = mdp.alphabet
    self.char_to_idx = {a: i for (i, a) in enumerate(self.alphabet)}

    self.ft_dim = len(self.alphabet) + 2

    self.policy_fwd = super().make_policy(self.args.sa_or_ssr, 'forward')
    self.policy_back = super().make_policy(self.args.sa_or_ssr, 'backward')

  @functools.lru_cache(maxsize=None)
  def featurize(self, state):
    """ Featurize BagState.

        Features
        - first len(alphabet) indices: count of that symbol
        - max count of symbol
        - (bool) max is >= substruct size
    """ 
    embed = [0.] * len(self.alphabet)
    content = state.content
    for char, idx in self.char_to_idx.items():
      if char in content:
        embed[idx] = float(content[char])
    
    max_group_size = state.max_group_size()
    embed += [max_group_size]
    embed += [float(bool(max_group_size >= self.mdp.substruct_size))]
    return torch.tensor(embed, device = self.args.device)
  
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
    """ Featurized Bag State -> encoding. """
    hid_dim = self.args.ssr_encoder_hid_dim
    n_layers = self.args.ssr_encoder_n_layers
    ssr_embed_dim = self.args.ssr_embed_dim
    net = network.make_mlp(
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

