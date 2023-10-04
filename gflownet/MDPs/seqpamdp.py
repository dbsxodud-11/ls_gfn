import copy
import functools
import numpy as np
from sklearn.preprocessing import OneHotEncoder

import torch

from .. import network, utils
from ..actor import Actor
from .basemdp import BaseState, BaseMDP

import enum
from dataclasses import dataclass


class SeqPAState(BaseState):
  """ String state, with prepend/append actions. """
  def __init__(self, content, is_leaf=False):
    """ content: string. """
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
    return self.content in other.content

  """
    Modify states.
    Construct new SeqPAState, given SeqPAAction.
        Return None if invalid action.
  """
  def _delfirst(self):
    new_content = copy.copy(self.content)
    return SeqPAState(new_content[1:])

  def _dellast(self):
    new_content = copy.copy(self.content)
    return SeqPAState(new_content[:-1])

  def _prepend(self, action):
    new_content = copy.copy(self.content)
    return SeqPAState(action.char + new_content)

  def _append(self, action):
    new_content = copy.copy(self.content)
    return SeqPAState(new_content + action.char)

  def _terminate(self):
    if not self.is_leaf:
      return SeqPAState(self.content, is_leaf=True)
    else:
      return None
  
  def _unterminate(self):
    if self.is_leaf:
      return SeqPAState(self.content, is_leaf=False)
    else:
      return None

class SeqPAActionType(enum.Enum):
  # Forward actions
  Stop = enum.auto()
  PrependChar = enum.auto()
  AppendChar = enum.auto()
  # Backward actions
  UnStop = enum.auto()
  DelFirst = enum.auto()
  DelLast = enum.auto()


@dataclass
class SeqPAAction:
  action: SeqPAActionType
  char: str = None


class SeqPrependAppendMDP(BaseMDP):
  """ MDP for building a string by prepending and appending chars.

      Action set is fixed and not a function of state.

      Forward actions: [stop, prepend A, prepend B, ..., append A, ...]
      Reverse actions: [Unstop, delete first char, delete last char]

      Cannot contain any CUDA elements: instance is passed
      to ray remote workers for substructure guidance, which need
      access to get_children & is_member.
  """
  def __init__(self, args, alphabet = list('0123'), forced_stop_len=8):
    self.args = args
    self.alphabet = alphabet
    self.alphabet_set = set(self.alphabet)
    self.char_to_idx = {a: i for (i, a) in enumerate(self.alphabet)}
    self.forced_stop_len = forced_stop_len

    self.fwd_actions = [SeqPAAction(SeqPAActionType.Stop)] + \
                       [SeqPAAction(SeqPAActionType.PrependChar, c)
                        for c in self.alphabet] + \
                       [SeqPAAction(SeqPAActionType.AppendChar, c)
                        for c in self.alphabet]
    self.back_actions = [SeqPAAction(SeqPAActionType.UnStop),
                         SeqPAAction(SeqPAActionType.DelFirst),
                         SeqPAAction(SeqPAActionType.DelLast)]

    self.state = SeqPAState
    self.parallelize_policy = False

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
    """ Applies SeqPAAction to state.
        Returns State or None (invalid transition). 
        Action Types: Stop, PrependChar, AppendChar
    """
    if state.is_leaf:
      return None
    if self.has_forced_stop(state) and action.action != SeqPAActionType.Stop:
        return None

    if action.action == SeqPAActionType.Stop:
      if self.has_stop(state):
        return state._terminate()
      else:
        return None

    if action.action == SeqPAActionType.PrependChar:
      return state._prepend(action)

    if action.action == SeqPAActionType.AppendChar:
      return state._append(action)

  def transition_back(self, state, action):
    """ Applies action to state. Returns State or None (invalid transition). 
        Action types: UnStop, DelFirst, DelLast
    """
    if state == self.root():
      return None
    if state.is_leaf and action.action != SeqPAActionType.UnStop:
      return None

    if action.action == SeqPAActionType.UnStop:
      if state.is_leaf:
        return state._unterminate()
      else:
        return None

    if action.action == SeqPAActionType.DelFirst:
      return state._delfirst()

    if action.action == SeqPAActionType.DelLast:
      return state._dellast()

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
class SeqPAActor(Actor):
  """ Holds SeqPAMDP and GPU elements: featurize & policies. """
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
    ohe_dim = lambda num_chars: num_chars * len(self.alphabet)
    full_seq_len = self.forced_stop_len
    full_ohe_dim = ohe_dim(full_seq_len)
    if len(state.content) == 0:
      embed = np.zeros((1, full_ohe_dim)).flatten()
    else:
      embed = np.concatenate(self.onehotencoder.transform(
          [[c] for c in state.content]))
      num_rem = self.forced_stop_len - len(state.content)
      padding = np.zeros((1, ohe_dim(num_rem))).flatten()
      embed = np.concatenate([embed, padding])
    return torch.tensor(embed, dtype=torch.float, device=self.args.device)

  def get_feature_dim(self):
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
