import numpy as np
from tqdm import tqdm
import torch
from collections import namedtuple, defaultdict

from ..data import Experience
from .. import utils


def uniform_policy(parent, children):
  return np.random.choice(children)


def unique_keep_order_remove_none(items):
  """ Remove duplicates, keeping order. Uses hashing. """
  return [x for x in list(dict.fromkeys(items)) if x is not None]


class BaseState():
  def __init__(self, content, is_leaf=False, is_root=False):
    self.content = self.canonicalize(content)
    self.is_leaf = is_leaf
    self.is_root = is_root

  def __repr__(self):
    """ Human-readable string representation of state. """
    raise NotImplementedError()

  def __eq__(self, other):
    return self.content_equals(other) and \
           self.is_leaf == other.is_leaf and \
           self.is_root == other.is_root

  def __hash__(self):
    """ Hash. Used for fast equality checking.
    
        Important for data structures that require unique states, e.g.,
        sampling unique datasets of x, or allXtoR dict that stores map
        {x: r} of all unique training data seen so far. 
        Importantly, policy needs to return logp over unique children
        or parents, which requires summing over equivalent states that
        can arise from different actions.
    """
    return hash(repr(self))

  def canonicalize(self, content):
    """ Standardize content. Ex: default sort for unordered items. """
    raise NotImplementedError()

  def content_equals(self, other):
    raise NotImplementedError()

  def is_member(self, other):
    """ Returns bool, whether there exists a path in the MDP from
        self state to other state. Critical for substructure guided
        GFlowNet.
    """
    raise NotImplementedError()


class BaseMDP:
  """ Markov Decision Process.
      
      Specifies relations between States, which States are leafs and their
      rewards. Also specifies forward and backward policy nets with
      high-dimensional outputs corresponding to actions, network
      architecture and logic for forward passing (translating actions into
      unique child/parent states).

      Inherited by object-specific MDPs (e.g., GraphMDP, BagMDP) in MDPs
      folder, and further inherited by task-specific scripts (e.g.,) in 
      experiments (exp) folder (e.g., TFBind8MDP) which specify reward
      functions, etc. 
  """
  def __init__(self):
    pass

  # Fundamentals
  def root(self):
    """ Return the root state. """
    raise NotImplementedError

  # Membership
  def is_member(self, query, target):
    """ Return bool, whether there exists an MDP path from query to target. """
    raise NotImplementedError

  """
    Children and parents
  """
  def get_children(self, state):
    """ Return list of children in deterministic order.
        Calls self.transition_fwd on actions from self.get_fwd_actions.
    """
    return [self.transition_fwd(state, act)
            for act in self.get_fwd_actions(state)]

  def get_parents(self, state):
    """ Return list of children in deterministic order.
        Calls self.transition_back on actions from self.get_back_actions.
    """
    return [self.transition_back(state, act)
            for act in self.get_back_actions(state)]

  def get_unique_children(self, state):
    """ Return unique states, keeping order. Used for substructure guide.
        Removes None
    """
    return unique_keep_order_remove_none(self.get_children(state))

  def get_unique_parents(self, state):
    """ Return unique states, keeping order.
        Used for maximum entropy gflownet with uniform backward policy.
    """
    return unique_keep_order_remove_none(self.get_parents(state))

  # Transitions
  def transition_fwd(self, state, action):
    """ Applies Action to state. Returns State or None (invalid transition). """
    raise NotImplementedError

  def transition_back(self, state, action):
    """ Applies Action to state. Returns State or None (invalid transition). """
    raise NotImplementedError

  # Actions
  def get_fwd_actions(self, state):
    """ Gets forward actions from state. Returns List of Actions.

        For many MDPs, this is independent of state. The num actions
        returned must match the policy's output dim. List of actions
        is used to associate policy output scores with states, so it
        must be in a consistent, deterministic order given state.
    """
    raise NotImplementedError

  def get_back_actions(self, state):
    """ See get_fwd_actions. """
    raise NotImplementedError

  # Specs
  def reward(self, x):
    """ Leaf State -> float """
    raise NotImplementedError

  def has_stop(self, state):
    """ State -> bool """
    raise NotImplementedError

  def has_forced_stop(self, state):
    raise NotImplementedError

  # Featurization
  def featurize(self, state):
    raise NotImplementedError

  def get_net_io_shapes(self):
    """ Specify input, output shapes for fwd/back policies.

        Note that MDPs that allow stopping at arbitrary points
        need an output dimension for the stop action.

        Returns
        -------
        dict with fields ['forward/backward']['in/out'],
            specifying the input and output shape of networks.
    """
    raise NotImplementedError
