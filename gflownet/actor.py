from .policy import BasePolicySA, BasePolicySSR

class Actor:
  def __init__(self, args, mdp):
    self.args = args
    self.mdp = mdp

  def featurize(self, state):
    raise NotImplementedError

  """
    Networks
  """
  def net_forward_sa(self):
    """ Construct a nn.Module, with forward that acts on List of States.

        Typically, make a net (nn.Module) using network.py that takes
        torch.tensor as input, and wrap it with network.py's StateFeaturizeWrap
        to take List of States as input.
    """
    raise NotImplementedError

  def net_backward_sa(self):
    """ Construct a nn.Module, with forward that acts on List of States.

        Typically, make a net (nn.Module) using network.py that takes
        torch.tensor as input, and wrap it with network.py's StateFeaturizeWrap
        to take List of States as input.
    """
    raise NotImplementedError
    
  def net_encoder_ssr(self):
    """ Construct a nn.Module, with forward that acts on List of States.

        Typically, make a net (nn.Module) using network.py that takes
        torch.tensor as input, and wrap it with network.py's StateFeaturizeWrap
        to take List of States as input.
    """
    raise NotImplementedError

  def net_scorer_ssr(self):
    """ Construct a nn.Module with forward that acts on torch.tensor 
        that are encodings from net_encoder_ssr. """
    raise NotImplementedError

  """
    Construct policy
  """
  def make_policy(self, sa_or_ssr, direction):
    """ Construct policy: sa or ssr, forward or backward.

        Inputs
        ------
        sa_or_ssr: string, either 'sa' or 'ssr'
        direction: string, either 'forward' or 'backward'
        featurizer: bound method
        masker:
    """
    assert direction in ['forward', 'backward']
    forward = bool(direction == 'forward')
    state_map_f = self.mdp.get_children if forward else self.mdp.get_parents

    if sa_or_ssr == 'sa':
      net = self.net_forward_sa() if forward else self.net_backward_sa()
      return BasePolicySA(self.args, self.mdp, actor = self,
                          net = net,
                          state_map_f = state_map_f)

    if sa_or_ssr == 'ssr':
      return BasePolicySSR(self.args, self.mdp, actor = self,
                            encoder = self.net_encoder_ssr(),
                            scorer = self.net_scorer_ssr(),
                            state_map_f = state_map_f)
