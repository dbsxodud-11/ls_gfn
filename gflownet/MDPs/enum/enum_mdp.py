"""
  Enumerate over all data in an MDP.
  Depth-first search on MDP children, with early stopping.

  Example: Used for molblockmdp with neural net reward oracle.
  Generate all molecules for a given block list, then run neural net on
  all of them to gather reward statistics.
"""

import pickle
from attrdict import AttrDict
from queue import LifoQueue

from ..molblockmdp import MolMDP


def enumerate_x(args, mdp):
  """
    50 million bytes = 50 MB
    If each x is 1000 bytes, then 50 GB. 
    Consider chunking

    Each x is ~1.25 KB. 
    50 million => 63 GB. 
  """
  all_x = set()
  all_states = set()
  stack = LifoQueue()

  # DFS
  print(f'Starting DFS ...')
  stack.put(mdp.root())
  while not stack.empty():
    node = stack.get()
    if node.is_leaf:
      all_x.add(node)
    else:
      children = mdp.get_unique_children(node)
      for child in children:
        if child not in all_states:
          all_states.add(child)
          stack.put(child)
    print(len(all_x), stack.qsize())

  # Save
  print(f'Saving ...')
  with open(f'{args.enum_name}.pkl', 'wb') as f:
    pickle.dump(all_x, f)
  return


def test():
  global verbose
  verbose = True

  args = {
    'enum_name': 'block10_2_3',
    'blocks_file': 'datasets/mol/blocks_10.json',
    'can_stop_len': 2,
    'forced_stop_len': 3,
  }
  args = AttrDict(args)
  mdp = MolMDP(args)

  # Loads model 
  # import sys
  # sys.path.append('/home/shenm19/prj/reading/')
  # from sEH import model
  # model.load_args()
  # import code; code.interact(local=dict(globals(), **locals()))

  enumerate_x(args, mdp)
  return


if __name__ == '__main__':
  # python -m gflownet.MDPs.enum.enum_mdp
  test()