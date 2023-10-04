
import numpy as np, pandas as pd
from rdkit import Chem

from . import _blockgraphlists as BGL
from . import seqpamdp


class MolStrMDP(seqpamdp.SeqPrependAppendMDP):
  def __init__(self, args):
    blocks_file = args.get('blocks_file', 'datasets/mol/blocks_qm9_str.json')
    self.__init_from_blocks_file(blocks_file)

    symbols = '0123456789abcdefghijklmnopqrstuvwxyz' + \
              'ABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\()*+,-./:;<=>?@[\]^_`{|}~'
    assert len(self.blocks) <= len(symbols)
    self.alphabet = symbols[:len(self.blocks)]
    self.forced_stop_len = args.forced_stop_len

    super().__init__(args=args,
        alphabet=self.alphabet, forced_stop_len=self.forced_stop_len)
    self.bgle = BGL.BGLeditor(blocks_file)

  def __init_from_blocks_file(self, blocks_file):
    self.blocks = pd.read_json(blocks_file)
    self.block_smi = self.blocks['block_smi'].to_list()
    self.block_rs = self.blocks['block_r'].to_list()
    self.block_nrs = np.asarray([len(r) for r in self.block_rs])
    
    assert all(nr == 2 for nr in self.block_nrs)
    # print(f'Loaded {blocks_file=} with {len(self.blocks)} blocks.')
    print(f'Loaded blocks_file={blocks_file} with {len(self.blocks)} blocks.')
    return

  def state_to_bgl(self, state):
    """ Convert SeqPAState to mol.

        Start from left block, and add blocks to the last stem available.
        This uses the property that BGLeditor add_block appends new stems,
        in the order of block_rs for that block in the json file, to bgl.stems. 
    """
    if isinstance(state, str):
      block_ids = [self.alphabet.index(x) for x in state]
    else:
      block_ids = [self.alphabet.index(x) for x in state.content]

    bgl = BGL.make_empty_bgl()
    bgl = self.bgle.add_block(bgl, block_ids[0])
    for block_id in block_ids[1:]:
      last_stem_idx = len(bgl.stems) - 1
      bgl = self.bgle.add_block(bgl, block_id,
                                stem_idx = last_stem_idx,
                                new_atom_idx = 0)
    return bgl
  
  def state_to_mol(self, state):
    mol, _ = BGL.mol_from_bgl(self.state_to_bgl(state))
    return mol


class MolStrActor(seqpamdp.SeqPAActor):
  def __init__(self, args, mdp):
    super().__init__(args, mdp)


# testing
def randomwalk(mdp):
  import random
  traj = [mdp.root()]
  while not traj[-1].is_leaf:
    children = mdp.get_unique_children(traj[-1])
    traj.append(random.choice(children))
  return traj

def test():
  from attrdict import AttrDict
  args = {
    'blocks_file': 'datasets/mol/blocks_qm9_str.json',
    'forced_stop_len': 5,
  }
  args = AttrDict(args)
  
  mdp = MolStrMDP(args)
  traj = randomwalk(mdp)

  x = traj[-1]
  mol = mdp.state_to_mol(x)
  print(x.content)
  print(mol)
  import code; code.interact(local=dict(globals(), **locals()))
  return


if __name__ == '__main__':
  # python -m gflownet.MDPs.molstrmdp
  test()
