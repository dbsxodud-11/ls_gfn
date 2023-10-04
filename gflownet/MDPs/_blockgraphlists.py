'''
  Molecule representation:
  Block graph.
  Editing functions

  Adapted from 
  https://github.com/GFNOrg/gflownet/blob/master/mols/utils/molMDP.py
  and related scripts.
'''
from collections import defaultdict
import functools
from dataclasses import dataclass
import copy
import numpy as np, pandas as pd

import networkx as nx
from rdkit import Chem
import torch


"""
  Data classes
"""
@dataclass
class BlockGraphLists:
  """ Concise representation of a molecular block graph, using lists.

      Fields
      ------
      blockids: List, [int]
        IDs of each block in graph, in insertion order.
        ID is the index of that block among all possible blocks.
      blocksmis: List, [string]
        Smiles of each block in graph, in insertion order.
      slices: List
        Item i is the atom index where block i starts. Atoms are counted
        over the whole graph.
      numblocks: int
        Number of blocks currently in graph. Equal to len(blockids)
      jbonds: List, [[block_idx1, block_idx2,
                     bond_atomidx_in_block1, bond_atomidx_in_block2]]
        Edges, between bondatom1 in block1 - bondatom2 in block2.
        In insertion order.
        block_idx1, block_idx2 are represented as indices of blocks in graph
        so far.
      stems: List, [[block_idx1, bond_atomidx_in_block1]]
        Available atoms to construct new edges.
        Order is first by block insertion order, then atom order in block.
        block1 is represented as index of block in the graph so far
  """
  blockids: list      # index id of each block
  blocksmis: list     # smiles
  slices: list        # atom index at which every block starts
  numblocks: int
  jbonds: list        # [blockidx1, blockidx2, bondatom1, bondatom2]
  stems: list         # [blockidx1, bondatom1]


def make_empty_bgl():
  return BlockGraphLists(
    blockids = [],      # indexes of every block
    blocksmis = [],     # smiles
    slices = [0],       # atom index at which every block starts
    numblocks = 0,
    jbonds = [],        # [block1, block2, bondatom1, bondatom2]
    stems = [],         # [block1, bondatom1]
  )


class GraphLists:
  def __init__(self):
    """ General graph representation.

        Fields
        ------
        node_hash_features: List
          List of node features for hashing. 
          Items must be hashable. Prefer tuples of ints.
        edges: dict
          edges[node_idx] = [neighbor_idx1, ...]
        edge_features: dict
          edge_features[n1_idx][n2_idx] = x
          Items must be hashable. Prefer tuples of ints.
        node_atom_fts: List of AtomFeatures objects.
          Used for accessing other atom features, e.g., for neural net.
    """
    self.node_hash_features = []
    self.edges = defaultdict(list)
    self.edge_features = defaultdict(dict)
    self.node_atom_fts = []

  def __len__(self):
    """ Return number of nodes. """
    return len(self.node_hash_features)

  def add_node(self, atom, block_id):
    atom_fts = AtomFeatures(atom, block_id)
    self.node_hash_features.append(atom_fts.hash_features())
    self.node_atom_fts.append(atom_fts)
    return

  def add_edge(self, n1, n2, features):
    """ n1, n2: int. Add edges between nodes at index n1, n2. """
    # assert n1 < len(self) and n2 < len(self), \
    #     f'{n1=}, {n2=}, {len(self)=}'
    assert n1 < len(self) and n2 < len(self), \
        f'n1={n1}, n2={n2}, len(self)={len(self)}'
    self.edges[n1].append(n2)
    self.edges[n2].append(n1)
    self.edge_features[n1][n2] = features
    self.edge_features[n2][n1] = features

  def __hash__(self):
    """ Weisfeiler-Lehman hash. """
    hashes = [hash(nhf) for nhf in self.node_hash_features]
    summarize_hash = lambda hashes: hash(tuple(sorted(hashes)))
    node_idxs = list(range(len(self)))
    num_rounds = len(self)
    for i in range(num_rounds):
      hashes = [self.wl_gather(hashes, node_idx) for node_idx in node_idxs]
    return summarize_hash(hashes)

  def wl_gather(self, fts, node_idx):
    """ Gather node/edge features from neighbors of node_idx; then hash. """
    node_hash = hash(fts[node_idx])
    edge_hash = lambda node_ft, edge_ft: hash((hash(node_ft), hash(edge_ft)))
    neighbor_hash = []
    for neighbor_idx in self.edges[node_idx]:
      node_ft = fts[neighbor_idx]
      edge_ft = self.edge_features[node_idx][neighbor_idx]
      neighbor_hash.append(edge_hash(node_ft, edge_ft))
    return hash(tuple(sorted([node_hash] + neighbor_hash)))

  def to_nx(self):
    """ Convert GraphLists to nx graph. """
    graph = nx.Graph()
    for node_idx, node_fts in enumerate(self.node_atom_fts):
      graph.add_node(node_idx, 
                     hash_features=node_fts.hash_features(),
                     nn_features=node_fts.nn_features(),
                     atom_num=node_fts.atom_num,
                     block_id=node_fts.block_id)
    for node_idx in self.edges:
      for neighbor_idx in self.edges[node_idx]:
        graph.add_edge(node_idx, neighbor_idx,
                       nn_features=self.edge_features[node_idx][neighbor_idx])
    return graph


@functools.lru_cache(maxsize=None)
def get_mol_from_smiles(smi):
  """ SMILES string. Returns rdkit.Chem.rdchem.Mol object. """
  return Chem.MolFromSmiles(smi)


class AtomFeatures:
  def __init__(self, atom, block_id):
    self.atom_num = atom.GetAtomicNum()
    self.formal_charge = atom.GetFormalCharge()
    self.chiral_tag = int(atom.GetChiralTag())
    self.hybridization = int(atom.GetHybridization())
    self.num_explicit_hs = atom.GetNumExplicitHs()
    self.is_aromatic = int(atom.GetIsAromatic())
    self.block_id = block_id

  def hash_features(self):
    """ All features - Used for hashing and subgraph isomorphism """
    return (self.atom_num,
            self.formal_charge,
            self.chiral_tag,
            self.hybridization,
            self.num_explicit_hs,
            self.is_aromatic,
            self.block_id)

  def nn_features(self):
    """ Remove atom_num / block_id: ints need to be embedded """
    return (self.formal_charge,
            self.chiral_tag,
            self.hybridization,
            self.num_explicit_hs,
            self.is_aromatic)
ATOM_NODE_NN_FEATURE_DIM = 5
FAKE_ATOM_FTS = [0] * ATOM_NODE_NN_FEATURE_DIM


@functools.lru_cache(maxsize=None)
def featurize_bond(bond):
  """ rdkit.Atom object -> Tuple of features (all ints). """
  features = (int(bond.GetBondType()))
  return features
ATOM_EDGE_FEATURE_DIM = 1
FAKE_ATOM_EDGE_FTS = [0] * ATOM_EDGE_FEATURE_DIM


def block_single_bond_fts():
  """ Features of bond connecting blocks. """
  return (int(Chem.BondType.SINGLE))


def bgl_to_gl(bgl):
  """ Converts BlockGraphList (molecule) into GraphList (general purpose).
  
      Expand each block into atoms (in order of blocks)
      Add edges inside each block
      Add edges between blocks.
  """
  gl = GraphLists()

  num_atoms = 0
  for block_id, blocksmi in zip(bgl.blockids, bgl.blocksmis):
    mol = get_mol_from_smiles(blocksmi)

    # Add atoms in block
    for atom in mol.GetAtoms():
      gl.add_node(atom, block_id)

    # Add interior edges between atom inside block
    for bond in mol.GetBonds():
      bond_fts = featurize_bond(bond)
      start_atom_idx = num_atoms + bond.GetBeginAtomIdx()
      end_atom_idx = num_atoms + bond.GetEndAtomIdx()
      gl.add_edge(start_atom_idx, end_atom_idx, bond_fts)

    num_atoms += mol.GetNumAtoms()

  # Add exterior edges across blocks (single bonds)
  def get_atom_idx(block_idx, atom_in_block):
    return bgl.slices[block_idx] + atom_in_block

  for block_idx1, block_idx2, bondatom1, bondatom2 in bgl.jbonds:
    atom1_idx = get_atom_idx(block_idx1, bondatom1)
    atom2_idx = get_atom_idx(block_idx2, bondatom2)
    bond_fts = block_single_bond_fts()
    gl.add_edge(atom1_idx, atom2_idx, bond_fts)

  return gl


class BGLeditor():
  def __init__(self, blocks_file):
    """ Initialize BlockGraphLists editor from json blocks_file.

        Example json:
          {"block_name":{
            "0":"c1ccccc1_0",
            "1":"CO_0",
          },
          "block_smi":{"0":"c1ccccc1",
            "1":"CO",
          },
          "block_r":{
            "0":[0, 1, 2, 3, 4, 5],
            "1":[0, 0, 0, 1],
          }}

        Fields
        ------
        block_name: string, name
        block_smi:  string, SMILES
        block_r:    List [int], atom numbers for available stems /
                    connections to other blocks
    """
    print(f'Building BlockGraphLists Editor with {blocks_file}')
    self.blocks = pd.read_json(blocks_file)
    self.block_smi = self.blocks['block_smi'].to_list()
    self.block_rs = self.blocks['block_r'].to_list()
    self.block_nrs = np.asarray([len(r) for r in self.block_rs])
    self.block_mols = [Chem.MolFromSmiles(smi)
                       for smi in self.blocks['block_smi']]
    self.block_natm = np.asarray([b.GetNumAtoms() for b in self.block_mols])
    self.smi_to_natm = {smi: natm for smi, natm in
                        zip(self.block_smi, self.block_natm)}


  # Editing
  def add_block(self, bgl, block_id, stem_idx=None, new_atom_idx=None):
    ''' Forms new bgl.
        Stems: Available atoms in existing molecule (bgl) to attach to
        Attaches new block 'block_idx', connecting atom new_atom_idx
        to an available stem.

        Returns None if action is invalid.
    '''
    c = copy.deepcopy(bgl)
    c.blockids.append(block_id)
    c.blocksmis.append(self.block_smi[block_id])
    c.slices.append(c.slices[-1] + self.block_natm[block_id])
    c.numblocks += 1

    block_r = self.block_rs[block_id]

    if len(c.blockids) == 1:
      for r in block_r:
        c.stems.append([c.numblocks-1, r])
    else:
      # assert stem_idx is not None and new_atom_idx is not None, \
      #   f'If bgl has blocks, adding block must specify either stem_idx \
      #     or new_atom_idx.\
      #     \n{bgl=}\n{c=}\n{block_id=}\n{stem_idx=}\n{new_atom_idx=}'
      assert stem_idx is not None and new_atom_idx is not None, \
        f'If bgl has blocks, adding block must specify either stem_idx \
          or new_atom_idx.\
          \nbgl={bgl}\nc={c}\nblock_id={block_id}\nstem_idx={stem_idx}\nnew_atom_idx={new_atom_idx}'
      stem = c.stems[stem_idx]
      bond = [stem[0], c.numblocks-1, stem[1], block_r[new_atom_idx]]
      c.stems.pop(stem_idx)
      c.jbonds.append(bond)
      open_block_r = [r for i, r in enumerate(block_r) if i != new_atom_idx]
      for r in open_block_r:
        c.stems.append([c.numblocks-1, r])
    return c

  def delete_blocks(self, bgl, block_mask):
    """ Edits bgl in place.

        block_mask: binary vector, length of num. blocks in bgl.
          1 = keep, 0 = delete.
    """
    c = copy.deepcopy(bgl)
    # update number of blocks
    c.numblocks = np.sum(np.asarray(block_mask, dtype=np.int32))
    c.blocksmis = list(np.asarray(c.blocksmis)[block_mask])
    c.blockids = list(np.asarray(c.blockids)[block_mask])

    # update junction bonds
    reindex = np.cumsum(np.asarray(block_mask, np.int32)) - 1
    jbonds = []
    stems = []
    for bond in c.jbonds:
      if block_mask[bond[0]] and block_mask[bond[1]]:
        jbonds.append(np.array([reindex[bond[0]], reindex[bond[1]],
                                bond[2], bond[3]]))

      # need to add back stems when deleting block
      if bool(not block_mask[bond[0]]) and bool(block_mask[bond[1]]):
        new_stem = np.array([reindex[bond[1]], bond[3]])
        stems.append(new_stem)
      if bool(not block_mask[bond[1]]) and bool(block_mask[bond[0]]):
        new_stem = np.array([reindex[bond[0]], bond[2]])
        stems.append(new_stem)

    c.jbonds = jbonds

    # update r-groups
    for stem in c.stems:
      if block_mask[stem[0]]:
        stems.append(np.array([reindex[stem[0]], stem[1]]))
    c.stems = stems

    # update slices
    natms = [self.smi_to_natm[smi] for smi in c.blocksmis]
    c.slices = [0] + list(np.cumsum(natms))
    # note - prev. code returned reindex
    return c

  def delete_block(self, bgl, block_idx):
    """ Deletes block indexed {block_idx} from block graph. """
    mask = np.ones(bgl.numblocks, dtype=bool)
    mask[block_idx] = 0
    return self.delete_blocks(bgl, mask)

  # @functools.cached_property
  def num_atom_types(self):
    """ Return num. unique atoms. """
    return len(set(atom.GetAtomicNum() for mol in self.block_mols
                                       for atom in mol.GetAtoms()))

  # @functools.cached_property
  def atom_num_to_id_map(self):
    atom_nums = set(atom.GetAtomicNum() for mol in self.block_mols
                                       for atom in mol.GetAtoms())
    atom_nums = sorted(list(atom_nums))
    return {atom_num: idx for idx, atom_num in enumerate(atom_nums)}


'''
  convert graph representations to / from
'''
def bgl_to_nx(bgl):
  """ Constructs nx graph. Node features = smiles. No edge features. """
  g = nx.Graph()
  for i, smi in enumerate(bgl.blocksmis):
    g.add_node(i, smi=smi, idx=i, x=[float(bgl.blockids[i])])
  for i, j, _, _ in bgl.jbonds:
    g.add_edge(i, j)
  return g


"""
  mol to/from - deprecated
"""
def mol_to_nx(mol):
  """ Deprecated """
  G = nx.Graph()
  if mol is None:
    return G
  for atom in mol.GetAtoms():
    allfts = [
      atom.GetAtomicNum(),
      atom.GetFormalCharge(),
      atom.GetChiralTag(),
      atom.GetHybridization(),
      atom.GetNumExplicitHs(),
      atom.GetIsAromatic()
    ]
    ftstr = ''.join([str(s) for s in allfts])
    G.add_node(atom.GetIdx(), features=ftstr, atomnum=atom.GetAtomicNum())
  for bond in mol.GetBonds():
      G.add_edge(bond.GetBeginAtomIdx(),
                  bond.GetEndAtomIdx(),
                  bond_type=bond.GetBondType())
  return G


def mol_from_bgl(bgl):
  jun_bonds = np.asarray(bgl.jbonds)
  frags = [Chem.MolFromSmiles(frag_name) for frag_name in bgl.blocksmis]
  return mol_from_bgl_data(jun_bonds, frags)


def mol_from_bgl_data(jun_bonds, frags):
  jun_bonds = np.asarray(jun_bonds)
  if len(frags) == 0:
    return None, None
  nfrags = len(frags)

  # combine fragments into a single molecule
  mol = frags[0]
  for i in np.arange(nfrags - 1) + 1:
    mol = Chem.CombineMols(mol, frags[i])
  # add junction bonds between fragments
  frag_startidx = np.concatenate([[0], np.cumsum([frag.GetNumAtoms() for frag in frags])], 0)[:-1]

  if jun_bonds.size == 0:
    mol_bonds = []
  else:
    mol_bonds = frag_startidx[jun_bonds[:, 0:2]] + jun_bonds[:, 2:4]

  emol = Chem.EditableMol(mol)
  
  for bond in mol_bonds:
    emol.AddBond(int(bond[0]), int(bond[1]), Chem.BondType.SINGLE)
  mol = emol.GetMol()
  atoms = list(mol.GetAtoms())

  def _pop_H(atom):
    nh = atom.GetNumExplicitHs()
    if nh > 0: atom.SetNumExplicitHs(nh-1)

  for bond in mol_bonds:
    _pop_H(atoms[bond[0]])
    _pop_H(atoms[bond[1]])

  #print([(atom.GetNumImplicitHs(), atom.GetNumExplicitHs(),i) for i,atom in enumerate(mol.GetAtoms())])
  Chem.SanitizeMol(mol)
  return mol, mol_bonds


'''
  Testing
'''
def randomwalk(length = 8):
  editor = BGLeditor('datasets/mol/blocks_PDB_105.json')
  bgl = make_empty_bgl()
  while bgl.numblocks < length:
    if bgl.numblocks == 0:
      stem_idx = None
    elif len(bgl.stems) > 0:
      stem_idx = np.random.choice(len(bgl.stems))
    else:
      # print(f'No more available stems - finishing early')
      break
    block_id = np.random.choice(np.arange(len(editor.blocks)))
    bgl = editor.add_block(bgl, block_id=block_id, stem_idx=stem_idx,
                           new_atom_idx=0)

  mol, mol_bonds = mol_from_bgl(bgl)
  # from rdkit.Chem import Draw
  # Draw.MolToFile(mol, 'testmol.png')
  return bgl


def hundred_randomwalks():
  from tqdm import tqdm
  stats = defaultdict(lambda: 0)
  for i in tqdm(range(100)):
    bgl = randomwalk()
    stats[bgl.numblocks] += 1
  print(stats)
  # {8: 519, 7: 36, 6: 31, 5: 48, 4: 74, 3: 96, 2: 196}
  # Terminates early often
  return


def test_graphlists():
  bgl = randomwalk()
  gl = bgl_to_gl(bgl)
  print(hash(gl))
  import code; code.interact(local=dict(globals(), **locals()))
  return


def draw(mol, name):
  from rdkit.Chem import Draw
  Draw.MolToFile(mol, f'mol-{name}.png')
  return


def test_hash():
  # Manually make states with different internal repr, but should
  # be equal.
  print('Testing hashing with manual examples ...')
  edit = BGLeditor('datasets/mol/blocks_mws.json')

  # block idx 0 = benzene
  m1 = make_empty_bgl()
  m1 = edit.add_block(m1, 0)
  m1 = edit.add_block(m1, 0, stem_idx=0, new_atom_idx=0)

  m2 = make_empty_bgl()
  m2 = edit.add_block(m2, 0)
  m2 = edit.add_block(m2, 0, stem_idx=5, new_atom_idx=1)

  assert hash(bgl_to_gl(m1)) == hash(bgl_to_gl(m2))

  # block 3 is not symmetric
  m1 = make_empty_bgl()
  m1 = edit.add_block(m1, 5)
  m1 = edit.add_block(m1, 3, stem_idx=0, new_atom_idx=0)

  m2 = make_empty_bgl()
  m2 = edit.add_block(m2, 5)
  m2 = edit.add_block(m2, 3, stem_idx=0, new_atom_idx=1)

  # draw(mol_from_bgl(m1)[0], 'testm1')
  # draw(mol_from_bgl(m2)[0], 'testm2')
  # g1 = bgl_to_gl(m1)
  # g2 = bgl_to_gl(m2)
  # print(hash(g1), hash(g2))
  assert hash(bgl_to_gl(m1)) != hash(bgl_to_gl(m2))

  print('Tests passed.')
  return


if __name__ == '__main__':
  # python -m gflownet.MDPs._blockgraphlists
  # hundred_randomwalks()
  # test_graphlists()
  test_hash()
