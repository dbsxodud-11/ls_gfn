import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool

from . import utils

"""
  Wrappers
"""
class StateFeaturizeWrap(torch.nn.Module):
  def __init__(self, net, featurizer):
    """ Converts net from requiring torch.tensor input to [State] input.
        Wraps net: nn.module with a featurizer.

        Applies to SA forward, SA backward, and SSR encoder.
        Not used for SSR scorer; as it acts on torch.tensor embeddings.
    """
    super().__init__()
    self.featurizer = featurizer
    self.net = net

  def forward(self, batch):
    """ List of States -> torch.tensor"""
    inp = utils.batch([self.featurizer(state) for state in batch])
    return self.net(inp)


class GraphMaskSAWrap(torch.nn.Module):
  def __init__(self, net, featurizer, masker):
    """ Wraps GNN with featurizer and masker (for SA).

        Inputs
        ------
        net: torch.nn.Module
          Graph neural net that outputs same graph with different node features.
          Expects no pooling.
        featurizer: function
          Maps State -> torch_geometric.Data
        masker: function
          Maps State -> binary vector mask, on node indices.
          Masks states with no valid actions (e.g., can only add to
          specific stems/atoms, or only delete certain "leaf" blocks)
    """
    super().__init__()
    self.net = net
    self.featurizer = featurizer
    self.masker = masker

  def forward(self, batch):
    """ Predict SA scores on batch: List of States.
        Featurize, mask, and flatten
    """
    masks = [self.masker(state) for state in batch]
    inp_graphs = utils.batch([self.featurizer(state) for state in batch])

    # [B graphs] -> [B graphs]
    out_globals, out_graphs = self.net(inp_graphs)

    # Flatten with mask
    flat_out = []
    for out_global, out_graph, mask in zip(out_globals, out_graphs, masks):

      # mask output (updated node features): -> [n, o]
      masked_out = out_graph.x[mask]
      # flatten to vector: -> [n * o]
      flat = torch.cat([out_global, torch.flatten(masked_out)])

      # output concatenates global with flat
      flat_out.append(flat)

    # type: List, elements are tensors with grad
    return flat_out


"""
  Simple nets
"""
def make_mlp(l, act=nn.LeakyReLU(), tail=[], with_bn=False):
  """ Makes an MLP with no top layer activation. """
  net = nn.Sequential(*(sum(
    [[nn.Linear(i, o)] + \
      (([nn.BatchNorm1d(o), act] if with_bn else [act]) 
         if n < len(l) - 2 else [])
      for n, (i, o) in enumerate(zip(l, l[1:]))], []
  ) + tail))
  return net


def make_convnet(inp_side, 
                 kernel_size, 
                 num_channels,
                 mlp_hid_dim,
                 mlp_n_layers,
                 mlp_out_dim, 
                 act=nn.LeakyReLU()):
  conv_output_side = inp_side - kernel_size + 1
  conv_output_dim = num_channels * conv_output_side**2
  # print(f'Calculated {conv_output_dim=}')  
  print(f'Calculated conv_output_dim={conv_output_dim}')  

  net = nn.Sequential(
    nn.Conv2d(num_channels, num_channels, kernel_size = kernel_size),
    act,
    nn.Flatten(1, -1),
  )
  l = [conv_output_dim] + [mlp_hid_dim]*mlp_n_layers + [mlp_out_dim]
  for i, o in zip(l, l[1:]):
    net.append(nn.Linear(i, o))
    net.append(act)
  return net


"""
  Graph nets
"""
def make_nodesummary_gnn(node_ft_dim, out_dim):
  """ Simple gnn, acting on node features only.
      Output has fixed size.
  """
  class GCN(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.conv1 = GCNConv(node_ft_dim, 16)
      self.conv2 = GCNConv(16, out_dim)

    def forward(self, batch):
      x, edge_index = batch.x, batch.edge_index, 

      x = self.conv1(x, edge_index)
      x = F.relu(x)
      x = self.conv2(x, edge_index)
      x = global_mean_pool(x, batch=batch.batch)
      return x

  return GCN()


# SA
def make_molblock_back_sa_gnn(num_block_types,
                              num_edge_types,
                              embed_dim,
                              hid_dim,
                              n_layers,
                              out_node_dim,
                              global_out_dim):
  class GNN(torch.nn.Module):
    """ Graph neural net. Embeds nodes (block ids) and edges (stem ids).
        If pool, performs global_mean_pool over nodes.

        Parameters
        ----------
        num_node_types: int
          Number of distinct node types, for embedding.
        num_edge_types: int
          Number of distinct node types, for embedding.
        embed_dim: int
          Dimension of initial feature embedding (look-up table).
        out_node_dim: int
          Output dimension at each node.
        global_out_dim: int
        
        Input to model: torch_geometric.Data graph with
          Data.x: Categorical block ids, to be embedded
          Data.edge_attr: Categorical edge ids, to be embedded
    """
    def __init__(self):
      super().__init__()
      # Ensure even
      self.embed_dim = embed_dim
      if self.embed_dim % 2 == 1:
        self.embed_dim += 1

      self.node_embed = torch.nn.Embedding(num_block_types, self.embed_dim)
      self.edge_embed = torch.nn.Embedding(num_edge_types, self.embed_dim // 2)

      self.conv_layers = torch.nn.ModuleList()
      self.gine_mlps = torch.nn.ModuleList()
      l = [embed_dim] + [hid_dim]*n_layers + [out_node_dim]
      for i, o in zip(l, l[1:]):
        gine_mlp = make_mlp([i, o])
        self.gine_mlps.append(gine_mlp)
        layer = GINEConv(gine_mlp, edge_dim=self.embed_dim)
        self.conv_layers.append(layer)

      self.global_mlp = make_mlp([out_node_dim, global_out_dim])
      print(self.modules)

    def forward(self, batch):
      x = self.node_embed(batch.x)
      edge_index = batch.edge_index
      edge_attr = self.edge_embed(batch.edge_attr)
      # (num_edges, 2, embed_dim/2) -> (num_edges, embed_dim) by cat
      edge_attr = torch.reshape(edge_attr, (edge_attr.shape[0], self.embed_dim))

      layers, last_layer = self.conv_layers[:-1], self.conv_layers[-1]
      for layer in layers:
        x = layer(x, edge_index, edge_attr)
        x = F.relu(x)
      x = last_layer(x, edge_index, edge_attr)

      pooled_out = global_mean_pool(x, batch=batch.batch)
      # (b, out_node_dim) -> (b, global_out_dim)
      pooled_out = self.global_mlp(pooled_out)

      # x is [total num nodes, d]. reassign to batch node features
      batch.x = x
      # get [num graphs of torch_geometric.Data]
      out_graphs = batch.to_data_list()
      return pooled_out, out_graphs

  return GNN()


def make_molatom_fwd_sa_gnn(inp_node_dim, 
                            inp_edge_dim,
                            num_block_types,
                            num_atom_types,
                            hid_dim,
                            n_layers,
                            global_out_dim, 
                            out_node_dim):
  class GNN(torch.nn.Module):
    """ Graph neural net.

        Acts on an input torch_geometric.Data graph, with
          node features of size (inp_node_dim) and
          edge features of size (inp_edge_dim).

        Returns (global_pool_result, out_graph), where
          global_pool_result has shape (global_out_dim), and
          out_graph has node features with shape (out_node_dim). 

        Parameters
        ----------
        inp_node_dim: int
          Input graph's node feature dimension, before embedding atom id
          and block id.
        inp_edge_dim: int
          Input graph's edge feature dimension.
        num_block_types: int
          Num. block types, for embedding.
        num_atom_types: int
          Num. atom types, for embedding.
        hid_dim: int
          Hidden dimension for graph convolution
        n_layers: int
          Number of graph convolution layers
        global_out_dim: int
          Output global pooled dimension.
        out_node_dim: int
          Output node feature dimension.

        Input to model: torch_geometric.Data graph with
          graph.nn_features: features directly act-able
          graph.atom_ids: ints, to be embedded
          graph.block_ids: ints, to be embedded
    """
    def __init__(self):
      super().__init__()
      embed_dim = 4
      self.atom_embed = torch.nn.Embedding(num_atom_types, embed_dim)
      self.block_embed = torch.nn.Embedding(num_block_types, embed_dim)

      total_inp_node_dim = inp_node_dim + 2*embed_dim

      self.conv_layers = torch.nn.ModuleList()
      self.gine_mlps = torch.nn.ModuleList()
      l = [total_inp_node_dim] + [hid_dim]*n_layers + [out_node_dim]
      for i, o in zip(l, l[1:]):
        gine_mlp = make_mlp([i, o])
        self.gine_mlps.append(gine_mlp)
        layer = GINEConv(gine_mlp, edge_dim=inp_edge_dim)
        self.conv_layers.append(layer)

      self.global_mlp = make_mlp([out_node_dim, global_out_dim])
      print(self.modules)

    def reform_input(self, batch):
      """ batch.x is [atom_ids, block_ids, nn_features].
          Embed atom_id, block_ids, and concatenate to nn_features.
      """
      new_xs = [self.atom_embed(batch.x[:, 0]),
                self.block_embed(batch.x[:, 1]),
                batch.x[:, 2:] ]
      return torch.cat(new_xs, -1)

    def forward(self, batch):
      """ Acts on torch_geometric.Data graphs with fields concatenated into x:
            atom_ids: categorical ints to be embedded
            block_ids: categorical ints to be embedded
            nn_features: features that can be directly acted on
      """
      x = self.reform_input(batch)
      edge_index = batch.edge_index
      edge_attr = batch.edge_attr.float()

      layers, last_layer = self.conv_layers[:-1], self.conv_layers[-1]
      for layer in layers:
        x = layer(x, edge_index, edge_attr)
        x = F.relu(x)
      x = last_layer(x, edge_index, edge_attr)

      pooled_out = global_mean_pool(x, batch=batch.batch)
      # (b, out_node_dim) -> (b, global_out_dim)
      pooled_out = self.global_mlp(pooled_out)

      # x is [total num nodes, d]. reassign to batch node features
      batch.x = x
      # get [num graphs of torch_geometric.Data]
      out_graphs = batch.to_data_list()
      return pooled_out, out_graphs

  return GNN()

# SSR
def make_molblock_ssr_gnn(num_node_types, 
                          num_edge_types, 
                          embed_dim,
                          hid_dim,
                          n_layers,
                          out_dim):
  class GNN(torch.nn.Module):
    """ Embeds block and edge types.
        Returns scalar value after global pooling.
    """
    def __init__(self):
      super().__init__()
      # Ensure even
      self.embed_dim = embed_dim
      if self.embed_dim % 2 == 1:
        self.embed_dim += 1
      self.node_embed = torch.nn.Embedding(num_node_types, self.embed_dim)
      self.edge_embed = torch.nn.Embedding(num_edge_types, self.embed_dim // 2)

      self.conv_layers = torch.nn.ModuleList()
      self.gine_mlps = torch.nn.ModuleList()
      l = [embed_dim] + [hid_dim]*n_layers + [out_dim]
      for i, o in zip(l, l[1:]):
        gine_mlp = make_mlp([i, o])
        self.gine_mlps.append(gine_mlp)
        layer = GINEConv(gine_mlp, edge_dim=self.embed_dim)
        self.conv_layers.append(layer)

      print(self.modules)

    def forward(self, batch):
      x = self.node_embed(batch.x)
      edge_index = batch.edge_index
      edge_attr = self.edge_embed(batch.edge_attr)
      # (num_edges, 2, embed_dim/2) -> (num_edges, embed_dim) by cat
      edge_attr = torch.reshape(edge_attr, (edge_attr.shape[0], self.embed_dim))

      layers, last_layer = self.conv_layers[:-1], self.conv_layers[-1]
      for layer in self.conv_layers[:-1]:
        x = layer(x, edge_index, edge_attr)
        x = F.relu(x)
      x = last_layer(x, edge_index, edge_attr)

      x = global_mean_pool(x, batch=batch.batch)
      return x
    
  return GNN()
