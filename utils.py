import collections

import numpy
import scipy.sparse
import tensorflow as tf

def get_sparse_adj_tensor(num_nodes, num_edges, adj_name): 
  indices_ph = tf.placeholder(tf.int64, [num_edges, 2], name='{}_indices'.format(adj_name))

  values_ph = tf.placeholder(tf.float32, [num_edges], name='{}_values'.format(adj_name))

  dense_shape = [num_nodes, num_nodes]

  sp_adj_tensor = tf.SparseTensor(indices_ph, values_ph, dense_shape)

  return sp_adj_tensor


def get_sparse_adj_from_edge_list(edge_lists, normalize=True):
  # Will be used to construct (sparse) adjacency matrix.
  edge_sets = collections.defaultdict(set)
  for node, neighbors in edge_lists.items():
    edge_sets[node].add(node)   # Add self-connections
    for n in neighbors:
      edge_sets[node].add(n)
      edge_sets[n].add(node)  # Assume undirected.

  # Now, build adjacency list.
  adj_indices = []
  adj_values = []
  for node, neighbors in edge_sets.items():
    for n in neighbors:
      adj_indices.append((node, n))
      if normalize:
        adj_values.append(1 / (numpy.sqrt(len(neighbors) * len(edge_sets[n]))))
      else:
        adj_values.append(1)

  adj_indices = numpy.array(adj_indices, dtype='int32')
  adj_values = numpy.array(adj_values, dtype='float32')

  return adj_indices, adj_values