import collections

import pickle
import numpy
import scipy.sparse
import tensorflow as tf

def get_sparse_adj_tensor(num_nodes, num_edges, adj_name): 
  indices_ph = tf.placeholder(tf.int64, [num_edges, 2], name='{}_indices'.format(adj_name))

  values_ph = tf.placeholder(tf.float32, [num_edges], name='{}_values'.format(adj_name))

  dense_shape = [num_nodes, num_nodes]

  sp_adj_tensor = tf.SparseTensor(indices_ph, values_ph, dense_shape)

  return sp_adj_tensor


def get_sparse_adj_from_edge_list(base_path):
  edge_lists = pickle.load(open(base_path, 'rb'))

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
      if base_path.endswith('.graph'):
        adj_values.append(1 / (numpy.sqrt(len(neighbors) * len(edge_sets[n]))))
      else:
        adj_values.append(1)

  adj_indices = numpy.array(adj_indices, dtype='int32')
  adj_values = numpy.array(adj_values, dtype='float32')

  return adj_indices, adj_values

def make_dataset(dir_name, file_names_list, postfix):
  data_files = [os.path.join(dir_name, "{}.{}".format(fname, postfix)) for fname in file_names_list]
  dataset = tf.data.Dataset.from_tensor_slices(data_files)
  return dataset

def get_data_readers(file_names, get_labels=False):
  """Given file names, returns Datasets. 
  Args:
    file_names: text with one file_name per line.
    get_labels: If set, returns 4 Datasets: the containing the image files (x)
      and the second containing the segmentation labels (y). If not, returns
      only the first argument.
  
  Returns:
    instance of (tf.data.Dataset, tf.data.Dataset), or 
    (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, tf.data.Dataset) (if get_labels == True).
  """

  with open(file_names, 'r') as f:
    file_names_list = f.read()
  file_names_list = [fl for fl in file_names_list.splitlines()]

  graph_dataset = make_dataset(FLAGS.data_dir, file_names_list, 'graph')
  features_dataset = make_dataset(FLAGS.data_dir, file_names_list, 'allx')

  if get_labels == False:
    return (graph_dataset, features_dataset)

  featbased_dataset = make_dataset(FLAGS.data_dir, file_names_list, 'ally1')
  struct_dataset = make_dataset(FLAGS.data_dir, file_names_list, 'ally2')

  return (graph_dataset, features_dataset, featbased_dataset, struct_dataset)


