import os
import pickle
import sys

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from utils import *

def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensorValue(indices, coo.data, coo.shape)


def concatenate_csr_matrices_by_rows(matrix1, matrix2):
  """Concatenates sparse csr matrices matrix1 above matrix2.
  
  Adapted from:
  https://stackoverflow.com/questions/6844998/is-there-an-efficient-way-of-concatenating-scipy-sparse-matrices
  """
  new_data = np.concatenate((matrix1.data, matrix2.data))
  new_indices = np.concatenate((matrix1.indices, matrix2.indices))
  new_ind_ptr = matrix2.indptr + len(matrix1.data)
  new_ind_ptr = new_ind_ptr[1:]
  new_ind_ptr = np.concatenate((matrix1.indptr, new_ind_ptr))

  return scipy.sparse.csr_matrix((new_data, new_indices, new_ind_ptr))


def load_x(filename):
  if sys.version_info > (3, 0):
    return pickle.load(open(filename, 'rb'), encoding='latin1')
  else:
    return np.load(filename)

def ReadDataset(dataset_dir, dataset_name):
  """Returns dataset files given e.g. ind.pubmed as a dataset_name.
 
  Args:
    dataset_dir: `data` directory of planetoid datasets.
    dataset_name: One of "ind.citeseer", "ind.cora", or "ind.pubmed".

  Returns:
    Dataset object (defined below).
  """

  if True:
    base_path = os.path.join(dataset_dir, dataset_name)
    # num_nodes, graph_dataset, features_dataset, featbased_dataset, struct_dataset = get_data_readers(base_path, file_names, get_labels=True)

    # files_train_x, files_train_y = get_filename_data_readers(train_ids_file, True)
    # data_features = features_dataset.map(load_x)
    # data_adj = graph_dataset.map(get_sparse_adj_from_edge_list)
    # data_featbased = featbased_dataset.map(get_sparse_adj_from_edge_list)
    # data_struct = struct_dataset.map(get_sparse_adj_from_edge_list)

    # data_zipped = tf.data.Dataset.zip((data_features, data_adj, data_featbased, data_struct))
    # data_xy = data_xy.repeat()
    # data_xy = data_xy.shuffle(300).batch(self.batch_size)
    # self.itr_xy = tf.compat.v1.data.make_initializable_iterator(data_xy)
    # self.next_batch = self.itr_xy.get_next()

    features = load_x(base_path + '.x')
    adj_indices, adj_values = get_sparse_adj_from_edge_list(base_path + '.graph')
    featbased_ind, feabased_values = get_sparse_adj_from_edge_list(base_path + '.y1')
    struct_ind, struct_values = get_sparse_adj_from_edge_list(base_path + '.y2')

    num_nodes = features.shape[0]

  else:
    # TODO(ninoch): Add real data here 
    pass 

  return Dataset(
      num_nodes=num_nodes,
      features=features, 
      featbased_indices = featbased_ind, featbased_values = feabased_values,
      structural_indices = struct_ind, structural_values = struct_values,
      adj_indices=adj_indices, adj_values=adj_values)

class Dataset(object):
  """Dataset object giving access to sparse feature & adjacency matrices.

  Access the matrices, as a sparse tensor, using functions:
    Features: sparse_allx_tensor()
    Adjacency: sparse_adj_tensor().

  If you use these tensors in a tensorflow graph, you must supply their
  dependencies. In particular, feed them into your feed dictionary like:

  feed_dict = {}   # and populate it with your own placeholders etc.
  dataset.populate_feed_dict(feed_dict)  # makes adj and allx tensors runnable.
  """
  def __init__(self, num_nodes = None,
      features = None, 
      featbased_indices = None, featbased_values = None,
      structural_indices = None, structural_values = None,
      adj_indices = None, adj_values = None):
    self.num_nodes = num_nodes
    self.features = features


    self.featbased_indices = featbased_indices
    self.featbased_values = featbased_values

    self.structural_indices = structural_indices
    self.structural_values = structural_values

    self.adj_indices = adj_indices
    self.adj_values = adj_values

    self.adj = sp.coo_matrix((self.adj_values, (self.adj_indices[:, 0], self.adj_indices[:, 1])), shape=(self.num_nodes, self.num_nodes))
    self.featbased = sp.coo_matrix((self.featbased_values, (self.featbased_indices[:, 0], self.featbased_indices[:, 1])), shape=(self.num_nodes, self.num_nodes))
    self.structural = sp.coo_matrix((self.structural_values, (self.structural_indices[:, 0], self.structural_indices[:, 1])), shape=(self.num_nodes, self.num_nodes))

    self.mask = np.zeros((2000, 2000))
    for ind in range(50):
      self.mask[ind*40:(ind + 1)*40, ind*40:(ind + 1)*40] = 1
    self.mask -= np.eye(2000)


    # import IPython
    # IPython.embed()

    # print ("sum(adj) = {}, sum(y1) = {}, sum(y2) = {}".format(sum(self.adj.values), sum(self.featbased), sum(self.structural)))

    # self.sp_features_tensor = None
    # self.sp_adj_tensor = None
    # self.sp_featbased_tensor = None
    # self.sp_structural_tensor = None

  def show_info(self):
    print ("\tNumber of nodes = {}".format(self.num_nodes))
    print ("\tNumber of edges + self-loop = {}".format(len(self.adj_indices)))
    print ("\tNumber of feature-based edges = {}".format(len(self.featbased_indices)))
    print ("\tNumber of structural-based edges = {}".format(len(self.structural_indices)))

  def sparse_feature_tensor(self):
      xrows, xcols = self.features.nonzero()

      if True: # For synthetic data only 
        x_values = self.features[xrows, xcols]
      else:
        x_values = tf.ones([len(xrows)], dtype=tf.float32)

      dense_shape = self.features.shape

      # import IPython
      # IPython.embed()

      return sp.coo_matrix((x_values, (xrows, xcols)), shape=dense_shape)


    # if self.sp_features_tensor is None:
    #   xrows, xcols = self.features.nonzero()
    #   self.x_indices = np.concatenate(
    #       [np.expand_dims(xrows, 1), np.expand_dims(xcols, 1)], axis=1)

    #   if True: # For synthetic data only 
    #     x_values = self.features[xrows, xcols]
    #   else:
    #     x_values = tf.ones([len(xrows)], dtype=tf.float32)

    #   dense_shape = self.features.shape

    #   # import IPython
    #   # IPython.embed()

    #   self.sp_features_tensor = tf.SparseTensor(
    #       self.x_indices, x_values, dense_shape)

    # return self.sp_features_tensor

  def sparse_adj_tensor(self):
    if self.sp_adj_tensor is None:
      self.sp_adj_tensor = get_sparse_adj_tensor(self.num_nodes, len(self.adj_indices), "adj")

    return self.sp_adj_tensor

  def sparse_feat_based_tensor(self):
    if self.sp_featbased_tensor is None:
      self.sp_featbased_tensor = get_sparse_adj_tensor(self.num_nodes, len(self.featbased_indices), "y1")
    return self.sp_featbased_tensor


  def sparse_structural_tensor(self):
    if self.sp_structural_tensor is None:
      self.sp_structural_tensor = get_sparse_adj_tensor(self.num_nodes, len(self.structural_indices), "y2")

    return self.sp_structural_tensor

  def get_next_batch(self):
    # self.x_shapes = self.next_batch[0]
    # self.x = self.next_batch[1]
    # self.y = self.next_batch[2]

    # TODO[ninoch]: How to handle sparse y, with sigmoid cross entropy? A1, A2 are not sparse, but y1, y2 could be sparse.
    # return self.sparse_feature_tensor(), self.sparse_adj_tensor(), self.sparse_feat_based_tensor(), self.sparse_structural_tensor()
    # return self.sparse_feature_tensor(), self.sparse_adj_tensor(), tf.sparse.to_dense(self.sparse_feat_based_tensor()), tf.sparse.to_dense(self.sparse_structural_tensor())


    # import IPython
    # IPython.embed()

    # return convert_sparse_matrix_to_sparse_tensor(self.sparse_feature_tensor()), convert_sparse_matrix_to_sparse_tensor(self.adj), np.array(self.featbased.todense()), np.array(self.structural.todense()), self.mask

    khar = self.adj.todense()
    khar[khar > 0] = 1
    khar -= np.eye(2000)

    return convert_sparse_matrix_to_sparse_tensor(self.sparse_feature_tensor()), convert_sparse_matrix_to_sparse_tensor(self.adj), np.array(self.featbased.todense()), khar, self.mask

    # return convert_sparse_matrix_to_sparse_tensor(self.sparse_feature_tensor()), convert_sparse_matrix_to_sparse_tensor(self.adj), khar, khar, self.mask


