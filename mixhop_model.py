import copy
import json

import tensorflow as tf

def sparse_dropout(x, drop_prob, num_entries, is_training):
  """Dropout for sparse tensors."""
  keep_prob = 1.0 - drop_prob
  is_test_float = 1.0 - tf.cast(is_training, tf.float32)
  random_tensor = is_test_float + keep_prob
  random_tensor += tf.random_uniform([num_entries])
  dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
  pre_out = tf.sparse_retain(x, dropout_mask)
  return pre_out * (1./tf.maximum(is_test_float, keep_prob))


def psum_output_layer(x, num_classes):
  num_segments = int(x.shape[1]) / num_classes
  if int(x.shape[1]) % num_classes != 0:
    print('Wasted psum capacity: %i out of %i' % (
        int(x.shape[1]) % num_classes, int(x.shape[1])))
  sum_q_weights = tf.get_variable(
      'psum_q', shape=[num_segments], initializer=tf.zeros_initializer, dtype=tf.float32, trainable=True)
  tf.losses.add_loss(tf.reduce_mean((sum_q_weights ** 2)) * 1e-3 )
  softmax_q = tf.nn.softmax(sum_q_weights)  # softmax
  psum = 0
  for i in range(int(num_segments)):
    segment = x[:, i*num_classes : (i+1)*num_classes]
    psum = segment * softmax_q[i] + psum
  return psum


def adj_times_x(adj, x, adj_pow=1):
  """Multiplies (adj^adj_pow)*x."""
  for i in range(adj_pow):
    x = tf.sparse_tensor_dense_matmul(adj, x)
  return x

def reorder(z, dim_inds):
  print('This is the dim_inds')
  print(dim_inds)
  print ('this is shape of z: ', z.shape)
  z_feat = tf.gather(z, dim_inds[0], axis=1)
  print ('this is shape of feat: ', z_feat.shape)
  z_struct = tf.gather(z, dim_inds[1], axis=1)
  print ('this is shape of struct: ', z_struct.shape)
  combined = tf.concat([z_feat, z_struct], axis = 1)
  print ('this is shape of combined: ', combined.shape)
  return combined

def dense_flow(z):
  a1 = z[:, :z.shape[1]//2]
  a2 = z[:, z.shape[1]//2:]
  a1_mean = tf.layers.dense(a1, z.shape[1]//2)
  a1_log_sigma = tf.layers.dense(a1, z.shape[1]//2)
  a1_normal = tf.concat([a1_mean, a1_log_sigma], axis=1)
  a2_mean = tf.layers.dense(a2, z.shape[1]//2)
  a2_log_sigma = tf.layers.dense(a2, z.shape[1]//2)
  a2_normal = tf.concat([a2_mean, a2_log_sigma], axis=1)
  z_tmp = tf.concat([a1_normal, a2_normal], axis=1)
  return z_tmp

def _sample_z(z_mean, z_log_sigma_sq):
    eps_shape = tf.shape(z_mean)
    eps = tf.random_normal(eps_shape, 0, 1, dtype=tf.float32 )
    # z = mu + sigma * epsilon
    z = tf.add(z_mean,
               tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps))
    return z

def sample(z_var):
  a1 = z_var[:, :z_var.shape[1]//2]
  a2 = z_var[:, z_var.shape[1]//2:]
  a1_sample = _sample_z(a1[:, :z_var.shape[1]//4], a1[:, z_var.shape[1]//4:])
  a2_sample = _sample_z(a2[:, :z_var.shape[1]//4], a2[:, z_var.shape[1]//4:])
  return tf.concat([a1_sample, a2_sample], axis=1)



def glorot(shape, name=None):
    import numpy as np
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


W1 = tf.Variable(glorot([1, 9]))
B1 = tf.Variable(tf.random_normal([1]))

W2 = tf.Variable(glorot([1, 9]))
B2 = tf.Variable(tf.random_normal([1]))

def decode(z):
  x_t = tf.transpose(z)
  x_t = tf.matmul(z, x_t)
  return x_t

def decoder_layer(z):
  # z1 = decode(z[:, :z.shape[1]//2])
  # z2 = decode(z[:, z.shape[1]//2:])

  z1 = z[:, :z.shape[1]//2]
  z2 = z[:, z.shape[1]//2:]

  A_feat = tf.matmul(tf.multiply(z1, W1), tf.transpose(z1)) + B1 
  A_struct = tf.matmul(tf.multiply(z2, W2), tf.transpose(z2)) + B2

  combined = tf.concat([A_feat, A_struct], axis=1)
  return combined #TODO: check whether combining them is better

def mixhop_layer(x, sparse_adjacency, adjacency_powers, dim_per_power,
                 kernel_regularizer=None, layer_id=None, replica=None):
  """Constructs MixHop layer.

  Args:
    sparse_adjacency: Sparse tensor containing square and normalized adjacency
      matrix.
    adjacency_powers: list of integers containing powers of adjacency matrix.
    dim_per_power: List same size as `adjacency_powers`. Each power will emit
      the corresponding dimensions.
    layer_id: If given, will be used to name the layer
  """
  #
  replica = replica or 0
  layer_id = layer_id or 0
  segments = []
  for p, dim in zip(adjacency_powers, dim_per_power):
    net_p = adj_times_x(sparse_adjacency, x, p)

    with tf.variable_scope('r%i_l%i_p%s' % (replica, layer_id, str(p))):
      layer = tf.layers.Dense(
          dim,
          kernel_regularizer=kernel_regularizer,
          activation=None, use_bias=False)
      net_p = layer.apply(net_p)

    segments.append(net_p)
  return tf.concat(segments, axis=1)


MODULE_REFS = {
    'tf': tf,
    'tf.layers': tf.layers,
    'tf.nn': tf.nn,
    'tf.sparse': tf.sparse,
    'tf.contrib.layers': tf.contrib.layers
}

class MixHopModel(object):
  """Builds MixHop architectures. Used as architectures can be learned.
  
  Use like:
    model = MixHopModel(sparse_adj, x, is_training, kernel_regularizer)
    ...
    model.add_layer('<module_name>', '<fn_name>', args_to_fn)
    model.add_layer( ... )
    ...

  Where <module_name> must be a string defined in MODULE_REFS, and <fn_name>
  must be a function living inside module indicated by <module_name>, finally,
  args_to_fn are passed as-is to the function (with name <fn_name>), with the
  exception of arguments:
    pass_kernel_regularizer: if argument is present, then we pass
      kernel_regularizer argument with value given to the constructor.
    pass_is_training: if argument is present, then we pass is_training argument
      with value given to the constructor.
    pass_training: if argument is present, then we pass training argument with
      value of is_training given to the constructor.
  
  In addition <module_name> can be:
    'self': invokes functions in this class.
    'mixhop_model': invokes functions in this file.

  See example_pubmed_model() for reference.
  """
  
  def __init__(self, sparse_adj, sparse_input, is_training, kernel_regularizer):
    self.is_training = is_training
    self.kernel_regularizer = kernel_regularizer 
    self.sparse_adj = sparse_adj
    self.sparse_input = sparse_input
    self.layer_defs = []
    self.activations = [sparse_input]

  def save_architecture_to_file(self, filename):
    with open(filename, 'w') as fout:
      fout.write(json.dumps(self.layer_defs, indent=2))

  def load_architecture_from_file(self, filename):
    if self.layer_defs:
      raise ValueError('Model is (partially) initialized. Cannot load.')
    layer_defs = json.loads(open(filename).read())
    for layer_def in layer_defs:
      self.add_layer(layer_def['module'], layer_def['fn'], *layer_def['args'],
                     **layer_def['kwargs'])

  def show_model_info(self):
    print ('############ MODEL ##############')
    for ind in range(len(self.layer_defs)):
      print (self.layer_defs[ind]['module'], self.layer_defs[ind]['fn'])
      print (self.activations[ind].get_shape())
      print ('---------------------------------')
    print ('##################################')

  def add_layer(self, module_name, layer_fn_name, *args, **kwargs):
    #
    self.layer_defs.append({
        'module': module_name,
        'fn': layer_fn_name,
        'args': args,
        'kwargs': copy.deepcopy(kwargs),
    }) # TODO: the decoders should be added here
    #
    if 'pass_training' in kwargs:
      kwargs.pop('pass_training')
      kwargs['training'] = self.is_training
    if 'pass_is_training' in kwargs:
      kwargs.pop('pass_is_training')
      kwargs['is_training'] = self.is_training
    if 'pass_kernel_regularizer' in kwargs:
      kwargs.pop('pass_kernel_regularizer')
      kwargs['kernel_regularizer'] = self.kernel_regularizer
    #
    fn = None
    if module_name == 'mixhop_model':
      fn = globals()[layer_fn_name]
    elif module_name == 'self':
      fn = getattr(self, layer_fn_name)
    elif module_name in MODULE_REFS:
      fn = getattr(MODULE_REFS[module_name], layer_fn_name)
    else:
      raise ValueError(
          'Module name %s not registered in MODULE_REFS' % module_name)
    self.activations.append(
        fn(self.activations[-1], *args, **kwargs))

  def mixhop_layer(self, x, adjacency_powers, dim_per_power,
                   kernel_regularizer=None, layer_id=None, replica=None):
    return mixhop_layer(x, self.sparse_adj, adjacency_powers, dim_per_power,
                        kernel_regularizer, layer_id, replica)

def example_pubmed_model(
    sparse_adj, x, num_x_entries, is_training, kernel_regularizer, input_dropout,
    layer_dropout, num_classes=3):
  """Returns PubMed model with test performance ~>80.4%.
  
  Args:
    sparse_adj: Sparse tensor of normalized adjacency matrix.
    x: Sparse tensor of feature matrix.
    num_x_entries: number of non-zero entries of x. Used for sparse dropout.
    is_training: boolean scalar Tensor.
    kernel_regularizer: Keras regularizer object.
    input_dropout: Float in range [0, 1.0). How much to drop out from input.
    layer_dropout: Dropout value for dense layers.
  """ 
  model = MixHopModel(sparse_adj, x, is_training, kernel_regularizer)

  model.add_layer('mixhop_model', 'sparse_dropout', input_dropout,
                  num_x_entries, pass_is_training=True)
  model.add_layer('tf', 'sparse_tensor_to_dense')
  model.add_layer('tf.nn', 'l2_normalize', axis=1)

  # MixHop Conv layer
  model.add_layer('self', 'mixhop_layer', [0, 1, 2], [17, 22, 21], layer_id=0,
                  pass_kernel_regularizer=True)

  model.add_layer('tf.contrib.layers', 'batch_norm')
  model.add_layer('tf.nn', 'tanh')

  model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)
  # MixHop Conv layer
  model.add_layer('self', 'mixhop_layer', [0, 1, 2], [3, 1, 6], layer_id=1,
                  pass_kernel_regularizer=True)
  model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)
  # MixHop Conv layer
  model.add_layer('self', 'mixhop_layer', [0, 1, 2], [2, 4, 4], layer_id=2,
                  pass_kernel_regularizer=True)
  model.add_layer('tf.contrib.layers', 'batch_norm')
  model.add_layer('tf.nn', 'tanh')
  model.add_layer('tf.layers', 'dropout', layer_dropout, pass_training=True)
  
  # Classification Layer --> TODO: This part should be change to adapt to our loss
  # model.add_layer('tf.layers', 'dense', num_classes, use_bias=False,
  #                 activation=None, pass_kernel_regularizer=True)
  return model
