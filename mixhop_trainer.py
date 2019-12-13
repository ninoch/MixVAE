
# Standard imports.
import collections
import json
import os
import pickle

# Third-party imports.
from absl import app
from absl import flags
import numpy
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.keras import regularizers as keras_regularizers
import simple_dataset as mixhop_dataset
import mixhop_model
import numpy as np
import matplotlib.pyplot as plt

from utils import masked_softmax_cross_entropy

# IO Flags.
flags.DEFINE_string('dataset_dir',
                    # os.path.join(os.environ['HOME'], 'data/planetoid/data'),
                    os.path.join('data/synthetic/'),
                    'Directory containing all datasets. We assume the format ')
flags.DEFINE_string('results_dir', 'results',
                    'Evaluation results will be written here.')
flags.DEFINE_string('train_dir', 'trained_models',
                    'Directory where trained models will be written.')
flags.DEFINE_string('run_id', '',
                    'Will be included in output filenames for model (in '
                    '--train_dir) and results (in --results_dir).')
flags.DEFINE_boolean('retrain', False,
                     'If set, model will retrain even if its results file '
                     'exists')

# Dataset Flags.
flags.DEFINE_string('dataset_name', 'ind.pubmed', '')
flags.DEFINE_integer('num_train_nodes', -20,
                     'Number of training nodes. If < 0, then the number is '
                     'converted to positive and that many training nodes are '
                     'used per class. -20 recovers setting in Kipf & Welling.')
flags.DEFINE_integer('num_validate_nodes', 500, '')

# Model Architecture Flags.
flags.DEFINE_string('architecture', '',
                    '(Optional) path to model architecture JSON file. '
                    'If given, none of the architecture flags matter anymore: '
                    'the contents of the file will entirely specify the '
                    'architecture. For example, see architectures/pubmed.json')
flags.DEFINE_string('hidden_dims_csv', '60',
                    'Comma-separated list of hidden layer sizes.')
flags.DEFINE_string('output_layer', 'wsum',
                    'One of: "wsum" (weighted sum) or "fc" (fully-connected).')
flags.DEFINE_string('nonlinearity', 'relu', '')
flags.DEFINE_string('adj_pows', '1',
                    'Comma-separated list of Adjacency powers. Setting to "1" '
                    'recovers valinna GCN. Setting to "0,1,2" uses '
                    '[A^0, A^1, A^2]. Further, you can feed as '
                    '"0:20:10,1:10:10", where the syntax is '
                    '<pow>:<capacity in layer1>:<capacity in layer2>. The '
                    'number of layers equals number of entries in '
                    '--hidden_dims_csv, plus one (for the output layer). The '
                    'capacities do *NOT* have to add-up to the corresponding '
                    'entry in hidden_dims_csv. They will be re-scaled if '
                    'necessary.')

# Training Flags.
flags.DEFINE_integer('num_train_steps', 100, 'Number of training steps.')
flags.DEFINE_integer('early_stop_steps', 50, 'If the validation accuracy does '
                     'not increase for this many steps, training is halted.')
flags.DEFINE_float('l2reg', 5e-4, 'L2 Regularization on Kernels.')
flags.DEFINE_float('lasso_reg', 0, 'Group Lasso Regularizer.')
flags.DEFINE_integer('architecture_search_steps', 0,
                     'Number of steps with L2 regularizer.')

flags.DEFINE_float('input_dropout', 0.7, 'Dropout applied at input layer')
flags.DEFINE_float('layer_dropout', 0.9, 'Dropout applied at hidden layers')
flags.DEFINE_string('optimizer', 'GradientDescentOptimizer',
                    'Name of optimizer to use. Must be member of tf.train.')
flags.DEFINE_float('learn_rate', 0.5, 'Learning Rate for the optimizer.')
flags.DEFINE_float('lr_decrement_ratio_of_initial', 0.01,
                   'Learning rate will be decremented by '
                   'this value * --learn_rate.')
flags.DEFINE_float('lr_decrement_every', 40,
                   'Learning rate will be decremented every this many steps.')
flags.DEFINE_integer('num_nodes', 2000 , 'Number of nodes in the training graph')
flags.DEFINE_integer('num_features', 4, 'Number of features for each node')
FLAGS = flags.FLAGS


def GetEncodedParams():
  """Summarizes all flag values in a string, to be used in output filenames."""
  return '_'.join([
      'ds-%s' % FLAGS.dataset_name,
      'r-%s' % FLAGS.run_id,
      'opt-%s' % FLAGS.optimizer,
      'lr-%g' % FLAGS.learn_rate,
      'l2-%g' % FLAGS.l2reg,
      'o-%s' % FLAGS.output_layer,
      'act-%s' % FLAGS.nonlinearity,
      'tr-%i' % FLAGS.num_train_nodes,
      'pows-%s' % FLAGS.adj_pows.replace(',', 'x').replace(':', '.'),
  ])


class AccuracyMonitor(object):
  """Monitors and remembers model parameters @ best validation accuracy."""

  def __init__(self, sess, early_stop_steps):
    """Initializes AccuracyMonitor.
    
    Args:
      sess: (singleton) instance of tf.Session that is used for training.
      early_stop_steps: int with number of steps to allow without any
        improvement on the validation accuracy.
    """
    self._early_stop_steps = early_stop_steps
    self._sess = sess
    # (validate accuracy, test accuracy, step #), recorded at best validate
    # accuracy.
    self.best = (0, 0, 0)
    # Will be populated to dict of all tensorflow variable names to their values
    # as numpy arrays.
    self.params_at_best = None 

  def mark_accuracy(self, validate_accuracy, test_accuracy, i):
    curr_accuracy = (float(validate_accuracy), float(test_accuracy), i)
    self.curr_accuracy = curr_accuracy
    if curr_accuracy > self.best:
      self.best = curr_accuracy
      all_variables = tf.global_variables()
      all_variable_values = self._sess.run(all_variables)
      params_at_best_validate = (
          {var.name: val
           for var, val in zip(all_variables, all_variable_values)})
      self.params_at_best = params_at_best_validate

    if i > self.best[-1] + self._early_stop_steps:
      return False
    return True


# TODO(haija): move to utils.
class AdjacencyPowersParser(object):
  
  def __init__(self):
    powers = FLAGS.adj_pows.split(',')

    has_colon = None
    self._powers = []
    self._ratios = []
    for i, p in enumerate(powers):
      if i == 0:
        has_colon = (':' in p)
      else:
        if has_colon != (':' in p):
          raise ValueError(
              'Error in flag --adj_pows. Either all powers or non should '
              'include ":"')
      #
      components = p.split(':')
      self._powers.append(int(components[0]))
      if has_colon:
        self._ratios.append(list(map(float, components[1:])))
      else:
        self._ratios.append([1])

  def powers(self):
    return self._powers

  def output_capacity(self, num_classes):
    if all([len(s) == 1 and s[0] == 1 for s in self._ratios]):
      return num_classes * len(self._powers)
    else:
      return sum([s[-1] for s in self._ratios])

  def divide_capacity(self, layer_index, total_dim):
    sizes = [l[min(layer_index, len(l)-1)] for l in self._ratios]
    sum_units = numpy.sum(sizes)
    size_per_unit = total_dim / float(sum_units)
    dims = []
    for s in sizes[:-1]:
      dim = int(numpy.round(s * size_per_unit))
      dims.append(dim)
    dims.append(total_dim - sum(dims))
    return dims


class ColumnLassoRegularizer(keras_regularizers.Regularizer):
  """Applies Lasso Regularization on every column of parameter matrices.
  
  Two-stage training (Section 4.2).
  """

  def get_config(self):
    return {'coef': float(self.coef)}

  def __init__(self, coef=0.):
    self.coef = coef

  def __call__(self, x):
    regularization = 0.
    num_columns = int(x.shape[1])
    k = keras_regularizers.K
    for c in range(num_columns):
      regularization = k.sqrt(k.sum(k.square(x[:, c]))) + regularization

    return regularization * self.coef


class CombinedRegularizer(keras_regularizers.Regularizer):
  
  def __init__(self, lasso=0., l2=0.):
    self.lasso = ColumnLassoRegularizer(lasso)
    self.l2 = keras_regularizers.l2(l2)
    self.config = {'lasso': lasso, 'l2': l2}

  def get_config(self):
    return self.config

  def __call__(self, x):
    return self.lasso(x) + self.l2(x)

def create_dim_inds(dim_list):
    z1_inds = []
    z2_inds = []
    offset = 0
    print('This is the dim list:  ')
    print(dim_list)
    for i in range(len(dim_list)):
        for j in range(dim_list[i]//2):
            z1_inds.append(offset + j)
            z2_inds.append(offset + int(dim_list[i]/2) + j)
        offset += dim_list[i]
    return np.array(z1_inds), np.array(z2_inds)

def build_model(sparse_adj, x, is_training, kernel_regularizer, num_x_entries):
    model = mixhop_model.MixHopModel(
        sparse_adj, x, is_training, kernel_regularizer)
    if False: # FLAGS.architecture:
        model.load_architecture_from_file(FLAGS.architecture)
    else:
        model.add_layer('mixhop_model', 'sparse_dropout', FLAGS.input_dropout,
                        num_x_entries, pass_is_training=True)
        model.add_layer('tf', 'sparse_tensor_to_dense')
        model.add_layer('tf.nn', 'l2_normalize', axis=1)

        power_parser = AdjacencyPowersParser()
        layer_dims = list(map(int, FLAGS.hidden_dims_csv.split(',')))

        for j, dim in enumerate(layer_dims):
            if j != 0:
                model.add_layer('tf.layers', 'dropout', FLAGS.layer_dropout,
                                pass_training=True)
            capacities = power_parser.divide_capacity(j, dim)
            model.add_layer('self', 'mixhop_layer', power_parser.powers(), capacities,
                            layer_id=j, pass_kernel_regularizer=True)

            if j != len(layer_dims) - 1:
                model.add_layer('tf.contrib.layers', 'batch_norm')
                model.add_layer('tf.nn', FLAGS.nonlinearity)

        model.add_layer('mixhop_model', 'reorder', create_dim_inds(
            power_parser.divide_capacity(len(layer_dims) - 1, layer_dims[-1])))  # TODO: Verify the capacity
        model.add_layer('mixhop_model', 'decoder_layer')

    net = model.activations[-1]  # TODO: check this output

    ### TRAINING.
    sliced_output = net  # tf.gather(net, ph_indices)
    A1 = sliced_output[:, :net.shape[1] // 2]
    A1 = tf.reshape(A1, [A1.shape[0], A1.shape[0]])
    A2 = sliced_output[:, net.shape[1] // 2:]
    A2 = tf.reshape(A2, [A2.shape[0], A2.shape[0]])

    return A1, A2, model

def evaluate_model(A1, A2, y1_ph, y2_ph, mask_ph):
    y1_weight = (tf.math.reduce_sum(mask_ph) - tf.math.reduce_sum(y1_ph)) / tf.math.reduce_sum(y1_ph)
    y2_weight = (tf.math.reduce_sum(mask_ph) - tf.math.reduce_sum(y2_ph)) / tf.math.reduce_sum(y2_ph)

    label_loss = masked_softmax_cross_entropy(A1, y1_ph, mask_ph, y1_weight)
    label_loss += masked_softmax_cross_entropy(A2, y2_ph, mask_ph, y2_weight)

    correct_prediction1 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(A1), 0.5), tf.int32), tf.cast(y1_ph, tf.int32))
    correct_prediction1 = tf.cast(correct_prediction1, tf.float32) 
    correct_prediction1 *= (mask_ph / tf.reduce_mean(mask_ph))
    acc1 = tf.reduce_mean(correct_prediction1)

    correct_prediction2 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(A2), 0.5), tf.int32), tf.cast(y2_ph, tf.int32))
    correct_prediction2 = tf.cast(correct_prediction2, tf.float32) 
    correct_prediction2 *= (mask_ph / tf.reduce_mean(mask_ph))
    acc2 = tf.reduce_mean(correct_prediction2)

    return label_loss, acc1, acc2


def save_model(acc_monitor):
    encoded_params = GetEncodedParams()
    output_results_file = os.path.join(
        FLAGS.results_dir, encoded_params + '.json')
    output_model_file = os.path.join(
        FLAGS.train_dir, encoded_params + '.pkl')
    if os.path.exists(output_results_file) and not FLAGS.retrain:
        print('Exiting early. Results are already computed: %s. Pass flag '
              '--retrain to override' % output_results_file)
        return
    if not os.path.exists(FLAGS.results_dir):
        os.makedirs(FLAGS.results_dir)
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    with open(output_results_file, 'w') as fout:
        results = {
            'at_best_validate': acc_monitor.best,
            'current': acc_monitor.curr_accuracy,
        }
        fout.write(json.dumps(results))

    with open(output_model_file, 'wb') as fout:
        pickle.dump(acc_monitor.params_at_best, fout)
    print('Wrote model to ' + output_model_file)
    print('Wrote results to ' + output_results_file)


def main(unused_argv):

  ### LOAD DATASET
  dataset = mixhop_dataset.ReadDataset(FLAGS.dataset_dir, FLAGS.dataset_name)
  dataset.show_info()
  # 9630.0 2090.0 7756.0

  eval_dataset = mixhop_dataset.ReadDataset(FLAGS.dataset_dir, 'sim_eval')
  eval_dataset.show_info()


  test_dataset = mixhop_dataset.ReadDataset(FLAGS.dataset_dir, 'sim_test')
  test_dataset.show_info()


  ### MODEL REQUIREMENTS (Placeholders, adjacency tensor, regularizers)
  x_ph = tf.sparse_placeholder(tf.float32, [FLAGS.num_nodes, FLAGS.num_features], name='x') # dataset.get_next_batch() #TODO: check the shape
  sparse_adj_ph = tf.sparse_placeholder(tf.float32, [FLAGS.num_nodes, FLAGS.num_nodes], name='sparse_adj') #dataset.sparse_adj_tensor() #TODO: check it to be placeholder, shape, sparsity
  y1_ph = tf.placeholder(tf.float32, [FLAGS.num_nodes, FLAGS.num_nodes], name='y1') #TODO: check the shape of placeholder
  y2_ph = tf.placeholder(tf.float32, [FLAGS.num_nodes, FLAGS.num_nodes], name='y2') #TODO: check the shape of placeholder
  mask_ph = tf.placeholder(tf.float32, [FLAGS.num_nodes, FLAGS.num_nodes], name='mask') #TODO: check the shape of placeholder
  is_training = tf.placeholder_with_default(True, [], name='is_training')

  pows_parser = AdjacencyPowersParser()  # Parses flag --adj_pows
  num_x_entries = tf.constant(FLAGS.num_nodes * FLAGS.num_features)
  kernel_regularizer = CombinedRegularizer(FLAGS.l2reg, FLAGS.l2reg) #  keras_regularizers.l2(FLAGS.l2reg)


  ### BUILD MODEL
  A1, A2, model = build_model(sparse_adj_ph, x_ph, is_training, kernel_regularizer, num_x_entries)
  model.show_model_info()
  learn_rate = tf.placeholder(tf.float32, [], 'learn_rate')
  label_loss, acc1, acc2 = evaluate_model(A1, A2, y1_ph, y2_ph, mask_ph)
  tf.losses.add_loss(label_loss)
  loss = tf.losses.get_total_loss()
  
  if FLAGS.optimizer == 'MomentumOptimizer':
    optimizer = tf.train.MomentumOptimizer(lr, 0.7, use_nesterov=True)
  else:
    optimizer_class = getattr(tf.train, FLAGS.optimizer)
    optimizer = optimizer_class(learn_rate)
  train_op = slim.learning.create_train_op(
      loss, optimizer, gradient_multipliers=[])

  ### CRAETE SESSION
  # Now that the graph is frozen
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  LAST_STEP = collections.Counter()
  accuracy_monitor = AccuracyMonitor(sess, FLAGS.early_stop_steps)

  # Step function makes a single update, prints accuracies, and invokes
  # accuracy_monitor to keep track of test accuracy and parameters @ best
  # validation accuracy


  def construct_feed_dict(lr, is_tr, x_batch, adj_batch, y1_batch = None, y2_batch = None, mask_batch=None):
      feed_dict = {}

      feed_dict[is_training] = is_tr
      if lr is not None:
        feed_dict[learn_rate] = lr

      feed_dict[x_ph.indices], feed_dict[x_ph.values] = x_batch.indices, x_batch.values
      feed_dict[sparse_adj_ph.indices], feed_dict[sparse_adj_ph.values] = adj_batch.indices, adj_batch.values

  # if is_tr == True:
      feed_dict[y1_ph] = y1_batch
      feed_dict[y2_ph] = y2_batch
      feed_dict[mask_ph] = mask_batch
      
      return feed_dict


  def step(dataset, lr=None, columns=None):
    i = LAST_STEP['step']
    LAST_STEP['step'] += 1

    x_batch, adj_batch, y1_batch, y2_batch, mask_batch = dataset.get_next_batch()
    feed_dict = construct_feed_dict(lr, True, x_batch, adj_batch, y1_batch, y2_batch, mask_batch)

    # Train step
    train_preds_A1, train_preds_A2, loss_value, _, a1, a2= sess.run((A1, A2, label_loss, train_op, acc1, acc2), feed_dict = feed_dict)

    
    if np.isnan(loss_value).any():
      print('NaN value reached. Debug please.')
      import IPython; IPython.embed()

    masked_A1 = np.where(mask_batch, train_preds_A1, 0)
    masked_A2 = np.where(mask_batch, train_preds_A2, 0)

    print ("Loss = {0:.2f}".format(loss_value), )
    print ("\t acc1 = {0:.4f}, acc2 = {1:.4f}".format(a1, a2), )
    print ("\t A1: +={}, -={}".format(np.sum(np.array(masked_A1) > 0), np.sum(np.array(masked_A1) < 0)), )
    print ("\t A2: +={}, -={}".format(np.sum(np.array(masked_A2) > 0), np.sum(np.array(masked_A2) < 0)))

    #TODO: add validation set -> monitor accuracy here
    x_dev, adj_dev, y1_dev, y2_dev, mask_dev = eval_dataset.get_next_batch() #TODO: for nazanin - a developement set is required
    feed_dict = construct_feed_dict(lr, False, x_dev, adj_dev, y1_dev, y2_dev, mask_dev)
    _, _, _, a1, a2 = sess.run((A1, A2, train_op, acc1, acc2), feed_dict=feed_dict)
    keep_going = accuracy_monitor.mark_accuracy(a1 + a2,  a1 + a2, i)
    #
    # print('%i. (loss=%g). Acc: train=%f val=%f test=%f  (@ best val test=%f)' % (
    #     i, loss_value, train_accuracy, validate_accuracy, test_accuracy,
    #     accuracy_monitor.best[1]))
    # if keep_going:
    #   return True
    # else:
    #     print('Early stopping')
    return keep_going, loss_value, a1, a2

  ### TRAINING LOOP
  lr = FLAGS.learn_rate
  lr_decrement = FLAGS.lr_decrement_ratio_of_initial * FLAGS.learn_rate
  loss_arr = []
  acc1_arr = []
  acc2_arr = []
  for i in range(FLAGS.num_train_steps):
    keep_going, loss_val, ac1, ac2 = step(dataset, lr=lr)
    if i == 40:
      x_batch, adj_batch, y1_batch, y2_batch, mask_batch = dataset.get_next_batch()
      feed_dict = construct_feed_dict(lr, True, x_batch, adj_batch, y1_batch, y2_batch, mask_batch) 
      z = sess.run(model.activations[-3], feed_dict=feed_dict)
      import IPython; IPython.embed()

    loss_arr.append(loss_val)
    acc1_arr.append(ac1)
    acc2_arr.append(ac2)
    if not keep_going:
      break

    if i > 0 and i % FLAGS.lr_decrement_every == 0:
      lr -= lr_decrement
      if lr <= 0:
        break

  save_model(accuracy_monitor)


  print(accuracy_monitor.best)
  plt.plot(acc1_arr, 'b-')
  plt.plot(acc2_arr, 'r.')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.show()
  plt.plot(loss_arr, 'g+')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.show()

  ## Test data
  ops = [v.assign(accuracy_monitor.params_at_best[v.name]) for v in tf.global_variables()]
  sess.run(ops)
    # Test data
  x_batch, adj_batch, y1_batch, y2_batch, mask_batch = test_dataset.get_next_batch()
  feed_dict = construct_feed_dict(lr, False, x_batch, adj_batch, y1_batch, y2_batch, mask_batch)
  train_preds_A1, train_preds_A2, loss_value, _, a1, a2= sess.run((A1, A2, label_loss, train_op, acc1, acc2), feed_dict = feed_dict)
  pr_1 = np.mean(np.abs(y1_batch - np.where(mask_batch, train_preds_A1, 0)))
  pr_2 = np.mean(np.abs(y2_batch - np.where(mask_batch, train_preds_A2, 0)))
  print("Loss = {0:.2f}, Train distance to label A1 = {1:.5f}, Train distance to label A2 = {2:.5f}".format(loss_value, pr_1, pr_2))
  print("\t acc1 = {0:.4f}, acc2 = {1:.4f}".format(a1, a2))
  import IPython
  IPython.embed()
if __name__ == '__main__':
  app.run(main)
