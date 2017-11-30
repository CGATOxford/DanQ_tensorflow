'''
this app is for training data from the DeepSEA project,
using template from 
https://github.com/uci-cbcl/DanQ/blob/master/DanQ-JASPAR_train.py
and 
https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow
/examples/tutorials/mnist/mnist_with_summaries.py
'''

'''
Refer to https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0
Basically, we will use high level tensorflow's API: Estimator + Experiment + Dataset
'''
import tensorflow as tf
from tensorflow.contrib.learn import learn_runner
from tensorflow.contrib import rnn
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib import slim
import numpy as np
import h5py
import scipy.io
import argparse


# tf needs >= 1.3.0
assert tf.__version__>='1.3.0', 'tensorflow version needs to be no lower than 1.3.0'

tf.logging.set_verbosity(tf.logging.DEBUG)


# Set up the FLAGS, which is basically a wrapper for argparse. The aim of this
# is to write demo code. It may be better for the future to implement arg parsing
# with argparse in the future if this code becomes stable

# FLAGS = tf.app.flags.FLAGS

# Define the model and data directories i.e. model_dir where model will be saved
# tf.app.flags.DEFINE_string(flag_name='model_dir', default_value='', docstring='A directory where the model will be saved')
# tf.app.flags.DEFINE_string(flag_name='test_mat', default_value='', docstring='A testing matrix to evaluate the model')
# tf.app.flags.DEFINE_string(flag_name='train_mat', default_value='', docstring='A training dataset to train the model')


# In the future it may be worth parameterising this with an ini file

def run_experiment(argv=None):
    '''Run the experiment'''
    
    # Define the hyperparameters
    params = tf.contrib.training.HParams(
        learning_rate=0.01,
        n_classes=919,
        train_steps=100,
        min_eval_frequency=10)
    
    # set the run config and directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)
    
    learn_runner.run(
        experiment_fn=experiment_fn,
        run_config=run_config,
        schedule="train_and_evaluate",
        hparams=params)
        
        
def experiment_fn(run_config, params):
    '''Create and experiment to train and evaluate the model'''
    
    run_config = run_config.replace(
                    save_checkpoints_steps=params.min_eval_frequency)
    
    # Define the classifier
    
    estimator = get_estimator(run_config, params)

    # Setup data loaders
    
    # testing, using valid_mat instead
    trainmat = h5py.File(FLAGS.train_mat, "r")
    testmat = scipy.io.loadmat(FLAGS.test_mat)
    
    danq_train = train_input_fn, train_input_hook = get_train_inputs(
            batch_size=10, data=trainmat, test=False)
    
    danq_test = eval_input_fn, eval_input_hook = get_test_inputs(
            batch_size=10, data=testmat, test=True)
    
    # Define the experiment

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=params.train_steps,
        min_eval_frequency=params.min_eval_frequency,
        train_monitors=[train_input_hook], # deprecated monitor is here
        eval_hooks=[eval_input_hook],
        eval_steps=None)

    return experiment
    
def get_estimator(run_config, params):
    '''Return the model as a tensorflow object'''
    
    return tf.estimator.Estimator(
            model_fn=model_fn,
            params=params,
            config=run_config)


def model_fn(features, labels, mode, params):
    '''Model function is used in the estimator and is required for running model'''
    
    is_training = mode == ModeKeys.TRAIN
    
    # Define the models architecture
    logits = architecture(features, is_training=is_training)
    predictions = tf.argmax(logits, axis=1)

    # Loss functions and not needed during inference
    
    loss = None
    train_op = None
    eval_metric_ops = {}
    
    if mode != ModeKeys.INFER:
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float32), logits=logits))
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)
    return tf.estimator.EstimatorSpec(mode=mode,
                                     predictions=predictions,
                                     loss=loss,
                                     train_op=train_op,
                                     eval_metric_ops=eval_metric_ops)
                                     
def get_train_op_fn(loss, params):
    """Get the training Op.

    Args:
         loss (Tensor): Scalar Tensor that represents the loss function.
         params (HParams): Hyperparameters (needs to have `learning_rate`)

    Returns:
        Training Op
    """
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def get_eval_metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.

    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=tf.reshape(labels, [919,1]),
            predictions=predictions,
            name='accuracy')
    }

def BDNN(x):
    '''Bidirectional neural network'''
    
    forward_lstm = rnn.LSTMCell(320, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
    backward_lstm = rnn.LSTMCell(320, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
    
    brnn, _ = tf.nn.bidirectional_dynamic_rnn(forward_lstm, backward_lstm, x, dtype=tf.float32)

    return brnn

def architecture(inputs, is_training, scope='DanQNN'):
    
    '''
    This function creates a CNN followed by a bidirectional LSTM RNN as per DanQ publication
    We aim to have this implimented in tensorflow as it will be easier to
    modify the implimentation for other uses if we incorporate with tensorboard.
    '''

    nb_filter = 320
    subsample = 1
    input_length = 1000
    border = "VALID"
    reuse = None
    max_pool_size = 13
    max_strides = 1
    filter_length = 26
    mode = None

    with tf.variable_scope(scope):

        conv1d = tf.layers.conv1d(tf.cast(inputs, tf.float32), filters=nb_filter, strides=subsample, 
                                  padding=border, kernel_size=filter_length, reuse=reuse, data_format="channels_last")
        
        max1 = tf.layers.max_pooling1d(conv1d, pool_size=max_pool_size, strides=max_strides)
        
        max1 = tf.layers.dropout(max1, rate=0.2,training=mode == tf.estimator.ModeKeys.TRAIN)
        

        brnn = BDNN(max1)


        brnn = tf.layers.dropout(brnn, rate=0.5,training=mode == tf.estimator.ModeKeys.TRAIN)

        brnn = tf.contrib.layers.flatten(brnn)

        # original code of 75 in DanQ is something related to the batch size, 
        #  or train steps. when 75, last layer's output gives (1284,919)
        # so 1284 * 75 / 100  = 963 (100 is my batch size AND train step)
        brnn = tf.reshape(brnn, [-1, 963*640])

        fc1 = tf.layers.dense(brnn, units=925, activation=tf.nn.relu)
        
        fc2 = tf.layers.dense(fc1, units=919, activation=tf.nn.sigmoid)

        return fc2


# Define data loaders #####################################
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

# Define the training inputs
def get_train_inputs(batch_size, data, test=False):
    """Return the input function to get the training data.
    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        data (Object): Object holding the loaded data.
        test (boolean): if test, then load valid mat for testing purposes
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Training_data'):
            # Get  data
            DNA = np.swapaxes(np.array(data['trainxdata']).T,2,1)
            labels = np.array(data['traindata']).T
            # Define placeholders
            DNA_placeholder = tf.placeholder(
                DNA.dtype, DNA.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            # note that cpu only accepts NHWC, i.e. channel last, 
            # therefore the transpose. if gpu, a plain transpose, combined with
            # 'channels_first' for conv1d would suffice.
            dataset = tf.data.Dataset.from_tensor_slices(
                (DNA_placeholder,labels_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={DNA_placeholder: DNA,
                               labels_placeholder: labels})
            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return train_inputs, iterator_initializer_hook

def get_test_inputs(batch_size, data, test=False):
    """Return the input function to get the test data.
    Args:
                batch_size (int): Batch size of training iterator that is returned
                      by the input function.
        data (Object): Object holding the loaded data.
        test (boolean): if test, then load valid mat for testing purposes
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def test_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
                (tf.transpose(DNA_placeholder,[2,0,1]), tf.transpose(labels_placeholder)))
        """
        with tf.name_scope('Test_data'):
            # Get data. labels need not transform, but DNA does!
            # and a different way of transform from train data!!! need optimise!
            DNA = np.swapaxes(data['validxdata'],1,2)
            labels =data['validdata']
            # Define placeholders
            DNA_placeholder = tf.placeholder(
                DNA.dtype, DNA.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.data.Dataset.from_tensor_slices(
                (DNA_placeholder, labels_placeholder))
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={DNA_placeholder: DNA,
                               labels_placeholder: labels})
            return next_example, next_label

    # Return function and hook
    return test_inputs, iterator_initializer_hook

# Run script ##############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="model")
    parser.add_argument("--test_mat", default="Data/valid.mat")
    parser.add_argument("--train_mat", default="Data/train4test.mat")
    FLAGS, _ = parser.parse_known_args()
    tf.app.run(
        main=run_experiment
    )
