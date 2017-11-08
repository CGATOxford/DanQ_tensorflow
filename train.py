'''
this app is for training data from the DeepSEA project,
using template from 
https://github.com/uci-cbcl/DanQ/blob/master/DanQ-JASPAR_train.py
and 
https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow
/examples/tutorials/mnist/mnist_with_summaries.py
'''
import tensorflow as tf
import os
import sys
import argparse
import h5py
import scipy.io

def train():
    '''
    build the model
    conv1d: input_dim=4,
            input_length=1000,
            nb_filter=1024,
            filter_length=30,
            border_mode="valid",
            activation="relu",
            subsample_length=1 
    note that conv1d will be initialised with JASPER  
    '''
    # input data
    trainmat = h5py.File('data/deepsea_train/train.mat')
    validmat = scipy.io.loadmat('data/deepsea_train/valid.mat')
    testmat = scipy.io.loadmat('data/deepsea_train/test.mat')
    
    X_train = np.transpose(np.array(trainmat['trainxdata']), axes=(2,0,1))
    y_train = np.transpose(trainmat['traindata']).T

    # define variables 
    learning_rate = 0.001
    num_steps = 2000
    batch_size = 1000

    input_dim = 4
    input_length = 1000
    num_classes = 919
    dropout_cnn = 0.2

    sess = tf.InteractiveSession()
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, input_dim], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, num_classes], name='y-input')

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv1d(x, W, b, strides=1):
        """
        Conv1d wrapper with bias and relu activation
        
        x : tensor
        W : filter
        b : bias
        stride : number of filter is moved
        """
        x = tf.nn.conv1d(x, W, strides, "VALID")
        x = tf.nn.bias_add(x, b)
        
        return tf.nn.relu(x)
    
    def maxpool1D(x, pool_size=13, strides=13, k=1):
        """
        return a 1d max pooling layer
        
        value : tensor of the format specified by data format
        ksize : The size of window for input of tensor
        strides : The stride of sliding window
        
        """
        return tf.nn.max_pooling1d(inputs=x, pool_size=pool_size, strides=strides)

    def conv_net(x, weights, biases, dropout):
        
        conv1d = conv1d(x, weights, biases, strides = 1)
        conv1d = maxpool1D(conv1, k=1)
        
        ## Fully connected layer
        
        # Reshape to fit fully connected layer
        fc1 = tf.reshape()
        fc1 = tf.add(tf.matmul(fc1, weights['w1']), biases['b1'])
        fc1 = tf.nn.relu(fc1)
        
        # Apply dropout
        
        fc1 = tf.nn.dropout(fc1, dropout)
        
        # Output class prediction
        
        out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
        
        return out


# initialise the weights
weights = {'w1' : tf.Variable(tf.random_normal([])),
           'out' : tf.Variable(tf.random_normal([]))}

# initalise biase with size of output 
biases = {'b1' : tf.Variable(tf.random_normal([])),
         'out' : tf.Variable(tf.random_normal([]))}

# Construct model

logits = conv_net(X_train, weights, biases, dropout)
prediction = tf.nn.sigmoid(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y_train))
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimiser.minimize(loss_op)

# Evaluate model

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_train, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialise the variables

init = tf.global_variables_initializer()

 
def main(**FLAGS):
    # parameters:
    #  training data
    #  valid data
    #  test data
    #  max epochs = 32
    #  using JASPER = True
    # load training data

if __name__ == '__main__':
    
    main(**FLAGS)
