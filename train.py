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
    sess = tf.InteractiveSession()
    # Input placeholders
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 4], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 919], name='y-input')

    # We can't initialize these variables to 0 - the network will get stuck.
    def weight_variable(shape):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

 
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
