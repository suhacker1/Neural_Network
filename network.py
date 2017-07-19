#Dependencies
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import tensorflow as tf

#Initializing datasets as numpy arrays stored in variables
train_images = np.load('/Users/suhahussain/Documents/MachineLearning/Data/train-images-idx3.npy')
train_labels = np.load('/Users/suhahussain/Documents/MachineLearning/Data/train-labels-idx1.npy')
test_images = np.load('/Users/suhahussain/Documents/MachineLearning/Data/t10k-images-idx3.npy')
test_labels = np.load('/Users/suhahussain/Documents/MachineLearning/Data/t10k-labels-idx1.npy')
#Note: These datasets were taken from the MNIST database and converted into numpy arrays in conversion.py

#Global varable definition
global images
global labels
global number

#Partitions data into images and labels and defines a function that creates a new batch of data for training and testing
class data(object):
    def __init__(self, images, labels):
        self.i = images
        self.l = labels
        #Function to create a batch of data
    def nextbatch(self,images, labels, batch_size):
        #Permutes images and labels in unison
        permutation = np.random.permutation(len(images))
        shuffled_images, shuffled_labels = images[permutation], labels[permutation]
        #Cuts the size down into a batch
        batch_images = np.array(shuffled_images[0:batch_size])
        batch_labels = np.array(shuffled_labels[0:batch_size])
        #Formats labels as one-hot vectors
        batch_labels = np.eye(10)[batch_labels]
        return batch_images, batch_labels

#Definition of the Neural Network
def deepnn(x):
    x_image = tf.cast((tf.reshape(x, [-1, 28, 28, 1])),tf.float32)

    # First convolutional layer - maps one grayscale image to 32 feature maps.
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # Pooling layer - downsamples by 2X.
    h_pool1 = max_pool_2x2(h_conv1)

    # Second convolutional layer -- maps 32 feature maps to 64.
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # Second pooling layer.
    h_pool2 = max_pool_2x2(h_conv2)

    # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
    # is down to 7x7x64 feature maps -- maps this to 1024 features.
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Map the 1024 features to 10 classes, one for each digit
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    return y_conv, keep_prob


def conv2d(x, W):
    #conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    #max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    #weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    #bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(_):
    # Create the model; input is images
    x = tf.placeholder(tf.float32, [None, 28, 28])

    # Define loss and optimizer; input is the label
    y_ = tf.placeholder(tf.float32, [None, 10])

    # Build the graph for the deep net; defines the predicted label and the keep probability as results of the neural net
    y_conv, keep_prob = deepnn(x)

    #Prediction and optimization: Determining the accuracy of the network from the correct prediction and the actual prediction
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Starts Tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        uno = data(train_images,train_labels)
        #Training
        for number in range(20000):
            batch_images, batch_labels = uno.nextbatch(uno.i,uno.l,30) #Creates a batch of data from train_labels and train_images
            if number % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x:
                    batch_images, y_: batch_labels, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (number, train_accuracy)) #Prints accuracy every 100 steps
            train_step.run(feed_dict={x: batch_images, y_: batch_labels, keep_prob: 0.5})
        #Testing
        new_test_labels = np.eye(10)[test_labels]
        print('test accuracy %g' % accuracy.eval(feed_dict={x: test_images, y_: new_test_labels, keep_prob: 1.0}))

if __name__ == '__main__':
    tf.app.run(main=main)
