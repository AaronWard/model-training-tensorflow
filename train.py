
'''
This is the model that will be used to train the deep convolutional neaurl network.

@Author : Aaron Ward 
'''

import tensorflow as tf
import os, os.path

###########################################################################################################


# Import and pre process images here


data_path = "/data"

###########################################################################################################


# Define initial variables
batch_size = 100
num_class = 6
num_epochs = 25

# Define placeholders
x = tf.placeholder('float', [None, 2500]) # 50 x 50 = 2500
y = tf.placeholder('float')

#define variables for dropout
keep_rate = .8
keep_prop = tf.placeholder(tf.float32)

###########################################################################################################


# Extract features
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #move one pixel at s time

#
def maxpool2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #pool 2 pixels at a time



###########################################################################################################


# Define the weights and biases as dictionaries and
# define structure of the network
def convolutional_network(x):
    weights = {
        'weights_conv1' : tf.Variable(tf.random_normal([5,5,1,64])),
        'weights_conv2' : tf.Variable(tf.random_normal([5,5,64,128])),
        'weights_conv3' : tf.Variable(tf.random_normal([5,5,128,256])),
        'weights_fully_con' : tf.Variable(tf.random_normal([7*7*256,4096])),
        'weights_out' : tf.Variable(tf.random_normal([4096, num_class]))
    }


    biases = {
        'bias_conv1' : tf.Variable(tf.random_normal([64])),
        'bias_conv2' : tf.Variable(tf.random_normal([128])),
        'bias_conv3' : tf.Variable(tf.random_normal([256])),
        'bias_fully_con' : tf.Variable(tf.random_normal([4096])),
        'bias_out' : tf.Variable(tf.random_normal([num_class]))
    }

    x = tf.reshape(x, shape=[-1, 50, 50, 1])

    # 3 comvolutional and 3 max pooling layers
    conv1 = tf.nn.relu(conv2d(x, weights['weights_conv1']) + biases['bias_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['weights_conv2']) + biases['bias_conv2'])
    conv2 = maxpool2d(conv2)

    conv3 = tf.nn.relu(conv2d(conv2, weights['weights_conv3']) + biases['bias_conv3'])
    conv3 = maxpool2d(conv3)

    # The fully connected layer
    fully_con = tf.reshape(conv3, [-1, 7*7*256])
    fully_con = tf.nn.relu(tf.matmul(fully_con, weights['weights_fully_con']) + biases['bias_fully_con'])
    fc = tf.nn.dropout(fully_con, keep_rate) # Apply dropout

    output = tf.matmul(fully_con, weights['weights_out']) + biases['bias_out']
    return output



###########################################################################################################

def train_network(x):
    pred = convolutional_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels= y))
    optimzer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # Initialize all the variables

        for epoch in range(num_epochs):
            loss = 0

            # Loop through 
            #
            #
            #
            # 
            print('Epoch : ', epoch+1, ' of ', num_epochs, ' - Loss: ', loss)

        # correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # acc = tf.reduce_mean(tf.cast(correct, 'float'))
        # print('Accuracy:', acc)

###########################################################################################################

train_network(x)
