'''
This is a testing script for the trained model

@Author : Aaron Ward 
'''
import tensorflow as tf
import os, os.path
import pandas as pd
import numpy as np
from numpy import ndarray
import skimage
from skimage import data, io, filters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #Suppress AVX Warnings

ROOT_PATH = os.getcwd()
TESTING_DIR = os.getcwd() + '/data/testing'
TRAINED_MODEL = os.getcwd() + '/output/trained_model.ckpt'


####################################### DATA PREPROCESSING - Labeling ################################################
'''
This function traverses throwe ach directory of training images
Two lists are made:
    - The RGB image values are added to the images list
    - For every photo in say the 'angry' directory of images, a 
      corresponding label is added to the label list

'''
def load_data(TESTING_DIR):
    images = []
    labels = []
    directories = [d for d in os.listdir(TESTING_DIR) 
                if os.path.isdir(os.path.join(TESTING_DIR, d))]

    # Traverse through each directory and make a list
    # of files names if they end in the PNG format
    for d in directories:
        label_directory = os.path.join(TESTING_DIR, d)
        file_names = [os.path.join(label_directory, f) 
                        for f in os.listdir(label_directory) 
                          if f.endswith(".png")]
        #Traverse through each file, add the image data
        # and label to the 2 lists
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))

    return images, labels

images, labels = load_data(TESTING_DIR)

images = np.array(images)
# labels = np.array(labels)


####################################### DATA PREPROCESSING - Imaging #######################################
'''
This cell is for image downsampling and transformation
This is on the fly to resize the images to a 50x50 size
'''
from skimage import transform, exposure
# from skimage.color import rgb2gray

print('Down scaling images...')
images = [transform.resize(image, (50, 50)) for image in images]

# print('equalizing exposure...')
# images = [exposure.equalize_adapthist(image, clip_limit=0.0001)for image in images50]


#################################### VARIABLE INITIATATION #################################################
'''
This cell is for initializing variables for the tensorflow session and 
placeholders for holding the data.

'''
# Define initial variables
batch_size = 100
num_class = 6

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 50, 50])
y = tf.placeholder(dtype = tf.int32, shape = [None])


# ######################################## HELPER FUNCTIONS #################################################

'''
This cell just contains helper functions for defining convolution
and maxpooling layers

'''
# Extract features
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #move one pixel at s time

#
def maxpool2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #pool 2 pixels at a time


########################################## NETWORK DEFINITION ################################################
'''
This cell contains a function that is used define the weights and biases of each layer in the
network. It is called by the train_network function. It also lays out the
structure of the network that goes as follows:
conv1 -> maxpooling -> conv2 -> maxpooling - > conv3 -> fully connected layer (with dropout) -> output layer

'''

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
        'bias_out' : tf.Variable(tf.random_normal([batch_size, num_class]))
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
    # fc = tf.nn.dropout(fully_con, keep_rate) # Apply dropout

    output = tf.matmul(fully_con, weights['weights_out']) + biases['bias_out']
    return output

######################################## BATCHING ###################################################
'''
This cell is for segmenting the training data in to batches to relieve the GPU of being overloaded
with data.

'''
num_images = len(images)
num_labels = len(labels)

# ## KEEP THESE FOR DEBUGGING
print(num_images, ' images')
print(num_labels, ' labels')
print('batch size ', batch_size)
print('Number of batches ', int(num_images/batch_size))

batch_start= 0
batch_end = 100

BATCHES_IMAGES = []
BATCHES_LABELS = []


for i in range(int(num_images/batch_size)):
    temp_batch = images[batch_start:batch_end]
    BATCHES_IMAGES.append(temp_batch)
    batch_start = batch_start + 100
    batch_end = batch_end + 100

batch_start = 0
batch_end = 100

for i in range(int(num_labels/batch_size)):
    temp_batch = labels[batch_start:batch_end]
    BATCHES_LABELS.append(temp_batch)
    batch_start = batch_start + 100
    batch_end = batch_end + 100

'''
This cell contains a function that runs the tensorflow session, it is called with the x placeholders.
The session is ran by first initializing all the tensorflow variables, then iterated through
the number of epochs and feed the image data and labels using feed_dict.
The loss/cost and accuracy is evaluated and printed to the console.

'''

def train_network(x):
    pred = convolutional_network(x)
    cost = tf.reduce_sum(tf.square(pred - y))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        saver = tf.train.Saver()


        pr, loss_value = sess.run([pred], [loss], feed_dict={x: images, y: labels})

        for i in range(len(pr)):
            err = abs(pr[i] - y[i])
            print("Output:", pr[i], '\t - Label Y: ', y[i], '\t - Error: ', err, '\t - Loss: ', loss_value )
            average_err += err
        average_err = average_err/len(pr)
        print('Average Error: ', ave)


        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', acc)
############################################################################################################
train_network(x)
