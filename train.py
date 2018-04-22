
# coding: utf-8

# In[1]:

'''
This is the model that will be used to train the deep convolutional neaural network.

@Author : Aaron Ward 
'''
import tensorflow as tf
import os, os.path
import pandas as pd
import time
import numpy as np
from numpy import ndarray
import skimage
from skimage import data, io, filters
import random
import matplotlib
import matplotlib.pyplot as plt
# sets the backend of matplotlib to the 'inline' backend
get_ipython().magic("matplotlib inline")

print('imported')


# In[2]:

'''
Basic import directories and parameters initialized

Remember to use os.getcwd() initializing partha on local 
machine

'''
TRAINING_DIR = '/data/training'
TESTING_DIR = '/data/testing'
MODEL_PATH = '/output/trained_model.ckpt'
SAVE = '/output/'

# Define initial variables
batch_size = 100
num_class = 6
num_epochs = 100


# In[3]:

####################################### DATA PREPROCESSING - Labeling ################################################
'''
This function traverses throwe ach directory of training images
Two lists are made:
    - The RGB image values are added to the images list
    - For every photo in say the 'angry' directory of images, a 
      corresponding label is added to the label list

'''
def load_data(TRAINING_DIR):
    images = []
    labels = []
    directories = [d for d in os.listdir(TRAINING_DIR) 
                if os.path.isdir(os.path.join(TRAINING_DIR, d))]
    # Need to sort these because
    # floyd hum jumbled up the order
    directories = sorted(directories, key=int)

    # Traverse through each directory and make a list
    # of files names if they end in the PNG format
    for d in directories:
        label_directory = os.path.join(TRAINING_DIR, d)
        file_names = [os.path.join(label_directory, f) 
                        for f in os.listdir(label_directory) 
                          if f.endswith(".png")]
        #Traverse through each file, add the image data
        # and label to the 2 lists
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))

    return images, labels

images, labels = load_data(TRAINING_DIR)

print('Data impored...')


# In[4]:

'''
Shuffle the entire dataset and labels

'''
from sklearn.utils import shuffle
images, labels = shuffle(images, labels)


# In[5]:

'''
This cell is  for converting the lists to numpy arrays 

'''
num_images = len(images)
images = np.array(images, object)
labels = np.array(labels, dtype = np.int32)

_labels = np.zeros((num_images, num_class))
_labels[np.arange(num_images), labels] = 1.0
labels = _labels

print(labels[1])
print(labels[2])
print(labels[3])
print(labels[4])


# In[6]:

####################################### DATA PREPROCESSING - Labeling ################################################
'''
import test data and labels
'''
def load_test_data(TESTING_DIR):
    test_images = []
    test_labels = []
    directories = [d for d in os.listdir(TESTING_DIR) 
                if os.path.isdir(os.path.join(TESTING_DIR, d))]
    # Need to sort these because
    # floyd hum jumbled up the order
    directories = sorted(directories, key=int)

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
            test_images.append(skimage.data.imread(f))
            test_labels.append(int(d))

    return test_images, test_labels

test_images, test_labels = load_data(TESTING_DIR)

test_images = np.array(test_images, object)
test_labels = np.array(test_labels, object)

# Convert labels into a one hot vector 
test_labels = pd.get_dummies(test_labels)
print('imported...')


# In[7]:

####################################### DATA PREPROCESSING - Imaging #######################################
'''
This cell is for image downsampling and transformation
This is on the fly to resize the images to a 50x50 size

'''
from skimage import transform, exposure

print('Down scaling train images...')
images = [transform.resize(image, (50, 50)) for image in images]

print('Down scaling test images...')
test_images = [transform.resize(test_image, (50, 50)) for test_image in test_images]

print('Images Downscaled...')


# In[8]:

'''
This cell is for initializing variables for the tensorflow session and 
placeholders for holding the data.

'''
# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 50, 50], name='X_placeholder')
y = tf.placeholder(dtype = tf.int32, shape= [None, num_class], name="Y_placeholder")
is_training = tf.placeholder( dtype = tf.bool, shape = (), name = "is_training" )

#define variables for dropout
keep_rate = .8
keep_prop = tf.placeholder(tf.float32)
print('initialized')


# In[3]:

######################################### NETWORK STRUCTURE #################################################
'''
This cell is for defining the stucture of the neural network.
The network has 11 convolutional layers and 2 fully connected layers

'''
import tensorflow.contrib.slim as slim

def convolutional_network(x, is_training):
    conv_net = tf.reshape(x, shape=[-1, 50, 50, 1]) # add channel dimensions

    #used for scoping layers arguments
    with slim.arg_scope([slim.conv2d],
            padding = "SAME",
            activation_fn = tf.nn.relu,
            stride = 1,
            weights_initializer = tf.truncated_normal_initializer(stddev=0.01),
            weights_regularizer = slim.l2_regularizer(0.0005),
            normalizer_fn = slim.batch_norm,
            normalizer_params = {'scale' : True,'trainable' : False, 'is_training' : is_training }):
        
        conv_net = slim.conv2d(conv_net, 32, 3)
        conv_net = slim.conv2d(conv_net, 64, 3)
        conv_net = slim.conv2d(conv_net, 64, 3)
        conv_net = slim.max_pool2d(conv_net, 3, stride = 1 )
        conv_net = slim.conv2d(conv_net, 96, 3)
        conv_net = slim.conv2d(conv_net, 96, 3)
        conv_net = slim.max_pool2d(conv_net, 2, stride = 2)
        
        conv_net = slim.conv2d(conv_net, 128, 3)
        conv_net = slim.conv2d(conv_net, 128, 3)
        conv_net = slim.max_pool2d(conv_net, 2, stride = 2)

        conv_net = slim.conv2d(conv_net, 128, 3)
        conv_net = slim.conv2d(conv_net, 128, 3)
        conv_net = slim.max_pool2d(conv_net, 2, stride = 2)

        conv_net = slim.conv2d(conv_net, 128, 3)
        conv_net = slim.max_pool2d(conv_net, 2, stride = 1)
        
        conv_net = slim.dropout(conv_net, keep_prob = keep_rate, is_training = is_training )

    # Fully Connect Layer
    with slim.arg_scope([slim.fully_connected ],weights_regularizer = slim.l2_regularizer(0.0005)):
        conv_net = slim.flatten(conv_net)
        output = slim.fully_connected(conv_net, num_class, activation_fn = None)
        prediction = tf.nn.softmax(output)

        return output, prediction
    
print('Network defined...')


# In[12]:

'''
Shuffle the batches on the fly.

'''

def randomize(batch_x, batch_y):
    batch_x, batch_y = shuffle(batch_x, batch_y)
    return batch_x, batch_y


# In[4]:

######################################### NETWORK TRAINING #################################################
'''
This cell contains a function that runs the tensorflow session, it is called with the x placeholders.
The session is ran by first initializing all the tensorflow variables, then iterated through
the number of epochs and feed the image data and labels using feed_dict.
The loss/cost and accuracy is evaluated and printed to the console.

'''

def train_network(x):
    output, prediction = convolutional_network(x, is_training)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))
    total_losses = tf.losses.get_total_loss( add_regularization_losses=True ) + loss
    
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # needed for batch normalization
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer( learning_rate=0.002 ).minimize(total_losses)
    
    correct = tf.equal(tf.argmax(prediction,1),  tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(correct, 'float'))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # Initialize all the variables
        saver = tf.train.Saver()

        time_full_start = time.clock()
        
        print("RUNNING SESSION...")
        for epoch in range(num_epochs):
            train_batch_x = []
            train_batch_y = []
            epoch_loss= 0
            epoch_total_loss = 0
            accuracy = 0
            time_epoch_start = time.clock()
            i = 0
            number_of_batches = 0
            
            #For all images in the DS, batch into sizes of 100
            while i < len(images):
                start = i
                end = i + batch_size
                train_batch_x = images[start:end]
                train_batch_y = labels[start:end]
                
                #Randomize the batches even more
                train_batch_x, train_batch_y = randomize(train_batch_x, train_batch_y)
                
                #Feed batches into tensorflow
                op, ac, loss_value, total_loss_value = sess.run([train_op, acc, loss, total_losses],feed_dict={x: train_batch_x,
                                                                    y: train_batch_y, is_training : True})
                epoch_loss += loss_value
                epoch_total_loss += total_loss_value
                accuracy += ac
                i += batch_size
                number_of_batches += 1
            
            accuracy /= number_of_batches
            
            print('Epoch:', epoch+1, 'total loss: ', epoch_total_loss  ,' loss: ', epoch_loss ,' acc: {: %}'.format(accuracy))
            
            time_epoch_end = time.clock()
            print('Time elapse: ', time_epoch_end - time_epoch_start)

        time_full_end = time.clock()
        print('Full time elapse:', time_full_end - time_full_start)

        if epoch_loss < 100:
            save_path = saver.save(sess, MODEL_PATH)
            print("Model saved in file: " , save_path)


        # Evaluate on unseen test data
        print('Accuracy:', acc.eval({x: test_images, y: test_labels, is_training : False }))


# In[14]:

train_network(x)


# In[ ]:

######################################### NETWORK TESTING #################################################
'''
Simple  cell for loading the model and testing the labels predicted for a range of
images
'''
def test(x):
    output, prediction = convolutional_network_v2(x, False )
#     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph('/output/trained_model.ckpt.meta')
        saver.restore(sess, '/output/trained_model.ckpt' )
        print('session restored...')

        pred_ = tf.nn.softmax(output)

        predicted = sess.run(pred_, feed_dict={x: test_images[400:410]})[0]
        
        print('Actual Labels for ten images\n', test_labels[400:410])
        print('\nPredicited Labels for ten images\n', predicted[400:410])
test(x)


# In[ ]:



