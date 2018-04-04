'''
This is the model that will be used to train the deep convolutional neaural network.

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
TRAINING_DIR = os.getcwd() + '/data/training'

####################################### DATA PREPROCESSING - Labeling ################################################
# list = os.listdir(TRAINING_DIR)
# # Add label names to list of labels
# for folders in list:
#     if(folders != "desktop.ini"):
#         labels.append(folders)

# # Creat one hot encodings
# dummy_vars = pd.get_dummies(labels)
# # print(dummy_vars)

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

images = np.array(images)
labels = np.array(labels)

####################################### DATA VISUALISATION #######################################
''''
This cell is for displaying data visualy using Matplotlib
This does not have any bearing on the model so can be commented out when
not being used.

copy and past the code from visualization.py into

'''
import matplotlib.pyplot as plt 

####################################### DATA PREPROCESSING - Imaging #######################################
'''
This cell is for image downsampling and transformation
This is on the fly to resize the images to a 50x50 size
'''
from skimage import transform, exposure
from skimage.color import rgb2gray

print('Down scaling images...')
images = [transform.resize(image, (50, 50)) for image in images]

# print('equalizing exposure...')
# images = [exposure.equalize_adapthist(image, clip_limit=0.0001)for image in images50]


# Plot the down sized / augmented image
# random = [295, 3098, 997, 4999]
# for i in range(len(random)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[random[i]])
#     plt.subplots_adjust(wspace=0.1)
# plt.savefig('exp.jpg')



#################################### VARIABLE INITIATATION #################################################

# Define initial variables
batch_size = 128
num_class = 6
num_epochs = 25

# Initialize placeholders 

# x = tf.placeholder('float', [5582, 50, 50])
x = tf.placeholder(dtype = tf.float32, shape = [None, 50, 50])
y = tf.placeholder(dtype = tf.int32, shape = [None])

#define variables for dropout
keep_rate = .8
keep_prop = tf.placeholder(tf.float32)

# ######################################## HELPER FUNCTIONS #################################################

# Extract features
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') #move one pixel at s time

#
def maxpool2d(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') #pool 2 pixels at a time

# # ######################################## NETWORK DEFINITION ################################################


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


# ####################################### TENSORFLOW SESSION ###################################################

# # Flatten the input data
# images_flat = tf.contrib.layers.flatten(x)
# # Fully connected layer 
# logits = tf.contrib.layers.fully_connected(images_flat, 6, tf.nn.relu)
# # Define a loss function
# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
# # Define an optimizer 
# train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# # Convert logits to label indexes
# correct_pred = tf.argmax(logits, 1)
# # Define an accuracy metric
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# print("images_flat: ", images_flat)
# print("logits: ", logits)
# print("loss: ", loss)
# print("predicted_labels: ", correct_pred)


# tf.set_random_seed(1234)
# sess = tf.Session()

# sess.run(tf.global_variables_initializer())

# for i in range(201):
#         print('EPOCH', i)
#         _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images, y: labels})
#         if i % 10 == 0:
#             print("Loss: ", loss)
#         print('DONE WITH EPOCH')

# sess.close()



def train_network(x):
    pred = convolutional_network(x)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels= y))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = pred))
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # Initialize all the variables

        print("RUNNING SESSION...")

        for epoch in range(num_epochs):
            for _ in range(1):
                _, loss_value = sess.run([train_op, loss], feed_dict={x: images, y: labels})
                print("feed")
            print('Epoch : ', epoch+1, ' of ', num_epochs, ' - Loss: ', loss_value)

        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', acc)

# ###########################################################################################################

train_network(x)
