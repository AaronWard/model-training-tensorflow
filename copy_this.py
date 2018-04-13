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
print('imported')



TRAINING_DIR = '/data/training'
MODEL_PATH = '/output/trained_model.ckpt'

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

print(images)
print(labels)

# images = np.array(images)
# labels = np.array(labels)


#plot the images with their labels to make sure they are correct

import matplotlib.pyplot as plt 
unique_labels = set(labels)
# Initialize the figure
plt.figure(figsize=(30, 30))
# Set a counter
i = 1
for label in unique_labels:
    # You pick the first image for each label
    image = images[labels.index(label)]
    plt.subplot(1, 6, i)
    plt.axis('off')
    plt.title("Label {0} ({1})".format(label, labels.count(label)))
    i += 1
    plt.imshow(image)
plt.show()



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

print(' ------------ IMAGES DOWNSCALED ------------')



#################################### VARIABLE INITIATATION #################################################
'''
This cell is for initializing variables for the tensorflow session and 
placeholders for holding the data.

'''

# Define initial variables
batch_size = 100
num_class = 6
num_epochs = 25

# Initialize placeholders 
# x = tf.placeholder('float', [5582, 50, 50])
x = tf.placeholder(dtype = tf.float32, shape = [None, 50, 50])
y = tf.placeholder(dtype = tf.int32, shape = [None])

#define variables for dropout
keep_rate = .8
keep_prop = tf.placeholder(tf.float32)
print('initialized')


######################################### HELPER FUNCTIONS #################################################

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

print('helpers defined')


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

print('network defined')


######################################## TENSORFLOW SESSION ###################################################
'''
This cell contains a function that runs the tensorflow session, it is called with the x placeholders.
The session is ran by first initializing all the tensorflow variables, then iterated through
the number of epochs and feed the image data and labels using feed_dict.
The loss/cost and accuracy is evaluated and printed to the console.

'''

def train_network(x):
    print('Starting training...')
    pred = convolutional_network(x)
    # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels= y))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = pred))
    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # Initialize all the variables
        saver = tf.train.Saver()

        print("RUNNING SESSION...")
        for epoch in range(num_epochs):
            _, loss_value = sess.run([train_op, loss], feed_dict={x: images, y: labels})
            print("feed")
            print('Epoch : ', epoch+1, ' of ', num_epochs, ' - Loss: ', loss_value)

        correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        acc = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', acc)

        save_path = saver.save(sess, MODEL_PATH)
        print("Model saved in file: " , save_path)
        

train_network(x)
