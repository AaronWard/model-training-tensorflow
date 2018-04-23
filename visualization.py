
# label_names = [0,1,2,3,4,6]
# num_angry = 0
# num_fear = 0
# num_happy = 0
# num_neutral = 0
# num_sad = 0
# num_suprise = 0

# for l in labels:
#     x = l
#     if x == label_names[0]:
#         num_angry += 1
#     elif x == label_names[1]:
#         num_fear += 1
#     elif x == label_names[2]:
#         num_happy += 1
#     elif x == label_names[3]:
#         num_neutral += 1
#     elif x == label_names[4]:
#         num_sad += 1
#     else:
#         num_suprise += 1
# label_count = [num_angry, num_fear, num_happy, num_neutral, num_sad, num_suprise]
# print(label_count)



''' Print a bar graph of the instances of classes '''
# objects = ('Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise')
# y_pos = np.arange(len(objects))
# performance = label_count
# plt.bar(y_pos, performance, align='center', alpha=0.5)
# plt.xticks(y_pos, objects)
# plt.ylabel('Images')
# plt.title('Classes for Training Set')
# plt.savefig('test.jpg')


''' Sub plot random images in the images list '''
# random = [295, 3098, 997, 4999]
# for i in range(len(random)):
#     plt.subplot(1, 4, i+1)
#     plt.axis('off')
#     plt.imshow(images[random[i]])
#     plt.subplots_adjust(wspace=0.1)

#     print("shape: {0}, min: {1}, max: {2}".format(images[random[i]].shape, 
#                                                 images[random[i]].min(), 
#                                                 images[random[i]].max()))
# plt.savefig('sub.jpg')



''' For printing our each unique label assigned with an image '''
# # Get the unique labels 
# unique_labels = set(labels)
# # Initialize the figure
# plt.figure(figsize=(30, 30))
# # Set a counter
# i = 1
# for label in unique_labels:
#     # You pick the first image for each label
#     image = images[labels.index(label)]
#     plt.subplot(1, 6, i)
#     plt.axis('off')
#     plt.title("Label {0} ({1})".format(label, labels.count(label)))
#     i += 1
#     plt.imshow(image)
# plt.savefig('image_label.jpg')




####################################################################################################

# ONE HOT ENCODER - NOT USED
# list = os.listdir(TRAINING_DIR)
# # Add label names to list of labels
# for folders in list:
#     if(folders != "desktop.ini"):
#         labels.append(folders)

# # Creat one hot encodings
# dummy_vars = pd.get_dummies(labels)
# # print(dummy_vars)


####################################################################################################
#Plotting the loss with pyplot 

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter

# loss = [0, 24.928, 50.30, 80.83, 90.13, 94.5, 95.35, 95.523, 97.32, 97.69, 
#         97.90, 98.095, 98.523, 97.08, 98.178, 97.952, 97.85, 97.785, 97.96, 
#         98.40, 98.6]

# epochs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ,15 ,16, 17, 18, 19, 20]

# plt.grid(True)
# plt.plot(epochs, loss)
# plt.title('Training Accuracy')
# plt.ylabel('Accuracy %')
# plt.xlabel('Epochs')
# plt.xticks(epochs)
# plt.savefig('accuracy.jpg')
# # Full time elapse: 607.854786



import matplotlib.pyplot as plt
 
# Assume m is 2D Numpy array with these values
# [[1.0 0   0   0  ]
#  [0.1 0.7 0.2 0  ]
#  [0   0   1.0 0  ]
#  [0   0   0   1.0]]

m = [
     [312,14,12,12,0,0],  #anger
     [9,338,0,0,3,0   ], #fear
     [3,9,338,0,0,0   ],  #happy
     [12,58,58,134,88,0 ], # neutral
     [32,3,0,1,313,1],
     [0,8,0,0,3,339 ] ]


emotions = ['Anger', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

df_cm = pd.DataFrame(m, index = [i for i in emotions],
                  columns = [i for i in emotions])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.title("Confusion Matrix for Test Images")
plt.xlabel('Predicted', weight='bold')
plt.ylabel('Actual', weight='bold')
plt.savefig('confusion.jpg')