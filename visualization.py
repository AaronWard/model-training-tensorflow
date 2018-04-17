
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

loss = [
    # 5534948111.29, 
    # 190246561.785,
    # 2687721.87646, 
    # 173743.435638,
    37462.5181961,
    18471.0896435,
    11579.234355,
    8794.70186961,
    7627.35857039,
    6429.21610296, 
    6418.67906547,
    5488.79455012,
    5493.91774639,
    5232.9456518,
    4939.2082068, 
    5058.37764013, 
    4985.056026, 
    4820.04455124, 
    4715.96226606, 
    4758.63074559, 
    4639.21305159, 
    4645.61671885,
    4711.5498182, 
    4623.56152188,
    4566.40746532]

epochs = [ 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 
        20, 21, 22, 23, 24, 25]

plt.grid(True)
plt.plot(epochs, loss)
plt.title('Loss Results for Training')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.xticks(epochs)
plt.savefig('image.jpg')
#Full time elapse: 17214.159914