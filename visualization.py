
# label_names = ['angry', 'fear', 'happy', 'neutral', 'sadness', 'suprise']
# num_angry = 0
# num_fear = 0
# num_happy = 0
# num_neutral = 0
# num_sad = 0
# num_suprise = 0

# for l in labels:
#     x = l + ""
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

