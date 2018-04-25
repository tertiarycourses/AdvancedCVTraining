# Advanced Computer Vision Training
# Module 3 Transfer Learning
# VGG16

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

# Step 1: Preprocess data

# Load the image file, resizing it to 224x224 pixels (required by this model)
img = image.load_img("./images/car-224.jpg", target_size=(224, 224))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Reshape it to (1, 150, 150, 3)
x = x.reshape((1,) + x.shape)

datagen = ImageDataGenerator(
      rotation_range=40,
      # width_shift_range=0.2,
      # height_shift_range=0.2,
      # shear_range=0.2,
      # zoom_range=0.2,
      # horizontal_flip=True,
      fill_mode='nearest')

import matplotlib.pyplot as plt
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 10 == 0:
        break

plt.show()