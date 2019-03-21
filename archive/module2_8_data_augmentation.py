# Advanced Computer Vision Training
# Module 2 Convnet
# Data Augmentation

import os
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
      # rotation_range=40,
      # width_shift_range=0.2,
      # height_shift_range=0.2,
      # shear_range=0.2,
      zoom_range=0.4,
      # horizontal_flip=True,
      # fill_mode='nearest'
)

img_path = './data/cats_and_dogs_small/test/cats/cat.1700.jpg'

from keras.preprocessing import image
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

# The .flow() command below generates batches of randomly transformed images.
# It will loop indefinitely, so we need to `break` the loop at some point!

import matplotlib.pyplot as plt
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 10 == 0:
        break

plt.show()