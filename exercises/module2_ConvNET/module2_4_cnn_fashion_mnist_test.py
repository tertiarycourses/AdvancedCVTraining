# Advanced Computer Vision Training
# Module 2 Convnet
# MNIST Live Demo

import keras

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_WINOGRAD_NONE_USED']='1'

from keras.models import load_model

path = "f_mnist_cnn.h5"
model = load_model(path)

from keras.preprocessing import image
img = image.load_img("./images/shoe.png", grayscale=True, target_size=(28,28))

# Convert the image to a numpy array
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)

prediction = model.predict(x)
print(prediction)