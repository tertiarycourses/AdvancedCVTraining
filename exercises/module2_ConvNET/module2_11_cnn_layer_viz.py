# Advanced Computer Vision Training
# Module 2 Convnet
# CNN Layer Visualization

from keras.models import load_model

model = load_model('cats_and_dogs_small_2.h5')
# print(model.summary())  # As a reminder.

img_path = './data/cats_and_dogs_small/test/cats/cat.1700.jpg'

# We preprocess the image into a 4D tensor
from keras.preprocessing import image
import numpy as np

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.


from keras import models

# Extracts the outputs of the top 8 layers:
layer_outputs = [layer.output for layer in model.layers[:23]]

# Creates a model that will return these outputs, given the model input:
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(img_tensor)
layer_activations = activations[0]

import matplotlib.pyplot as plt
for i in range(8):
    plt.figure(i)
    plt.imshow(layer_activations[0, :, :,i], cmap='viridis')
plt.show()