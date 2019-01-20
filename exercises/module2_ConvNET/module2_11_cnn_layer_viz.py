# Advanced Computer Vision Training
# Module 2 Convnet
# CNN Layer Visualization

from keras.models import load_model
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.models import Model
import numpy as np

model = load_model('cats_and_dogs_small_2.h5')
# print(model.summary())  # As a reminder.

img_path = './data/cats_and_dogs_small/test/cats/cat.1700.jpg'

# We preprocess the image into a 4D tensor
img = image.load_img(img_path, target_size=(150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img /= 255.

# Extracts the outputs of the top 8 layers:
outputs = [layer.output for layer in model.layers[:8]]

# Creates a model that will return these outputs, given the model input:
activation = Model(inputs=model.input, outputs=outputs)
activations = activation.predict(img)
layer_activations = activations[0]

for i in range(8):
    plt.figure(i)
    plt.imshow(layer_activations[0, :, :,i], cmap='viridis')
plt.show()