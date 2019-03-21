# Advanced Computer Vision Training
# Module 2 Convnet
# MNIST Live Demo

import keras

from keras.models import load_model

model = load_model("mnist_cnn.h5")

from keras.preprocessing import image
img = image.load_img("./images/shoe.png", grayscale=True, target_size=(28,28))

# Convert the image to a numpy array
x = image.img_to_array(img)
# x = x.reshape((1,) + x.shape)
x = np.expand_dims(x, axis=0)

prediction = model.predict(x)
print(prediction)