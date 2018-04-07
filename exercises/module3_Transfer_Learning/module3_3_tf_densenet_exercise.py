# Advanced Computer Vision Training
# Module 3 Transfer Learning
# Exercise: DenseNET

from keras.applications.densenet import DenseNet121
from keras.preprocessing import image

img = image.load_img('images/football-299.jpg',
                     target_size=(224,224))

x = image.img_to_array(img)

import numpy as np

x = np.expand_dims(x,axis=0)


model = DenseNet121()
#print(model.summary())
#
from keras.applications.densenet import preprocess_input

x = preprocess_input(x)

y = model.predict(x)

from keras.applications.densenet import decode_predictions

y_pred = decode_predictions(y)

print("The image is ",y_pred)