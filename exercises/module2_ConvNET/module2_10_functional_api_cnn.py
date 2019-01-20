# Advanced Computer Vision Training
# Module 3 Transfer Learning
# Keras Functional API

import keras
from keras.layers import Input, Dense
from keras.models import Model

# Step 1 Load the data
from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 784)
X_test = X_test.reshape(-1, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Step 2: Create the Model
inputs = Input(shape=(784,)) # This returns a tensor

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
yhat = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=yhat)


# Step 3: Create the Model
model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train, y_train)  # starts training

# Step 5: Evaluate the Model
loss,acc = model.evaluate(X_test, y_test)
print(acc)