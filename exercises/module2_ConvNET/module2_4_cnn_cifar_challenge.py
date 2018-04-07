# Advanced Computer Vision Training
# Module 2 Convnet
# Create small cats and dogs dataset

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import keras
from keras.utils import np_utils
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Conv2D,MaxPool2D,Dropout,Flatten

# Step 1 Load the Model
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Step 2: Build the CNN Model
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),activation='relu',padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3),activation='relu',padding='same'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(3,3),activation='relu',padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

print(model.summary())

# Step 3: Compile the Model
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train,y_train,epochs=2,batch_size=100)

# Step 5: Evalute the Model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Step 6: Save the Model
model.save('cifar_cnn.h5')
