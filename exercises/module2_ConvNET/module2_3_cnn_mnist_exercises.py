# Advanced Computer Vision Training
# Module 2 Convnet
# MNIST CNN exercise

import keras

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

from keras.models import Sequential
from keras.layers import Conv2D,MaxPool2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(28,28,1),
                 activation='relu',padding = 'same'))

model.add(MaxPool2D((2,2,)))
model.add(Conv2D(64,(3,3), activation='relu',padding = 'same'))
model.add(MaxPool2D((2,2,)))
model.add(Conv2D(128,(3,3), activation='relu',padding = 'same'))
model.add(MaxPool2D((2,2,)))
model.add(Conv2D(256,(3,3), activation='relu',padding = 'same'))
model.add(MaxPool2D((2,2,)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(10,activation='softmax'))

# print(model.summary())

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train,y_train,batch_size=100,epochs=2)

loss,accuracy = model.evaluate(X_test,y_test)
print("Accuracy: %.2f%%" % (accuracy*100))

model.save('mnist_cnn_2.h5')


