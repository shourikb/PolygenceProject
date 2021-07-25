from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas

import Algorithms

print("[INFO] accessing MNIST...")
((trainX, trainY), (testX, testY)) = mnist.load_data()
trainX = Algorithms.main()
testX = Algorithms.getTest()
trainY = trainX.pop("Success")
testY = testX.pop("Success")
trainX = trainX.to_numpy()
testX = testX.to_numpy()

# each image in the MNIST dataset is represented as a 28x28x1
# image, but in order to apply a standard neural network we must
# first "flatten" the image to be simple list of 28x28=784 pixels
trainX = trainX.reshape((trainX.shape[0], 6))
testX = testX.reshape((testX.shape[0], 6))
# scale data to the range of [0, 1]
trainX = trainX.astype("float32")
testX = testX.astype("float32")

trainX = trainX[0:100,:]

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

trainY = trainY[0:100,:]


model = Sequential()
model.add(Dense(256, input_shape=(6,), activation="sigmoid"))
model.add(Dense(128, activation="sigmoid"))
model.add(Dense(10, activation="sigmoid"))
model.add(Flatten())


print("[INFO] training network...")
sgd = SGD(0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=10, batch_size=128)


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(predictions.argmax(axis=1))
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

print("Train X", trainX.shape)
print("Test X", testX.shape)
print("Train Y", trainY.shape)
print("Test Y", testY.shape)

model.save("modelsAndCheckpoints")

'''
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("Output.jpg")
'''