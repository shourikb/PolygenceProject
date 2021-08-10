from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import tensorflow as tf

import Algorithms

print("[INFO] accessing MNIST...")
#((trainX, trainY), (testX, testY)) = mnist.load_data()
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

trainX = trainX[0:20,:]

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

trainY = trainY[0:20,:]


model = Sequential()
model.add(Dense(512, input_shape=(6,), activation="relu"))
#model.add(Dense(16, activation="relu"))
model.add(Dense(2, activation="softmax"))
model.add(Flatten())


print("[INFO] training network...")
#sgd = SGD(0.0001)

adam = Adam(0.0001)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=64)

'''
adam = Adam(0.001)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=64)

adam = Adam(0.00001)
model.compile(loss="binary_crossentropy", optimizer=adam, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=50, batch_size=64)
'''


print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=128)
print(predictions.argmax(axis=1))
print(classification_report(testY, predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

print("Train X", trainX)
print("Test X", testX)

#model.save("modelsAndCheckpoints")

frozen_out_path = "modelsAndCheckpoints"

frozen_graph_filename = "frozen_graph"

full_model = tf.function(lambda x: model(x))
full_model = full_model.get_concrete_function(
    tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

layers = [op.name for op in frozen_func.graph.get_operations()]
print("-" * 60)
print("Frozen model layers: ")
for layer in layers:
    print(layer)
print("-" * 60)
print("Frozen model inputs: ")
print(frozen_func.inputs)
print("Frozen model outputs: ")
print(frozen_func.outputs)
# Save frozen graph to disk
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pb",
                  as_text=False)
# Save its text representation
tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                  logdir=frozen_out_path,
                  name=f"{frozen_graph_filename}.pbtxt",
                  as_text=True)

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