import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import boston_housing

import Algorithms

class SimpleLinearRegression:
    def __init__(self, initializer='random'):
        if initializer == 'ones':
            self.var = 1.
        elif initializer == 'zeros':
            self.var = 0.
        elif initializer == 'random':
            self.var = tf.random.uniform(shape=[], minval=0., maxval=1.)

        self.m = tf.Variable(1., shape=tf.TensorShape(None))
        self.b = tf.Variable(self.var)

    def predict(self, x):
        return tf.reduce_sum(self.m * x, 1) + self.b

    def mse(self, true, predicted):
        return tf.reduce_mean(tf.square(true - predicted))

    def update(self, X, y, learning_rate):
        with tf.GradientTape(persistent=True) as g:
            loss = self.mse(y, self.predict(X))

        print("Loss: ", loss)

        dy_dm = g.gradient(loss, self.m)
        dy_db = g.gradient(loss, self.b)

        self.m.assign_sub(learning_rate * dy_dm)
        self.b.assign_sub(learning_rate * dy_db)

    def train(self, X, y, learning_rate=0.01, epochs=5):

        if len(X.shape) == 1:
            X = tf.reshape(X, [X.shape[0], 1])

        self.m.assign([self.var] * X.shape[-1])

        for i in range(epochs):
            print("Epoch: ", i)

            self.update(X, y, learning_rate)

if __name__ == "__main__":

    #tf.compat.v1.disable_eager_execution()
    print("check")
    training_df = Algorithms.main()
    features = ['ToGo', 'Down', 'YardLine']
    x_train = tf.cast(training_df[features].values, tf.int32)
    y_train = tf.cast(training_df['Success'].values, tf.int32)

    y_train = y_train.numpy()
    x_train = x_train.numpy()

    testing_df = Algorithms.getTest()
    x_test = tf.cast(testing_df[features].values, tf.int32)
    y_test = tf.cast(testing_df['Success'].values, tf.int32)

    x_test = x_test.numpy()
    y_test = y_test.numpy()

    (xx_train, yy_train), (xx_test, yy_test) = boston_housing.load_data()



    #xx_train = tf.zeros((3, 3051), dtype=tf.dtypes.float32)
    #yy_train = tf.zeros((1, 3051), dtype=tf.dtypes.float32)

    '''
    i = 0
    for features_tensor, target_tensor in x_train:
        xx_train[i] = features_tensor
        yy_train[i] = target_tensor
        print(f'features:{features_tensor} success:{target_tensor}')
        i+=1
    '''
    #x_test = 0
    #y_test = 0

    mean_label = y_train.mean(axis=0)
    std_label = y_train.std(axis=0)

    mean_feat = x_train.mean(axis=0)
    std_feat = x_train.std(axis=0)

    x_train = (x_train-mean_feat)/std_feat
    y_train = (y_train-mean_label)/std_label

    linear_model = SimpleLinearRegression('zeros')
    linear_model.train(x_train, y_train, learning_rate=0.1, epochs=1000)

    o_xtest = x_test
    x_test = (x_test-mean_feat)/std_feat

    pred = linear_model.predict(x_test)
    pred *= std_label
    pred += mean_label

    plt.scatter(o_xtest[:,1], pred)
    plt.show()



'''
tf.disable_v2_behavior()

np.random.seed(101)
tf.random.set_seed(101)

x = np.linspace(0, 50, 50)
y = np.linspace(0, 50, 50)

x += np.random.uniform(-4, 4, 50)
y += np.random.uniform(-4, 4, 50)

n = len(x)

plt.scatter(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title("Training Data")
plt.show()

tf.compat.v1.disable_eager_execution()

X = tf.compat.v1.placeholder("float")
Y = tf.compat.v1.placeholder("float")

W = tf.Variable(np.random.randn(), name = "W")
b = tf.Variable(np.random.randn(), name = "b")

learning_rate = 0.01
training_epochs = 100

y_pred = tf.add(tf.multiply(X, W), b)

cost = tf.reduce_sum(tf.pow(y_pred-Y, 2) / (2*n))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init)

    print("hello")

    for epoch in range(training_epochs):

        tf.compat.v1.disable_eager_execution()

        for(_x, _y) in zip(x, y):
            sess.run(optimizer, feed_dict = {X : _x, Y : _y})

        if(epoch + 1) % 50 == 0:
            c = sess.run(cost, feed_dict = {X : _x, Y : _y})
            print("Epoch", (epoch+1), ": cost = ", c, "W =", sess.run(W), "b =", sess.run(b))

    training_cost = sess.run(cost, feed_dict = {X: x, Y: y})
    weight = sess.run(W)
    bias = sess.run(b)
'''