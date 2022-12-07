"""
INSERT YOUR NAME HERE
Woo Hyun Maeng
"""

from __future__ import division
from __future__ import print_function
import sys
try:
    import _pickle as pickle
except:
    import pickle
import numpy as np
import matplotlib.pyplot as plt

# This is a class for a LinearTransform layer which takes an input
# weight matrix W and computes W x as the forward step
class LinearTransform(object):

    def __init__(self, W, b):
        # DEFINE __init function
        self.w = W
        self.b = b
        self.Dw = np.zeros_like(self.w)
        self.Db = np.zeros_like(self.b)

    def forward(self, x):
        # DEFINE forward function
        self.x = x
        z = np.dot(self.x, self.w) + self.b
        return z

                        # ltf2 's : grad_output = dldz2 from SCE
                        # ltf1 's : grad_output = dldz1 from ReLu
    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        # DEFINE backward function
        self.dz2da1 = self.w
        self.dldw = np.dot(self.x.T, grad_output)
        self.dldb = grad_output.sum(axis=0, keepdims=True)

        self.Dw = momentum * self.Dw - learning_rate * self.dldw
        self.Db = momentum * self.Db - learning_rate * self.dldb

        self.dldw = (self.dldw + l2_penalty * self.w)
        self.dldb = (self.dldb + l2_penalty * self.b)

        self.w = self.w + self.Dw
        self.b = self.b + self.Db

        self.dlda1 = np.dot(grad_output, self.dz2da1.T)
        return self.dlda1   # this will be fed to relu.backward(), so this is ltf2 grad_output


# This is a class for a ReLU layer max(x,0)
class ReLU(object):
    def forward(self, x):
        # DEFINE forward function
        self.x = x
        self.a1 = np.maximum(0, self.x)
        return self.a1

                      # grad_output : dlda1 from LTF1
    def backward(self, grad_output, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        # DEFINE backward function
        self.dldz1 = grad_output * self.relu_deriv(self.x)
        return self.dldz1

    # ADD other operations in ReLU if needed
    def relu_deriv(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


# This is a class for a sigmoid layer followed by a cross entropy layer, the reason
# this is put into a single layer is because it has a simple gradient form
class SigmoidCrossEntropy(object):
    def forward(self, x):
        # DEFINE forward function
        self.x = x
        self.a2 = self.sigmoid(x)
        return self.a2

                      # grad_output : y
    def backward(self, grad_output=None, learning_rate=0.0, momentum=0.0, l2_penalty=0.0):
        # DEFINE backward function
        dldz2 = (self.a2 - grad_output) / grad_output.shape[0]
        return dldz2

    # ADD other operations and data entries in SigmoidCrossEntropy if needed
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


# This is a class for the Multilayer perceptron
class MLP(object):
    def __init__(self, input_dims, hidden_units):
        # INSERT CODE for initializing the network
        self.input_dims = input_dims
        self.hidden_units = hidden_units
        self.w1 = 1.0 * np.random.rand(input_dims, hidden_units)
        self.b1 = np.zeros((1, hidden_units))
        self.w2 = 1.0 * np.random.rand(hidden_units, 1)
        self.b2 = np.zeros((1, 1))

        self.ltf1 = LinearTransform(self.w1, self.b1)
        self.ltf2 = LinearTransform(self.w2, self.b2)
        self.relu = ReLU()
        self.sce = SigmoidCrossEntropy()

    def train(self, x_batch, y_batch, learning_rate, momentum, l2_penalty):
        self.z1 = self.ltf1.forward(x_batch)
        self.a1 = self.relu.forward(self.z1)
        self.z2 = self.ltf2.forward(self.a1)
        self.a2 = self.sce.forward(self.z2)

        self.dldz2 = self.sce.backward(y_batch, learning_rate, momentum, l2_penalty)
        self.dlda1 = self.ltf2.backward(self.dldz2, learning_rate, momentum, l2_penalty)
        self.dldz1 = self.relu.backward(self.dlda1, learning_rate, momentum, l2_penalty)
        self.ltf1.backward(self.dldz1, learning_rate, momentum, l2_penalty)

    def evaluate(self, x, y):
        # INSERT CODE for testing the network
        z1 = self.ltf1.forward(x)
        a1 = self.relu.forward(z1)
        z2 = self.ltf2.forward(a1)
        a2 = self.sce.forward(z2)

        epsilon = 1e-12
        a2 = np.clip(a2, epsilon, 1 - epsilon)

        accuracy = np.sum([(a2 > 0.5) * 1 == y]) / y.shape[0]
        loss = np.sum(-(y * np.log(a2) + (1 - y) * np.log(1 - a2))) / y.shape[0]
        return loss, accuracy


if __name__ == '__main__':
    if sys.version_info[0] < 3:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'))
    else:
        data = pickle.load(open('cifar_2class_py2.p', 'rb'), encoding='bytes')
    train_x = data[b'train_data'] / 255
    train_y = data[b'train_labels']
    test_x = data[b'test_data'] / 255
    test_y = data[b'test_labels']

    num_examples, input_dims = train_x.shape
    # INSERT YOUR CODE HERE
    # YOU CAN CHANGE num_epochs AND num_batches TO YOUR DESIRED VALUES
    num_epochs = 30
    num_batches = 200
    batch_size = num_examples // num_batches
    hidden_units = 10
    mlp = MLP(input_dims, hidden_units)
    # INSERT YOUR CODE FOR EACH EPOCH HERE
    for epoch in range(num_epochs):
        # Shuffle data and make mini batch
        indices = np.arange(train_x.shape[0])
        np.random.shuffle(indices)
        train_a = train_x[indices, :]
        train_b = train_y[indices, :]

        for b in range(num_batches):
            # INSERT YOUR CODE FOR EACH MINI_BATCH HERE
            c = b * batch_size
            d = c + batch_size
            x_batch = train_a[c:d, :]
            y_batch = train_b[c:d, :]
            mlp.train(x_batch, y_batch, 1e-5, 0.8, 1e-4)
            total_loss, total_accuracy = mlp.evaluate(x_batch, y_batch)
            print('\r[Epoch {}, mb {}]    Avg.Loss = {:.3f}'.format(epoch+1, b + 1, total_loss), end='')
            sys.stdout.flush()

        # INSERT YOUR CODE AFTER ALL MINI_BATCHES HERE
        # MAKE SURE TO COMPUTE train_loss, train_accuracy, test_loss, test_accuracy
        train_loss, train_accuracy = mlp.evaluate(train_x, train_y)
        test_loss, test_accuracy = mlp.evaluate(test_x, test_y)
        print('\n')
        print('    Train Loss: {:.3f}    Train Acc.: {:.2f}%'.format(train_loss, 100. * train_accuracy))
        print('    Test Loss:  {:.3f}    Test Acc.:  {:.2f}%'.format(test_loss, 100. * test_accuracy))
