# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 19:42:34 2023

@author: aluga.com
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 19:16:33 2023

@author: aluga.com
"""

import tensorflow as tf

from keras import models
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.utils.vis_utils import model_to_dot
from IPython.display import SVG
from keras.utils import plot_model
from tensorflow.keras import regularizers
from keras.optimizers import Adam, SGD
# Custom Activation Function
def custom_activation(x):
    return tf.nn.tanh(x)

# Custom loss function: Mean Square Absolute Percentage Error (MSAPE)\
def msape_loss(y_true, y_pred):
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), 1e-15))
    return 100 * tf.reduce_mean(tf.square(diff))

def custom_loss(y_true, y_pred):
    return msape_loss(y_true, y_pred)

# Create a custom neuron layer
class CustomNeuron(tf.keras.layers.Layer):
    def __init__(self, units=1,  **kwargs):
        super(CustomNeuron, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.weight = self.add_weight("weight", shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.bias = self.add_weight("bias", shape=(self.units,), initializer="ones", trainable=True)
        super(CustomNeuron, self).build(input_shape)

    def call(self, inputs):
        #weighted_sum = tf.matmul(inputs, self.weight)
        #result = tf.reduce_prod(weighted_sum) + self.bias
        result = tf.reduce_prod(tf.expand_dims(inputs, axis=2) * tf.expand_dims(self.weight, axis=0), axis=1) + self.bias
        return result

# Custom Layer
class CustomLayer(tf.Module):
    def __init__(self, num_inputs, num_neurons):
        self.neurons = [CustomNeuron(num_inputs) for _ in range(num_neurons)]

    def __call__(self, inputs):
        output = tf.concat([neuron(inputs) for neuron in self.neurons], axis=1)
        return custom_activation(output)

# Custom Neural Network
class CustomNetwork(tf.Module):
    def __init__(self, num_inputs, num_layers, num_neurons_per_layer):
        self.layers = [CustomLayer(num_inputs, num_neurons_per_layer) for _ in range(num_layers)]

    def __call__(self, inputs):
        output = inputs
        for layer in self.layers:
            output = layer(output)
        return output

# Create a simple model with the custom neuron
model = tf.keras.Sequential([CustomNeuron(units=10, input_shape=(2,))])
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))

# Generate some dummy data
x_train = tf.random.normal((1000, 2)) * 100
y_train = tf.reduce_prod(x_train, axis=1)

# Compile the model
model.compile(optimizer='adam', loss=custom_loss)
# Compile the model

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=32)


# Generate some dummy test data
x_test = tf.random.normal((10, 2)) * 100 
x_test_numpy = x_test.numpy()

y_test = tf.reduce_prod(x_test, axis=1)
y_test_num =y_test.numpy()

# Make predictions on the test data
predictions = model.predict(x_test)

# Print the predictions
print("x_test_numpy:", x_test_numpy)
print("x_test:", y_test_num)
print("Predictions:", predictions)