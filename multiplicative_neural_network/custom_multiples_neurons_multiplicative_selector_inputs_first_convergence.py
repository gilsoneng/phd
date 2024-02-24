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
from keras.utils import plot_model

# Custom Activation Function
def custom_activation(x):
    #return x
    return tf.nn.linear(x)

# Custom loss function: Mean Square Absolute Percentage Error (MSAPE)\
def msape_loss(y_true, y_pred):
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), 1e-15))
    return 100 * tf.reduce_mean(tf.square(diff))

def custom_loss(y_true, y_pred):
    return msape_loss(y_true, y_pred)



# Use tf.map_fn to apply the custom function element-wise to tensors C and S


# Create a custom neuron layer
class CustomNeuron(tf.keras.layers.Layer):
    def __init__(self, units=1,  **kwargs):
        super(CustomNeuron, self).__init__(**kwargs)
        self.units = units
        
    #@tf.function
    #def alpha1_function(self, y):
        # Replace this with your custom logic
        #alpha_1 = 1-(1/(1+tf.exp(-y)))
       # return alpha_1
    
    #def alpha2_function(self, y):
        # Replace this with your custom logic
     #   alpha_2 = 1/(1+tf.exp(-y))
     #   return alpha_2

    def build(self, input_shape):
        #self.weight = self.add_weight("weight", shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True)
        self.weight = self.add_weight("weight_selector", shape=(input_shape[-1], self.units), initializer="zeros", trainable=True)
        self.bias = self.add_weight("bias", shape=(input_shape[-1], self.units), initializer="ones", trainable=True)
        #self.alpha_1 = self.alpha1_function(self.weight_selector)
        #self.alpha_2 = self.alpha2_function(self.weight_selector)
        super(CustomNeuron, self).build(input_shape)

    def call(self, inputs):
        # expand_matrix = tf.expand_dims(inputs, axis=2) * tf.expand_dims(self.weight, axis=0)
        # result = tf.map_fn(lambda args: custom_function(args[0], args[1]), (expand_matrix, self.weight_selector), dtype=tf.float32)
        # result = tf.reduce_prod(expand_matrix, axis=1) + self.bias
        #self.alpha_1 = self.alpha1_function(self.weight_selector)
        #self.alpha_2 = self.alpha2_function(self.weight_selector)
        expand_matrix = tf.expand_dims(inputs, axis=2) * tf.expand_dims(self.weight, axis=0) + tf.expand_dims(self.bias, axis=0)
        result = tf.reduce_prod(expand_matrix, axis=1)# + self.bias
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
model = tf.keras.Sequential()
#model.add(Dense(10, activation='linear'))
model = tf.keras.Sequential([CustomNeuron(units=2, input_shape=(4,))])
# model.add(Dense(10, activation='linear'))
# model.add([CustomNeuron(units=4, input_shape=(10,))])
# model = tf.keras.Sequential()
# model.add(Dense(100, activation='linear'))
# model.add(Dense(100, activation='linear'))
# model.add(Dense(100, activation='linear'))
#model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))

# Generate some dummy data
# x_train = tf.random.normal((1000, 2)) * 100
x_train = tf.random.normal((1000,4), mean = 1000, stddev = 70)
xtrain_numpy = x_train.numpy()

#y_train = x_train[:, 0]*x_train[:, 1]#+3*x_train[:, 2]*x_train[:, 3]
y_train = x_train[:, 0]*x_train[:, 1]+0.1*x_train[:, 2]*x_train[:, 3]
#y_train = -x_train[:, 0]*x_train[:, 0]+10*x_train[:, 0]+100
#y_train = tf.reduce_prod(x_train, axis=1)
# y_train = tf.reduce_sum(x_train, axis=1) 
#y_train = tf.math.sin(x_train[:0])+x_train[1]
y_train_num = y_train.numpy()

# Compile the model

# opt = tf.keras.optimizers.SGD(learning_rate=0.0001)
# opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

# model.compile(optimizer=opt, loss=custom_loss)

model.compile(optimizer='adam', loss=custom_loss)
# opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
# model.compile(optimizer=opt, loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=1000, batch_size=32)

# Generate some dummy test data*
#x_test = tf.random.normal((10, 2)) * 100
x_test = tf.random.normal((10, 4), mean = 200, stddev = 70)
x_test_numpy = x_test.numpy()

#y_test = x_test[:, 0]*x_test[:, 1]#+3*x_test[:, 2]*x_test[:, 3]
y_test = x_test[:, 0]*x_test[:, 1]+0.1*x_test[:, 2]*x_test[:, 3]
#y_test = -x_test[:, 0]*x_test[:, 0]+10*x_test[:, 0]+100
#y_test = tf.reduce_prod(x_test, axis=1)
# y_test = tf.reduce_sum(x_test, axis=1)
# y_test = tf.math.sin(x_test)
y_test_num = y_test.numpy()

# Make predictions on the test data
predictions = model.predict(x_test)

# Print the predictions
print("x_test_numpy:", x_test_numpy)
print("y_test:", y_test_num)
print("Predictions:", predictions)