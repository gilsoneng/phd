# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 14:39:40 2023

@author: aluga.com
"""

import tensorflow as tf
import numpy as np

# Generate synthetic data
num_samples = 1000
input_dim = 20
output_dim = 10

X_train = np.random.randn(num_samples, input_dim)
y_train = np.random.randint(0, output_dim, size=num_samples)

class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, units=32):
        super(CustomLayer, self).__init__()
        self.units = units
    
    def build(self, input_shape):
        # Create the trainable weights for the layer
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )
    
    def call(self, inputs):
        # Define the forward pass of the layer
        output_weights = tf.matmul(inputs, self.w) + self.b
        print(output_weights)
        output_reduce_prod = tf.math.reduce_prod(output_weights, axis=None, keepdims=False, name=None)
        print(output_reduce_prod)
        return output_weights

# Create a simple model using the custom layer
model = tf.keras.Sequential([
    #CustomLayer(units=64),
    tf.keras.layers.ReLU(),
    CustomLayer(units=32),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model
epochs = 10
batch_size = 32

model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

weights = model.layers[0].get_weights()

weights = model.layers[1].get_weights()

X_train_with_weights = np.dot(X_train, weights[0]) + weights[1]

output_reduce_prod = tf.math.reduce_prod(X_train_with_weights)

output_reduce_prod_array = output_reduce_prod.np()