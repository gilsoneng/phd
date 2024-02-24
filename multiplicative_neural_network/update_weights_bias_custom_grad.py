# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 21:57:55 2023

@author: aluga.com
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Custom activation function
def custom_activation(x):
    return tf.math.sin(x)

# Custom gradient of the activation function
def custom_activation_gradient(x):
    return tf.math.cos(x)

# Create a simple Sequential model with custom activation
model = keras.Sequential([
    layers.Dense(64, activation=custom_activation, input_shape=(10,)),
    layers.Dense(1, activation='linear')
])

# Dummy input data
x_train = tf.random.normal((10, 10))
y_true = tf.random.normal((10, 10))

# Define your custom loss function
def custom_loss(y_true, y_pred):
    # Your custom loss calculation
    loss = tf.reduce_mean(tf.square(y_true - y_pred))
    return loss

# Function to calculate the custom gradient using tf.GradientTape
def custom_gradient(model, x, y_true):
    #x = x_train
    with tf.GradientTape(persistent=True) as tape:
        # Apply custom activation to the hidden layer
        x_hidden = custom_activation(model.layers[0](x))

        # Forward pass with custom activation
        y_pred = model.layers[1](x_hidden)

        # Calculate custom loss
        loss = custom_loss(y_true, y_pred)

    # Calculate gradients using the custom loss and custom activation gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Include custom activation gradients in the gradient calculation
    gradients[0] = tape.gradient(x_hidden, model.layers[0].trainable_variables, output_gradients=[gradients[0]])

    return gradients

# Custom method for updating weights and biases with momentum
def custom_update(model, gradients, learning_rate=0.01):
    for layer, grad in zip(model.layers[0].trainable_variables, gradients[0]):
        # Update weights with momentum
        delta = learning_rate * grad# + momentum * prev_grad
        layer.assign_sub(delta)
        #model.layers[0].optimizer.m.assign(delta)

    for layer, grad in zip(model.layers[1].trainable_variables, gradients[1]):
        # Update biases with momentum
        delta = learning_rate * grad# + momentum * prev_grad
        layer.assign_sub(delta)
        #model.layers[1].optimizer.m.assign(delta)

# Setting up the optimizer with momentum term
model.layers[0].optimizer = tf.optimizers.Adam(learning_rate=0.01)
model.layers[1].optimizer = tf.optimizers.Adam(learning_rate=0.01)

# Training loop
epochs = 100

for epoch in range(epochs):
    epoch = 1
    # Calculate custom gradients
    gradients = custom_gradient(model, x_train, y_true)

    # Update weights and biases using the custom method with momentum
    custom_update(model, gradients)

    # Print the updated weights and biases after each epoch
    print(f"Epoch {epoch + 1}/{epochs}")
    for i, layer in enumerate(model.layers):
        if isinstance(layer, layers.Dense):
            print(f"Layer {i + 1}")
            print("Weights:")
            print(layer.weights[0].numpy())  # Weights
            print("Biases:")
            print(layer.weights[1].numpy())  # Biases

    # Calculate and print the custom loss
    y_pred = model(x_train)
    loss = custom_loss(y_true, y_pred)
    print(f"Loss: {loss.numpy()}")

# After training, you can use the model for predictions
y_pred_final = model(x_train)
