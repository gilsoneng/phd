# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 20:24:32 2023

@author: aluga.com
"""

# please I need a Python code using TensorFlow
# tensor X input with z variables and n inputs
# tensor Y as a product of X z variables
# tensor W with the shape (k, z)
# tensor P as the multiplication X * W

import tensorflow as tf
import numpy as np


# Define the dimensions
z = 3  # Number of variables in X
n = 5  # Number of inputs in X
k = 2  # Number of outputs in P

# Create a placeholder for the input tensor X
# X = tf.placeholder(tf.float32, shape=(None, n), name='X')

# Create a placeholder for the weight tensor W
# X = tf.Variable(tf.random.normal(shape=(n, z), mean = 0, stddev = 10, dtype=tf.float32), name='X')
# X_numpy = X.numpy()
# x_array = np.array([[1, 2, 3],
#                     [0.1, 3, 6],
#                     [1, 1, 2],
#                     [1, 3, 2],
#                     [5, 1, 2]])

x_array = np.array([[100, 200, 300, 100],
                    [100, 300, 600, 100],
                    [100, 100, 200, 100],
                    [100, 300, 200, 100],
                    [500, 100, 200, 100]])

X = tf.convert_to_tensor(x_array, dtype=tf.float32)
X_numpy = X.numpy()

# Create a placeholder for the weight tensor W
# W = tf.Variable(tf.random.normal(shape=(z, k), dtype=tf.float32), name='W')
# W_numpy = W.numpy()

# Create a placeholder for the weight tensor W
# W = tf.Variable(tf.ones(shape=(z, k), dtype=tf.float32), name='X')
# W_numpy = W.numpy()

w_array = np.array([[1, 1],
                    [-0.01, 1],
                    [1, 1],
                    [1, 1],])

W = tf.convert_to_tensor(w_array, dtype=tf.float32)
W_numpy = W.numpy()

b_array = np.array([[0, 0],
                    [0.99, 0],
                    [0, 0],
                    [0, 0],])

B = tf.convert_to_tensor(b_array, dtype=tf.float32)
B_numpy = B.numpy()

# # Compute P as the product of X and W
# P = tf.matmul(X, W)  # Use transpose to match dimensions for multiplication
# P_numpy = P.numpy()

# O = tf.math.reduce_prod(P, axis = 1)
# O_numpy = O.numpy()


with tf.GradientTape() as g:
    g.watch(X)
    # Compute P as the product of X and W
    P = tf.expand_dims(X, axis=2) * tf.expand_dims(W, axis=0) + tf.expand_dims(B, axis=0)
    P_numpy = P.numpy()

    O = tf.math.reduce_prod(P, axis = 1)
    O_numpy = O.numpy()
    
    dO_dX = g.gradient(O, X)

with tf.GradientTape() as g:
    g.watch(W)
    # Compute P as the product of X and W
    P = tf.expand_dims(X, axis=2) * tf.expand_dims(W, axis=0) + tf.expand_dims(B, axis=0)
    P_numpy = P.numpy()

    O = tf.math.reduce_prod(P, axis = 1)
    O_numpy = O.numpy()
    
    dO_dW = g.gradient(O, W)

print("X Variable:")
print(X_numpy)
print("W Variable:")
print(W_numpy)
print("B Variable:")
print(B_numpy)
print("d/dx")
print(dO_dX)
print("d/dw")
print(dO_dW)

w_array = np.array([[1, 1],
                    [-0.01, 1],
                    [1, 1],
                    [1, 1],])

W = tf.convert_to_tensor(w_array, dtype=tf.float32)
W_numpy = W.numpy()

b_array = np.array([[0, 0],
                    [0.99, 0],
                    [0, 0],
                    [0, 0],])

B = tf.convert_to_tensor(b_array, dtype=tf.float32)
B_numpy = B.numpy()
# # Compute P as the product of X and W
# P = tf.matmul(X, W)  # Use transpose to match dimensions for multiplication
# P_numpy = P.numpy()

# O = tf.math.reduce_prod(P, axis = 1)
# O_numpy = O.numpy()


with tf.GradientTape() as g:
    g.watch(X)
    # Compute P as the product of X and W
    P = tf.expand_dims(X, axis=2) * tf.expand_dims(W, axis=0) + tf.expand_dims(B, axis=0)
    P_numpy = P.numpy()

    O = tf.math.reduce_prod(P, axis = 1)
    O_numpy = O.numpy()
    
    dO_dX = g.gradient(O, X)

with tf.GradientTape() as g:
    g.watch(W)
    # Compute P as the product of X and W
    P = tf.expand_dims(X, axis=2) * tf.expand_dims(W, axis=0) + tf.expand_dims(B, axis=0)
    P_numpy = P.numpy()

    O = tf.math.reduce_prod(P, axis = 1)
    O_numpy = O.numpy()
    
    dO_dW = g.gradient(O, W)
    

print("X Variable:")
print(X_numpy)
print("W Variable:")
print(W_numpy)
print("B Variable:")
print(B_numpy)
print("d/dx")
print(dO_dX)
print("d/dw")
print(dO_dW*0.00001)


#--------------------------------------------------------------------

def custom_function(y):
    # Replace this with your custom logic
    alpha_2 = 1/(1+tf.exp(-y))
    alpha_1 = 1-alpha_2
    return alpha_1, alpha_2

k_array = np.array([[20, 20],
                    [-20, 20],
                    [20, 20],
                    [20, 20],])

k_array = tf.convert_to_tensor(k_array,dtype=tf.float32)

w, b = custom_function(k_array)

dEdW = np.array([[20, 20],
                 [-1000, 20],
                 [20, 20],
                 [20, -10000000],])

dEdW = tf.convert_to_tensor(dEdW,dtype=tf.float32)

lr = 0.1

dEdW_dW = (1/(1+tf.exp(-dEdW)) -1/2)*lr

k_array = k_array + dEdW_dW

k_array

def apply_saturation_rule(x):
    return tf.cond(x >= 20, lambda: 20, lambda: tf.cond(x <= -20, lambda: -20, lambda: x))

# Vectorize the rule function using TensorFlow's vectorized_map
k_array = tf.map_fn(lambda x: tf.map_fn(apply_saturation_rule, x), k_array)

k_array

b, w = custom_function(k_array)

w

b


dEdW = np.array([[0.1, 0.1],
                 [0.1, 0.1],
                 [0.1, 0.1],
                 [0.1, 0.1],])

dEdW = tf.convert_to_tensor(dEdW,dtype=tf.float32)

model.layers[0].trainable_variables[0].assign_add(dEdW)
