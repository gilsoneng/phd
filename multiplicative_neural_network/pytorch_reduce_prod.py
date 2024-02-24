# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 15:42:29 2023

@author: aluga.com
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define a custom activation function
class CustomActivation(nn.Module):
    def forward(self, x):
        return torch.relu(x)  # Example custom activation function (sine)

# Define a custom neuron class
class CustomNeuron(nn.Module):
    def __init__(self, input_size):
        super(CustomNeuron, self).__init__()
        self.input_size = input_size
        # self.weights = nn.Parameter(torch.randn(input_size))
        # self.bias = nn.Parameter(torch.randn(1))
        self.weights = nn.Parameter(torch.randn(input_size))
        self.bias = nn.Parameter(torch.randn(1))
        self.activation = CustomActivation()  # Custom activation function

    def forward(self, x):
        # weighted_sum = torch.sum(self.weights * x) + self.bias
        weighted_prod = torch.prod(self.weights * x,0) + self.bias
        x = self.activation(weighted_prod)
        return x
        # return weighted_sum


# Create a custom neural network with 3 layers and 10 neurons each
class CustomNetwork(nn.Module):
    def __init__(self, input_size, num_layers, num_neurons):
        super(CustomNetwork, self).__init__()
        self.input_size = input_size
        self.num_layers = num_layers
        self.num_neurons = num_neurons

        # Create layers with custom neurons
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            layer = nn.Sequential(
                *[CustomNeuron(input_size) for _ in range(self.num_neurons)]
            )
            self.layers.append(layer)
            input_size = self.num_neurons

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# Generate synthetic training data
num_samples = 10000
input_dim = 2
output_dim = 1

X_train = np.random.randn(num_samples, input_dim) * 100
# y_train = np.random.randn(num_samples, output_dim)

y_train = X_train[:,:1]*X_train[:,1:2]

# Normalize data
# X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)

# Convert data to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

X_train_np = X_train.numpy()
y_train_np = y_train.numpy()

# Create an instance of the custom neural network
custom_net = CustomNetwork(input_size=input_dim, num_layers=3, num_neurons=10)

# Create an instance of the custom neuron
neuron = CustomNeuron(input_size=input_dim)

# Define the loss function and optimizer
# criterion = nn.MSELoss()

# optimizer = optim.SGD(neuron.parameters(), lr=0.01) # Reduced learning rate

# criterion = nn.L1Loss() # nn.MSELoss()
criterion = nn.L1Loss() # nn.MSELoss()
optimizer = optim.SGD(custom_net.parameters(), lr=0.001)  # Reduced learning rate

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    # epoch = 1
    outputs = neuron(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

print('Training finished.')

# Test the trained neuron
test_input = torch.FloatTensor([5.0, 4.0])
predicted_output = neuron(test_input)
print('Predicted Output:', predicted_output.item())


test_input = torch.FloatTensor([100, 100])
predicted_output = neuron(test_input)
print('Predicted Output:', predicted_output.item())


test_input = torch.FloatTensor([40, 2])
predicted_output = neuron(test_input)
print('Predicted Output:', predicted_output.item())

test_input = torch.FloatTensor([40, 5])
predicted_output = neuron(test_input)
print('Predicted Output:', predicted_output.item())
