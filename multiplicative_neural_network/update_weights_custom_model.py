import tensorflow as tf

from keras import models
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.datasets import mnist
from IPython.display import SVG
from keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from keras.utils import plot_model

# Custom Activation Function
def custom_activation(x):
    #return x
    return x

# Custom loss function: Mean Square Absolute Percentage Error (MSAPE)\
def msape_loss(y_true, y_pred):
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), 1e-15))
    return 100 * tf.reduce_mean(tf.square(diff))

def custom_loss(y_true, y_pred):
    return msape_loss(y_true, y_pred)

# Use tf.map_fn to apply the custom function element-wise to tensors C and S

# Define the mean squared error loss function
def mean_squared_error(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Create a custom neuron layer
class CustomNeuron(tf.keras.layers.Layer):
    def __init__(self, units=1,  **kwargs):
        super(CustomNeuron, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.weight = self.add_weight("weight", shape=(input_shape[-1], self.units), initializer="ones", trainable=True)
        self.bias = self.add_weight("bias", shape=(input_shape[-1], self.units), initializer="zeros", trainable=True)
        super(CustomNeuron, self).build(input_shape)

    def call(self, inputs):
        expand_matrix = tf.expand_dims(inputs, axis=2) * tf.expand_dims(self.weight, axis=0) + tf.expand_dims(self.bias, axis=0)
        #result = tf.reduce_prod(expand_matrix, axis=1)# + self.bias
        result = tf.reduce_sum(expand_matrix, axis=1)# + self.bias
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
model.add(Dense(1, activation='linear'))

# Generate some dummy data
x_train = tf.random.normal((1000,4), mean = 100, stddev = 100)
xtrain_numpy = x_train.numpy()

#y_train = x_train[:, 0]*x_train[:, 1]+0.1*x_train[:, 2]*x_train[:, 3]
y_train = x_train[:, 0]-0.5*x_train[:, 1]+0.1*x_train[:, 2]+x_train[:, 3]
y_train_num = y_train.numpy()

# Compile the model

model.compile(optimizer='adam', loss=custom_loss)

# Train the model
# model.fit(x_train, y_train, epochs=1000, batch_size=32)

# Generate some dummy test data*
x_test = tf.random.normal((10, 4), mean = 100, stddev = 100)
x_test_numpy = x_test.numpy()

y_true = x_test[:, 0]-0.5*x_test[:, 1]+0.1*x_test[:, 2]+x_test[:, 3]
y_true_num = y_true.numpy()

# ---------------------------------------------------------------------------------
# Training loop
epochs = 10000
learning_rate = 0.00001
optimizer = tf.optimizers.SGD(learning_rate)

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(x_test)
        loss = mean_squared_error(y_true, y_pred)

    # Calculate gradients
    gradients = tape.gradient(loss, [model.layers[0].weight, model.layers[0].bias])

    # Custom weight and bias updates
    model.layers[0].weight.assign_sub(learning_rate * gradients[0])
    model.layers[0].bias.assign_sub(learning_rate * gradients[1])

    # Print loss every 10 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.numpy()}')

# Print final weights and bias
print('Final Weights:', model.layers[0].weight.numpy())
print('Final Bias:', model.layers[0].bias.numpy())

# ---------------------------------------------------------------------------------

# Make predictions on the test data
predictions = model.predict(x_test)

# Print the predictions
print("x_test_numpy:", x_test_numpy)
print("y_test:", y_true_num)
print("Predictions:", predictions)