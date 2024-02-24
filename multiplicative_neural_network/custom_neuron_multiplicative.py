import tensorflow as tf
import numpy as np

# Custom loss function: Mean Square Absolute Percentage Error (MSAPE)
def msape_loss(y_true, y_pred):
    diff = tf.abs((y_true - y_pred) / tf.maximum(tf.abs(y_true), 1e-10))
    return 100 * tf.reduce_mean(tf.square(diff))
# Generate synthetic data for training
np.random.seed(0)
X_train = np.random.rand(100, 2)  # 100 samples with 2 variables
y_train = np.prod(X_train, axis=1)  # Output is the product of the input variables

# Custom neuron for sum operation
class CustomNeuron(tf.keras.layers.Layer):
    def __init__(self, units=10, **kwargs):
        super(CustomNeuron, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.weight = self.add_weight(name="weight", shape=(input_shape[-1], self.units),
                                               initializer="random_normal", trainable=True)
        self.bias = self.add_weight(name="bias", shape=(self.units,), initializer="zeros", trainable=True)
        super(CustomNeuron, self).build(input_shape)

    def call(self, inputs):
        # Apply a dense layer with a single unit to simulate the sum operation
        # custom_operation = tf.multiply(inputs, self.weight)  # Element-wise multiplication
        # custom_operation = tf.reduce_sum(custom_operation, axis=1)  # Sum along the last axis
        # return custom_operation + self.biasB_transposed = tf.transpose(B)
        custom_operation = tf.matmul(inputs, self.weight)  + self.bias
        return custom_operation # tf.reduce_sum(custom_operation, axis=-1)

# Define the neural network model
model = tf.keras.Sequential([
    CustomNeuron(input_shape=(2,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss=msape_loss)

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32)

# Generate synthetic data for prediction
X_test = np.array([[0.5, 0.6], [0.2, 0.8]])

# Make predictions
predictions = model.predict(X_test)

print("Predictions:")
print(predictions)
