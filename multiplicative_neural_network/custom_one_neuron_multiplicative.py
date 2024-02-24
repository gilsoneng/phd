import tensorflow as tf
import numpy as np

# Generate random continuous input data with two variables (X_train) as float32
X_train = np.random.rand(100, 2).astype(np.float32)*100  # 100 samples with 2 continuous variables each

# Calculate the corresponding continuous output data (y_train) by summing the two variables
# y_train = np.sum(X_train, axis=1)

y_train = X_train[:, 0] * X_train[:, 1]
# Define a custom single neuron model
class CustomNeuron(tf.Module):
    def __init__(self):
        self.W = tf.Variable(tf.random.normal([2, 1], dtype=tf.float32), name='weight')
        self.b = tf.Variable(tf.zeros([1], dtype=tf.float32), name='bias')

    def __call__(self, x):
        return tf.math.reduce_prod(tf.matmul(x, self.W),1) + self.b

# Instantiate the custom neuron model
neuron = CustomNeuron()

# Define a custom mean absolute error percentage (MAPE) loss function
def mean_absolute_error_percentage(y_true, y_pred):
    absolute_error = tf.abs(y_true - y_pred)
    percentage_error = tf.divide(absolute_error, tf.maximum(tf.abs(y_true), 1e-6))
    return 100.0 * tf.reduce_mean(percentage_error)

# Create an optimizer
optimizer = tf.optimizers.Adam(learning_rate=.01)
# optimizer = tf.optimizers.SGD(learning_rate=1)

# Training loop
num_epochs = 10000

for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        predictions = neuron(X_train)
        loss = mean_absolute_error_percentage(y_train, predictions)

    gradients = tape.gradient(loss, [neuron.W, neuron.b])
    optimizer.apply_gradients(zip(gradients, [neuron.W, neuron.b]))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.numpy()}')

# Test the model
X_test = np.random.rand(10, 2).astype(np.float32)*100   # New continuous input data for testing with two variables
# y_test = np.sum(X_test, axis=1)  # True output is the sum of the two variables
y_test =  X_test[:, 0] * X_test[:, 1] # True output is the sum of the two variables

predictions = neuron(X_test)  # Predictions from the model

print("True Values:")
print(y_test)
print("Predicted Values:")
print(predictions.numpy())

predictions_np = predictions.numpy()

# X_test_np = X_test.numpy()
# y_test_np = y_test.numpy()