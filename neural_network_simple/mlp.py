import numpy as np

# Define sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the input data and target
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Set random seed for reproducibility
np.random.seed(0)

# Initialize weights randomly with mean 0
input_size = 3
hidden_size = 2
output_size = 1

weights_input_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
weights_hidden_output = 2 * np.random.random((hidden_size, output_size)) - 1

# Training the neural network
epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output_layer_output = sigmoid(output_layer_input)

    # Calculate the error
    error = y - output_layer_output

    # Backpropagation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate

# Print the final output after training
print("Final output after training:")
print(output_layer_output)
