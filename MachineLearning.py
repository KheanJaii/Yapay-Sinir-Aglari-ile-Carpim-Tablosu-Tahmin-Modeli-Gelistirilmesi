import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

data = []
for i in range(1, 11):
    for j in range(1, 11):
        data.append([i/100, j/100, (i*j)/100])

data = np.array(data)

np.random.shuffle(data)

np.random.seed(1)

train_size = int(0.7 * len(data))
train_data, test_data = data[:train_size], data[train_size:]

input_layer_size = 2
hidden_layer_size = 16
output_layer_size = 1

weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)
weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size)


learning_rate = 0.1
epochs = 100000

beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8


m_weights_input_hidden = np.zeros_like(weights_input_hidden)
v_weights_input_hidden = np.zeros_like(weights_input_hidden)
m_weights_hidden_output = np.zeros_like(weights_hidden_output)
v_weights_hidden_output = np.zeros_like(weights_hidden_output)

for epoch in range(epochs):
    hidden_layer_input = np.dot(train_data[:, :2], weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    
    error = train_data[:, 2].reshape(-1, 1) - predicted_output

   
    output_error = error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_error.dot(weights_hidden_output.T) * sigmoid_derivative(hidden_layer_output)

    m_weights_hidden_output = beta1 * m_weights_hidden_output + (1 - beta1) * hidden_layer_output.T.dot(output_error)
    v_weights_hidden_output = beta2 * v_weights_hidden_output + (1 - beta2) * (hidden_layer_output.T.dot(output_error))**2
    weights_hidden_output += (learning_rate / (np.sqrt(v_weights_hidden_output) + epsilon)) * m_weights_hidden_output

    m_weights_input_hidden = beta1 * m_weights_input_hidden + (1 - beta1) * train_data[:, :2].T.dot(hidden_layer_error)
    v_weights_input_hidden = beta2 * v_weights_input_hidden + (1 - beta2) * (train_data[:, :2].T.dot(hidden_layer_error))**2
    weights_input_hidden += (learning_rate / (np.sqrt(v_weights_input_hidden) + epsilon)) * m_weights_input_hidden
    
    mse = mean_squared_error(train_data[:, 2], predicted_output.flatten())
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Mean Squared Error: {mse}")


test_inputs = test_data[:, :2]
hidden_layer_test = sigmoid(np.dot(test_inputs, weights_input_hidden))
output_layer_test = sigmoid(np.dot(hidden_layer_test, weights_hidden_output))
predicted_results = output_layer_test * 100


for i in range(len(test_data)):
    input_values = test_data[i, :2] * 100
    actual_result = test_data[i, 2] * 100
    predicted_result = predicted_results[i, 0]
    
    print(f"[{int(input_values[0])}x{int(input_values[1])}] = Modelin tahmin ettiği değer = [{predicted_result:.2f}] Gerçek sonuç: [{actual_result:.2f}]")
