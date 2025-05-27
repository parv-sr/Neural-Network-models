import numpy as np
import json
import matplotlib.pyplot as plt
import random 

with open(r"C:\\F DRIVE\\Python\\Neural Network models\\coffee_preference_dataset.json", "r") as f:
    data = json.load(f)


weight1 = [random.random(), random.random(), -random.random()]    # hidden neuron 1
bias1 = 0.4

weight2 = [random.random(), random.random(), -random.random()]     # hidden neuron 2
bias2 = 0.5

# New hidden layer (layer 2)
weight4 = [random.random(), random.random()]   # neuron 1 in layer 2 (takes a1, a2)
bias4 = 0.3

weight5 = [random.random(), random.random()]   # neuron 2 in layer 2
bias5 = 0.6

# New hidden layer (layer 3)
weight6 = [random.random(), random.random()]   # neuron 1 in layer 3 (takes a4, a5)
bias6 = 0.1

weight7 = [random.random(), random.random()]   # neuron 2 in layer 3
bias7 = 0.2

# Output layer (same)
weight3 = [random.random(), random.random()]            # output neuron (takes a6, a7)
bias3 = 0.2

learning_rate = 0.05
error_list = []
epochs = 100  
error_history = []


for k in range(epochs):

    for i in range(len(data["age"])):

        input_vec = [data["age"][i]/100, data["time"][i]/24, data["weather"][i]/2]
        y_actual = data["labels"][i]

        # Forward pass - Layer 1
        z1 = input_vec[0]*weight1[0] + input_vec[1]*weight1[1] + input_vec[2]*weight1[2] + bias1
        a1 = 1 / (1 + np.exp(-z1))

        z2 = input_vec[0]*weight2[0] + input_vec[1]*weight2[1] + input_vec[2]*weight2[2] + bias2
        a2 = 1 / (1 + np.exp(-z2))

        # Forward pass - Layer 2
        z4 = a1*weight4[0] + a2*weight4[1] + bias4
        a4 = 1 / (1 + np.exp(-z4))

        z5 = a1*weight5[0] + a2*weight5[1] + bias5
        a5 = 1 / (1 + np.exp(-z5))

        # Forward pass - Layer 3
        z6 = a4*weight6[0] + a5*weight6[1] + bias6
        a6 = 1 / (1 + np.exp(-z6))

        z7 = a4*weight7[0] + a5*weight7[1] + bias7
        a7 = 1 / (1 + np.exp(-z7))

        # Output layer
        z_out = a6*weight3[0] + a7*weight3[1] + bias3
        y_pred = 1 / (1 + np.exp(-z_out))

        error = -(y_actual*np.log(y_pred) + (1 - y_actual)*np.log(1 - y_pred))
        error_list.append(error)

        # ---------------------
        # Backpropagation step
        # ---------------------

        delta_out = y_pred - y_actual

        grad_w3_0 = delta_out * a6
        grad_w3_1 = delta_out * a7
        grad_b3 = delta_out

        delta_6 = delta_out * weight3[0] * a6 * (1 - a6)
        delta_7 = delta_out * weight3[1] * a7 * (1 - a7)

        grad_w6 = [delta_6 * a4, delta_6 * a5]
        grad_b6 = delta_6

        grad_w7 = [delta_7 * a4, delta_7 * a5]
        grad_b7 = delta_7

        delta_4 = (delta_6 * weight6[0] + delta_7 * weight7[0]) * a4 * (1 - a4)
        delta_5 = (delta_6 * weight6[1] + delta_7 * weight7[1]) * a5 * (1 - a5)

        grad_w4 = [delta_4 * a1, delta_4 * a2]
        grad_b4 = delta_4

        grad_w5 = [delta_5 * a1, delta_5 * a2]
        grad_b5 = delta_5

        delta_1 = (delta_4 * weight4[0] + delta_5 * weight5[0]) * a1 * (1 - a1)
        delta_2 = (delta_4 * weight4[1] + delta_5 * weight5[1]) * a2 * (1 - a2)

        grad_w1 = [delta_1 * input_vec[0], delta_1 * input_vec[1], delta_1 * input_vec[2]]
        grad_b1 = delta_1

        grad_w2 = [delta_2 * input_vec[0], delta_2 * input_vec[1], delta_2 * input_vec[2]]
        grad_b2 = delta_2

        # Update weights and biases
        weight3[0] -= learning_rate * grad_w3_0
        weight3[1] -= learning_rate * grad_w3_1
        bias3 -= learning_rate * grad_b3

        for j in range(2):
            weight6[j] -= learning_rate * grad_w6[j]
            weight7[j] -= learning_rate * grad_w7[j]
            weight4[j] -= learning_rate * grad_w4[j]
            weight5[j] -= learning_rate * grad_w5[j]

        bias6 -= learning_rate * grad_b6
        bias7 -= learning_rate * grad_b7
        bias4 -= learning_rate * grad_b4
        bias5 -= learning_rate * grad_b5

        for j in range(3):
            weight1[j] -= learning_rate * grad_w1[j]
            weight2[j] -= learning_rate * grad_w2[j]

        bias1 -= learning_rate * grad_b1
        bias2 -= learning_rate * grad_b2

        avg_error = sum(error_list) / len(error_list)

    error_history.append(avg_error)


print(f"Last prediction: {y_pred * 100:.2f}% chance that you like coffee")
print(f"Average error after 1 epoch: {avg_error:.4f}")

plt.plot(error_history)
plt.show()
