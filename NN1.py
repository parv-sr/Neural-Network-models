import numpy as np
import json
import matplotlib.pyplot as plt

with open(r"C:\F DRIVE\Python\Neural Network models\coffee_preference_dataset.json", "r") as f:
    data = json.load(f)


weight1 = [0.05, 0.06, -0.2]    # hidden neuron 1
bias1 = 0.4

weight2 = [0.04, 0.03, 0.1]     # hidden neuron 2
bias2 = 0.5

weight3 = [0.1, 0.3]            # output neuron
bias3 = 0.2

learning_rate = 0.01
error_list = []

epochs = 100
error_history = []


for k in range(epochs):

    for i in range(len(data["age"])):

        input_vec = [data["age"][i], data["time"][i], data["weather"][i]]
        y_actual = data["labels"][i]

        # Forward pass
        z1 = input_vec[0]*weight1[0] + input_vec[1]*weight1[1] + input_vec[2]*weight1[2] + bias1
        a1 = 1 / (1 + np.exp(-z1))

        z2 = input_vec[0]*weight2[0] + input_vec[1]*weight2[1] + input_vec[2]*weight2[2] + bias2
        a2 = 1 / (1 + np.exp(-z2))

        z_out = a1*weight3[0] + a2*weight3[1] + bias3
        y_pred = 1 / (1 + np.exp(-z_out))


        error = -(y_actual*np.log(y_pred) + (1 - y_actual)*np.log(1 - y_pred))
        error_list.append(error)

        # ---------------------
        # Backpropagation step
        # ---------------------

        delta_out = y_pred - y_actual

        # Gradients for output weights and bias
        grad_w3_0 = delta_out * a1
        grad_w3_1 = delta_out * a2
        grad_b3 = delta_out

        # Backpropagate error to hidden layer
        delta_1 = delta_out * weight3[0] * a1 * (1 - a1)
        delta_2 = delta_out * weight3[1] * a2 * (1 - a2)

        # Gradients for hidden weights and biases
        grad_w1 = [delta_1 * input_vec[0], delta_1 * input_vec[1], delta_1 * input_vec[2]]
        grad_b1 = delta_1

        grad_w2 = [delta_2 * input_vec[0], delta_2 * input_vec[1], delta_2 * input_vec[2]]
        grad_b2 = delta_2


        weight3[0] -= learning_rate * grad_w3_0
        weight3[1] -= learning_rate * grad_w3_1
        bias3 -= learning_rate * grad_b3

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