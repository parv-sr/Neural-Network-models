import torch 
import torch.nn.functional as F
import json
import matplotlib.pyplot as plt

with open(r"C:\\F DRIVE\\Python\\Neural Network models\\coffee_preference_dataset.json", "r") as f:
    data = json.load(f)


weight_vec1 = torch.tensor([0.1, 0.2, -0.3], requires_grad=True)  #neuron 1
bias1 = torch.tensor(0.4, requires_grad=True)

weight_vec2 = torch.tensor([-0.4, 0.3, 0.2], requires_grad=True)  # neuron 2
bias2 = torch.tensor(0.1, requires_grad=True)


weight_out = torch.tensor([0.1, 0.3], requires_grad=True)
bias_out = torch.tensor(0.6, requires_grad=True)

epochs = 10
eta = 0.1
error_list = []
error_history = []

# forward pass (training loop from here)

for k in range(epochs):

    for i in range(len(data["age"])):
        input_vec = torch.tensor([data["age"][i]/100, data["time"][i]/24, data["weather"][i]/2])
        y_actual = torch.tensor(data["labels"][i], dtype=torch.float32)

        z1 = torch.dot(input_vec, weight_vec1) + bias1
        z2 = torch.dot(input_vec, weight_vec2) + bias2

        a1 = torch.sigmoid(z1)
        a2 = torch.sigmoid(z2)

        activation_vec = torch.stack([a1, a2])

        z_out = torch.dot(weight_out, activation_vec) + bias_out
        y_pred = torch.sigmoid(z_out)

        loss = F.binary_cross_entropy(y_pred, y_actual)
        error_list.append(loss)


        # backpropagation (gradient descent from here onwards)

        loss.backward()
        
        with torch.no_grad():
            weight_vec1 -= eta * weight_vec1.grad
            weight_vec2 -= eta * weight_vec2.grad
            bias1 -= eta * bias1.grad
            bias2 -= eta * bias2.grad
            weight_out -= eta * weight_out.grad
            bias_out -= eta * bias_out.grad

        weight_vec1.grad.zero_()
        weight_vec2.grad.zero_()
        bias1.grad.zero_()
        bias2.grad.zero_()
        weight_out.grad.zero_()
        bias_out.grad.zero_()



        # pytorch inbuilt optimisation: 
        #params = [weight_vec1, bias1, weight_vec2, bias2, weight_out, bias_out]
        #optimiser = torch.optim.SGD(params, lr=0.1)
        
        
    avg_error = sum(error_list)/len(error_list)
    error_history.append(avg_error)



avg_error_after_all_epochs = sum(error_history)/len(error_history)
print(avg_error_after_all_epochs)

plt.plot(error_history)
plt.show()






