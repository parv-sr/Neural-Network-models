import tkinter as tk
from tkinter import ttk
import json
import torch

with open(r"C:\F DRIVE\Python\Neural Network models\Coffee preference predictor\coffee_preference_model_saved_weights.json", "r") as f:
    data = json.load(f)


root = tk.Tk()
root.title("Coffee Preference Predictor")
root.geometry("400x300")
root.resizable(False, False)


tk.Label(root, text="Enter your age:").pack(pady=(10, 0))
age_entry = tk.Entry(root)
age_entry.pack()


tk.Label(root, text="Select time of day:").pack(pady=(10, 0))
time_options = [str(i) for i in range(24)]
time_var = tk.StringVar(root)
time_var.set("12")
tk.OptionMenu(root, time_var, *time_options).pack()



tk.Label(root, text="Select weather:").pack(pady=(10, 0))
weather_options = {
    "Sunny": 0,
    "Rainy": 1,
    "Cloudy": 2
}
weather_var = tk.StringVar()
weather_dropdown = ttk.Combobox(root, textvariable=weather_var, values=list(weather_options.keys()), state="readonly")
weather_dropdown.pack()




def predict():
    try:
        age_nn = int(age_entry.get())
        time_nn = int(time_var.get())
        weather_nn = weather_options[weather_var.get()]

        input_vector = torch.tensor([age_nn/100, time_nn/24, weather_nn/2], dtype=torch.float32)

        w1 = torch.tensor(data["weight_vec1"], dtype=torch.float32)
        b1 = torch.tensor(data["bias1"], dtype=torch.float32)

        w2 = torch.tensor(data["weight_vec2"], dtype=torch.float32)
        b2 = torch.tensor(data["bias2"], dtype=torch.float32)

        w_out = torch.tensor(data["weight_out"], dtype=torch.float32)
        b_out = torch.tensor(data["bias_out"], dtype=torch.float32)

        z1 = torch.dot(input_vector, w1) + b1
        z2 = torch.dot(input_vector, w2) + b2

        a1 = torch.sigmoid(z1)
        a2 = torch.sigmoid(z2)

        activation_vector = torch.stack([a1, a2])

        z_out = torch.dot(w_out, activation_vector) + b_out

        predicted_value = torch.sigmoid(z_out).item()


        for widget in root.pack_slaves():
            if isinstance(widget, tk.Label) and "You " in widget.cget("text"):
                widget.destroy()


        if 0 < predicted_value < 0.2:
            text = "You probably wouldn't like coffee"
        elif 0.2 <= predicted_value < 0.5:
            text = "You might like coffee right now"
        elif 0.5 <= predicted_value < 0.8:
            text = "You would probably enjoy a cup of coffee right now"
        else:
            text = "You almost definitely will love coffee right now"

        response_label = tk.Label(root, text=text, wraplength=350, justify="left", fg="blue")
        response_label.pack(pady=(15, 10))

        print(predicted_value)

    except Exception as e:
        print("Error:", e)


predict_btn = tk.Button(root, text="Predict", command=predict)
predict_btn.pack()



root.mainloop()