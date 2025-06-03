import torch 
import torch.nn as nn 
import json

with open(r"C:\F DRIVE\Python\Neural Network models\House price predictor\house_cost_dataset.json", "r") as f:
    data = json.load(f)


class HousePriceModel(nn.Module):
    def __init__(self):
        super(HousePriceModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 5), 
            nn.ReLU(),
            nn.Linear(5, 5), 
            nn.ReLU(),
            nn.Linear(5, 5), 
            nn.ReLU(),
            nn.Linear(5, 5), 
            nn.ReLU(),
            nn.Linear(5, 5), 
            nn.ReLU(),
            nn.Linear(5, 1), 
        )

    def forward(self, x):
        return self.model(x)