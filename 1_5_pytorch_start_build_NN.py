"""
신경망은 데이터에 대한 연산을 수행하는 layer module로 구성되어 있음
torch.nn : 신경망을 구성하는데 필요한 모든 구성 요소를 제공
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# CUDA?
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Define class
class NeuralNetwork(nn.Module): # nn.Module의 하위클래스로 정의
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x): # 상속받은 모든 클래스는 forward method에 입력 데이터에 대한 연산 구현
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device) # instance 생성 -> device
print(model)



X = torch.rand(1, 28, 28, device=device) 
logits = model(X) # 모델 입력 호출

print("logits")
print(logits)

pred_probab = nn.Softmax(dim=1)(logits) # softmax

print("Softmax output")
print(pred_probab)

y_pred = pred_probab.argmax(1) # max값

print(f"Predicted class: {y_pred}")