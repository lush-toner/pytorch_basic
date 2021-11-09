"""
데이터는 학습에 필요한 최종 처리된 형태로 제공되지 않기 때문에 변형은 필수다.
"""


import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

# transform 했을 때
ds = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=ToTensor(), # normalize
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)) # one-hot vector
)

# transform 안했을 때
ds_ori = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=False
)

import numpy as np
print("No transform")
print(ds_ori[0][0])


print("---------")
print(ds[0][0])