"""
torch.utils.data.Dataset -> 샘플 & label 저장
torch.utils.data.DataLoader -> Dataset을 샘플에 쉽게 접근할 수 있도록 iterable로 warp한다datetime A combination of a date and a time. Attributes: (
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="./data", # 저장 경로
    train=True, # 학습용 or 테스트용
    download=False, # 다운로드 여부, 로컬에 데이터 있으면 false
    transform=ToTensor() # input과 label을 변형
)

test_data = datasets.FashionMNIST(
    root="./data", 
    train=False,
    download=False,
    transform=ToTensor()
)

# Dataset Iterating 과 visualization
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

# figure = plt.figure(figsize=(8, 8))
# cols, rows = 3, 3
# for i in range(1, cols * rows + 1):
#     sample_idx = torch.randint(len(training_data), size=(1,)).item()
#     img, label = training_data[sample_idx]
#     figure.add_subplot(rows, cols, i)
#     plt.title(labels_map[label])
#     plt.axis("off")
#     plt.imshow(img.squeeze(), cmap="gray")
# plt.show()


# 파일에서 사용자 정의 데이터셋 만들기
"""
Dataset 클래스는 반드시 3개 함수 구현

1. __init__
2. __len__
3. __getitem__
"""