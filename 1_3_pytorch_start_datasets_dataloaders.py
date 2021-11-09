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
    download=True, # 다운로드 여부, 로컬에 데이터 있으면 false
    transform=ToTensor() # input과 label을 변형
)

test_data = datasets.FashionMNIST(
    root="./data", 
    train=False,
    download=True,
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

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()


# 파일에서 사용자 정의 데이터셋 만들기
"""
Dataset 클래스는 반드시 3개 함수 구현

1. __init__
  Dataset 객체가 생성(instantiate)될 때 한 번만 실행. 
  여기서는 이미지와 annotation_file이 포함된 디렉토리와 두 가지 변형(transform)을 초기화.

2. __len__
  데이터셋의 샘플 개수 리턴

3. __getitem__
  주어진 인덱스 idx 에 해당하는 샘플을 데이터셋에서 불러오고 반환
  인덱스 기반, 디스크에서 이미지의 위치 식별
  이미지나 label을 transform 할 수 있음

"""

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# Dataloader로 학습용 데이터 준비
"""
Dataset : 데이터셋의 feature를 가져오고 하나의 샘플에 label을 지정하는 일을 한 번에 진행
DataLoader : 모델 학습 시 mini-batch, epoch마다 shuffling 등 -> iterable
"""
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Dataloader를 통해 iterate

"""
    DataLoader 에 데이터셋을 불러온 뒤, 필요에 따라 데이터셋을 iterate
    shuffle true로 모든 배치 순회 뒤 데이터가 섞인다.
"""

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")