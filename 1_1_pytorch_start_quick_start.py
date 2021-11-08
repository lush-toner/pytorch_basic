# 빠른 시작
# 데이터 작업 하기

"""
파이토치 데이터 작업을 위한 기본 요소
1. torch.utils.data.Dataset : samples와 label을 저장
2. torch.utils.data.DataLoader : Dataset을 iterable로 감싼다.

"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

import matplotlib.pyplot as plt

# torchvision.datasets 모듈은 CIFAR, COCO 등과 같은 다양한 실제 vision 데이터에 대한 Dataset을 포함하고 있다.

train_data = datasets.FashionMNIST(
    root = "./data",
    train = True,
    download = True,
    transform = ToTensor(),
)

test_data = datasets.FashionMNIST(
    root = "./data",
    train =False,
    download = True,
    transform = ToTensor()
)

# Dataset 을 Dataloader의 인자로 전달
# 이는 데이터셋을 iterable한 객체로 감싼다
# batch, sampling, shuffling, multiprocess data loading 가능


batch_size = 64

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print("Shape of X [N, C, H, W]: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

# =========================================
# 모델 만들기
# =========================================

"""
Pytorch에서 신경망 모델은 nn.Module을 상속받는 클래스를 생성하여 정의
__init__ : 신경망의 계층(layer)을 정의
forward  : 신경망에 데이터를 어떻게 전달할지
"""

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

# Define Model

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# 매개 변수 최적화

loss_fn = nn.CrossEntropyLoss() # loss function
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3) # optimizer

# 각 Training loop에서 모델은 batch로 제공하는 학습 데이터셋에 대한 prediction과 예측 오류를 역전파하여 모델 매개변수 조정

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# model이 제대로 학습하고 있는지 확인하기 위해 test셋으로 성능 확인
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item() # max value index predict 후, y와 비교
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


# 모델 저장하기
"""
일반적인 모델 저장 방법


모델의 매개변수들을 포함하여 internal state dictorionary를 serialize 하는 것
serialize :  데이터 구조나 오브젝트 상태를 동일하거나 다른 컴퓨터 환경에 저장하고 나중에 재구성할 수 있는 포맷으로 변환하는 과정
"""

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# 모델 불러오기

model = NeuralNetwork()
model.load_state_dict(torch.load("model.pth"))


# 모델 불러온 후 예층 가능


classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1] # x : image, y : label
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')