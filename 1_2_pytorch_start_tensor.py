"""
Tensor
  - 배열이나 행렬과 매우 유사한 특수한 자료구조
  - input, output, parameters를 encode 함
  - numpy의 ndarray와 유사, numpy array와도 내부 메모리 공유 가능
  - auto differentiation에 대해 최적화되어 있다.
"""

import torch
import numpy as np

"""
Tensor 초기화
"""

# 1 데이터로부터 직접 생성

data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2 Numpy 배열로부터 생성

np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 다른 텐서로부터 생성하기

x_ones = torch.ones_like(x_data) # x_data의 속성을 유지합니다.
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_data의 속성을 덮어씁니다.
print(f"Random Tensor: \n {x_rand} \n")

# 무작위 또는 상수 값을 사용
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")

# 텐서의 속성 -> tensor의 shape, datatype, 어디에 저장되는지 확인

tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")

# 텐서의 연산(Operation)

# GPU 존재시 텐서를 이동

if torch.cuda.is_available():
  tensor = tensor.to("cuda")


# 연산 이용

# tensor = torch.tensor([
#   [1,2,3,4], 
#   [5,6,7,8],
#   [9,10,11,12],
#   [13,14,15,16]])


tensor = torch.ones(4, 4)

print('First row: ',tensor[0])
print('First column: ', tensor[:, 0])
print('Last column:', tensor[..., -1]) # ... : Ellipsis
tensor[:,1] = 0
print(tensor)

# Tensor concat
t1 = torch.cat([tensor, tensor, tensor], dim=1)

# 산술 연산(Arithmetic operations)
"""
두 텐서 간의 행렬 곱(matrix multiplication) 
y1, y2, y3 -> same value
"""
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(tensor)
torch.matmul(tensor, tensor.T, out=y3)

"""
요소별 곱(element-wise product)을 계산.
z1, z2, z3 -> same value
"""
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)

# 단일 요소(single-element) 텐서
"""
텐서의 모든 값을 하나로 aggregate 하여 요소가 하나인 텐서의 경우, item()을 사용하여 python 숫자 값으로 변환ㄴ
"""
print("==========================")
print("tensor.item()")
print("==========================")
print(tensor, "\n")

tensor.add_(5)
print(tensor)
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# in-place
"""
result를 피연산자에 저장하는 연산을 in-place 연산
접미사 _가 있음

** 메모리를 일부 절약하지만, 기록(history)이 즉시 삭제되어 derivateive 계산에 문제가 발생할 수 있다.
"""
print("==========================")
print("in-place")
print("==========================")

print(tensor, "\n")
tensor.add_(5)
print(tensor)


# Numpy 변환(bridge)
"""
CPU 상의 텐서와 Numpy 배열은 메모리 location을 공유하기 때문에, 하나를 변경하면 다른 하나도 변경됨
"""
print("==========================")
print("Numpy bridge")
print("==========================")

t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")

t.add_(1)
print(f"t: {t}")
print(f"n: {n}")


# Numpy 배열을 tensor로 변환

n = np.ones(5)
t = torch.from_numpy(n)

print(f"t: {t}")
print(f"n: {n}")
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")