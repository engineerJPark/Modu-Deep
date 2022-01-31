# Lecture 8 : Perceptron, XOR Prob, Backpropagation

## 왜 인공신경망이라고 하는가?

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled.png)

실제 신경망은 Axon에서 일정한 양의 자극(threshold)를 넘지 못하면 Terminal로 신호를 보내지 못한다. 이를 본따서 ‘인공신경망’, Perceptron이라고 한다.

## Fully Connected Layer

이렇게 모든 input node가 output node에 연결된 구조의 Layer를 Fully Connected Layer라고 한다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%201.png)

여기에 Activation Function을 붙이면 Perceptron이 된다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%202.png)

이런 구조의 Perceptron은 Linear Classifier를 만들 목적으로 만들어졌다. Linear Classifier는 말 그대로 두 Class를 구분짓는 선형적인 관계식을 찾는 것이다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%203.png)

## AND OR 그리고 XOR Problem

다음과 같은 AND와 OR를 Linear Classifier로 구현하고자 하였다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%204.png)

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%205.png)

위 사진과 같이 구현하면 될 거다.

하지만 심각한 문제가 있었으니, 바로 XOR는 단일한 Linear Classifier로 해결이 안되고, Multi Layer를 이용해야 하나, Weight를 update할 BackPropagation이 아직 발견되지 않은 것이다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%206.png)

아래는 Lienar Classifier로는 안된다는 참회를 한 그림.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%207.png)

아래에 실제로 구현할 것이다.

## Multi Layer Perceptron : Backpropagation

다음과 같은 구조이다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%208.png)

Y1이 예측값이고 O라고 하고 실제 label의 값을 G라고 하면, Loss = G - O

output부터 input 방향으로 미분을 전달해주는 것이다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%209.png)

두 가지 pass를 거치게된다.

1. forward : 초기값을 지정해준다. 그리고 output방향으로 점차 미분 전의 함수값을 전달한다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%2010.png)

1. backward : 미분을 하고 그 값을 뒤에 전달한다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%2011.png)

Layer가 많은 경우도 똑같다. 그냥 바로 **앞뒤 Node에 대한 미분만 구한 다음, 그걸 곱하면 되는 것이다!**

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%2012.png)

Sigmoid의 미분도 이런 방식으로 한다.

![Untitled](Lecture%208%20Perceptron,%20XOR%20Prob,%20Backpropagation%20944ce2a2e9ab45c39caa89e051e1dafb/Untitled%2013.png)

---

XOR 문제를 실제로 구현해서 테스트해보자.

```python
# XOR by Single Layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

linear = torch.nn.Linear(2,1, bias=True)
sigmoid = torch.nn.Sigmoid()
model = nn.Sequential(linear, sigmoid)
optimizer = optim.SGD(model.parameters(), lr = 0.1)

for epoch in range(10000 + 1):
    cost = F.binary_cross_entropy(model(X), Y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print("{} : {}", epoch, cost.item())

print(model(X))
print(Y)
```

진짜로 학습이 안되는 걸 볼 수 있다.

```python
{} : {} 9600 0.6931471824645996
{} : {} 9700 0.6931471824645996
{} : {} 9800 0.6931471824645996
{} : {} 9900 0.6931471824645996
{} : {} 10000 0.6931471824645996

tensor([[0.5000],
        [0.5000],
        [0.5000],
        [0.5000]], grad_fn=<SigmoidBackward0>)
tensor([[0.],
        [1.],
        [1.],
        [0.]])
```

---

이번에는 Multi Layer를 이용해서 해보자.

```python
# Backpropagation implementation
# XOR by Multiple Layer

import torch

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# ???
w1 = torch.Tensor(2,1)
b1 = torch.Tensor(2)
w2 = torch.Tensor(2,1)
b2 = torch.Tensor(1)

def sigmoid(x):
    return 1. / (1. + torch.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

learning_rate = 0.1

for step in range(10000 + 1):
    # forward pass. Multi Layer
    l1 = torch.add(torch.matmul(X, w1), b1) # w1*X + b1
    a1 = sigmoid(l1) # sigmoid(w1*X + b1)
    l2 = torch.add(torch.matmul(a1, w2), b2) # w2*a1 + b2
    Y_pred = sigmoid(l2) # sigmoid(w2*a1 + b2)
    
    cost = -torch.mean(Y * torch.log(Y_pred) + (1 - Y) * torch.log(1 - Y_pred)) # B-Cross Entropy
    
    # backwward Propagation : chain rule까지 적용된 결과
    d_Y_pred = (Y_pred - Y) / (Y_pred * (1. - Y_pred) + 0.00000001) # Loss : B-Cross Entropy의 미분
    # Layer 2
    d_l2 = d_Y_pred * sigmoid_prime(l2)
    d_b2 = d_l2 # 1 * d_l2
    d_w2 = torch.matmul(torch.transpose(a1, 0, 1), d_l2) # w2*X + b2를 w2로 미분하면 X만 남는다.
    # d_w2 = torch.matmul(torch.transpose(a1, 0, 1), d_b2)
    # Layer 1
    d_a1 = torch.matmul(d_b2, torch.transpose(w2,0,1))
    d_l1 = d_a1 * sigmoid_prime(l1)
    d_b1 = d_l1 
    d_w1 = torch.matmul(torch.transpose(X, 0, 1), d_l1)
    # d_w1 = torch.matmul(torch.transpose(X, 0, 1), d_b1)
    
    # Update process
    w1 = w1 - learning_rate * d_w1
    b1 = b1 - learning_rate * torch.mean(d_b1, 0)
    w2 = w2 - learning_rate * d_w2
    b2 = b2 - learning_rate * torch.mean(d_b2, 0)
    
    if step % 100 == 0:
        print(step, cost.item())

print(Y_pred)
print(Y)

# ...
# 9600 0.004000889137387276
# 9700 0.003951025195419788
# 9800 0.003902316326275468
# 9900 0.003854866838082671
# 10000 0.003808571957051754

# tensor([[0.0023],
#         [0.9952],
#         [0.9952],
#         [0.0033]])
# tensor([[0.],
#         [1.],
#         [1.],
#         [0.]])
```

```python
# 이어서 xor를 high level 구현으로 풀어본다.
# Single Layer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

linear1 = nn.Linear(2,2,bias=True)
linear2 = nn.Linear(2,1,bias=True)
sigmoid = nn.Sigmoid()
model = nn.Sequential(linear1, sigmoid, linear2, sigmoid)
optimizer = optim.SGD(model.parameters(), lr=1)

for step in range(10000 + 1):
    hypothesis = model(X)
    cost = F.binary_cross_entropy(hypothesis, Y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, cost.item())
print(model(X))
```

```python
# 더 깊은 Layer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
Y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

model = nn.Sequential(nn.Linear(2,10,bias=True),
                      nn.Sigmoid(),
                      nn.Linear(10,10,bias=True),
                      nn.Sigmoid(),
                      nn.Linear(10,10,bias=True),
                      nn.Sigmoid(),
                      nn.Linear(10,1,bias=True),
                      nn.Sigmoid())
optimizer = optim.SGD(model.parameters(), lr=1)

for step in range(10000 + 1):
    hypothesis = model(X)
    cost = F.binary_cross_entropy(hypothesis, Y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, cost.item())
print(model(X))
```