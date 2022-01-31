# Lecture 7 : Methods for Optimizing the Results

## MLE : Maximum Likelihood Estimation

0과 1에 대한 분포 : Binary Distribution = Bernoulli Distribution

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled.png)

이를 이용해서 $\theta$에 대한 함수 $f(\theta)$가 되는 것이다. 

즉, MLE는 observation을 가장 잘 설명하는 $\theta$를 찾아내는 과정이다.

관측한 data를 가장 잘설명하는 distribution의 parameter를 찾는 것.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%201.png)

그럼 최대 지점은 어떻게 구할 것인가?

미분을 해서 0이 되는 지점을 구한다. 즉 Gradient Des/Asc를 사용한다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%202.png)

## Overfitting

하지만 MLE는 반드시 overfitting 문제를 직면한다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%203.png)

위의 그림 중 파란색과 같은 경계선이 아니라 빨간선처럼 training data에 대해서만 잘 맞는 결과를 내보내면 이를 overfitting이라고 한다.

학습 과정을 그래프로 나타내면 다음과 같을 것이다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%204.png)

여기서 validation loss가 최소인 지점에서 학습을 멈추면 될 것이다.

overfitting을 해결하기 위해서는 다음과 같이 observation(data)를 구성한다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%205.png)

Test set은 단 한번만 마지막에 사용하고, validation용으로 Dev set을 사용하는 것이다.

이 외에도 overfitting은 다음과 같은 방법으로 방지 가능하다.

- More Data
- Less Feature
- **Regularization**

## Regularization → Overfitting 방지

1. Early Stopping : Validation Loss가 더 이상 낮아지지 않을 때 실행.
2. Reducing Network size
3. Weight Decay : weight의 크기를 제한하는 것. 학습을 하면서 점점 작아지도록
4. Dropout : 몇개의 NN unit을 종료한다.
5. Batch Normalization : 

## Deep Neural Network에 대한 기본적인 접근법

1. Neural network 구조를 만든다.
2. train → check overfitted
    1. overfit 될 때까지 model의 size를 늘린다. 더 많은 layer와 unit의 개수 의미
    2. overfit 됐다면 regularization(drop out, batch normalization)을 가한다.
3. 무한 반복?

뒤는 코드를 보세요?

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

x_train = torch.FloatTensor([[1, 2, 1],
                             [1, 3, 2],
                             [1, 3, 4],
                             [1, 5, 5],
                             [1, 7, 5],
                             [1, 2, 5],
                             [1, 6, 6],
                             [1, 7, 7]
                            ])
y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])
```

```python
class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,3)
    def forward(self, x):
        return self.linear(x)
    
model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

```python
def train(model, optimizer, x_train, y_train):
    epochs = 100
    for epoch in range(epochs + 1):
        prediction = model(x_train)
        cost = F.cross_entropy(prediction, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
        ))
```

```python
def test(model, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1] # prediction 중에서 가장 큰 값을 내놓는다.
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)
    
    print('Accuracy: {}% Cost: {:.6f}'.format(
         correct_count / len(y_test) * 100, cost.item()
    ))
```

```python
train(model, optimizer, x_train, y_train)

# Epoch    0/100 Cost: 2.203666
# Epoch    1/100 Cost: 1.989424
# Epoch    2/100 Cost: 1.803548
# Epoch    3/100 Cost: 1.647425
# Epoch    4/100 Cost: 1.520720
# Epoch    5/100 Cost: 1.421076
# Epoch    6/100 Cost: 1.344574
# Epoch    7/100 Cost: 1.286687
# Epoch    8/100 Cost: 1.243165
# Epoch    9/100 Cost: 1.210481
# Epoch   10/100 Cost: 1.185905
# Epoch   11/100 Cost: 1.167381
# Epoch   12/100 Cost: 1.153379
# Epoch   13/100 Cost: 1.142754
# Epoch   14/100 Cost: 1.134649
# Epoch   15/100 Cost: 1.128421
# Epoch   16/100 Cost: 1.123588
# Epoch   17/100 Cost: 1.119791
# Epoch   18/100 Cost: 1.116760
# Epoch   19/100 Cost: 1.114298
# Epoch   20/100 Cost: 1.112257
# Epoch   21/100 Cost: 1.110527
# Epoch   22/100 Cost: 1.109028
# Epoch   23/100 Cost: 1.107700
# Epoch   24/100 Cost: 1.106501
# Epoch   96/100 Cost: 1.050260
# Epoch   97/100 Cost: 1.049548
# Epoch   98/100 Cost: 1.048838
# Epoch   99/100 Cost: 1.048129
# Epoch  100/100 Cost: 1.047422

test(model, x_test, y_test)

# Accuracy: 0.0% Cost: 1.861790
```

보면 이미 overfitting이 일어난 상태임을 알 수 있다.

![2022-01-17-21-39-00.png](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/2022-01-17-21-39-00.png)

# Learning Rate

$\theta \leftarrow \theta - \alpha \nabla_\theta L(x;\theta)$

Learning Rate가 너무 크면 다음과 같이 발산한다.

![2022-01-17-21-39-00.png](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/2022-01-17-21-39-00%201.png)

너무 작으면 아예 학습이 안되고.

# Data Preprocessing

데이터를 전처리하는 과정을 통해서 문제를 보다 안정적으로 풀 수 있다.

 

```python
x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
```

정규분포화 한다.

![2022-01-17-21-44-13.png](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/2022-01-17-21-44-13.png)

```python
mu = x_train.mean(dim = 0)
sigma = x_train.std(dim = 0)
norm_x_train = (x_train - mu) / sigma
print(norm_x_train) # gaussian distribution을 따른다.

class MultivariateLienarRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
        
    def forward(self, x):
        return self.linear(x)
    
model = MultivariateLienarRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(model, optimizer, x_train, y_train):
    epochs = 100
    for epoch in range(epochs + 1):
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
        ))
```

```python
train(model, optimizer, norm_x_train, y_train)

# Epoch    1/100 Cost: 28484.753906
# Epoch    2/100 Cost: 27337.447266
# Epoch    3/100 Cost: 26237.316406
# Epoch    4/100 Cost: 25182.328125
# Epoch    5/100 Cost: 24170.564453
# Epoch    6/100 Cost: 23200.175781
# Epoch    7/100 Cost: 22269.408203
# Epoch    8/100 Cost: 21376.582031
# Epoch    9/100 Cost: 20520.101562
# Epoch   10/100 Cost: 19698.441406
# Epoch   11/100 Cost: 18910.128906
# Epoch   12/100 Cost: 18153.785156
# Epoch   13/100 Cost: 17428.066406
# Epoch   14/100 Cost: 16731.705078
# Epoch   15/100 Cost: 16063.479492
# Epoch   16/100 Cost: 15422.226562
# Epoch   17/100 Cost: 14806.833008
# Epoch   18/100 Cost: 14216.231445
# Epoch   19/100 Cost: 13649.401367
# Epoch   20/100 Cost: 13105.369141
# Epoch   21/100 Cost: 12583.198242
# Epoch   22/100 Cost: 12081.996094
# Epoch   23/100 Cost: 11600.904297
# Epoch   24/100 Cost: 11139.104492
# Epoch   96/100 Cost: 605.487915
# Epoch   97/100 Cost: 581.528809
# Epoch   98/100 Cost: 558.518677
# Epoch   99/100 Cost: 536.420349
# Epoch  100/100 Cost: 515.196899
```

![2022-01-17-22-47-45.png](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/2022-01-17-22-47-45.png)

최적화를 하지 않으면 첫 column만 줄어들고 두 번째 column은 그대로 유지. 이미 작으니깐
이를 해결하기 위해 normalization을 하는 것이다.

# 김성훈 교수님의 설명

 

Learning Rate가 크면 다음과 같이 overshooting이 발생한다. 

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%206.png)

Learning Rate가 너무 작으면 수렴을 안한다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%207.png)

이를 위해서는 

1. Cost function을 확인한다.
2. Learning Rate를 낮춘다.

등고선으로 본다면 다음과 같이 이동할 것이다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%208.png)

그런데 등고선이 다음과 같이 찌그러져 있으면(=하나의 feature가 작거나 크다면) 그 하나의 feature에 대한 값이 수렴하지 않고 발산한다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%209.png)

그래서...

## Standardization

다음과 같이 zero-centered로 만들거나 일정한 범위로 들게 normalized하게  한다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2010.png)

보통 다음과 같은 식으로 계산한다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2011.png)

## Overfitting

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2012.png)

오른쪽과 같이 training data에 대해서만 model이 정확한 경우이다.

해결법

1. training data 많게
2. feature의 개수 줄여라
3. Regularization을 한다.

## Regularization

일단 Overfitting이 된 상황이라면 아래 그림과 같이 구분선이 구부러진 상황이다.

Regularization을 통해서 이를 펼 수 있다.

또한, Regularization은 weight의 어떠한 값이 과도하게 큰 값을 가지지 않도록 하는 효과를 지닌다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2013.png)

수식으로는 다음과 같이 작성한다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2014.png)

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2015.png)

## Training/Testing Dataset에 대하여

training set으로만 학습 예측을 하면 의미가 없다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2016.png)

validation set을 이용해서 hyperparameter tuning을 하고, testing은 단 한 번만 한다.

### Online Learning ???

학습해야 할 양이 너무 많은 경우 사용한다.

나눠서 하나씩 추가하고, 추가하는 과정에서 중간마다 기존의 learning 결과가 남아있다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2017.png)

실제 데이터는 다음과 같이 train-test set으로 나뉘어 있다.

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2018.png)

# MNIST dataset

![Untitled](Lecture%207%20Methods%20for%20Optimizing%20the%20Results%2020e11bbced684549a8784cd42d8f4187/Untitled%2019.png)

28*28*1인 dataset이다.

참고로 Pytorch의 경우 이미지의 형식이 0~ 1, Channel * Height * Width인데, 보통의 이미지의 경우 0 ~ 255, Height * Width * Channel이다.

이를 ToTensor()로 변환할 수 있다.

실제 방법은 코드를 참고하자.

# Epoch / Batch size / Iteration

### Epoch

하나의 training set을 전부 train하면 한 epoch가 끝난 것이다. 즉, 한 번의 forward pass와 backward pass가 끝난 것을 한 epoch라고 부른다. 참고로 batch size로 나눈 것과 상관없이 모든 training set을 순환한 것을 의미한다.

### Batch size

training set이 크면 그걸 잘라서 학습하는데, 그 자른 크기를 batch size라고 한다. 보통은 2의 제곱수를 사용한다.

### Iteration

한 epoch가 되기까지, 몇 개의 batch가 학습에 사용했는지.

예를 들면, 1000개의 sample에  500크기의 batchsize를 둔다고 하면 2번의 iteration을 거쳐야 1번의 epoch가 끝난다.

## Softmax로 다운받은 데이터 사용해보기

```python
import torchvision.datasets as dsets
import torchvision.transforms as transforms

batch_size = 128
training_epochs = 100

mnist_train = dsets.MNIST(root="MNIST_data/",
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
mnist_test = dsets.MNIST(root="MNIST_data/",
                          train=False,
                          transform=transforms.ToTensor(),
                          download=True)

# 불러온 값을 사용하기 위해 dataloader를 사용한다.
data_loader=torch.utils.data.DataLoader(dataset=mnist_train,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True) # batch_size에 안맞고 남는 data 버린다.

for epoch in range(training_epochs):
    for X, Y in data_loader:
        X = X.view(-1,28*28) # reshape image to 784, but Label should be onehotecoded
        # 참고로 X는 (batch size, 1, 28, 28)이었는데 그 이후 (batch size, 784)가 된다.
```

```python
linear = nn.Linear(784,10,bias=True)
training_epochs = 15
batch_size = 128

criterion = nn.CrossEntropyLoss() # F.cross_entropy는 정의 할 때 모든 걸 설정해야한다.
optimizer = optim.SGD(linear.parameters(), lr=0.1)

for epoch in range(training_epochs):
    avg_cost = 0
    len_total_batch = len(data_loader)
    for X, Y in data_loader:
        X = X.view(-1, 28*28)
        optimizer.zero_grad()
        hypothesis = linear(X)
        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / len_total_batch
        
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

with torch.no_grad(): # test할 때 꼭 넣어주자. 오류가 줄어든다.
    X_test = mnist_test.test_data.view(-1,28*28).float()
    Y_test = mnist_test.test_labels
    
    prediction = linear(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print("Accuracy : ", accuracy.item())
```

**with torch.no_grad(): 를 test할 때 꼭 넣어주자.** 

```python
# Visualization
import matplotlib.pyplot as plt
import random

r = random.randint(0, len(mnist_test) - 1)
x_single_data = mnist_test.test_data[r:r+1].view(-1,28*28).float()
Y_single_data = mnist_test.test_labels[r:r+1]

print("Label : ", Y_single_data.item())
single_prediction = linear(x_single_data)
print("Prediction : ", torch.argmax(single_prediction, 1).item())

plt.imshow(mnist_test.test_data[r:r+1].view(28,28))
plt.show()
```

# Reference

[torch.Tensor.to — PyTorch 1.10.1 documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to)

[Tensor Attributes — PyTorch 1.10.1 documentation](https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device)