# Lecture 6 : Softmax Classification = Multinomial Logistics Regression

# Softmax Classification (=Multinomial Classfication)

여러개의 Classification에 관한 Prob이다.

binary classification, 즉, sigmoid를 여러번 해서 구할 수도 있기는 하다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled.png)

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%201.png)

---

**중요한 것은, softmax가 단순히 classification만이 목적이 아니라, 확률 분포를 근사한다는 사실에 주목해야한다.**

classification은 단순히 이를 응용한 것에 불과하다.

---

하지만 위의 방식대로 한다면 각 sample 별로 연산을 따로 해야하는 문제가 있다. 이는 연산양을 늘린다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%202.png)

따라서 위 그림처럼 하나의 행렬을 만들어서 연산을 할 것이다.

이제 이것 대신에, softmax에 넣어서 확률로 바꿔준다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%203.png)

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%204.png)

이렇게 행렬곱 결과로 나온 것을 softmax를 거쳐서, 각 class에 대한 확률을 내뱉는 것이다. 확률이므로 이 값들의 합은 1이 된다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%205.png)

softmax를 거친 결과 나온 확률을 one hot incoding을 통해서 [1,0,0]과 같은 형태로 만들어준다. 어떤 class인가를 확실하게 언급하는 것이다.

# Cost Function : Cross Entropy

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%206.png)

Cross Entropy는 두 확률이 얼마나 비슷한지에 대한 지표가 된다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%207.png)

확률 분포 P에서 $x$를 샘플링하고, 그 $x$에 대해서 $Q$를 구한다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%208.png)

---

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%209.png)

왼쪽이 예측값, 오른쪽이 실제값이 된다. 그리고 중앙의 식이 그 차이인 cross-entropy이다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2010.png)

이렇게 정의되고, $L_i$는 예측값, $-\log y_i$는 실제값에 log를 씌운 것이다.

다음 그림은 실제값이 A인 일 때, 예측을 A로 했을 때와 B로 했을 때이다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2011.png)

다음 그림은 실제값이 B인 일 때, 예측을 A로 했을 때와 B로 했을 때이다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2012.png)

즉 예측과 실제가 맞으면 cost가 0이되고, 틀리면 무한이 되므로, 적합한 cost function임을 알 수 있다.

---

# Logistic Cost와 Cross Entropy는 그 의미가 같다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2013.png)

두 함수가 같은 이유는 두개의 예측 및 결과만 있기에  $-\sigma(Li * log(Si) = -(L1log(S1))-(L2log(S2))$ 입니다.

실제 값 $L1,L2$은 1과 0, 그리고 서로 반대의 값을 지닐수밖에 없기 때문에 $L2 = 1-L1$ 일 수밖에 없습니다. (0 또는 1, 1또는 0)

S1, S2은 예측값이기 때문에 1,0 만 나오는 것은 아니지만 둘의 합은 1이 될 수밖에 없습니다. (0.3, 0.7 등 그러나 어쨌든 $1-L0 = L1$)

따라서 $-L1*log(S1)-(1-L1)log(1-S1)$ 가 됩니다.
$L1 = y, S1 = H(x)$ 로 바꾸면 $-ylog(H(x))-(1-y)*log(1-H(x))$가 되는 것입니다.

---

# 다시 Cost Function을 보자.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2014.png)

이번에도 Cost function은 실제값과 예측값 사이의 distance로 정의해주면 된다. 뒤에서 코드로 보자.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2015.png)

Optimizing은 이 Cost Function을 편미분해서 구한다.

## Probability Distribution

### PMF : discrete

point의 높이가 확률을 의미

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2016.png)

### PDF : continuous

not point, 면적이 확률을 의미

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2017.png)

### softmax 구현하기

softmax를 거쳐서 나오는 확률은 다음과 같다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2018.png)

---

이런 경우의 텐서에서 argmax를 구하면 다음과 같다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2019.png)

참고로 one hot incoding 이후, softmax를 거쳐 나온 것([0.85,0.01,0.02,0.05,0.07])이나, 원래 data label에서 one hot incoding을 거친 결과물([0,0,1,0,0,0])이나 어차피 똑같이 '확률'을 의미한다. 단순히 data label에서 얻은 확률은 '이것이 100% 맞다.'는 의미일 뿐.

하나의 학습에서 모든 class의 예측값으로 나온 softmax의 합은 1이 된다.

모델의 맨 마지막에 softmax를 둔다. 이후, maximum likelihood estimation을 통해 훈련한다.

### Cross Entropy 구현하기

cross entropy는 그 식이 다음과 같다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2020.png)

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2021.png)

$y$는 실제값이므로 $P(x)$이고, $\hat y$는 예측값 $Q(x)$를 의미한다. $Q(x)$는 가끔 $P_\theta(x)$라고도 한다.

one hot incoding은 `.scatter()`나 `.scatter()_`(이경우 inclass 연산)를 사용한다.

이에대한 참고문서는

[torch.Tensor.scatter_ — PyTorch 1.10.1 documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html)

[torch.Tensor.scatter_ — PyTorch 1.10.1 documentation](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html)

참고로 후자의 method를 써야 메모리가 새로 할당되지 않는다. inclass method는 메모리 할당이 없다.

Cost가 되는 cross entropy를 code로 옮기면 다음과 같다.

![Untitled](Lecture%206%20Softmax%20Classification%20=%20Multinomial%20Log%209e60ac29e165448e96f9bce4fb9de1b7/Untitled%2022.png)

high level 표현은 github에서 코드를 직접 확인하자.

참고로 functional 패키지의 cross_entropy는 이미 softmax가 들어있다. 이에 주의.

구현할 때 다음과 같은 사항을 잊지말자.

Binary Classification이면 :

- sigmoid
- Binary Cross Entropy

Multinomial Classification이면 :

- softmax
- Cross Entropy

# 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# softmax 구현
z = torch.FloatTensor([1,2,3])

# torch 내장 softmax
hypothesis = F.softmax(z, dim = 0)
print(hypothesis)
print(hypothesis.sum())

# Cross Entropy 구현
z = torch.rand(3,5,requires_grad=True)
hypothesis = F.softmax(z, dim=1)
print(hypothesis)
y = torch.randint(5,(3,)).long() # data type long으로 변경
print(y)

y_one_hot = torch.zeros_like(hypothesis)
y_one_hot.scatter_(1, y.unsqueeze(1), 1)

# cross entropy
cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean() # 하나의 지표로 나타내기 위해서 mean으로 처리해버린다.
print(cost)

# 다시 풀어서 정리하면 다음과 같다.

# Cross Entropy
# low level에서의 cross entorpy와 loss
torch.log(F.softmax(z, dim=1))
(y_one_hot * -torch.log(F.softmax(z, dim=1))).sum(dim=1).mean()

# hight level에서의 cross entropy와 loss 
F.log_softmax(z, dim=1)
F.nll_loss(F.log_softmax(z, dim=1), y) # negative log likelihood
```

대신에 다음을 사용할수도 있다.

F.cross_entropy(z, y)

F.log_softmax(z, dim=1)

F.nll_loss(F.log_softmax(z, dim=1), y)

첫번째 인수는 log probability이고, 두번째 인수는 target이다.

### 이제는 cross entropy로 학습을 하는 과정을 거쳐보자...

```python
x_train = torch.FloatTensor([[1,2,1,1],
                            [2,1,3,2],
                            [3,1,3,4],
                            [4,1,5,5],
                            [1,7,5,5],
                            [1,2,5,6],
                            [1,6,6,6],
                            [1,7,7,7]])
y_train = torch.LongTensor([2,2,2,1,1,1,0,0])

W = torch.zeros((4,3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W,b], lr=0.01)

epochs = 1000
for epoch in range(epochs + 1):
    hypothesis = F.softmax(x_train.matmul(W) + b, dim=1)
    y_one_hot = torch.zeros_like(hypothesis)
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1) # y_train이 주는 index에 따라서 1을 뿌린다는 뜻이다.
    cost = (y_one_hot * -torch.log(F.softmax(hypothesis, dim=1))).sum(dim=1).mean()
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
            ))
```

```python
x_train = torch.FloatTensor([[1,2,1,1],
                            [2,1,3,2],
                            [3,1,3,4],
                            [4,1,5,5],
                            [1,7,5,5],
                            [1,2,5,6],
                            [1,6,6,6],
                            [1,7,7,7]])
y_train = torch.LongTensor([2,2,2,1,1,1,0,0])

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3) # feature의 개수만 받는다. input, output 모두. 그리고 W,b를 내장으로 가지고 있다.
        
    def forward(self, x):
        return self.linear(x)    

model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 1000
for epoch in range(epochs + 1):
    hypothesis = model(x_train)
    cost = F.cross_entropy(hypothesis, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
            ))
```

nn.Lienar에 대해선 [https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)