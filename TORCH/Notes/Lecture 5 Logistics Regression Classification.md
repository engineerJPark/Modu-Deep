# Lecture 5 : Logistics Regression/Classification

선형회귀에서 주요요소는 다음과 같았다.

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled.png)

하지만,

Logistics : 이진분류 = 참/거짓, 0/1 중 하나만 고르는 문제이다. 선형회귀를 그대로 적용할 수 없다.

# Classification에서 만나는 문제

1. outlier를 만났을 때, 그 출력이 0과 1 뿐이라서 linear regression 표현식이 크게 변한다.

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%201.png)

1. $H(x) = Wx + b$가 0과 1사이가 안나오는 문제가 있다.

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%202.png)

이 문제를 해결하기 위해 sigmoid function (혹은 logistic function)을 도입한다. 강제로 output을 0과 1 사이에 두는 것이다.

아래 그림은 sigmoid function의 graph이다.

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%203.png)

즉 수식을 다음과 같이 고려한다.

$Z = WX + b, \space H(X) = g(Z), \space g = sigmoid$

이를 이용해서 binary classification에서의 새로운 hypothesis를 정의한다.

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%204.png)

## Cost Function

위의 새로운 hypothesis와 기존의 cost function을 이용해서 그래프를 그리면 다음과 같이 local minimum이 많이 나온다. 즉 global minimum으로 가기가 어려워진다.

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%205.png)

이를 해결하기 위해 새로운 cost function을 설정해야한다.

---

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%206.png)

이렇게 새로운 Cost Function을 정의한다. y = 0, y = 1인 경우에 따라서 함수가 달라진다.

위 식은 Cost의 평균을 Cost Function이라고 보는 의미이고, 그 아랜 그 Cost의 정의

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%207.png)

기본적으로 $\log$를 사용하는 이유는 자연상수 의 효과를 절충하여 선형성을 주기 위해서이다.

그리고 위 사진에서 Cost Function의 정의를 보면 $H(x)$가 와 일치할 때 cost가 0이 되는 것을 볼 수 있다.

---

이제 if 절을 없애자. 즉, y = 1이냐 0이냐에 따라서 각 term을 없애는 방식으로 정의하면 될 것이다.

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%208.png)

---

즉, 기본적으로 2개를 분류하는 것은 sigmoid함수를 이용해서 hypothesis를 정의하는 것 부터해서 시작한다.

![Untitled](Lecture%205%20Logistics%20Regression%20Classification%20a6c6b08d003a44f7bc71b3388dc67aca/Untitled%209.png)

---

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F

torch.manual_seed(1)

x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
y_data = [[0], [0], [0], [1], [1], [1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

W = torch.zeros((2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr = 0.001)
epochs = 10000
for epoch in range(epochs + 1):
    hypothesis = 1/(1 + torch.exp(-(x_train.matmul(W) + b)))
    # hypothesis = torch.sigmoid(x_train.matmul(W) + b) # 이렇게 해도 된다.

    loss = -(y_train * torch.log(hypothesis) + (1 - y_train) * torch.log(1 - hypothesis))
    cost = loss.mean() # 하나의 스칼라로 저장
    # cost = F.binary_cross_entropy(hypothesis, y_train) # 이렇게 해도 된다.
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} Cost: {:.6f}'.format(
            epoch, epochs, cost.item()
            ))
```