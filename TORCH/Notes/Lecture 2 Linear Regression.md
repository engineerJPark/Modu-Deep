# Lecture 2 : Linear Regression

공부 시간에 따른 시험 점수를 예측하는 문제를 푼다고하자.

![Untitled](Lecture%202%20Linear%20Regression%206c281771b6a14b94a4b1b0639cc429cb/Untitled.png)

다음과 같이 train data의 입출력을 지정한다.

![Untitled](Lecture%202%20Linear%20Regression%206c281771b6a14b94a4b1b0639cc429cb/Untitled%201.png)

예측값과 실제 target의 차이를 측정한다. 이를 Loss function이라고 한다. 기본적으로 예측값과 실제값의 차이이므로, 모델의 정확도를 나타낸다고 보면 된다.

![Untitled](Lecture%202%20Linear%20Regression%206c281771b6a14b94a4b1b0639cc429cb/Untitled%202.png)

즉,

1. data를 받아온다.
2. 모델을 통해 관계식을 만든다.
3. Loss function을 만든다.
4. optimizing을 통해 loss function의 값을 줄여서 관계식을 정확하게 만든다.

![Untitled](Lecture%202%20Linear%20Regression%206c281771b6a14b94a4b1b0639cc429cb/Untitled%203.png)

![Untitled](Lecture%202%20Linear%20Regression%206c281771b6a14b94a4b1b0639cc429cb/Untitled%204.png)

```python
import torch
import torch.optim as optim

# data는 torch.tensor을 이용한다.
# 입력과 출력을 따로 설정한다.

x_train = torch.FloatTensor([1],[2],[3])
y_train = torch.FloatTensor([2],[4],[6])
# Hypothesis를 정의한다.
# W, b는 0으로 초기화

W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
hypothesis = x_train * W + b

# Objective/Loss/Cost function을 정의한다.

cost = torch.mean((hypothesis - y_train) ** 2)

# Optimizer : Cost function을 미분한 결과를 이용해서 W, b를 점차 원하는 값으로 줄여나가는 알고리즘

optimizer = optim.SGD([W,b], lr=0.01)

# 항상 붙어다니는 세줄
# gradient 초기화 -> gradient 계산 -> step으로 parameter update

optimizer.zero_grad()
cost.backward()
optimizer.step()
```