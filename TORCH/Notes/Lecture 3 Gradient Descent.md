# Lecture 3 : Gradient Descent

hypothesis는 다음과 같다.

![Untitled](Lecture%203%20Gradient%20Descent%2078eaebf5f8c348779609488d1e152b5d/Untitled.png)

쉬운 예시를 위해 다음과 같은 상황을 가정하자.

![Untitled](Lecture%203%20Gradient%20Descent%2078eaebf5f8c348779609488d1e152b5d/Untitled%201.png)

![Untitled](Lecture%203%20Gradient%20Descent%2078eaebf5f8c348779609488d1e152b5d/Untitled%202.png)

그러면 다음과 같은 상황이 최상의 상황인 것이다.

![Untitled](Lecture%203%20Gradient%20Descent%2078eaebf5f8c348779609488d1e152b5d/Untitled%203.png)

cost function은 그래프가 다음과 같다.

![Untitled](Lecture%203%20Gradient%20Descent%2078eaebf5f8c348779609488d1e152b5d/Untitled%204.png)

![Untitled](Lecture%203%20Gradient%20Descent%2078eaebf5f8c348779609488d1e152b5d/Untitled%205.png)

이 최소값을 컴퓨터로 찾을 때에는 gradietn descent를 사용한다.

![Untitled](Lecture%203%20Gradient%20Descent%2078eaebf5f8c348779609488d1e152b5d/Untitled%206.png)

```python
import torch
import torch.optim as optim

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

W = torch.zeros(1)
lr = 0.01

nb_epochs = 100
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W
    cost = torch.mean((hypothesis - y_train) ** 2)
    gradient = torch.sum((W * x_train - y_train) * x_train)
    
    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), cost.item()))
    
    W -= lr * gradient
```

```python
Epoch    0/100 W: 0.000, Cost: 4.666667
Epoch    1/100 W: 0.140, Cost: 3.451467
Epoch    2/100 W: 0.260, Cost: 2.552705
Epoch    3/100 W: 0.364, Cost: 1.887980
Epoch    4/100 W: 0.453, Cost: 1.396350
Epoch    5/100 W: 0.530, Cost: 1.032741
Epoch    6/100 W: 0.595, Cost: 0.763815
Epoch    7/100 W: 0.652, Cost: 0.564918
Epoch    8/100 W: 0.701, Cost: 0.417813
Epoch    9/100 W: 0.743, Cost: 0.309015
Epoch   10/100 W: 0.779, Cost: 0.228547
Epoch   11/100 W: 0.810, Cost: 0.169034
Epoch   12/100 W: 0.836, Cost: 0.125017
Epoch   13/100 W: 0.859, Cost: 0.092463
Epoch   14/100 W: 0.879, Cost: 0.068385
Epoch   15/100 W: 0.896, Cost: 0.050578
Epoch   16/100 W: 0.910, Cost: 0.037407
Epoch   17/100 W: 0.923, Cost: 0.027667
Epoch   18/100 W: 0.934, Cost: 0.020462
Epoch   19/100 W: 0.943, Cost: 0.015134
Epoch   20/100 W: 0.951, Cost: 0.011193
Epoch   21/100 W: 0.958, Cost: 0.008278
Epoch   22/100 W: 0.964, Cost: 0.006123
Epoch   23/100 W: 0.969, Cost: 0.004528
Epoch   24/100 W: 0.973, Cost: 0.003349
Epoch   96/100 W: 1.000, Cost: 0.000000
Epoch   97/100 W: 1.000, Cost: 0.000000
Epoch   98/100 W: 1.000, Cost: 0.000000
Epoch   99/100 W: 1.000, Cost: 0.000000
Epoch  100/100 W: 1.000, Cost: 0.000000
```

다음과 같이 할 수도 있다.

```python
import torch
import torch.optim as optim

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

W = torch.zeros(1, requires_grad=True)
optimizer = optim.SGD([W], lr=0.01)

nb_epochs = 100
for epoch in range(nb_epochs + 1):
    hypothesis = x_train * W
    cost = torch.mean((hypothesis - y_train) ** 2)
        
    print('Epoch {:4d}/{} W: {:.3f}, Cost: {:.6f}'.format(epoch, nb_epochs, W.item(), cost.item()))
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
```

```python
Epoch    0/100 W: 0.000, Cost: 4.666667
Epoch    1/100 W: 0.093, Cost: 3.836207
Epoch    2/100 W: 0.178, Cost: 3.153533
Epoch    3/100 W: 0.255, Cost: 2.592344
Epoch    4/100 W: 0.324, Cost: 2.131022
Epoch    5/100 W: 0.387, Cost: 1.751795
Epoch    6/100 W: 0.444, Cost: 1.440053
Epoch    7/100 W: 0.496, Cost: 1.183788
Epoch    8/100 W: 0.543, Cost: 0.973126
Epoch    9/100 W: 0.586, Cost: 0.799953
Epoch   10/100 W: 0.625, Cost: 0.657597
Epoch   11/100 W: 0.660, Cost: 0.540574
Epoch   12/100 W: 0.691, Cost: 0.444376
Epoch   13/100 W: 0.720, Cost: 0.365297
Epoch   14/100 W: 0.746, Cost: 0.300290
Epoch   15/100 W: 0.770, Cost: 0.246852
Epoch   16/100 W: 0.791, Cost: 0.202923
Epoch   17/100 W: 0.811, Cost: 0.166812
Epoch   18/100 W: 0.829, Cost: 0.137127
Epoch   19/100 W: 0.845, Cost: 0.112724
Epoch   20/100 W: 0.859, Cost: 0.092664
Epoch   21/100 W: 0.872, Cost: 0.076174
Epoch   22/100 W: 0.884, Cost: 0.062619
Epoch   23/100 W: 0.895, Cost: 0.051475
Epoch   24/100 W: 0.905, Cost: 0.042315
Epoch   96/100 W: 1.000, Cost: 0.000000
Epoch   97/100 W: 1.000, Cost: 0.000000
Epoch   98/100 W: 1.000, Cost: 0.000000
Epoch   99/100 W: 1.000, Cost: 0.000000
Epoch  100/100 W: 1.000, Cost: 0.000000
```