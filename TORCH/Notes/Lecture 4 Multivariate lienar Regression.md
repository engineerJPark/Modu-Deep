# Lecture 4 : Multivariate lienar Regression

이번에는 data가 늘어났다.

![Untitled](Lecture%204%20Multivariate%20lienar%20Regression%208104dee8e8894de98220be61907e589f/Untitled.png)

관계식은 다음과 같이 쓴다.

![Untitled](Lecture%204%20Multivariate%20lienar%20Regression%208104dee8e8894de98220be61907e589f/Untitled%201.png)

Loss function과 gradient descent는 여전히 똑같다.

![Untitled](Lecture%204%20Multivariate%20lienar%20Regression%208104dee8e8894de98220be61907e589f/Untitled%202.png)

![Untitled](Lecture%204%20Multivariate%20lienar%20Regression%208104dee8e8894de98220be61907e589f/Untitled%203.png)

다음은 구현 코드이다.

Low level implement

```python
import torch
import torch.optim as optim

x_train = torch.FloatTensor([[73,80,75],
                             [93,88,93],
                             [89,91,80],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185,],[180],[196],[142]])

W = torch.zeros((3,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

optimizer = optim.SGD([W,b], lr=0.0001)

epochs = 100

for epoch in range(epochs + 1):
    hypothesis = x_train.mm(W) + b # matmul or @

    cost = torch.mean((hypothesis - y_train)**2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {}, Cost: {:.6f}'.format(epoch, epochs, hypothesis.squeeze().detach(), cost.item()))
```

high Level implementation

```python
import torch
import torch.optim as optim # optimizing algorith 사용
import torch.nn as nn # Module 상속
import torch.nn.functional as F # loss function 사용

# forward() : hypothesis 반환
# backward() : gradient 계산
class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1) # 3개의 feature input과 1개의 feature output(그냥 class를 물어보는 것이므로)
        
    def forward(self, x): # linear를 반환
        return self.linear(x)
    
x_train = torch.FloatTensor([[73,80,75],
                             [93,88,93],
                             [89,91,80],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185,],[180],[196],[142]])

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr =0.00001) # parameter를 받을 때는 model.paramter로 받는다.
epochs = 10000

for epoch in range(epochs + 1):
    hypothesis = model(x_train)
    cost = F.mse_loss(hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print('Epoch {:4d}/{} hypothesis: {}, Cost: {:.6f}'.format(
          epoch, epochs, hypothesis.squeeze().detach(), cost.item()))
```

nn. Module에 관해서는

https://pytorch.org/docs/stable/generated/torch.nn.Module.html

그냥 pytorch에서 neural network 정의할 때 반드시 상속시키는 class 이다.

nn.Linear에 관해서는

https://pytorch.org/docs/1.9.1/generated/torch.nn.Linear.html

https://m.blog.naver.com/fbfbf1/222480437930

feature의 개수만 받고 sample의 개수에 대해서는 신경쓰지 않는다.

model.paramter에 관해서는

https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.parameters

그냥 optimizer에 parameter를 전달하기 위해서 사용한다고 생각하면 되겠다.

**# Loading Big Data**

이제는 많은 데이터를 받아오는 경우를 생각해보자

많은 데이터를 사용해야한다. 그러면 일부분씩 나눠서 학습한다.

전체 데이터를 사용할 때보다는 좀더 거친 epoch-loss graph를 보인다.

![Untitled](Lecture%204%20Multivariate%20lienar%20Regression%208104dee8e8894de98220be61907e589f/Untitled%204.png)

Minibatch Stochastic Gradient Descent를 하면 다음과 같이 descent가 불균일하게 된다. 거시적으로만 Cost가 줄어드는 셈.

![Untitled](Lecture%204%20Multivariate%20lienar%20Regression%208104dee8e8894de98220be61907e589f/Untitled%205.png)

```python
import torch

# 그동안은 다음과 같이 Data를 불러오거나 생성해서 사용하였다.

x_train = torch.FloatTensor([[79,88,75],
                             [93,88,93],
                             [89,91,90],
                             [96,98,100],
                             [73,66,70]])
y_train = torch.FloatTensor([[152],[185],[180],[196],[142]])

# 앞으로 큰 데이터는 다음과 같이 한다.

from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [[79,88,75],
                       [93,88,93],
                       [89,91,90],
                       [96,98,100],
                       [73,66,70]]
        self.y_data = [[152],[185],[180],[196],[142]]
    
    def __len__(self): # 이 데이터셋의 총 데이터 수
        return len(self.x_data)
    
    def __getitem__(self.x_data): # idx데이터 반환
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y

dataset = CustomDataset()
```

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset,
                        batch_size = 2, # batch size 정의. 메모리의 효율적 사용을 위해 2의 제곱수로 설정
                        shuffle=True) # epoch마다 데이터셋을 섞는다. 데이터가 학습되는 순서 바꿈.

# enumerate(dataloader) : minibatch 인덱스와 데이터를 받는다.
# len(dataloader) : 한 epoch당 minibatch의 개수이다.

nb_epoch = 20
for epoch in range(nb_epoch + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epoch, batch_idx + 1, len(dataloader), cost.item
        ))
```

enumerate()에 대해서는 다음 링크로 확인한다.

https://wikidocs.net/16045

위의 경우 enumerate의 결과는 (idx, (x_train, y_train))일 것이다.