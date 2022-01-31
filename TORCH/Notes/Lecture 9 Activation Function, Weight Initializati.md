# Lecture 9 : Activation Function, Weight Initialization, Dropout, Batch Normalization

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled.png)

# ReLU의 등장 배경

sigmoid는 앞뒤 부분이 1과 0에 가깝다. 이런 특성이 backpropagation 할 때 input layer에서 gradient가 vanish하게 만든다.

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%201.png)

특히 이 sigmoid가 많이 모여있으면 더 한다.

 

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%202.png)

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%203.png)

# ReLU

아래 사진과 같이 생겨서 x>0에서는 gradient가 유지된다. 물론 x<0에서는 gradient가 사라진다.

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%204.png)

이외에도 다음과 같은 activation function이 존재한다.

```python
torch.nn.sigmoid()
torch.nn.relu()
torch.nn.tanh()
torch.nn.leaky_relu()
```

# Optimizer

다음과 같은 optimizer가 있다. 

```python
torch.optim.SGD
torch.optim.Adadelta
torch.optim.Adagrad
torch.optim.Adam
torch.optim.SparseAdam
torch.optim.Adamax
torch.optim.ASGD
torch.optim.LBFGS
torch.optim.RMSprop
torch.optim.Rprop
```

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%205.png)

```python

```

```python

```

---

# Weight Initialization

알고 보니 weight가 초기에 가지고 있는 값이 학습에 지대한 영향을 주었다.

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%206.png)

0으로 초기화를 하면 gradient가 모두 0이 되버리기에 0 초기화는 안된다.

# RBM and DBM

## RBM = Restricted Boltzmann Machine

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%207.png)

layer 사이의 연결하지 않고, 다른 layer끼리 Fully Connected한다.

X → Y : encoding Y → X : decoding

## Pre-training

하나의 layer 학습하고 고정 후 그 다음 layer 쌓고 학습 후 고정하고...를 반복

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%208.png)

## Fine-tuning

Pre-training이 된 layer를 모두 쌓고(RBM을 쌓고)

우리가 하는 보통의 방법, input에서 output, 그리고 Loss까지 나오면 이를 이용해서 backpropagation을 하고, parameter를 update하는 것.

아래는 RBM을 도입한 예시

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%209.png)

# Xavier / He Initialization

간단하게 initialization을 할 수 있는 방법론. RBM 안쓰고 초기화한다.

논문은 가끔 등장하니 기억해두자

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%2010.png)

$n_{in}, n_{out}$은 각각 layer의 input, layer의 output을 의미한다.

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%2011.png)

실제 구현을 해보자.

```python

```

```python

```

# Dropout

아래가 각각 학습된 경계선의 결과이다.

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%2012.png)

참고로 overfitting의 경우는 다음과 같은 문제가 있다.

머신러닝의 목표는 학습 데이터에 대한 정확한 관계식이 아니라, 관계식을 통해 테스트 데이터가 적절하게 예측이 되는지가 중요하다. 그런데 학습데이터에만 딱 맞는 관계식이 나오고 테스트 데이터에 대해서는 엉망이니 overfitting이 좋지 않다고 하는 것.

두가지로 분류하는 학습을 한다고 하자

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%2013.png)

구불구불한 선이 overfitting된 것이고, 완만한 것이 good이다.

overfitting되면 train에서만 성능이 좋고 test에서는 성능이 안좋아진다. train set과 test set이 완벽히 일치하지 않기 때문이다.

차라리 적절히 학습된 경우가 더 정확도가 높다.

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%2014.png)

train set이 내려가면서 test set에 대한 rate가 증가하면 overfitting이라고 판단할 수 있다.

### 해결방법

1. more training data
2. reduce features
3. regularization
4. dropout

여기서는 dropout을 알아본다.

![Untitled](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Untitled%2015.png)

train할 때, 무작위로 node를 종료하고 남은 것만 이용해서 연산을 진행한다.

test할 때에는 모든 node를 사용한다.

model.train()으로 train mode를 맞춰서 dropout을 on하고, model.eval()로 eval mode로 맞춰서 dropout을 off한다.

# Batch Normalization

앞서 말한 Gradient Vanishing과 Exploding을 해결할 수 있는 방법 중 하나인 Batch Normalization을 알아보자

![Screenshot_20220124-074949_Chrome.jpg](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Screenshot_20220124-074949_Chrome.jpg)

![Screenshot_20220124-075009_Chrome.jpg](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Screenshot_20220124-075009_Chrome.jpg)

Gradient Vanishing과 Exploding의 근본 문제는 바로 train set과 test set의 분포가 다르다는 점이다.

![Screenshot_20220124-075028_Chrome.jpg](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Screenshot_20220124-075028_Chrome.jpg)

또한 Layer를 하나 거칠 때마다 입력된 데이터의 분포가 점점 변하게 될 것이다.

![Screenshot_20220124-075044_Chrome.jpg](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Screenshot_20220124-075044_Chrome.jpg)

이를 해결하기 위해 B.N을 도입해서 모든 layer의 데이터 출력 분포가 일정하게 조정한다.

감마와 베타는 trainable parameter이고, mean, var는 batch마다 달라진다.

![Screenshot_20220124-075105_Chrome.jpg](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Screenshot_20220124-075105_Chrome.jpg)

B.N은 Train 할 때만 적용하는 방법이다.

Test할 때에는 각 Batch에서 얻은 mean, var를 평균내서 구한 고정된 값인 learning mean, learning var을 이용해서 inference를 한다.

![Screenshot_20220124-075121_Chrome.jpg](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Screenshot_20220124-075121_Chrome.jpg)

batch norm은 보통 ReLU 직전에 사용한다.

![Screenshot_20220124-075141_Chrome.jpg](Lecture%209%20Activation%20Function,%20Weight%20Initializati%2063db1e41adfe4b1cade7de2ce902fb84/Screenshot_20220124-075141_Chrome.jpg)