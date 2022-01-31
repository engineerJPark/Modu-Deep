# Lecture 11 : RNN

Sequential data를 위해서 도입된 Neural Network

데이터의 순서가 중요한 경우 주로 도입한다.

word, sentence, time series 등.....

그 전까지는 position index라는 것을 이용한다. → 몇 번째 벡터라는 것을 알려주는 숫자.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled.png)

# RNN

---

RNN은 순서도 학습한다...는 것이다. 

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%201.png)

루프를 풀어서 표현하면 왼쪽 그림이 된다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%202.png)

loop를 도는 data를 hidden state라고 한다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%203.png)

hiddent state를 통해서 그 전의 입력값에 의한 영향을 반영한다. 따라서 모델이 데이터의 순서에 영향을 받게 된다.

여기서 $h_t$는 hidden state, $x_t$는 input을 의미한다. 

$f_W(x)$는 parameter를 가지고 있는 Cell A를 의미한다.

RNN은 모든 셀이 parameter를 공유하며(같은 parameter를 사용하며), 항상 같은 $f_W(x)$를 사용한다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%204.png)

다음과 같은 예시를 들 수 있다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%205.png)

결과물 $y_t$는 다음과 같이 계산한다.

즉 여기서 $W_{hh}, W{xh}, W{hy}$를 학습하는 것이다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%206.png)

하나의 character를 받고 다음에 올 단어가 뭔지 맞추는 모델을 만든다고 가정하자.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%207.png)

각 character를 표현하는 가장 쉬운 방법은 one hot encoding하는 것이다. character vector, multi label이라고 볼 수도 있다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%208.png)

초기에는 $h_t = 0$으로 두고 시작한다. 지속적으로 input과 hidden state를 섞어줌으로써 직전의 hidden state가 영향을 미치게한다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%209.png)

그 다음 $y = W_{hy}h_t$를 한다. 그 다음 softmax에 넣어서 다음 character를 예측한다.

## 다양한 RNN 구조

---

---

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2010.png)

1. Image Captioning : 이미지 to 문장
2. Semantic Classification : 문장 to 감정(sentiment, 하나의 레이블)
3. Machine Translation : Seq of words → Seq of words. 문장 to 문장 : 빈 부분은 그 출력값을 쓰지 않는다는 의미이다. 주로 번역에 사용
4. Video Classification on frame level : video to feature captioning

# Multi Layer RNN

---

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2011.png)

이렇게 여러 개의 Layer를 쌓아서 NN을 구성할 수 있다.

## Advanced model

Multi Layer RNN이 너무 계산 오래걸리고 그래서 개선한 모델 : LSTM, GRU

모델 복잡도는 LSTM이 가장 높다. GRU은 중간 정도.

# RNN input/output shape

---

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2012.png)

model의 형상은 위와 같고, 아직 그 shape은 모른다.

단 두줄이 전부다.

```python
rnn = torch.nn.RNN(input_size, hidden_size)
outputs, _status = rnn(input_data)
```

## Input size

데이터를 구성하는 element에 대한 one hot encoding(경우에 따라선 word-embedding)을 한다.

```python
# 1-hot encoding
h = [1, 0, 0, 0]
e = [0, 1, 0, 0]
l = [0, 0, 1, 0]
o = [0, 0, 0, 1]

input_size = 4 # 벡터의 길이
```

shape 중 하나의 차원 값을 구했다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2013.png)

## Hidden State

우리가 출력하고자 하는 class의 개수(감정의 개수 등)가 곳 hidden_state의 class 개수가 된다.

hidden state의 shape는 기본적으로 output과 같아야한다. 왜냐하면 hidden state든 output y든 같은 값을 통해서 구해내는 것이기 때문이다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2014.png)

```python
hidden_size = 2
```

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2015.png)

## Sequence Length

아래 사진 처럼 이제 input data가 차례로 입력될 것이다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2016.png)

입력되는 단어의 문자 길이 = sequence length가 된다. 즉 하나의 단어가 앞서 말한 element 몇 개로 만들어졌는지가 sequence length를 결정한다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2017.png)

```python
# 1-hot encoding
x1 = [1, 0, 0, 0]
x2 = [0, 1, 0, 0]
x3 = [0, 0, 1, 0]
x4 = [0, 0, 1, 0]
x5 = [0, 0, 0, 1]

input_size = 4 # 벡터의 길이
```

Pytorch는 model이 자동으로 seqence length를 파악한다.

## Batch Size

---

다른 것은 batch_size이다.

pytorch는 batch size model에 전달 안해도 된다. 알아서 해줌.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2018.png)

구현 코드다. 종합하면..

input shape : (batch size, seqence length, element size)

output shape : (batch size, seqeunce length, label size)

```python
import torch
import torch.nn as nn
import numpy as np

h = [1,0,0,0]
e = [0,1,0,0]
l = [0,0,1,0]
o = [0,0,0,1]
input_size = 4
hidden_size = 2

input_data = torch.Tensor(np.array([[h, e, l, l, o],
                                    [e, o, l, l, l],
                                    [l, l, e, e, l]], dtype=np.float32)) # shape 3,5,4
rnn = nn.RNN(input_size, hidden_size)
outputs, _status = rnn(input_data)
outputs, outputs.shape # shape 3,5,2
```

# HiHello Example

---

하나의 character가 들어오면 다음 character를 예측하는 모델을 만들고자 한다.

모델의 학습이 어디까지 진행됐는지 기억하는 RNN의 hidden state의 역할이 중요하다.

one hot ㄷncoding을 하면 숫자의 크기에 따라서 의미를 준 것이기 때문에, 크기 영향을 없애기 위해 index가 가능한 data로 표현한다.

주로 categorical한 data는 one hot encoding을 한다고 생각하면 된다.

아래는 예시 코드로, 기본적으로 아래의 형식을 따른다고 생각하면 된다.

```python
char_set = ['h', 'i', 'e', 'l', 'o']
x_data = [[0,1,0,2,3,3]]
x_one_hot = [[[1,0,0,0,0],
              [0,1,0,0,0],
              [1,0,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,0],
              [0,0,0,1,0]]]
y_data = [[1,0,2,3,3,4]]
```

### Implementation

```python
char_set = ['h', 'i', 'e', 'l', 'o']

# hyperparameter
input_size = len(char_set)
hidden_size = len(char_set)
learning_rate = 0.1

# data
x_data = [[0,1,0,2,3,3]]
x_one_hot = [[[1,0,0,0,0],
              [0,1,0,0,0],
              [1,0,0,0,0],
              [0,0,1,0,0],
              [0,0,0,1,0],
              [0,0,0,1,0]]]
y_data = [[1,0,2,3,3,4]]

rnn = nn.RNN(input_size, hidden_size, batch_first=True) # batch_first = True로 두면 batch dimension이 가장 앞으로 온다.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate)

epochs = 200
for i in range(epochs):
    optimizer.zero_grad()
    output, _status = rnn(X) # 원래 _status는 다음 hidden state이지만 당장은 쓰지 않는다. 
    loss = criterion(output.view(-1, input_size), Y.view(-1))
    loss.backward()
    optimizer.step()
    
    result = torch.argmax(output.view(-1, input_size), 1)
    accuracy = (result == Y).float().mean()
    result_str = ''.join([char_set[c] for c in torch.squeeze(result)])
    
    print("Epoch : {}, Loss : {}, accuracy : {}, true Y : {}, prediction : {}".format(i + 1, loss, accuracy, Y, result_str))
```

## Cross Entropy Loss

---

categorical classification이므로 cross entropy loss를 사용한다.

```python
criterion = nn.CrossEntropyLoss()
...
loss = criterion(outputs.view(-1, input_size), Y.view(-1)) # row vector 되도록
```

# CharSequence

---

### Implementation

```python
sample = "if you want you"

# make dictionary
char_dic = {c: i for i, c in enumerate(list(set(sample)))}

# hyperparameter
dic_size = len(char_dic)
hidden_size = len(char_dic)
learning_rate = 0.1

# data setting
sample_idx = [char_dic[c] for c in sample] # 문장을 index로 표현한다.
print(sample_idx)
x_data = [sample_idx[:-1]] # 맨 마지막 data 제거 
print(x_data)
x_one_hot = [np.eye(dic_size)[x] for x in x_data] # 문장을 index로 표현한 것을 one hot encoding
print(x_one_hot)
y_data = [sample_idx[1:]] # 맨 처음 data 제거 
print(y_data)

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)

rnn = nn.RNN(dic_size, hidden_size, batch_first=True) # batch_first = True로 두면 batch dimension이 가장 앞으로 온다.

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate)

epochs = 100
for i in range(epochs):
    optimizer.zero_grad()
    output, _status = rnn(X) # 원래 _status는 다음 hidden state이지만 당장은 쓰지 않는다. 
    loss = criterion(output.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()
    
    result = torch.argmax(output.view(-1, dic_size), 1)
    accuracy = (result == Y).float().mean()
    result_str = ''.join([char_set[c] for c in torch.squeeze(result)])
    
    print("Epoch : {}, Loss : {}, accuracy : {}, true Y : {}, prediction : {}".format(i + 1, loss, accuracy, Y, result_str))
```

`np.eye()`는 identity를 만들어준다. 이를 indexing해서 one hot vector를 가져오는 것이다.

`batch_first = True`로 두면 batch dimension이 가장 앞으로 온다.

원래 `_status`는 다음 hidden state이지만 당장은 쓰지 않는다. 

# Long Sequence

---

긴 문장의 경우 그대로 사용할 수는 없기에, 이를 작은 단위로 자르려고 한다.

text에 window를 씌우는 것이다. 한칸씩 shift하면서 전에 거는 X 그 다음 것은 Y라고 명명한다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2019.png)

```python
x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i : i + sequence_length]
    y_str = sentence[i + 1 : i + sequence_length + 1]
    print(i, x_str, '->', y_str)
    
    # get index to make one hot vector
    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

x_one_hot = [np.eye(dic_size)[c] for c in x_data] # one hot encoding
```

# Stacking RNN

그동안 이렇게 단일 RNN cell만 있던 것을

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2020.png)

앞으로는 이렇게 layer를 쌓고 마지막에 FC Layer를 추가해줄 것이다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2021.png)

```python
# rnn = nn.RNN(dic_size, hidden_size, batch_first=True) # batch_first = True로 두면 batch dimension이 가장 앞으로 온다.
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True) # num_layers를 주면 여러 층을 만들 수 있다.
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

RNNnet = Net(dic_size, hidden_size, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(rnn.parameters(), learning_rate)
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Random seed to make results deterministic and reproducible
torch.manual_seed(0)

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea.")
# make dictionary
char_set = list(set(sentence))
char_dic = {c: i for i, c in enumerate(char_set)}
char_dic

# hyperparameter
dic_size = len(char_dic)
hidden_size = len(char_dic)
sequence_length = 10 # 임의 결정
learning_rate = 0.1

x_data = []
y_data = []

for i in range(0, len(sentence) - sequence_length):
    x_str = sentence[i : i + sequence_length]
    y_str = sentence[i + 1 : i + sequence_length + 1]
    print(i, x_str, '->', y_str)
    
    # get index to make one hot vector
    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

x_one_hot = [np.eye(dic_size)[c] for c in x_data] # one hot encoding

X = torch.FloatTensor(x_one_hot)
Y = torch.LongTensor(y_data)
X.shape

# rnn = nn.RNN(dic_size, hidden_size, batch_first=True) # batch_first = True로 두면 batch dimension이 가장 앞으로 온다.
class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers=layers, batch_first=True) # num_layers를 주면 여러 층을 만들 수 있다.
        self.fc = nn.Linear(hidden_dim, hidden_dim, bias=True)
    def forward(self, x):
        x, _status = self.rnn(x)
        x = self.fc(x)
        return x

RNNnet = Net(dic_size, hidden_size, 2)
RNNnet

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(RNNnet.parameters(), learning_rate)

epochs = 100
for i in range(epochs):
    optimizer.zero_grad()
    output = RNNnet(X)
    
    # loss = criterion(output, Y)
    loss = criterion(output.view(-1, dic_size), Y.view(-1))
    loss.backward()
    optimizer.step()
    
    print(output, output.shape)
    print(output.view(X.shape[0], -1, dic_size))
    result = torch.argmax(output.view(X.shape[0], -1, dic_size), dim=2)
    print(result, result.shape)
    
    predicted_str = ''
    for j, predicted_char in enumerate(result):
        if j == 0:
            predicted_str += ''.join([char_set[t] for t in predicted_char])
        else:
            predicted_str += char_set[predicted_char[-1]]
            
    print(predicted_str)
```

# Time Series

---

시간을 따라서 변하는 데이터를 시계열 데이터라고 한다.

...

일단 다음 가정을 한다.

8일차의 주가는 앞선 7일간의 데이터에 영향을 받아 결정된다.

 

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2022.png)

여기서 hidden state를 봐자.

RNN의 구조에 따라서 hidden state의 dimenstion이 output dimension이 같다.

RNN을 통과함에 따라 데이터가 압축되는데, 이 때문에 모델에 가해지는 부담이 증가된다.

이것이 문제가 될 수 있으므로, 충분한 공간을 보장해준 후 이를 이용해서 FC를 통과한 결과가 주가를 결정하게 한다. 

따라서 데이터가 유통되는 부분 + 예측하는 부분으로 구성한다. 

```markdown
압축의 개념이 잘 이해가 되지 않는데요...(왜 부담이 생기는지)

hidden size가  10일때에 비해 1이면 압축하는 과정이 추가되는 건가요?? 이게 직관적으로 잘 이해가 되지않습니다.
hidden size가 몇이든 행렬곱을 hidden state와 input각각해서 더해주고 activation을 통과해서 넘기는 걸로 이해했는데
1이면 처리후 압축을 한다고하셔서...이부분이 잘 이해가 되지않습니다.

---

안녕하세요, 우선, 질문 주셔서 감사합니다.
여기서 말씀드린 압축은, 다음과 같은 의미입니다.

RNN cell에 입력되는 hidden state는 이전 step을 거치면서 Cell에서 추출된 정보(feature)의 벡터입니다.
이 정보를 표현하기 위한 용량이 바로 hidden size라고 할 수 있습니다. 

그런데, hidden size를 너무 작게 쓰게 되면 (예를 들어, hiddenSize = 1) 
실제로 feature를 표현하기 위한 용량보다 적은 용량으로 다음 cell에 정보를 전달해야 하고, 
여기서 정보의 “압축”이 발생하게 됩니다.

이러한 맥락에서 네트워크가 feature를 다음  cell로 잘 전달할 수 있도록, 충분한 hidden size를 설정해야 한다는 의미로 생각해주시면 좋을 것 같습니다.

따로 압축을 하는게 아니라 hidden size가 작아서 생기는 정보의 압축
```

# Implementation

---

Scaling 작업 이유

800정도의 주가 데이터와 1000000 정도의 거래량 데이터가 있다.

모델이 출력해야하는 데이터의 크기가 800 or 10000000이 된다는 것이다.

이런 데이터의 크기 또한 학습의 대상이 된다. 즉, 모델의 학습 부담이 생긴다.

그래서 이를 [0,1] 사이의 값으로 변환한다.

```python

```

# Seq2Seq

---

구조는 다음과 같다.

![Screenshot_20220130-142348_YouTube.jpg](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Screenshot_20220130-142348_YouTube.jpg)

이렇게 끝에서 상반된 단어를 내보내면 적절한 답변을 못하는 RNN

![Screenshot_20220130-141737_YouTube.jpg](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Screenshot_20220130-141737_YouTube.jpg)

그래서 Seq2Seq를 도입한다.

![SmartSelect_20220130-142058_YouTube.jpg](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/SmartSelect_20220130-142058_YouTube.jpg)

# 과정 및 구성요소

---

preprocess

input/output source가 몇 개의 단어로 구성되어있는지 찾아내는 과정

Encoder/Decoder

Encoder는 받은 메세지를 vector로 만들고, 그 vector를 hidden state로 decoder에서 사용한다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2023.png)

Decoder까지 거치고 나면 GRU를 한 번 거쳐서, Fully Connected Layer를 한 번 거쳐서 목표하는 output 단어 개수에 맞춰준다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2024.png)

Input size는 들어가는 input source의 단어개수이고,

Output size는 나오는 output source의 단어 개수이다. 

input/output source를 one hot encoding한 결과를 사용하는 것이다.

Embedding은 적은 차원으로 줄이는 매트릭스

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2025.png)

tensorize

sentence를 one hot vector로 변환

Start of Sentence = 0

문장의 시작을 나타낸다.

End of Sentence = 1

문장의 끝을 나타낸다.

Encoder의 구조

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2026.png)

Decoder의 구조.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2027.png)

Decoder에서 나온 결과를 다음 두가지 중 하나로 사용할 수 있다.

1. 다음 GRU의 입력으로 넣기
2. decoder input을 target data로 바꾸기 = teacher forcing : 빠른 수렴, 불안정한 학습

## Implementation

---

```python
import torch
import torch.nn as nn
import torch.optim as optim
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(0)
if device == 'cuda':
    torch.cuda.manual_seed(0)

raw = ["I feel hungry.	나는 배가 고프다.",
       "Pytorch is very easy.	파이토치는 매우 쉽다.",
       "Pytorch is a framework for deep learning.	파이토치는 딥러닝을 위한 프레임워크이다.",
       "Pytorch is very clear to use.	파이토치는 사용하기 매우 직관적이다."]

# fix token for "start of sentence" and "end of sentence"
SOS_token = 0
EOS_token = 1

class Vocab:
    def __init__(self):
        self.vocab2index = {"<SOS>": SOS_token, "<EOS>": EOS_token}
        self.index2vocab = {SOS_token: "<SOS>", EOS_token: "<EOS>"}
        self.vocab_count = {} # 특정 단어의 개수
        self.n_vocab = len(self.vocab2index) # vocab vector의 길이
    def add_vocab(self, sentence):
        for word in sentence.split(" "):
            if word not in self.vocab2index: # 기존에 없던 것 추가됐을 때
                self.vocab2index[word] = self.n_vocab
                self.vocab_count[word] = 1
                self.index2vocab[self.n_vocab] = word
                self.n_vocab += 1
            else:
                self.vocab_count[word] += 1

# filter out the long sentence from source and target data
def filter_pair(pair, source_max_length, target_max_length):
    # 논리값 출력
    # source와 target 모두 max_length이내이면 통과시킨다.
    return len(pair[0].split(" ")) < source_max_length and len(pair[1].split(" ")) < target_max_length

# read & process
def preprocess(corpus, source_max_length, target_max_length):
    pairs = []
    for line in corpus: # 한 문장씩 뽑아와서 빈 칸 없애고, 소문자로하고, 공백기준으로 원문과 번역으로 나눈다.
        pairs.append([s for s in line.strip().lower().split("\t")])
    print("read {} sentences pairs".format(len(pairs)))
    
    pairs = [pair for pair in pairs if filter_pair(pair, source_max_length, target_max_length)]
    print("trimmed to {} sentence pairs".format(len(pairs)))
    
    source_vocab = Vocab()
    target_vocab = Vocab()
    
    for pair in pairs:
        source_vocab.add_vocab(pair[0]) # 원문
        target_vocab.add_vocab(pair[1]) # 번역문
    print("source vocab size = ", source_vocab.n_vocab)
    print("target vocab size = ", target_vocab.n_vocab)
    return pairs, source_vocab, target_vocab

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    def forward(self, x, hidden):
        x = self.embedding(x).view(1,1,-1)
        x, hidden = self.gru(x, hidden)
        return x, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size) # softmax skipped
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, x, hidden):
        x = self.embedding(x).view(1,1,-1)
        x, hidden = self.gru(x, hidden)
        x = self.out(x[0])
        x = self.softmax(x)
        return x, hidden

def tensorize(vocab, sentence):
    indexes = [vocab.vocab2index[word] for word in sentence.split(" ")]
    indexes.append(vocab.vocab2index["<EOS>"])
    return torch.Tensor(indexes).long().to(device).view(-1,1)

# training
def train(pairs, source_vocab, target_vocab, encoder, decoder, n_iter, print_every=1000, learning_rate=0.01, loss_total=0):
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    
    training_batch = [random.choice(pairs) for _ in range(n_iter)] # batch 생성
    # vocab to tensor
    training_source = [tensorize(source_vocab, pair[0]) for pair in training_batch]
    training_target = [tensorize(target_vocab, pair[1]) for pair in training_batch]
    
    # criterion = nn.CrossEntropyLoss() # [1,16]이 들어가야한다. 하지만 입력이 [1]이 되므로 안됨
    criterion = nn.NLLLoss()
    
    for i in range(1, n_iter + 1):
        source_tensor = training_source[i - 1]
        target_tensor = training_target[i - 1]
        
        encoder_hidden = torch.zeros([1,1,encoder.hidden_size]).to(device) # 초기 hidden은 0
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        
        source_length = source_tensor.size(0)
        target_length = target_tensor.size(0)
        
        loss = 0
        for enc_input in range(source_length):
            _, encoder_hidden = encoder(source_tensor[enc_input], encoder_hidden) # 입력만 주구장창
            
        decoder_input = torch.Tensor([[SOS_token]]).long().to(device) # decoder의 시작은 SOS
        decoder_hidden = encoder_hidden # encoder hidden은 decoder hidden으로 전이된다.
        
        for di in range(target_length): # decoder 학습
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]
        
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        loss_iter = loss.item() / target_length
        loss_total += loss_iter
    
        if i % print_every == 0:
            loss_avg = loss_total / print_every
            loss_total = 0
            print("{} - {} loss = {:05.4f}".format(i, i / n_iter * 100, loss_avg))

# evaluate the result
def evaluate(pairs, source_vocab, target_vocab, encoder, decoder, target_max_length):
    for pair in pairs:
        print(">", pair[0])
        print("=", pair[0])
        source_tensor = tensorize(source_vocab, pair[0])
        source_length = source_tensor.size()[0]
        encoder_hidden = torch.zeros([1,1,encoder.hidden_size]).to(device) # encoder의 초기 hidden은 0
        
        for ei in range(source_length):
            _, encoder_hidden = encoder(source_tensor[ei], encoder_hidden)
        
        decoder_input = torch.Tensor([[SOS_token]], device=device).long()
        decoder_hidden = encoder_hidden
        decoded_words = []
        
        # topk : key/value에서 value를 기준으로 K번째로 높은 값을 구하는 것이다.
        # values, indexs = torch.topk(predict, k=k, dim=-1)
        # https://pytorch.org/docs/stable/generated/torch.topk.html#torch.topk
        for di in range(target_max_length):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            _, top_index = decoder_output.data.topk(1)
            if top_index.item() == EOS_token: # EOS인지 확인
                decoded_words.append("<EOS>")
                break
            else:
                decoded_words.append(target_vocab.index2vocab[top_index.item()])
            decoder_input = top_index.squeeze().detach()
            
        predict_words = decoded_words
        predict_sentence = " ".join(predict_words)
        print("<", predict_sentence)
        print("")

SOURCE_MAX_LENGTH = 10
TARGET_MAX_LENGTH = 12

load_pairs, load_source_vocab, load_target_vocab = preprocess(raw, SOURCE_MAX_LENGTH, TARGET_MAX_LENGTH)
print(random.choice(load_pairs)) # list에서 아무거나 하나 뽑아준다.

enc_hidden_size = 16
dec_hidden_size = 16
enc = Encoder(load_source_vocab.n_vocab, enc_hidden_size).to(device)
dec = Decoder(dec_hidden_size, load_target_vocab.n_vocab).to(device)

train(load_pairs, load_source_vocab, load_target_vocab, enc, dec, 5000, print_every=1000)
evaluate(load_pairs, load_source_vocab, load_target_vocab, enc, dec, TARGET_MAX_LENGTH)
```

# Packed Sequence and Padded Sequence

---

길이가 서로다른 Sequence를 다룰 때 사용하는 두가지 기술이 있다.

1. padding
2. packing

## 방법론

padding은 가장 긴 sequence를 기준으로 빈 공간에 0과 같은 pad token을 넣는다.

packing은 sequence를 가장 긴 것이 위로 오도록 정렬하고, 각 sequence의 길이를 따로 저장한다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2028.png)

각 객체에 대한 함수 구조는 다음 그림을 따른다.

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2029.png)

## Implementation

---

```python
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

# example data generating. batch size = 5, longest seq = 13
# Random word from random word generator
data = ['hello world',
        'midnight',
        'calculation',
        'path',
        'short circuit']

# Make dictionary
# 다음과 같이 해석한다 : char (for seq in (data for char in seq))
char_set = ['<pad>'] + list(set(char for seq in data for char in seq)) # Get all characters and include pad token 
char2idx = {char: idx for idx, char in enumerate(char_set)} # Constuct character to index dictionary
print('char_set:', char_set)
print('char_set length:', len(char_set))

# Convert character to index and make list of tensors
X = [torch.LongTensor([char2idx[char] for char in seq]) for seq in data]

# Check converted result
for sequence in X:
    print(sequence)
print(X)

# 위의 tensor는 모두 그 size가 제각각이다.
# Make length tensor (will be used later in 'pack_padded_sequence' function)
lengths = [len(seq) for seq in X]
print('lengths:', lengths)
```

하나의 batch로 만들어주기 위해서 일반적으로 제일 긴 sequence 길이에 맞춰 뒷부분에 padding을 추가(일반적으로 많이 쓰이는 Padding 방식)

PyTorch에서는 `PackedSequence`라는 것을 쓰면 padding 없이도 정확히 필요한 부분까지만 병렬 계산을 할 수도 있다.

따로 PaddedSequence라는 class는 존재하지 않고, PaddedSequence는 sequence중에서 가장 긴 sequence와 길이를 맞추어주기 위해 padding을 추가한 일반적인 ****Tensor****를 말한다.

이때, `pad_sequence`라는 PyTorch 기본 라이브러리 함수를 이용한다. input이 ****Tensor들의 list****이다.

list 안에 있는 각각의 Tensor들의 shape가 `(?, a, b, ...)` 라고 할때, (여기서 ?는 각각 다른 sequence length 입니다.)

input Tensor의 shape = `(sequence length, a, b, ...)`

output Tensor shape = `(longest sequence length in the batch, batch_size, a, b, ...)`

`batch_first=True`이면 `(batch_size, longest sequence length in the batch, a, b, ...)` shape를 가지는 Tensor가 리턴된다.

`padding_value=42`와 같이 파라미터를 지정해주면, padding하는 값도 정해진다.

```python
# Make a Tensor of shape (Batch x Maximum_Sequence_Length)
padded_sequence = pad_sequence(X, batch_first=True) # X is now padded sequence
print(padded_sequence)
print(padded_sequence.shape)

padded_sequence = pad_sequence(X) # X is now padded sequence
print(padded_sequence)
print(padded_sequence.shape)

# padding을 추가하지 않고 정확히 주어진 sequence 길이까지만 모델이 연산을 하게끔 만드는 PyTorch의 자료구조이다.
# 주어지는 input (list of Tensor)는 길이에 따른 내림차순으로 정렬이 되어있어야 한다.
# sorted 함수에 대해서는 다음 링크를 확인
# https://blockdmask.tistory.com/466

# Sort by descending lengths
sorted_idx = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True) # docu 참고
sorted_X = [X[idx] for idx in sorted_idx] # sorting한 결과

# Check converted result
for sequence in sorted_X:
    print(sequence)

packed_sequence = pack_sequence(sorted_X)
print(packed_sequence) # 여기서의 batch_size는 각 sequence를 처리하는 step 마다의 batch를 의미한다. 함수값으로 입력되는 sequence의 length가 아니라!
print(packed_sequence.data.shape)

# RNN에 투입하기
# character의 index를 one-hot character embedding한 값을 RNN의 input으로 넣어준다.
# one-hot embedding using PaddedSequence
eye = torch.eye(len(char_set)) # Identity matrix of shape (len(char_set), len(char_set))
embedded_tensor = eye[padded_sequence]
print(embedded_tensor.shape) # shape: (Batch_size, max_sequence_length, number_of_input_tokens)

# one-hot embedding using PackedSequence
embedded_packed_seq = pack_sequence([eye[X[idx]] for idx in sorted_idx])
print(embedded_packed_seq.data.shape)

# declare RNN
rnn = torch.nn.RNN(input_size=len(char_set), hidden_size=30, batch_first=True)

rnn_output, hidden = rnn(embedded_tensor)
print(rnn_output.shape) # shape: (batch_size, max_seq_length, hidden_size)
print(hidden.shape)     # shape: (num_layers * num_directions, batch_size, hidden_size)

rnn_output, hidden = rnn(embedded_packed_seq)
print(rnn_output.data.shape)
print(hidden.data.shape)

#`pad_packed_sequence` -> (Tensor, list_of_lengths)

unpacked_sequence, seq_lengths = pad_packed_sequence(embedded_packed_seq, batch_first=True)
print(unpacked_sequence.shape)
print(seq_lengths)

# `pack_padded_sequence` : sequence 길이반드시 input, input sequence가 내림차순 정렬 되어있어야함

# sorting
embedded_padded_sequence = eye[pad_sequence(sorted_X, batch_first=True)]
print(embedded_padded_sequence.shape)

# pack_padded_sequence
sorted_lengths = sorted(lengths, reverse=True)
new_packed_sequence = pack_padded_sequence(embedded_padded_sequence, sorted_lengths, batch_first=True)
print(sorted_lengths)
print(new_packed_sequence.data.shape)
print(new_packed_sequence.batch_sizes) # 여기서의 batch_size는 각 sequence를 처리하는 step 마다의 batch를 의미한다. 함수값으로 입력되는 sequence의 length가 아니라!
```

![Untitled](Lecture%2011%20RNN%20025f110894834a468084c933dffee59d/Untitled%2030.png)

packed Sequence에서 말하는 batch size는 위 그림과 같이, 하나의 Sequence에 따른 batch를 의미하는 것이다.