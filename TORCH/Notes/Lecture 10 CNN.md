# Lecture 10 : CNN

# Convolution

이미지 위에서 stride 값 만큼 filter(kernel)을 이동하면서 겹쳐지는 부분의 각 원소의 값을 곱해서 모두 더한 값을 출력으로 하는 연산

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%201.png)

## Convolution Layer의 다른 주요 사항들

stride : kernel이 이동하는 킉

padding : 1이면 한 칸 씩 띠를 둘러주고, 2이면 두 칸 씩 띠를 둘러주고.

zero padding : 띠를 둘러주는 padding의 내용물이 0이라는 의미이다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%202.png)

입력채널 1, 출력채널 1, 커널 크기 3*3이면 다음과 같이 쓴다.

```python
nn.Conv2d(1,1,3) # 입력채널, 출력채널, 커널크기
```

위 코드를 보면 channel의 수는 convolution 함수에 의해서 강제로 결정된다. 물론 이는 채널의 수가 convolution layer가 적용하는 필터의 개수에 의해 결정되기 때문이다.

반면, 이미지의 height와 width는 다음 계산식에 의해 결정된다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%203.png)

## Perceptron에서의 Convolution Layer 적용 방법

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%204.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%205.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%206.png)

이렇게 가중치처럼 적용되고, bias까지 고려하면 다음과 같이 한 부분이 계산된다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%207.png)

# Max/AveragePooling

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%208.png)

범위 중에서 최대값/평균값을 계산해서 output을 내보낸다.

선언과 사용은 다음과 같이 한다.

```python
maxpool = nn.MaxPool2D(kernel_size, stride, padding, ...)
maxpool((batch, channel, height, width))
```

## 참고)원래는 Convolution보다는 Cross-Correlation이라고 부르는 것이 맞다는 전설

Convoltuion은 함수 f가 필터 g에 대해서 얼마나 겹치는 지를 확인하는 함수이다. 그런데, 기본적으로 원래의 Convolution공식으로는 filter g를 뒤집어서 계산해야 한다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%209.png)

하지만 Cross-correlation 연산은 기본적으로는 filter를 뒤집지 않는다. 즉 앞서 했던 연산과 동일하다는 의미이다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2010.png)

# Building CNN model with MNIST

코드를 짤 때 가장 큰 틀

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2011.png)

이런 구조를 만들어 보자.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2012.png)

```python

```

다른 구조도 만들어보자.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2013.png)

```python

```

# Visdom

# Imagefolder

# Advanced CNN Models

channel of output은 filter의 개수에 의해 결정된다.

**즉, CNN에서는 Channel 개수 = Filter 개수**

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2014.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2015.png)

$1 \times 1$ convolution을 이용해서 parameter의 수와 연산 수를 줄일 수 있다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2016.png)

다음의 Net을 다음과 같이 구현할 수 있다. 마지막으로는 list처럼 concatenate한다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2017.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2018.png)

 최종적으로는 다음과 같이 구현된다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2019.png)

하지만 더 많은 Layer를 쌓는다고 성능이 향상되지는 않는다. 한계점이 좋지

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2020.png)

 

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2021.png)

그래서 RESNET과 같은 방법을 이용하는 network를 도입하곤 한다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2022.png)

특징

모두 $3 \times 3$ conv layer

spatial size / 2 → number of filters x 2

no hidden Fully Connected Layer

no Dropout

단 하나의 limitation이 있는데, 아래 그림의 단위에서 input size와 output size의 크기가 같아야한다는 것이다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2023.png)

오른쪽의 layer를 흔히 bottleneck layer라고 하는데, operation의 수를 줄이는데 효과적이다.

참고로 이렇게 많은 실험 결과 가장 잘 나온 결과가 위의 limitation이라고 한다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2024.png)

## LeNet (참고)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2025.png)

## AlexNet(참고)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2026.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2027.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2028.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2029.png)

## GoogLeNet(참고)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2030.png)

## VGG

특징 : conv는 모두 $3 \times 3$, stride 1, padding 1

```python

```

Q : $3 \times 224 \times 224$ image를 기준으로 만들어졌는데, 다른 image size에 대해서는 어떻게 적용해야하나?

A : Fully Connected Layer의 i/o feature를 조절해줘야 한다. 혹은 Maxpooling의 개수를 조절해준다.

참고로 $units = Channels \times Height \times Width$이다.

CIFAR-10에 적용해본다.

```python

```

# RESNET

Resnet의 경우 매우 큰 depth를 가지고 있고, 이를 jumping을 이용해서 해결한다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2031.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2032.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2033.png)

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2034.png)

특징

모두 $3 \times 3$ conv layer

spatial size / 2 → number of filters x 2

no hidden Fully Connected Layer

no Dropout

단 하나의 limitation이 있는데, 아래 그림의 단위에서 input size와 output size의 크기가 같아야한다는 것이다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2023.png)

오른쪽의 layer를 흔히 bottleneck layer라고 하는데, operation의 수를 줄이는데 효과적이다.

참고로 이렇게 많은 실험 결과 가장 잘 나온 결과가 위의 limitation이라고 한다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2024.png)

주요 구현은 다음과 같다.

주석으로 입력값의 shape이 어떻게 변해가는지 표시해주자

Basic Block

```python

```

Bottleneck

```python

```

downsample은 stride가 1보다 큰 경우, out이 identity보다 커져서 += 연산이 안되는 경우를 해결하기 위해 주어진 것이다.

RESNET

```python

```

self.inplanes : 

zero_init_residual : 

RESNET에서의 downsample은 channel을 맞추기 위해 사용한다.

# RESNET to CIFAR10

이미지가 작으므로 중간의 maxpool2d는 제거한다.

![Untitled](Lecture%2010%20CNN%20f7637e2917c64674839409e172f3e03d/Untitled%2035.png)