'''
MobileNet v1
- standard conv, depthwise conv, pointwise conv 총 3가지 convolution layer 사용
- 총 28개의 레이어로 구성되어 있음
    - 첫번째 conv layer : 일반 convolutional layer를 사용하며 모든 컨볼루션 레이어 뒤에 Batch Normalization과 ReLu가 이어짐

- Depthwise Conv layer (공간 방향의 convolution)
: 기존의 Conv2D가 각 채널만의 공간적 특징을 추출하는 것이 불가능하기 때문에 고안해낸 방법
   Depthwise Conv의 역할은 각 채널마다 Spatial Feature 를 추출하는 것
   즉, 각 채널마다 독립적으로 컨벌루션 곱을 실행함 -> 기존 Conv2D에 비해 파라미터 감소, 연산속도 향상

Height, Width, Channels의 Convolutional Output을 채널 단위로 분리하여, 
각각의 Conv Filter를 적용하여 Output을 만들고, 
그 결과를 다시 합치면 Conv Filter가 훨씬 적은 파라미터를 갖고 동일한 크기의 Output을 낼 수 있다.
이 때, 동일한 채널 내에서만 컨벌루션 곱을 진행하게 되는데 (채널 사이는 독립적), 이는 입력채널의 개수 == 출력채널의 개수의 관계가 성립되어야함

- Pointwise Conv layer (채널 방향의 convolution) (1x1 convolution)
: 1D Convolution 으로 여러 개의 채널을 하나의 새로운 채널로 합치는 역할
  흔히 1x1 Conv라고 불리는 필터. 주로 기존 텐서의 결과를 논리적으로 다시 셔플해서 뽑아내는 것을 목적으로 함

PC Filter의 크기는 1x1으로 고정되어 있음 -> PC는 채널들에 대한 연산만 수행함 따라서 출력의 크기는 변하지 않으며 채널의 수를 조절할 수 있는 역할을 함
차원 감소를 위해 사용하는 레이어 (Dimensional Reduction) -> 연산량 감소

DC : 각 input channel에 single convolution filter를 적용하여 네트워크를 경량화함
PC : DC의 결과를 PC를 통하여 다음 layer의 input으로 합쳐주는 역할으르 함
- 활성화 함수는 ReLU6를 사용하여 block을 구성

* Depthwise Separable Convolution (채널방향 + 공간방향)
'''


from keras import models, layers
from keras import Input
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, initializers, regularizers, metrics
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import BatchNormalization, DepthwiseConv2D, Conv2D, Activation, Dense
from keras.layers import GlobalAveragePooling2D, ZeroPadding2D, Reshape, Dropout

import os
import matplotlib.pyplot as plt
import numpy as np
import math

def depthwise_bn_relu(x, s, padd):
    # 3*3 window, stride 및 padding 지정, conv layer에서 bias 사용x
    x = DepthwiseConv2D((3, 3)), stride = (s, s), padding = padd, use_bias = False)(x)
    # 배치 정규화 : 각 미니 배치별로 학습 전에 정규화 처리
    x = BatchNormalization()(x)
    # 활성화 함수 relu
    x = Activation('relu')(x)
    return x

def pointwise_bn_relu(x, number_of_filter):
    # 필터 수, window size 1x1, stride 1,1 padding 사용, bias x
    x = Conv2D(number_of_filter, (1, 1), stride = (1, 1), padding = 'same', use_bias = False)(x)
    # 배치 정규화
    x = BatchNormalization()(x)
    # 활성화 relu
    x = Activation('relu')(x)
    return x


# 입력 텐서 = H:224, W:224, C:3
input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')

# 패딩 - 원하는 위치에 패딩 값 설정
# padding = ((top_pad, bottom_pad), (left_pad, right_pad))
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(input_tensor)

# 1. Standard Conv / stride 2 : Conv2D(필터 32, window = 3x3, stride(2,2), padding X, bias X) - 배치정규화 - relu
# Filter Shape = 3x3x3x32
# Input Size = 224x224x3
x = Conv2D(32, (3, 3), stride=(2, 2), padding='valid', use_bias=False)(x)
x = BatchNormalization()(x)
x = Activation('relu')

# 2. Depthwise Conv / stride 1 :  stride(1,1), padding same - 배치정규화 - relu
# Filter Shape = 3x3x32 (하나의 Feature map(input channel)에 대해 하나의 필터를 사용)
# Input Size = 112x112x32 (->stride2,filter32라서)
x = depthwise_bn_relu(x, 1, 'same')

# 3. Pointwise Conv / stride 1 : Conv2D(filter = 64, window = 1x1, stride = 1x1, padding same, bias X)
# Filter Size = 1x1x32x64
# Input Size = 112x112x32
x = pointwise_bn_relu(x, 64)
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)  # 패딩

# 4. Depthwise Conv / window 3 / stride 2 / padding X
# Filter Size = 3x3x64
# Input Size = 112x112x64
x = depthwise_bn_relu(x, 2, 'valid')

# 5. Pointwise Conv / filter 128 / window 1 / stride 1 / padding same
# Filter Size = 1x1x64x128
# Input Size = 56x56x64
x = pointwise_bn_relu(x, 128)

# 6. Depthwise Conv / window 3 / stride 1 / padding same
# Filter Size = 3x3x128
# Input Size = 56x56x128
x = depthwise_bn_relu(x, 1, 'same')

# 7. Pointwise Conv / filter 128 / window 1 / stride 1 / padding same
# Filter Size = 1x1x128x128
# Input Size = 56x56x128
x = pointwise_bn_relu(x, 128)
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)

# 8. Depthwise Conv / window 3 / stride 2 / padding X
# Filter Size = 3x3x128
# Input Size = 56x56x128
x = depthwise_bn_relu(x, 2, 'valid')

# 9. Pointwise Conv / filter 256 / window 1 / stride 1 / padding same
# Filter Size = 1x1x128x256
# Input Size = 28x28x128
x = pointwise_bn_relu(x, 256)

# 10. Depthwise Conv / window 3 / stride 1 / padding same
# Filter Size = 3x3x256
# Input Size = 28x28x256
x = depthwise_bn_relu(x, 1, 'same')

# 11. Pointwise Conv / filter 256 / window 1 / stride 1 / padding same
# Filter Size = 1x1x256x256
# Input Size = 28x28x256
x = pointwise_bn_relu(x, (256))
x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)

# 12. Depthwise Conv / window 3 / stride 2 / padding X
# Filter Size = 3x3x256
# Input Size = 28x28x256
x = depthwise_bn_relu(x, 2, 'valid')

# 13. Pointwise Conv / filter 512 / window 1 / stride 1 / padding same
# Filter Size = 1x1x256x512
# Input Size = 14x14x256
x = pointwise_bn_relu(x, 512)

# 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
# Depthwise Conv / window 3 / stride 1 / padding same
# Pointwise Conv / filer 512 / window 1 / stride 1 / padding same
# 14. Depth / Filter Size = 3x3x512 / Input Size = 14x14x512
# 15. Point / Filter Size = 1x1x512x512 / Input Size = 14x14x512
# 16, 17, 18, 29, 20, 21
# 22. Depth / Filter Size = 3x3x512 / Input Size = 14x14x512
# 23. Point / Filter Size = 1x1x512x512 / Input Size = 14x14x512
for _ in range(5):
    x = depthwise_bn_relu(x, 1, 'same')
    x = pointwise_bn_relu(x, 512)

x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)

# 24. Depthwise Conv / window 3 / stride 2 / padding X
# Filter Size = 3x3x512
# Input Size = 14x14x512
x = depthwise_bn_relu(x, 2, 'valid')

# 25. Pointwise Conv / filter 1024 / window 1 / stride 1 / padding same
# Filter Size = 1x1x512x1024
# Input Size = 7x7x512
x = pointwise_bn_relu(x, 1024)

# 26. Depthwise Conv / window 3 / stride 2 / padding same
# Filter Size = 3x3x1024
# Input Size = 7x7x1024
x = depthwise_bn_relu(x, 2, 'same')

# 27. Pointwies Conv / filter 1024 / window 1 / stride 1 / padding same
# Filter Size = 1x1x1024x1024
# Input Size =  7x7x1024     #padding!
x = pointwise_bn_relu(x, 1024)

# 28. Avg Pool : 해당 영역의 평균값을 계산 / output_size 1 / 1x1 으로 축소
# Height x Width x Depth -> 1x1xDepth
# Filter Size = Pool 7x7
# Input Size = 7x7x1024
x = GlobalAveragePooling2D()(x)

x = Reshape((1, 1, 1024))(x)  # Reshape

# 29. Dropout
x = Dropout(0.001)(x)

# 30. Standard Conv / filter 1000 / window = 1x1 / stride(1,1) / padding same / softmax
# Fully Convolution - Reshape
# Filter Size = 1024x1000
# Input Size = 1x1x1024
x = Conv2D(1000, (1, 1), strides=(1, 1), padding='same')(x)

# Classifier : Input Size = 1x1x1000
x = Activation('softmax')(x)

# Output
output_tensor = Reshape((1000,))(x)

my_mobile = Model(input_tensor, output_tensor)
my_mobile.summary()


















