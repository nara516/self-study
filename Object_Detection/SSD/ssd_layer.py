import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Sequential

"""
create_vgg16_layers() 
- vgg16_conv4 : Pre Trainning된 VGG16 Model에서 Conv5_3 layer 까지 지정하는 곳
- vgg16_conv7 : Pre Trainning된 VGG16 Model에서 Fclayer6(Dense1), Fclayer7(Dense2)를 통과한 layer를 지정하는 곳
"""
def create_vgg16_layers():
    vgg16_conv4 = [
        layers.Conv2D(64, 3, padding='same', activation='relu'),    # conv 1-1
        layers.Conv2D(64, 3, padding='same', activation='relu'),    # conv 1-2
        layers.MaxPool2D(2, 2, padding='same'),                     # pooling 1

        layers.Conv2D(128, 3, padding='same', activation='relu'),   # conv 2-1
        layers.Conv2D(128, 3, padding='same', activation='relu'),   # conv 2-2
        layers.MaxPool2D(2, 2, padding='same'),                     # pooling 2

        layers.Conv2D(256, 3, padding='same', activation='relu'),   # conv 3-1
        layers.Conv2D(256, 3, padding='same', activation='relu'),   # conv 3-2
        layers.Conv2D(256, 3, padding='same', activation='relu'),   # conv 3-3
        layers.MaxPool2D(2, 2, padding='same'),                     # pooling 3

        layers.Conv2D(512, 3, padding='same', activation='relu'),   # conv 4-1
        layers.Conv2D(512, 3, padding='same', activation='relu'),   # conv 4-2
        layers.Conv2D(512, 3, padding='same', activation='relu'),   # conv 4-3
        layers.MaxPool2D(2, 2, padding='same'),                     # pooling 4

        layers.Conv2D(512, 3, padding='same', activation='relu'),   # conv 5-1
        layers.Conv2D(512, 3, padding='same', activation='relu'),   # conv 5-2
        layers.Conv2D(512, 3, padding='same', activation='relu'),   # conv 5-3
    ]

    x = layers.Input(shape=[None, None, 3])
    out = x

    for layer in vgg16_conv4:
        out = layer(out)

    # PreTrainning된 VGG16 Model에서 Conv5_3 Layer까지 지정
    vgg16_conv4 = tf.keras.Model(x, out)

    # PreTrainning된 VGG16 Model에서 FcLayer6(Dense1), FcLayer7(Dense2)를 통과한 Layer를 지정
    vgg16_conv7 = [
        # Difference from original VGG16:
        # 5th maxpool layer has kernel size = 3 and stride = 1
        layers.MaxPool2D(3, 1, padding='same'),             # pooling 5
        # atrous conv2d for 6th block
        layers.Conv2D(1024, 3, padding='same',
                      dilation_rate=6, activation='relu'),
        layers.Conv2D(1024, 1, padding='same', activation='relu'),
    ]

    x = layers.Input(shape=[None, None, 512])
    out = x
    for layer in vgg16_conv7:
        out = layer(out)

    vgg16_conv7 = tf.keras.Model(x, out)
    return vgg16_conv4, vgg16_conv7


"""
create_extra_layers()
- 다양한 Scale의 FeatureMap에서 Object Detection을 하기위한 Extra Feature layers 선언
"""
def create_extra_layers():
    """ Create extra layers
        8th to 11th blocks
    """
    extra_layers = [
        # 8th block output shape: B, 512, 10, 10
        Sequential([
            layers.Conv2D(256, 1, activation='relu'),
            layers.Conv2D(512, 3, strides=2, padding='same',
                          activation='relu'),
        ]),

        # 9th block output shape: B, 256, 5, 5
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, strides=2, padding='same',
                          activation='relu'),
        ]),

        # 10th block output shape: B, 256, 3, 3
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ]),

        # 11th block output shape: B, 256, 1, 1
        Sequential([
            layers.Conv2D(128, 1, activation='relu'),
            layers.Conv2D(256, 3, activation='relu'),
        ])
    ]

    return extra_layers


"""
create_conf_head_layers(num_classes)
- Object의 Class를 확인하기 위한 layer
- 각각은 Default Box의 개수 * Class 로서 Dimension을 이룸
- 논문과 같이 Convolution의 Filter Size는 3x3
"""
def create_conf_head_layers(num_classes):
    """ Create layers for classification
    """
    conf_head_layers = [
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 4th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 7th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 8th block
        layers.Conv2D(6 * num_classes, kernel_size=3,
                      padding='same'),  # for 9th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same'),  # for 10th block
        layers.Conv2D(4 * num_classes, kernel_size=3,
                      padding='same')  # for 11th block
    ]

    return conf_head_layers

# Object의 Localization을 확인하기 위한 Layer이다.
# 각각은 Default Box의 개수 * (cx,cy,w,h) Dimension을 이루고
# 논문과 같이 Convolution의 Filter의 Size는 3x3이다.

"""
create_loc_head_layers()
- Object의 Localization을 확인하기 위한 layer
- 각각은 Default Box의 개수 * (cx, cy, w, h) Dimension
- Convolution의 filter size는 3x3
"""
def create_loc_head_layers():
    """ Create layers for regression
    """
    loc_head_layers = [
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(6 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same'),
        layers.Conv2D(4 * 4, kernel_size=3, padding='same')
    ]

    return loc_head_layers