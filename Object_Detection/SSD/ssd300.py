from __future__ import division

import keras.backend as K
import numpy as np
from keras.layers import Input, Lambda, Activation, Conv2D, MaxPooling2D, ZeroPadding2D, Reshape, Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization



"""
model = ssd_300(image_size=(300, 300, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)             
"""
def ssd_300(image_size,                 # (h,w,c) 형식의 입력크기
            n_classes,                  # int형, positive class의 수
            mode='training',            # training, inference, inference_fast
            l2_regularization=0.0005,   # L2 regularization 비율. 모든 conv layer에 적용됨 (비활성화 시 0)
            min_scale=None,             # anchor boxes의 크기의 최소 스케일
            max_scale=None,             #                      최대 스케일
            scales=None,
            aspect_ratios_global=None,  # 앵커 박스 종횡비
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],  # 각 prediction layer의 aspect ratio
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],         # 컬러 이미지에 대한 채널 별 평균 정규화
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,                 # 최소 분류 신뢰도 non-maximum suppression stage for the respective class
            iou_threshold=0.45,                     # IoU
            top_k=200,                              # non maximum suppression 이후 각 배치에 대해 유지될 highest scoring predictions
            nms_max_output_size=400,                # NMS 이후에 남겨질 maximal number of predictions
            return_predictor_sizes=False):
    n_predictor_layers = 6  # predictor conv layers 수. SSD300 의 경우 6
    n_classes += 1  # Account for the background class.
    l2_reg = l2_regularization  # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError(
            "`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError(
                "It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(
                    n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers + 1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(
                n_predictor_layers + 1, len(scales)))
    else:  # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers + 1)

    if len(variances) != 4:
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # anchor box 매개 변수 계산
    ############################################################################

    # 각 predictor layer의 aspect ratio 설정. anchor box layer 에만 필요
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # 각 predictor layer 에 대해 셀 당 예측할 box수 계산
    # predictor layer에 필요한 채널 수를 알기 위해 정보가 필요
   if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1)  # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else:  # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack(
                [tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[..., swap_channels[0]], tensor[..., swap_channels[1]], tensor[..., swap_channels[2]],
                            tensor[..., swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))

    # The following identity layer is only needed so that the subsequent lambda layers can be optional.
    x1 = Lambda(identity_layer, output_shape=(img_height, img_width, img_channels), name='identity_layer')(x)
    if not (subtract_mean is None):
        x1 = Lambda(input_mean_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_mean_normalization')(x1)
    if not (divide_by_stddev is None):
        x1 = Lambda(input_stddev_normalization, output_shape=(img_height, img_width, img_channels),
                    name='input_stddev_normalization')(x1)
    if swap_channels:
        x1 = Lambda(input_channel_swap, output_shape=(img_height, img_width, img_channels), name='input_channel_swap')(
            x1)


    # conv1_1. filter = 3x3x3x64 / input = 300x300x3
    conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
    # conv1_2. filter = 3x3x64x64 / input = 300x300x64
    conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
    # pool1. input = 300x300x64
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)
    ## output = 150x150x64


    # conv2_1. filter = 3x3x64x128 / input = 150x150x64
    conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
    # conv2_2. filter = 3x3x128x128 / input = 150x150x128
    conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
    # pool2. input = 150x150x128
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)
    ## output = 75x75x128


    # conv3_1. filter = 3x3x128x256 / input = 75x75x128
    conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
    # conv3_2. filter = 3x3x256x256 / input = 75x75x128
    conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
    # conv3_3. filter = 3x3x256x256 / input = 75x75x256
    conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
    # pool3. input = 75x75x256
    pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)
    ## output = 38x38x256


    # conv4_1. filter = 3x3x256x512 / input = 38x38x256
    conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
    # conv4_2. filter = 3x3x512x512 / input = 38x38x512
    conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
    # conv4_3. filter = 3x3x512x512 / input = 38x38x512
    conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
    # pool4. input = 38x38x512
    pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)
    ## output = 19x19x512


    # conv5_1. filter = 3x3x512x512 / input = 19x19x512
    conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
    # conv5_2. filter = 3x3x512x512 / input = 19x19x512
    conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
    # conv5_3. filter = 3x3x512x512 / input = 19x19x512
    conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
    # pool5. input = 19x19x512
    pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)
    ## output = 19x19x512


    # fc6 (conv6). filter = 3x3x512x1024 /  input = 19x19x512
    fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc6')(pool5)
    ## output = 19x19x1024

    # fc7 (conv7). filter = 1x1x1024x1024 / input = 19x19x1024
    fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=l2(l2_reg), name='fc7')(fc6)
    ## output = 19x19x1024


    ### Extra layer
    # conv6_1 (conv8_1). filter = 1x1x1024x256 / input = 19x19x1024
    conv6_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
    conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
    # conv6_2 (conv8_2). filter = 3x3x256x512 /  input = 19x19x256
    conv6_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv6_2')(conv6_1)
    ## output = 10x10x512

    # conv7_1 (conv9_1). filter = 1x1x512x128 / input = 10x10x512
    conv7_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_1')(conv6_2)
    conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
    # conv7_2 (conv9_2). filter = 3x3x128x256 / input = 10x10x512
    conv7_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv7_2')(conv7_1)
    ## output = 5x5x256


    # conv8_1 (conv10_1). filter = 1x1x256x128 / input = 5x5x256
    conv8_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_1')(conv7_2)
    # conv8_2 (conv10_2). filter = 3x3x128x256 / input = 5x5x128
    conv8_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv8_2')(conv8_1)
    ## output = 3x3x256
    # (5-3)/1 + 1 = 3


    # conv9_1 (conv11_1). filter = 1x1x256x128 / input = 3x3x256  *******************************************************
    conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_1')(conv8_2)
    # conv9_2 (conv11_2). filter = 3x3x128x256 /  input = 3x3x256
    conv9_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal',
                     kernel_regularizer=l2(l2_reg), name='conv9_2')(conv9_1)
    ## output = 1x1x256


    # Feed conv4_3 into the L2 normalization layer.
    conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

    ### base notwork 위에 convolutional predictor 구축

    #### class 예측
    # 각 상자에 대해 'n_classes' confidence value를 예측하므로 confidence predictor의 depth는 n_boxes * n_classes 를 가짐
    # Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`

    # filter = 3x3x512x(4xclasses) / input = 38x38x512 / output = 38x38x(4xclasses)
    conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                                    kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
    # filter = 3x3x1024x(6xclasses) / input = 19x19x1024 / output = 19x19x(6xclasses)
    fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
    # filter = 3x3x512x(6xclasses) / input = 10x10x512 / output = 10x10x(6xclasses)
    conv6_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_conf')(conv6_2)
    # filter = 3x3x256x(6xclasses) / input = 5x5x256 / output = 5x5x(6xclasses)
    conv7_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_conf')(conv7_2)
    # filter = 3x3x256x(4xclasses) / input = 3x3x256 / output = 3x3x(4xclasses)
    conv8_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
    # filter = 3x3x256x(4xclasses) / input =1x1x256 / output = 1x1x(4xclasses)
    conv9_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal',
                               kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)

    #### 좌표 예측
    # 각 상자에 대해 4개의 좌표를 예측하므로 localization predictor의 depth는 n_boxes * 4를 가짐
    # Output shape of the localization layers: `(batch, height, width, n_boxes * 4)

    # filter =  3x3x512x(4x4) / input = 38x38x512 / output = 38x38x(4x4)
    conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                                   kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
    # filter = 3x3x1024x(6x4) / input = 19x19x1024 / output = 19x19x(6x4)
    fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                          kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
    # filter = 3x3x512x(6x4) / input = 10x10x512 / output = 10x10x(6x4)
    conv6_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv6_2_mbox_loc')(conv6_2)
    # filter = 3x3x256x(6x4) / input = 5x5x256 / output = 5x5x(6x4)
    conv7_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv7_2_mbox_loc')(conv7_2)
    # filter = 3x3x256x(4x4) / input = 3x3x256 / output = 3x3x(4x4)
    conv8_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
    # filter = 3x3x256x(4x4) / input = 1x1x256 / output = 1x1x(4x4)
    conv9_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal',
                              kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)


    ### anchor box 생성 (called "priors" in the original Caffe/C++ implementation, so I'll keep their layer names)

    # Output shape of anchors: `(batch, height, width, n_boxes, 8)`
    conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1],
                                             aspect_ratios=aspect_ratios[0],
                                             two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0],
                                             this_offsets=offsets[0], clip_boxes=clip_boxes,
                                             variances=variances, coords=coords, normalize_coords=normalize_coords,
                                             name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
    fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2],
                                    aspect_ratios=aspect_ratios[1],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                                    clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords,
                                    name='fc7_mbox_priorbox')(fc7_mbox_loc)
    conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3],
                                        aspect_ratios=aspect_ratios[2],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2],
                                        this_offsets=offsets[2], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
    conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4],
                                        aspect_ratios=aspect_ratios[3],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3],
                                        this_offsets=offsets[3], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
    conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5],
                                        aspect_ratios=aspect_ratios[4],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4],
                                        this_offsets=offsets[4], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv8_2_mb_priorbox')(conv8_2_mbox_loc)
    conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6],
                                        aspect_ratios=aspect_ratios[5],
                                        two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5],
                                        this_offsets=offsets[5], clip_boxes=clip_boxes,
                                        variances=variances, coords=coords, normalize_coords=normalize_coords,
                                        name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)

    ### Reshape

    # class predictions 3차원 텐서로 reshape -> `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    conv4_3_norm_mbox_conf_reshape = Reshape((-1, n_classes), name='conv4_3_norm_mbox_conf_reshape')(
        conv4_3_norm_mbox_conf)
    fc7_mbox_conf_reshape = Reshape((-1, n_classes), name='fc7_mbox_conf_reshape')(fc7_mbox_conf)
    conv6_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv6_2_mbox_conf_reshape')(conv6_2_mbox_conf)
    conv7_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv7_2_mbox_conf_reshape')(conv7_2_mbox_conf)
    conv8_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv8_2_mbox_conf_reshape')(conv8_2_mbox_conf)
    conv9_2_mbox_conf_reshape = Reshape((-1, n_classes), name='conv9_2_mbox_conf_reshape')(conv9_2_mbox_conf)
    # box predictions 3차원 텐서로 reshape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss (4개의 상자 좌표)
    conv4_3_norm_mbox_loc_reshape = Reshape((-1, 4), name='conv4_3_norm_mbox_loc_reshape')(conv4_3_norm_mbox_loc)
    fc7_mbox_loc_reshape = Reshape((-1, 4), name='fc7_mbox_loc_reshape')(fc7_mbox_loc)
    conv6_2_mbox_loc_reshape = Reshape((-1, 4), name='conv6_2_mbox_loc_reshape')(conv6_2_mbox_loc)
    conv7_2_mbox_loc_reshape = Reshape((-1, 4), name='conv7_2_mbox_loc_reshape')(conv7_2_mbox_loc)
    conv8_2_mbox_loc_reshape = Reshape((-1, 4), name='conv8_2_mbox_loc_reshape')(conv8_2_mbox_loc)
    conv9_2_mbox_loc_reshape = Reshape((-1, 4), name='conv9_2_mbox_loc_reshape')(conv9_2_mbox_loc)
    # anchor box tensors 3차원으로 reshape `(batch, height * width * n_boxes, 8)`
    conv4_3_norm_mbox_priorbox_reshape = Reshape((-1, 8), name='conv4_3_norm_mbox_priorbox_reshape')(
        conv4_3_norm_mbox_priorbox)
    fc7_mbox_priorbox_reshape = Reshape((-1, 8), name='fc7_mbox_priorbox_reshape')(fc7_mbox_priorbox)
    conv6_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv6_2_mbox_priorbox_reshape')(conv6_2_mbox_priorbox)
    conv7_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv7_2_mbox_priorbox_reshape')(conv7_2_mbox_priorbox)
    conv8_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv8_2_mbox_priorbox_reshape')(conv8_2_mbox_priorbox)
    conv9_2_mbox_priorbox_reshape = Reshape((-1, 8), name='conv9_2_mbox_priorbox_reshape')(conv9_2_mbox_priorbox)

    ### Concatenate the predictions from the different layers

    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1, the number of boxes per layer
    # Output shape of `mbox_conf`: (batch, n_boxes_total, n_classes)
    mbox_conf = Concatenate(axis=1, name='mbox_conf')([conv4_3_norm_mbox_conf_reshape,
                                                       fc7_mbox_conf_reshape,
                                                       conv6_2_mbox_conf_reshape,
                                                       conv7_2_mbox_conf_reshape,
                                                       conv8_2_mbox_conf_reshape,
                                                       conv9_2_mbox_conf_reshape])

    # Output shape of `mbox_loc`: (batch, n_boxes_total, 4)
    mbox_loc = Concatenate(axis=1, name='mbox_loc')([conv4_3_norm_mbox_loc_reshape,
                                                     fc7_mbox_loc_reshape,
                                                     conv6_2_mbox_loc_reshape,
                                                     conv7_2_mbox_loc_reshape,
                                                     conv8_2_mbox_loc_reshape,
                                                     conv9_2_mbox_loc_reshape])

    # Output shape of `mbox_priorbox`: (batch, n_boxes_total, 8)
    mbox_priorbox = Concatenate(axis=1, name='mbox_priorbox')([conv4_3_norm_mbox_priorbox_reshape,
                                                               fc7_mbox_priorbox_reshape,
                                                               conv6_2_mbox_priorbox_reshape,
                                                               conv7_2_mbox_priorbox_reshape,
                                                               conv8_2_mbox_priorbox_reshape,
                                                               conv9_2_mbox_priorbox_reshape])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Activation('softmax', name='mbox_conf_softmax')(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if mode == 'training':
        model = Model(inputs=x, outputs=predictions)
    elif mode == 'inference':
        decoded_predictions = DecodeDetections(confidence_thresh=confidence_thresh,
                                               iou_threshold=iou_threshold,
                                               top_k=top_k,
                                               nms_max_output_size=nms_max_output_size,
                                               coords=coords,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width,
                                               name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    elif mode == 'inference_fast':
        decoded_predictions = DecodeDetectionsFast(confidence_thresh=confidence_thresh,
                                                   iou_threshold=iou_threshold,
                                                   top_k=top_k,
                                                   nms_max_output_size=nms_max_output_size,
                                                   coords=coords,
                                                   normalize_coords=normalize_coords,
                                                   img_height=img_height,
                                                   img_width=img_width,
                                                   name='decoded_predictions')(predictions)
        model = Model(inputs=x, outputs=decoded_predictions)
    else:
        raise ValueError(
            "`mode` must be one of 'training', 'inference' or 'inference_fast', but received '{}'.".format(mode))

    if return_predictor_sizes:
        predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                    fc7_mbox_conf._keras_shape[1:3],
                                    conv6_2_mbox_conf._keras_shape[1:3],
                                    conv7_2_mbox_conf._keras_shape[1:3],
                                    conv8_2_mbox_conf._keras_shape[1:3],
                                    conv9_2_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model