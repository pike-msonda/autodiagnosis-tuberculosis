# -*- coding: utf-8 -*-
'''ResNet50 model for Keras.
# Reference:
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
'''
from __future__ import print_function

import numpy as np
import warnings 
from keras.layers import Reshape
from keras.layers import Input
from keras.layers.merge import Add
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
# from keras imagenet_utils import decode_predictions, preprocess_input


TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


class ResNet50:


    def __init__(self, input_shape, classes,  weights_path=None):
        # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
            self.weights_path = weights_path
            self.input = Input(shape=(input_shape))
            self.classes = classes
            if K.image_dim_ordering() == 'tf':
                self.bn_axis = 3
            else:
                self.bn_axis = 1


    def identity_block(self, input_tensor, kernel_size, filters, stage, block):
        '''The identity_block is the block that has no conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        '''
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=self.bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, kernel_size=(kernel_size, kernel_size),
                        padding='same', name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=self.bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=self.bn_axis, name=bn_name_base + '2c')(x)

        x = Add()([x, input_tensor])
        x = Activation('relu')(x)
        return x


    def conv_block(self,input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
        '''conv_block is the block that has a conv layer at shortcut
        # Arguments
            input_tensor: input tensor
            kernel_size: defualt 3, the kernel size of middle conv layer at main path
            filters: list of integers, the nb_filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
        Note that from stage 3, the first conv layer at main path is with strides=(2,2)
        And the shortcut should have strides=(2,2) as well
        '''
        nb_filter1, nb_filter2, nb_filter3 = filters
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(nb_filter1, kernel_size=(1, 1), strides=strides,
                        name=conv_name_base + '2a')(input_tensor)
        x = BatchNormalization(axis=self.bn_axis, name=bn_name_base + '2a')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter2, kernel_size=(kernel_size, kernel_size), padding='same',
                        name=conv_name_base + '2b')(x)
        x = BatchNormalization(axis=self.bn_axis, name=bn_name_base + '2b')(x)
        x = Activation('relu')(x)

        x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c')(x)
        x = BatchNormalization(axis=self.bn_axis, name=bn_name_base + '2c')(x)

        shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                                name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=self.bn_axis, name=bn_name_base + '1')(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def model(self, include_top=True, weights='imagenet',
                input_tensor=None):
        '''Instantiate the ResNet50 architecture,
        optionally loading weights pre-trained
        on ImageNet. Note that when using TensorFlow,
        for best performance you should set
        `image_dim_ordering="tf"` in your Keras config
        at ~/.keras/keras.json.
        The model and the weights are compatible with both
        TensorFlow and Theano. The dimension ordering
        convention used by the model is the one
        specified in your Keras config file.
        # Arguments
            include_top: whether to include the 3 fully-connected
                layers at the top of the network.
            weights: one of `None` (random initialization)
                or "imagenet" (pre-training on ImageNet).
            input_tensor: optional Keras tensor (i.e. xput of `layers.Input()`)
                to use as image input for the model.
        # Returns
            A Keras model instance.
        '''
        if weights not in {'imagenet', None}:
            raise ValueError('The `weights` argument should be either '
                            '`None` (random initialization) or `imagenet` '
                            '(pre-training on ImageNet).')
        # Determine proper input shape
       

        x = ZeroPadding2D((3, 3))(self.input)
        x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
        x = BatchNormalization(axis=self.bn_axis, name='bn_conv1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = self.conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = self.conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = self.identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = self.conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = self.identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = AveragePooling2D((7, 7), name='avg_pool')(x)

        if include_top:
            x = Flatten()(x)
            x = Dense(self.classes, activation='softmax')(x)

        model = Model(self.input, x, name="ResNet50")

        # load weights
        # if weights == 'imagenet':
        #     print('K.image_dim_ordering:', K.image_dim_ordering())
        #     if K.image_dim_ordering() == 'th':
        #         if include_top:
        #             weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels.h5',
        #                                     TH_WEIGHTS_PATH,
        #                                     cache_subdir='models',
        #                                     md5_hash='1c1f8f5b0c8ee28fe9d950625a230e1c')
        #         else:
        #             weights_path = get_file('resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
        #                                     TH_WEIGHTS_PATH_NO_TOP,
        #                                     cache_subdir='models',
        #                                     md5_hash='f64f049c92468c9affcd44b0976cdafe')
        #         model.load_weights(weights_path)
        #         if K.backend() == 'tensorflow':
        #             warnings.warn('You are using the TensorFlow backend, yet you '
        #                         'are using the Theano '
        #                         'image dimension ordering convention '
        #                         '(`image_dim_ordering="th"`). '
        #                         'For best performance, set '
        #                         '`image_dim_ordering="tf"` in '
        #                         'your Keras config '
        #                         'at ~/.keras/keras.json.')
        #             convert_all_kernels_in_model(model)
        #     else:
        #         if include_top:
        #             weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        #                                     TF_WEIGHTS_PATH,
        #                                     cache_subdir='models',
        #                                     md5_hash='a7b3fe01876f51b976af0dea6bc144eb')
        #         else:
        #             weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
        #                                     TF_WEIGHTS_PATH_NO_TOP,
        #                                     cache_subdir='models',
        #                                     md5_hash='a268eb855778b3df3c7506639542a6af')
        #         model.load_weights(weights_path)
        #         if K.backend() == 'theano':
        #             convert_all_kernels_in_model(model)
        return model
