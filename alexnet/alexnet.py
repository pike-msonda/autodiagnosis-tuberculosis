import numpy as np
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, Input, merge
from custom_layers.spatial_pyramid_pooling import SpatialPyramidPooling
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, concatenate, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import backend as K
from scipy.misc import imread, imresize, imsave
from custom_layers.lrn_layer import LRN
from custom_layers.crosschannelnormalisation import splittensor
from custom_layers.spatial_pyramid_pooling import SpatialPyramidPooling
K.set_image_dim_ordering('th')
class AlexNet:

    def __init__(self, input_shape, classes, weights_path=None):
        self.init = Input(input_shape)
        self.classes = classes
        self.weights_path = weights_path

    def model(self):  

        # COVOLUTIONAL LAYER 1  
        x = Conv2D(96, (11,  11),strides=(4,4),activation='relu',
                            name='conv_1')(self.init)

        x = MaxPooling2D((3, 3), strides=(2,2))(x)
        x = LRN(name="convpool_1")(x) # normalisation instead of Batch Normalisation
        x = ZeroPadding2D((2,2))(x)

        # COVOLUTIONAL LAYER 2  
        x = concatenate([
            Conv2D(128,(5,5),activation="relu",name='conv_2_'+str(i+1))(

                splittensor(ratio_split=2,id_split=i)(x)

            ) for i in range(2)],axis=1,name="conv_2")

        x = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = LRN()(x)
        x = ZeroPadding2D((1,1))(x)

        # COVOLUTIONAL LAYER 3 
        x = Conv2D(384,(3,3),activation='relu',name='conv_3')(x)

        x = ZeroPadding2D((1,1))(x)
        x = concatenate([
            Conv2D(192,(3,3),activation="relu",name='conv_4_'+str(i+1))(

                splittensor(ratio_split=2,id_split=i)(x)

            ) for i in range(2)], axis=1,name="conv_4")
        x = ZeroPadding2D((1,1))(x)


        # COVOLUTIONAL LAYER 5 
        x = concatenate([
            Conv2D(128,(3,3),activation="relu",name='conv_5_'+str(i+1))(

                splittensor(ratio_split=2,id_split=i)(x)
            ) for i in range(2)],axis=1,name="conv_5")

        x = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(x)

        # import pdb; pdb.set_trace()
        # x = SpatialPyramidPooling([1, 2, 4])(x)
        # Flatten Tensor
        x = Flatten(name="flatten")(x)

        # FUlly connected layer 1
        x = Dense(4096, activation='relu',name='dense_1')(x)
        x = Dropout(0.5)(x)

        # FUlly connected layer 2
        x = Dense(4096, activation='relu',name='dense_2')(x)
        x = Dropout(0.5)(x)
        

        # OUTPUT Layer
        x = Dense(self.classes,name='dense_3')(x)
        ouput  = Activation("softmax",name="softmax")(x)


        model = Model(input=self.init, output=ouput, name="alexnet")

        if self.weights_path:
            model.load_weights(self.weights_path)

        return model

