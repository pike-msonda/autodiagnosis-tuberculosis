"""
    Author: Pike Msonda
    Description: AlexNet implementation using Keras api
"""

from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2

class AlexNet:

    def __init__(self, input_shape, classes):
        self.init = Input(input_shape)
        self.classes = classes

    def conv_layer(self, x, filters,kernel_size, padding= "same", 
            kernel_regularizer=l2(0), strides = (1,1),
                 max_pooling=True, activation="relu", name=None): 

        x = Conv2D(filters, kernel_size ,strides =strides,padding=padding, 
            activation=activation)(x)
        if (max_pooling == True):
            x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
        x = BatchNormalization()(x)
        return x

    def output_layer(self,x, classes):
        x = Dense(units=classes)(x)
        x = Activation('softmax')(x)
        return x
    
    def dense_layer(self,x,units):
        x = Dense(units)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)
        
        return x

    def model(self):    
        # 1st LAYER
        x =  self.conv_layer(self.init, filters=96, kernel_size=(11,11), strides=(4,4),
        padding="valid", max_pooling=True, activation='relu', name='conv_1')

        #2nd Layer
        x =  self.conv_layer(x, filters=256, kernel_size=(5,5), strides=(1,1),
            padding="valid", max_pooling=True, name="conv_2")

        # 3RD LAYER
        x = ZeroPadding2D((1,1))(x)

        x =  self.conv_layer(x, filters=384, kernel_size=(3,3), strides=(1,1),padding="valid",
        max_pooling=False, name="conv_3")
        
        x = ZeroPadding2D((1,1))(x)

        # 4Th LAYER
        x =  self.conv_layer(x, filters=384, kernel_size=(3,3), strides=(1,1), 
        padding="valid", max_pooling=False, name="conv_4")

        x = ZeroPadding2D((1,1))(x)

        # 5Th LAYER
        x =  self.conv_layer(x, filters=256, kernel_size=(3,3), strides=(1,1),
        padding="valid", max_pooling=True, name="conv_5")

        # 6 FLATTEN 
        x = Flatten()(x)
        
        # Fully Connected LAYER 1
        x = Dense(4096)(x)
        x = Activation('relu')(x)
        x = Dropout(0.5)(x)

        #FULLY CONNECTED LAYER 2
        x = Dense(4096)(x)
        x = Activation('relu')(x)

        #FULLY CONNECTED LAYER 3
        x = Dense(1000)(x)
        x =  Activation('relu')(x)

        # OUTPUT LAYER
        output = self.output_layer(x, self.classes)

        model = Model(self.init, output, name='AlexNet2')

        return model

    




       
    
