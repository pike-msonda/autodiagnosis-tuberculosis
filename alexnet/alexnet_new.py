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
from keras import regularizers


class NewAlexNet:

    def __init__(self, input_shape, classes):
        self.init = Input(input_shape)
        self.classes = classes
    
    def conv_layer(self, input, filters,kernel_size, strides, padding= "same",
        activation="relu", kernel_regularizer=regularizers.l2(0), use_bias=False):
        x = Conv2D(filters=filters,
            kernel_size=kernel_size,
            strides=strides, padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
            use_bias=use_bias)(input)
        x = ZeroPadding2D(padding=(1,1))(x)
        return x

    def model(self):
        # FIRST LAYER
        x = self.conv_layer(input=self.init, filters=3, kernel_size=(11,11), 
            strides=(1,1),padding="same")
        x = MaxPooling2D(pool_size=(4,4), strides=(2,2))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

        # SECOND LAYER
        x = self.conv_layer(x, filters=48, kernel_size=(55,55), 
            strides=(1,1),padding="same")
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

        # THIRD LAYER
        x = self.conv_layer(x, filters=128, kernel_size=(27,27), 
            strides=(1,1),padding="same")
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)

         # FOURTH LAYER
        x = self.conv_layer(x, filters=192, kernel_size=(13,13), 
            strides=(1,1),padding="same")
        x = ZeroPadding2D(padding=(1, 1))(x)

           # FIFTH LAYER
        x = self.conv_layer(x, filters=192, kernel_size=(13,13), 
            strides=(1,1),padding="same")
        x = ZeroPadding2D(padding=(1, 1))(x)

            # SIXTH LAYER
        x = self.conv_layer(x, filters=128, kernel_size=(27,27), 
            strides=(1,1),padding="same")
        x = MaxPooling2D(pool_size=(2,2), strides=(2,2))(x)
        x = ZeroPadding2D(padding=(1, 1))(x)


        x = Flatten()(x)
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Channel 1 - Cov Net Layer 8
        x = Dense(2048, activation='relu')(x)
        x = Dropout(0.5)(x)

        # Final Channel - Cov Net 9
        output = Dense(units=self.classes,
              activation='softmax')(x)
         
        model = Model(self.init, output, name='AlexNet')

        return model

if __name__ == "__main__":
    alexnet = NewAlexNet(input_shape=(224,224,3), classes=2)
    model = alexnet.model()
    model.summary()