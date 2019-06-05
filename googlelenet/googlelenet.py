"""
    Author: Pike Msonda
    Description: GoogleLeNet implementation using Keras api
"""

from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import ZeroPadding2D, merge, Reshape
from keras.layers.merge import concatenate
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras import layers


class GoogleLeNet:


    def __init__(self,input_shape, classes):
        self.input_shape = Input(input_shape)
        self.classes = classes

    def conv_layer(self, x,filters,kernel_size,padding="same",strides=(1,1)):

        x = Conv2D(filters, kernel_size,strides=strides,padding=padding)(x)

        x = BatchNormalization(axis=3, scale=False)(x)

        x = Activation("relu")(x)
        

        return x

    def concatenated_layer(self, x, specs, channel_axis):

        (br0, br1, br2, br3) = specs

        # import pdb; pdb.set_trace()

        branch_0 = self.conv_layer(x, br0[0], kernel_size=(1,1))

        
        branch_1 = self.conv_layer(x, br1[0], kernel_size=(1,1))
        branch_1 = self.conv_layer(branch_1, br1[1], kernel_size=(3,3))

        branch_2 = self.conv_layer(x, br2[0], kernel_size=(1,1))
        branch_2 = self.conv_layer(branch_2, br2[1], kernel_size=(3,3))

        branch_3 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)  
        branch_3 = self.conv_layer(branch_3, br3[0], kernel_size=(3,3))

        x = layers.concatenate(
            [branch_0, branch_1, branch_2, branch_3],
            axis=channel_axis)

        return x
    
    def model(self):
        
        channel_axis = 3

        x = self.conv_layer(self.input_shape, filters=64, kernel_size=(7, 7),
            strides=(2, 2),padding="same")
        
        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)  
        
        x = self.conv_layer(x, 64, kernel_size=(1, 1), strides=(1, 1), 
            padding="same") 
        
        x = self.conv_layer(x, 192, kernel_size=(3, 3), strides=(1, 1), 
            padding="same")

        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x) 

        # Now the '3' level inception units
        x = self.concatenated_layer(x, ((64,), (96,128), (16,32), (32,)), 
            channel_axis)

        x = self.concatenated_layer(x, ((128,), (128,192), (32, 96), ( 64,)), 
            channel_axis)

        x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        
        x = self.concatenated_layer(x, ((192,), ( 96,208), (16, 48), ( 64,)), 
            channel_axis)
        x = self.concatenated_layer(x, ((160,), (112,224), (24, 64), ( 64,)), 
            channel_axis)
        x = self.concatenated_layer(x, ((128,), (128,256), (24, 64), ( 64,)), 
            channel_axis)
        x = self.concatenated_layer(x, ((112,), (144,288), (32, 64), ( 64,)), 
            channel_axis)
        x = self.concatenated_layer(x, ((256,), (160,320), (32,128), (128,)), 
            channel_axis)
        
        
        x = MaxPooling2D((2, 2), strides=(2, 2))(x) 


          # Now the '5' level inception units
        x = self.concatenated_layer(x, ((256,), (160,320), (32,128), (128,)),   
            channel_axis)
        x = self.concatenated_layer(x, ((384,), (192,384), (48,128), (128,)),
             channel_axis) 

        x = AveragePooling2D((7, 7), strides=(1, 1), padding='valid')(x)  

        x = Dropout(0.2)(x)  # slim has keep_prob (@0.8), keras uses drop_fraction

        x = Conv2D(self.classes, (1, 1), strides=(1,1), padding='valid', 
            use_bias=True, name='Logits')(x)

        x = Flatten(name='Logits_flat')(x)

        x = Activation('softmax', name='Predictions')(x)

        model = Model(self.input_shape, x, name='GoogleLeNet2')


        return model