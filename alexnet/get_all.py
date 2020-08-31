import sys
sys.path.append("..") 
from data_utils import *
from googlenet.v1 import InceptionV1
from alexnet import AlexNet
from googlenet.v1_spp import InceptionV1 as InceptionV1SPP
from alexnet_spp import AlexNet as AlexNetSPP
from keras.applications import ResNet50
from resnet.resnet50 import ResNet50 as ResNet50SPP
from datetime import datetime
from utils.model_utils import ModelUtils
import tensorflow as tf
# tf.set_random_seed(1000)
# random.seed(1000)
# np.random.seed(1000)
MODEL_SIZE=(224, 224)

def alexNet():
    return AlexNet(input_shape=(224,224, 3), classes=2)
def alexNetSPP():
    return AlexNetSPP(input_shape=(224,224, 3), classes=2)

def googlenet():
    return InceptionV1( include_top=True, input_shape=(224, 224, 3), weights=None, classes=2)

def googlenetSPP():
    return InceptionV1SPP( include_top=True, input_shape=(224, 224, 3), weights=None, classes=2)

def resnet50():
    return ResNet50(include_top=True, input_shape=(224,224,3), weights=None, classes=2)

def resnet50SPP():
    return ResNet50SPP(include_top=True, input_shape=(224,224,3), weights=None, classes=2)

if __name__ == "__main__":
    start = datetime.now()
    util = ModelUtils(epochs=120)
    util.get_train_data()
    util.get_results([alexNet().model(), alexNetSPP().model()])
    # util.get_train_data(resize=MODEL_SIZE) 
    util.get_results([googlenet(), googlenetSPP(), resnet50(), resnet50SPP()])
    # util.get_train_data(resize=MODEL_SIZE)

    # util.get_results([googlenet(), googlenetSPP(), resnet50(), resnet50SPP()])



