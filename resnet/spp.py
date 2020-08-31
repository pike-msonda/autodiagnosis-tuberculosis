import sys
sys.path.append("..") 
from data_utils import *
from datetime import datetime
from keras.models import Model
# from keras.applications import ResNet50
from resnet.resnet50 import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, GlobalMaxPooling2D, Dropout
from utils.model_utils import ModelUtils
from custom_layers.spatial_pyramid_pooling import SpatialPyramidPooling

DATASET_PATH = '../data/train/'
TEST_PATH = 'D:\Data/test/'
TEST_PATH_NAME=os.path.join(TEST_PATH, 'china.pkl')
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')
MODEL_SIZE=(224, 224)



def make_model(classes=2):
     # CREATE MODEL 
    model = ResNet50(include_top=True, input_shape=(224, 224, 3),  weights=None, classes=2)
    return model

    
if __name__ == "__main__":
    start = datetime.now()
    model = make_model()

    model.summary()
    util = ModelUtils(epochs=120)
    # util.get_train_data(resize=(224, 224))
    # util.get_val_data()
    # util.get_test_data()
    # util.train(model)
    # util.evaluate()
    # util.save()
    # util.confusion_matrix()
    # util.plot_loss_accuracy()
    util.plot_multiple_roc(model, (224, 224))

    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))