import sys
sys.path.append("..") 
from data_utils import *
# from resnet50 import  ResNet50
from datetime import datetime
from keras.applications import ResNet50
from utils.model_utils import ModelUtils

DATASET_PATH = '../data/train/'
TEST_PATH = 'D:\Data/test/'
TEST_PATH_NAME=os.path.join(TEST_PATH, 'china.pkl')
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')

if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 
    model = ResNet50(include_top=True, input_shape=(227,227,3), weights=None,  classes=2)
    # model = resnet50.model()

    model.summary()

    util = ModelUtils(epochs=200)
    util.get_train_data()
    # util.get_test_data(resize=(227,227))
    util.train(model)
    util.evaluate()
    util.save()
    util.confusion_matrix()
    util.plot_loss_accuracy()
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))