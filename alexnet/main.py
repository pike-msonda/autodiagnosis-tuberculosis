import sys
sys.path.append("..") 
from data_utils import *
from alexnet import AlexNet
from datetime import datetime
from utils.model_utils import ModelUtils

DATASET_PATH = '../data/train/'
TEST_PATH = 'D:\Data/test/'
TEST_PATH_NAME=os.path.join(TEST_PATH, 'china.pkl')
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')

if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 
    alexnet = AlexNet(input_shape=(3,227,227), classes=2)

    model = alexnet.model()

    model.summary()

    util = ModelUtils(epochs=40)
    util.get_train_data()
    util.get_test_data()
    util.train(model)
    util.evaluate()
    util.save()
    util.confusion_matrix()
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))