import sys
sys.path.append("..") 
from data_utils import *
from datetime import datetime
from keras.applications import ResNet50
from keras.models import GlobalAveragePooling2D, Model, Dense
from utils.model_utils import ModelUtils

DATASET_PATH = '../data/train/'
TEST_PATH = 'D:\Data/test/'
TEST_PATH_NAME=os.path.join(TEST_PATH, 'china.pkl')
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')


def make_model(classes=2):
     # CREATE MODEL 
    model = ResNet50(include_top=False, weights='imagenet')
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions)
    return model

    
if __name__ == "__main__":
    start = datetime.now()
    model  = make_model()

    # util = ModelUtils(epochs=200)
    # util.get_train_data()
    # # util.get_test_data(resize=(227,227))
    # util.train(model)
    # util.evaluate()
    # util.save()
    # util.confusion_matrix()
    # util.plot_loss_accuracy()
    
    # time_elapsed = datetime.now() - start 
    # print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))