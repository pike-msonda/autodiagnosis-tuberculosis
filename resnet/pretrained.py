import sys
sys.path.append("..") 
from data_utils import *
from datetime import datetime
from keras.models import Model
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense
from utils.model_utils import ModelUtils

DATASET_PATH = '../data/train/'
TEST_PATH = 'D:\Data/test/'
TEST_PATH_NAME=os.path.join(TEST_PATH, 'china.pkl')
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')

MODEL_SIZE=(224, 224)
def make_model(classes=2):
     # CREATE MODEL 
    model = ResNet50(include_top=False, weights='imagenet')
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(classes, activation='softmax')(x)
    model = Model(inputs=model.input, outputs=predictions, name="resent_pretrained")
    return model

    
if __name__ == "__main__":
    start = datetime.now()
    model  = make_model()

    # model.summary()
    util = ModelUtils(epochs=100)
    util.get_train_data(resize=MODEL_SIZE)
    util.get_val_data(resize=MODEL_SIZE)
    util.get_test_data(resize=MODEL_SIZE)
    util.mean_subtraction()
    util.train(model)
    util.evaluate()
    util.save()
    util.confusion_matrix()
    util.plot_loss_accuracy()
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))