import sys
sys.path.append("..") 
from data_utils import *
from datetime import datetime
from googlenet import GoogleNet
from utils.model_utils import ModelUtils


if __name__ == "__main__":
    start = datetime.now()
    googlelenet = GoogleNet(input_shape=(3, 224, 224), classes=2)
    model = googlelenet.model()
    model.summary()

    util = ModelUtils(epochs=2)
    util.get_train_data(resize=(224, 224))
    util.get_test_data(resize=(224, 224))
    util.train(model)
    util.evaluate()
    # util.save()
    util.confusion_matrix()
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))