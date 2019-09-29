import sys
sys.path.append("..") 
from data_utils import *
from keras.applications.inception_v3 import InceptionV3
from datetime import datetime
from utils.model_utils import ModelUtils


if __name__ == "__main__":
    start = datetime.now()
    model = InceptionV3( include_top=True, input_shape=(256, 256, 3), weights=None, classes=2)
    # model = googlelenet.model()
    model.summary()

    util = ModelUtils(epochs=80)
    util.get_train_data()
    util.get_test_data()
    util.train(model)
    util.evaluate()
    util.save()
    util.confusion_matrix()
    util.plot_loss_accuracy()
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
