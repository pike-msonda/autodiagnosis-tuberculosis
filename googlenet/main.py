import sys
sys.path.append("..") 
from data_utils import *
from datetime import datetime
from utils.model_utils import ModelUtils


if __name__ == "__main__":
    start = datetime.now()
    model = InceptionV1(input_shape=(227, 227, 3), classes=2)
    # model = googlelenet.model()
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
