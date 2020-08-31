import sys
sys.path.append("..") 
from data_utils import *
from alexnet import AlexNet
from datetime import datetime
from utils.model_utils import ModelUtils
import tensorflow as tf
tf.set_random_seed(1000)
random.seed(1000)
np.random.seed(1000)
MODEL_SIZE=(227, 227)

if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 
    alexnet = AlexNet(input_shape=(227,227, 3), classes=2)

    model = alexnet.model()

    model.summary()

    util = ModelUtils(epochs=120)
    util.get_train_data()
    # util.get_val_data(resize=(MODEL_SIZE))

    # util.train(model)
    # util.evaluate()
    # util.save()
    # util.confusion_matrix()
    # util.plot_roc_curve()
    # util.plot_loss_accuracy()
    util.plot_multiple_roc(model)
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))