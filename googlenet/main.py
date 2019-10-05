import sys
sys.path.append("..") 
from data_utils import *
# from googlenet.v1 import InceptionV1
from googlenet import create_googlenet

from datetime import datetime
from utils.model_utils import ModelUtils
MODEL_SIZE=(224, 224)

if __name__ == "__main__":
    start = datetime.now()
    # model = InceptionV1( include_top=True, input_shape=(224, 224, 3), weights=None, classes=2)
    # model = googlelenet.model()
    model = create_googlenet(weights_path=None, input_shape=(3, 224, 224))

    model.summary()

    util = ModelUtils(epochs=100)
    util.get_train_data(resize=MODEL_SIZE)
    util.get_val_data(resize=MODEL_SIZE)
    util.get_test_data(resize=MODEL_SIZE)
    util.train(model)
    util.evaluate()
    util.save()
    util.confusion_matrix()
    util.plot_loss_accuracy()
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
