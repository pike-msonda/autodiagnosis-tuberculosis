import sys
sys.path.append("..") 
from data_utils import *
from v1 import InceptionV1
from datetime import datetime
from utils.model_utils import ModelUtils
import tensorflow as tf
tf.set_random_seed(1000)
random.seed(1000)
np.random.seed(1000)
MODEL_SIZE=(224, 224)

if __name__ == "__main__":
    start = datetime.now()
    model = InceptionV1( include_top=True, input_shape=(224, 224, 3), weights=None, classes=2)
    # model = googlelenet.model()
    # model = create_googlenet(weights_path=None, input_shape=(3, 224, 224))

    model.summary()

    util = ModelUtils(epochs=120)
    util.get_train_data(resize=(224, 224))
    # util.train(model)
    # util.evaluate()
    # util.save()
    # util.confusion_matrix()
    # util.plot_roc_curve()

    # util.plot_loss_accuracy()
    util.plot_multiple_roc(model, (224, 224))
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
