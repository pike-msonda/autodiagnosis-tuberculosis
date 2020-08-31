import sys
sys.path.append("..") 
from data_utils import *
# from resnet50 import  ResNet50
from datetime import datetime
from keras.applications import ResNet50
from utils.model_utils import ModelUtils

MODEL_SIZE=(224, 224)

if __name__ == "__main__":
    start = datetime.now()
    # CREATE MODEL 
    model = ResNet50(include_top=True, input_shape=(224,224,3), weights=None, classes=2)
    # model = resnet50.model()

    model.summary()
    util = ModelUtils(epochs=120)
    util.get_train_data(resize=(224, 224))

    # util.train(model)
    # util.evaluate()
    # util.save()
    # util.confusion_matrix()
    # util.plot_loss_accuracy()
    util.plot_multiple_roc(model, (224, 224))
    
    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))