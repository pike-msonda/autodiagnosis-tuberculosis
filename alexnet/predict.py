import sys
import numpy as np
from alexnet import AlexNet
from alexnet_new import NewAlexNet
sys.path.append("..") 
from data_utils import *
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from callbacks.history_callback import HistoryCallback
from keras.models import model_from_json
from keras import optimizers
from sklearn import metrics

IMAGE_NAME = '21_100.jpg'


if __name__ == "__main__":
    start = datetime.now()
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005)
    image_to_predict = load_image(IMAGE_NAME)
    image = pil_to_nparray(image_to_predict)
    image = np.expand_dims(image, axis=0)
    json_file = open('../models/alexnet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close() 
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("../models/alexnet.h5")

    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd,
        metrics=['acc'])
    #MODEL PREDICTION
    prediction = loaded_model.predict(image)
    print(onehot_to_cat (prediction))
