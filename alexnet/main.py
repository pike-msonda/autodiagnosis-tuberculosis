import numpy as np
import sys
sys.path.append("..") 
from data_utils import *
import tensorflow as tf
from alexnet import AlexNet
from datetime import datetime
from sklearn.model_selection import train_test_split 
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from keras.models import load_model
from keras import optimizers
from sklearn import metrics
# from keras.backend import manual_variable_initialization(True)
DATASET_PATH = '../data/train/'
TEST_PATH = 'D:\Data/test/'
TEST_PATH_NAME=os.path.join(TEST_PATH, 'china.pkl')
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')

if __name__ == "__main__":
    start = datetime.now()
    trainX, trainY = build_image_dataset_from_dir(os.path.join(DATASET_PATH, 'china'),
        dataset_file=IMAGESET_NAME,
        resize=None,
        filetypes=['.png'],
        convert_to_color=True,
        shuffle_data=True,
        categorical_Y=True)

    testX, testY = build_image_dataset_from_dir(os.path.join(TEST_PATH, 'china'),
        dataset_file=TEST_PATH_NAME,
        resize=(227, 227),
        filetypes=['.png'],
        convert_to_color=True,
        shuffle_data=True,
        categorical_Y=True)

    X_train, X_test, y_train, y_test = train_test_split(trainX,trainY, test_size=0.30, 
        random_state=1000)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005,nesterov=False)

    alexnet = AlexNet(input_shape=(3,227,227), classes=2)
    model = alexnet.model()
    model.summary()  
    if not (os.path.exists('../models/alexnet.h5')):
        print("Training with {0}".format(len(trainX)))
        print("Testing with {0}".format(len(testX)))

        # COMPILE MODEL
       
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        #TRAIN MODEL
        model.fit(np.moveaxis(X_train, -1, 1) ,y_train, batch_size=32, nb_epoch=10, verbose=1, 
           validation_data=(np.moveaxis(X_test,-1,1), y_test), shuffle=True)
        
        score= model.evaluate(np.moveaxis(testX, -1, 1), testY, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]))
        store_model(model, "../models/", "alexnet")
        y_pred =  model.predict(np.moveaxis(testX, -1, 1))
        print(precision_recall_fscore_support(onehot_to_cat(np.moveaxis(testX, -1, 1)), onehot_to_cat(y_pred)))
        labels = list(set(get_labels(testY))) 
        cm = confusion_matrix(get_labels(np.moveaxis(testX, -1, 1)),get_labels(y_pred))
        plot_confusion_matrix(cm, labels)
    else:
        labels = list(set(get_labels(testY))) 

        loaded_model = load_model("../models/alexnet.h5")

        import pdb; pdb.set_trace()
        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd,
         metrics=['acc'])
        score = loaded_model.evaluate(np.moveaxis(testX, -1, 1), testY, verbose=1)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]))
        y_pred =  loaded_model.predict(np.moveaxis(testX, -1, 1))
        # print(precision_recall_fscore_support(onehot_to_cat(testY), onehot_to_cat(y_pred)))
        labels = list(set(get_labels(testY))) 
        cm = confusion_matrix(get_labels(testY),get_labels(y_pred))
        plot_confusion_matrix(cm, labels)
        # plot_confusion_matrix(cm, labels, title="Confusion Matrix: AlexNet", cmap=plt.cm.Greens)

    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))