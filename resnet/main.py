import sys
sys.path.append("..")
from keras_applications import resnet50
from resnet50 import ResNet50
from resnet import ResNet
import numpy as np
from data_utils import *
from datetime import datetime
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from callbacks.history_callback import HistoryCallback
from keras.models import model_from_json
from keras import optimizers
from keras import layers
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split



DATASET_PATH = '../data'
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')

if __name__ == "__main__":

    # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    np.random.seed(1000)
    start = datetime.now()
    x, y = build_image_dataset_from_dir(os.path.join(DATASET_PATH, 'usa'),
                                        dataset_file=IMAGESET_NAME,
                                        resize=(224, 224),
                                        filetypes=['.png'],
                                        convert_to_color=True,
                                        shuffle_data=True,
                                        categorical_Y=True)

    sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay=0.0005)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
    resnet = ResNet50()
    model = resnet.model(include_top=True, weights=None)
    # model.summary()
    if not (os.path.exists('../models/resnet2.h5')):

        # resnet2 = ResNet(input_shape=(224,224, 3), classes=2)

        print("Training with {0}".format(len(X_train)))
        print("Testing with {0}".format(len(X_test)))

        # COMPILE MODEL
        model.compile(loss='categorical_crossentropy', optimizer=sgd,
                      metrics=['accuracy'])

        # # #TRAIN MODEL
        model.fit(X_train, y_train, batch_size=6, epochs=120, verbose=1,
                  validation_split=0.2, shuffle=True, callbacks=[HistoryCallback('../history/history.csv')])

        store_model(model, "../models/", "resnet2")

        score = model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        y_pred = model.predict(X_test)
        print(precision_recall_fscore_support(onehot_to_cat(
            y_test), onehot_to_cat(y_pred), average='binary'))
        labels = list(set(get_labels(y_test)))
        cm = confusion_matrix(get_labels(y_test), get_labels(y_pred))
        plot_confusion_matrix(cm, labels)

    else:

        labels = list(set(get_labels(y_test)))
        json_file = open('../models/resnet2.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("../models/resnet2.h5")

        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd,
                             metrics=['accuracy'])
        score = loaded_model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.4f%%" % (loaded_model.metrics_names[1], score[1]*100))
        print("%s: %.4f%%" % (loaded_model.metrics_names[0], score[0]))
        # MODEL PREDICTION
        prediction = loaded_model.predict(X_test)
  
        print(precision_recall_fscore_support(onehot_to_cat(
            y_test), onehot_to_cat(prediction), average='binary'))
        cm = confusion_matrix(get_labels(y_test), get_labels(prediction))
        plot_confusion_matrix(cm, labels, title="Confusion Matrix: ResNet50")
        # #import pdb; pdb.set_trace()

    time_elapsed = datetime.now() - start
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
