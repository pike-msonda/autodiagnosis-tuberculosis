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
from callbacks.history_callback import HistoryCallback
from keras.models import model_from_json
from keras import optimizers
from sklearn import metrics

DATASET_PATH = '../data/aug/all'
IMAGESET_NAME = os.path.join(DATASET_PATH, 'china.pkl')

if __name__ == "__main__":
    start = datetime.now()
    x, y = build_image_dataset_from_dir(os.path.join(DATASET_PATH, 'china'),
        dataset_file=IMAGESET_NAME,
        resize=None,
        filetypes=['.png'],
        convert_to_color=True,
        shuffle_data=True,
        categorical_Y=True)

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30, 
        random_state=1000)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=0.0005,nesterov=False)

    alexnet = AlexNet(input_shape=(227,227,3), classes=2)
    model = alexnet.model()
    model.summary()  
    if not (os.path.exists('../models/alexnet.h5')):
        print("Training with {0}".format(len(X_train)))
        print("Testing with {0}".format(len(X_test)))

        # COMPILE MODEL
       
        model.compile(loss='categorical_crossentropy', optimizer=sgd,\
        metrics=['accuracy'])

        #TRAIN MODEL
        model.fit(X_train,y_train, batch_size=32, epochs=80, verbose=1, 
           validation_split=0.3, shuffle=True) #callbacks=[HistoryCallback('../history/history.csv')])
        
        score= model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]))
        # store_model(model, "../models/", "alexnet")
        y_pred =  model.predict(X_test)
        print(precision_recall_fscore_support(onehot_to_cat(y_test), onehot_to_cat(y_pred)))
        labels = list(set(get_labels(y_test))) 
        cm = confusion_matrix(get_labels(y_test),get_labels(y_pred))
        plot_confusion_matrix(cm, labels)
    else:

        labels = list(set(get_labels(y_test))) 
        json_file = open('../models/alexnet.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close() 
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("../models/alexnet.h5")

        loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd,
         metrics=['acc'])
        score = loaded_model.evaluate(X_test, y_test, verbose=0)
        print(score)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]))
        #MODEL PREDICTION
        prediction = loaded_model.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test,prediction)
        auc = metrics.roc_auc_score(y_test,model.predict(X_test))
        print(precision_recall_fscore_support(onehot_to_cat(y_test), onehot_to_cat(prediction), average='binary'))
        cm = confusion_matrix(get_labels(y_test),get_labels(prediction))
        plot_confusion_matrix(cm, labels, title="Confusion Matrix: AlexNet", cmap=plt.cm.Greens)

    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))