import os    
os.environ['THEANO_FLAGS'] = "device=cuda"  
os.environ['KERAS_BACKEND'] = 'theano'
from datetime import datetime
import sys
sys.path.insert(0, '../convnets-keras')
import sys
sys.path.append("..") 
from data_utils import *
from sklearn.model_selection import train_test_split 
from keras.models import model_from_json
from keras import optimizers
from keras import backend as K
from theano import tensor as T
from keras.models import Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from conv.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D

DATASET_PATH = '../data/aug/all'
IMAGESET_NAME = os.path.join(DATASET_PATH, 'usa.pkl')

def mean_subtract(img):   
    img = T.set_subtensor(img[:,0,:,:],img[:,0,:,:] - 123.68)
    img = T.set_subtensor(img[:,1,:,:],img[:,1,:,:] - 116.779)
    img = T.set_subtensor(img[:,2,:,:],img[:,2,:,:] - 103.939)

    return img / 255.0

def get_alexnet(input_shape,nb_classes,mean_flag): 
	# code adapted from https://github.com/heuritech/convnets-keras

	inputs = Input(shape=input_shape)

	if mean_flag:
		mean_subtraction = Lambda(mean_subtract, name='mean_subtraction')(inputs)
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
		                   name='conv_1', init='he_normal')(mean_subtraction)
	else:
		conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu',
		                   name='conv_1', init='he_normal')(inputs)

	conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
	conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
	conv_2 = ZeroPadding2D((2,2))(conv_2)
	conv_2 = merge([
	    Convolution2D(128,5,5,activation="relu",init='he_normal', name='conv_2_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_2)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

	conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
	conv_3 = crosschannelnormalization()(conv_3)
	conv_3 = ZeroPadding2D((1,1))(conv_3)
	conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3',init='he_normal')(conv_3)

	conv_4 = ZeroPadding2D((1,1))(conv_3)
	conv_4 = merge([
	    Convolution2D(192,3,3,activation="relu", init='he_normal', name='conv_4_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_4)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

	conv_5 = ZeroPadding2D((1,1))(conv_4)
	conv_5 = merge([
	    Convolution2D(128,3,3,activation="relu",init='he_normal', name='conv_5_'+str(i+1))(
		splittensor(ratio_split=2,id_split=i)(conv_5)
	    ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

	dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

	dense_1 = Flatten(name="flatten")(dense_1)
	dense_1 = Dense(4096, activation='relu',name='dense_1',init='he_normal')(dense_1)
	dense_2 = Dropout(0.5)(dense_1)
	dense_2 = Dense(4096, activation='relu',name='dense_2',init='he_normal')(dense_2)
	dense_3 = Dropout(0.5)(dense_2)
	dense_3 = Dense(nb_classes,name='dense_3_new',init='he_normal')(dense_3)

	prediction = Activation("softmax",name="softmax")(dense_3)

	alexnet = Model(input=inputs, output=prediction)
    
	return alexnet

if __name__ == "__main__":
   
    start = datetime.now()
    x, y = build_image_dataset_from_dir(os.path.join(DATASET_PATH, 'usa'),
        dataset_file=IMAGESET_NAME,
        resize=None,
        filetypes=['.png'],
        convert_to_color=True,
        shuffle_data=True,
        categorical_Y=True)

    X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.30, 
        random_state=1000)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, decay=1e-6)

    # adam = optimizers.adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0005, amsgrad=False)

    model = get_alexnet((227,227,3), 2, False)
    # model = alexnet.model()
    model.summary()  
    if not (os.path.exists('../models/alexnet.h5')):
        print("Training with {0}".format(len(X_train)))
        print("Testing with {0}".format(len(X_test)))

        # COMPILE MODEL
       
        model.compile(loss='categorical_crossentropy', optimizer=sgd,\
        metrics=['accuracy'])

        #TRAIN MODEL
        model.fit(X_train,y_train, batch_size=16, nb_epoch=60, verbose=1, 
           validation_split=0.2, shuffle=True) #callbacks=[HistoryCallback('../history/history.csv')])
        
        score= model.evaluate(X_test, y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*227))
        store_model(model, "../models/", "alexnet")
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
        import pdb; pdb.set_trace()
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*227))
        #MODEL PREDICTION
        prediction = loaded_model.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test,prediction)
        auc = metrics.roc_auc_score(y_test,model.predict(X_test))
        print(precision_recall_fscore_support(onehot_to_cat(y_test), onehot_to_cat(prediction), average='binary'))
        cm = confusion_matrix(get_labels(y_test),get_labels(prediction))
        plot_confusion_matrix(cm, labels, title="Confusion Matrix: AlexNet", cmap=plt.cm.Greens)

    time_elapsed = datetime.now() - start 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))