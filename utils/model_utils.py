import os
import numpy as np
import keras as ke
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from data_utils import build_image_dataset_from_dir, get_labels, onehot_to_cat, plot_confusion_matrix, plot_accuracy_loss_graph
from keras import backend as K
from utils.data_sequence import DataSequence
from utils.augs import AUGMENTATIONS_TRAIN, AUGMENTATIONS_TEST
import tensorflow as tf
import random
tf.set_random_seed(1000)
random.seed(1000)
np.random.seed(1000)

FOLDER = 'all'
class ModelUtils():

    def __init__(self, epochs=2,test_split=0.30, validation_split=0.25):
        self.epochs=epochs
        self.test_split=test_split
        self.validation=validation_split
        self.batch_size = 32

    def get_train_data(self, name=FOLDER, folder='../data/train', resize=None):
        self.x, self.y = build_image_dataset_from_dir(os.path.join(folder, name),
            dataset_file=os.path.join(folder, name+'.pkl'),
            resize=resize,
            filetypes=['.png'],
            convert_to_color=True,
            shuffle_data=True,
            categorical_Y=True)
        
        self.trainX, self.valX, self.trainY, self.valY = train_test_split(self.x, self.y, test_size=self.validation, random_state=1000)
        mean = np.mean(self.x, axis=0)
        self.x -= mean
        self.valX -= mean

        print("Training on {0} and validating on {1}".format(len(self.trainX), len(self.valX)))
       
        

        # self.trainGen =  DataSequence(self.trainX, self.trainY, self.batch_size, AUGMENTATIONS_TRAIN)
        # self.valGen =  DataSequence(self.valX, self.valY, self.batch_size, AUGMENTATIONS_TEST)
        # self.testGen =  DataSequence(self.testX, self.testY, self.batch_size, AUGMENTATIONS_TEST)

    def get_test_data(self, name=FOLDER, folder='../data/test', resize=None):
        self.testX, self.testY = build_image_dataset_from_dir(os.path.join(folder, name),
            dataset_file=os.path.join(folder, name+'.pkl'),
            resize=resize,
            filetypes=['.png'],
            convert_to_color=True,
            shuffle_data=True,
            categorical_Y=True)

        mean = np.mean(self.testX, axis=0)
        self.testX -= mean
        print("Testing on {0} ".format(len(self.testX)))
        # self.testGen =  DataSequence(self.testX, self.testY, self.batch_size, AUGMENTATIONS_TEST)
        
    def train(self, model):
        self.model = model
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer(), 
            metrics=['accuracy'])
        aug = ImageDataGenerator(
            # rotation_range=90, 
			# zoom_range=[0.15, 0.25],
			# width_shift_range=0.2,
			# height_shift_range=0.2,
			shear_range=0.15,
			horizontal_flip=True,
            # vertical_flip=True,
			fill_mode="nearest")

        if(K.image_dim_ordering() == 'th'):
            self.x = np.moveaxis(self.x, -1, 1)
            self.valX = np.moveaxis(self.valX, -1, 1)
        
        if(os.path.exists('../models/'+self.model.name+'.h5')):
            self.model.load_weights('../models/'+self.model.name+'.h5') 
        else:
            # self.history = self.model.fit_generator(self.trainGen,
            #     epochs=self.epochs, verbose=1, shuffle=True,
            #     validation_data=self.valGen, workers=2, use_multiprocessing=False)

            self.history = self.model.fit_generator(aug.flow(self.x,self.y, batch_size=self.batch_size, shuffle=True, seed=1452),
                steps_per_epoch=len(self.x)/self.batch_size ,epochs=self.epochs, verbose=1, 
                validation_data=(self.valX, self.valY))


    def evaluate(self):
        score = self.model.evaluate(self.testX, self.testY)
        scoreVal = self.model.evaluate(self.valX, self.valY)
      
        print(score)
        print("%s: %.2f%%" % (self.model.metrics_names[-1], score[-1]))
        print("%s: %.2f%%" % (self.model.metrics_names[-1], scoreVal[-1]))


    def save(self, folder='../models'):
        self.model.save_weights(folder+'/'+self.model.name+'.h5')

    def optimizer(self):
        return SGD(lr=0.001, momentum=0.9, decay=0.0005)

    def confusion_matrix(self):
        predictions = self.model.predict(self.testX)
        labels = list(set(get_labels(self.testY))) 
        cm = confusion_matrix(get_labels(self.testY),get_labels(predictions))
        print("Confusion Matrix {}".format(cm))
        plot_confusion_matrix(cm, labels, title=self.model.name)

    def plot_loss_accuracy(self):
        plot_accuracy_loss_graph(self.history)