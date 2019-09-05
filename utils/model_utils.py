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
from callbacks.clr_callback import CyclicLR


FOLDER = 'turkey'
class ModelUtils():

    def __init__(self, epochs=2,test_split=0.30, validation_split=0.3):
        self.epochs=epochs
        self.test_split=test_split
        self.validation=validation_split
        self.batch_size = 32
       

    def get_train_data(self, name=FOLDER, folder='E:\Pike\Data/train', resize=None):
        self.x, self.y = build_image_dataset_from_dir(os.path.join(folder, name),
            dataset_file=os.path.join(folder, name+'.pkl'),
            resize=resize,
            filetypes=['.png'],
            convert_to_color=True,
            shuffle_data=True,
            categorical_Y=True)
        self.x, self.valX, self.y, self.valY = train_test_split(self.x, self.y, test_size=self.test_split, random_state=1000)
        mean = np.mean(self.x, axis=0)
        self.x -= mean
        self.valX -= mean
        # self.clr = CyclicLR(
        #     mode="triangular",
        #     base_lr=1e-7,
        #     max_lr=1e-2,
        #     step_size= 8 * (self.x.shape[0] // self.batch_size))


    def get_test_data(self, name=FOLDER, folder='E:\Pike\Data/test', resize=None):
        self.testX, self.testY = build_image_dataset_from_dir(os.path.join(folder, name),
            dataset_file=os.path.join(folder, name+'.pkl'),
            resize=resize,
            filetypes=['.png'],
            convert_to_color=True,
            shuffle_data=True,
            categorical_Y=True)
        
    def train(self, model):
        self.model = model
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer(), 
            metrics=['accuracy'])
        aug = ImageDataGenerator(
            # rotation_range=20, 
			zoom_range=0.15,
			width_shift_range=0.2,
			height_shift_range=0.2,
			shear_range=0.15,
			horizontal_flip=True,
            # vertical_flip=True,
			fill_mode="nearest")

        if(K.image_dim_ordering() == 'th'):
            self.x = np.moveaxis(self.x, -1, 1)
            self.valX = np.moveaxis(self.valX, -1, 1)
            self.testX = np.moveaxis(self.testX, -1, 1)
        
        if(os.path.exists('../models/'+self.model.name+'.h5')):
            self.model.load_weights('../models/'+self.model.name+'.h5') 
            self.model.evaluate(self.valX, self.valY, verbose=0)
        else:
            if(self.model.name == 'googlenet'):
                self.y = [self.y,self.y, self.y] # because GoogleNet has 3 outputs
                self.valY = [self.valY, self.valY, self.valY]

            self.history = self.model.fit_generator(aug.flow(self.x,self.y, batch_size=self.batch_size, shuffle=True),
                steps_per_epoch=len(self.x)/self.batch_size ,epochs=self.epochs, verbose=1, 
                validation_data=(self.valX, self.valY)
                # callbacks=[self.clr]
                )

        

    def evaluate(self):
        if(self.model.name == 'googlenet'):
            self.testY = [self.testY,self.testY, self.testY] # because GoogleNet has 3 outputs
        score = self.model.evaluate(self.valX, self.valY)
      
        print("%s: %.2f%%" % (self.model.metrics_names[-1], score[-1]))

    def save(self, folder='../models'):
        self.model.save_weights(folder+'/'+self.model.name+'.h5')

    def optimizer(self):
        return SGD(lr=0.001 , momentum=0.9, decay=0.0005,nesterov=False)

    def confusion_matrix(self):
        predictions = self.model.predict(self.valX)
        if(self.model.name == 'googlenet'):
            self.testY = self.testY[0]
            predictions = predictions[0]

        labels = list(set(get_labels(self.valY))) 
        cm = confusion_matrix(get_labels(self.valY),get_labels(predictions))
        print("Confusion Matrix {}".format(cm))
        plot_confusion_matrix(cm, labels, title=self.model.name)

    def plot_loss_accuracy(self):
        plot_accuracy_loss_graph(self.history)