import numpy as np
import cv2

"""
This code is highly influenced by the implementation of:
https://github.com/joelthchao/tensorflow-finetune-flickr-style/dataset.py
But changed abit to allow dataaugmentation (yet only horizontal flip) and 
shuffling of the data. 
The other source of inspiration is the ImageDataGenerator by @fchollet in the 
Keras library. But as I needed BGR color format for fine-tuneing AlexNet I 
wrote my own little generator.
"""

class ImageDataGenerator:
    def __init__(self, x, y, horizontal_flip=False, shuffle=False, 
                 mean = np.array([104., 117., 124.]), scale_size=(227, 227),
                 nb_classes = 2):
        
                
        # Init params
        self.horizontal_flip = horizontal_flip
        self.n_classes = nb_classes
        self.shuffle = shuffle
        self.mean = mean
        self.scale_size = scale_size
        self.pointer = 0
        
        self.read_class_list(x, y)
        
        if self.shuffle:
            self.shuffle_data()

    def read_class_list(self,x,y):
        """
        Scan the image file and get the image paths and labels
        """
        self.images = x
        self.labels = y
        self.data_size = len(y)
        
    def shuffle_data(self):
        """
        Random shuffle the images and labels
        """
        images = self.images.copy()
        labels = self.labels.copy()
        self.images = []
        self.labels = []
        
        #create list of permutated index and shuffle data accoding to list
        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])
                
    def reset_pointer(self):
        """
        reset pointer to begin of the list
        """
        self.pointer = 0
        
        if self.shuffle:
            self.shuffle_data()
        
    
    def next_batch(self, batch_size):
        """
        This function gets the next n ( = batch_size) images from the path list
        and labels and loads the images into them into memory 
        """
        # Get next batch of image (path) and labels
        paths = self.images[self.pointer:self.pointer + batch_size]
        labels = self.labels[self.pointer:self.pointer + batch_size]
        
        #update pointer
        self.pointer += batch_size
        
        # Read images
        images = np.ndarray([batch_size, self.scale_size[0], self.scale_size[1], 3])
        for i in range(len(paths)):
                                                                 
            images[i] = paths[i]


        #return array of images and labels
        return images, labels