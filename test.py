import os
import cv2 as openCv
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt
from PIL import Image

def read_images(folder=None):
    x_images = []
    images = [i for i in os.listdir(os.path.join(folder)) if i.endswith('.png')]
    for image in images:
        img = mpimg.imread(os.path.join(folder, image))
        if(img.shape[2] > 3):
            img = img[:, :, :3]
        x_images.append(img)
    return x_images

def random_crop(images, samples=2):
    x_random_crops = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (256, 256, 3))
    crops = []
    tf_cache = tf.image.random_crop(X, [227, 227,3], 1000)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for _ in range(samples):
            random_cropped_image = sess.run(tf_cache, feed_dict = {X: img})
            crops.append(random_cropped_image)
    x_random_crops = np.array(crops, dtype= np.float32)
    return x_random_crops

if __name__ == "__main__":
    img = mpimg.imread(os.path.join('test.png'))[:, :, :3]
    crops = random_crop(img, 10)
    first_crop = crops[0]
    second_crop = crops[1]
    for index, img in enumerate(crops):
        mpimg.imsave('test_'+str(index), img)

