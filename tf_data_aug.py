import os
import cv2 as openCv
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg 
from PIL import Image
IMAGE_SIZE = 256

CROP_SIZE = 227

IMAGE_PATH = ''

SEED = 1000

IMAGE_PATH='../data' # Store the transformed image into the project folder
IMAGE_PATH="../Data" #  Folder containing all the image to augment.
AUG_PATH = 'data/aug'

def resize_images(filepath, width=256, height=256):
    resized_images = []
    tf.reset_default_graph()
    imagePath = tf.placeholder(tf.string, name="inputFile")
    imagePlaceholder = tf.io.decode_png(tf.read_file(imagePath), dtype=tf.dtypes.uint8)
    resized_binary = tf.image.resize_images(imagePlaceholder, size=[IMAGE_SIZE,IMAGE_SIZE], method=tf.image.ResizeMethod.BILINEAR)
    images = [i for i in os.listdir(os.path.join(filepath)) if i.endswith('.png')]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for image in images:
            img = sess.run(resized_binary, feed_dict = {imagePath: os.path.join(filepath, image)})
            resized_images.append(img)
    np.array(resized_images, dtype = np.uint8)
    return resized_images

def read_images(folder=None):
    x_images = []
    images = [i for i in os.listdir(os.path.join(folder)) if i.endswith('.png')]
    for image in images:
        img = mpimg.imread(os.path.join(folder, image))
        if(img.shape[2] > 3):
            img = img[:, :, :3]
        x_images.append(img)
    return x_images

def save_images(filepath, images, prefix="untitled"):
    for index, image in enumerate(images):
        filename = filepath+'/'+prefix+'_'+str(index)+'.png'
        Image.fromarray(image, mode='RGB').save(filename)
        # import pdb; pdb.set_trace()
        # imageToSave = Image.fromarray(image)
        mpimg.imsave(filename, image)

def rotate_images(images):
    X_rotate = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    k = tf.placeholder(tf.int32)
    tf_img = tf.image.rot90(X, k = k)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in images:
            X_rotate.append(img) #append original image
            for i in range(3):  # Rotation at 90, 180 and 270 degrees
                rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
                X_rotate.append(rotated_img)
        
    X_rotate = np.array(X_rotate, dtype =np.uint8)
    return X_rotate
    
def flip_images(X_imgs):
    X_flip = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    tf_img1 = tf.image.flip_left_right(X)
    tf_img2 = tf.image.flip_up_down(X)
    tf_img3 = tf.image.transpose_image(X)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in X_imgs:
            flipped_imgs = sess.run([tf_img1, tf_img2, tf_img3], feed_dict = {X: img})
            X_flip.extend(flipped_imgs)
    X_flip = np.array(X_flip, dtype = np.uint8)
    return X_flip

def random_crop(images, samples=2):
    x_random_crops = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE, 3))
    
    tf_cache = tf.image.random_crop(X, [CROP_SIZE, CROP_SIZE,3], SEED)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for img in images:
            for _ in range(samples):
                random_cropped_image = sess.run(tf_cache, feed_dict = {X: img})
                x_random_crops.append(random_cropped_image)

    x_random_crops = np.array(x_random_crops, dtype= np.uint8)

    return x_random_crops

def add_augs():
    for parentdir in os.listdir(IMAGE_PATH):
        print("Reading sub-folders in {0} ".format(parentdir))
        for subdir in os.listdir(os.path.join(IMAGE_PATH, parentdir)):
            print("Reading sub-folders in {0} ".format(subdir))

            images = resize_images(os.path.join(IMAGE_PATH, parentdir, subdir))
            print("{} will be rotated and flipped".format(len(images)))
            rotated_images = rotate_images(images)
            print("Rotated {}".format(len(rotated_images)))
            flipped_images = flip_images(rotated_images)
            print("Flipped  {}".format(len(flipped_images)))

            cropped_images = random_crop(flipped_images)
            print("Cropped  {}".format(len(cropped_images)))
            # import pdb; pdb.set_trace()
            save_images(filepath='/'.join([AUG_PATH, 'all', parentdir, subdir]), images=cropped_images, prefix="cropped")
        
def create_dataset():
     for parentdir in os.listdir(AUG_PATH):
        if(parentdir == 'all'):
            pass
        else:
            print("Reading sub-folders in {0} ".format(parentdir))
            for subdir in os.listdir(os.path.join(AUG_PATH, parentdir)):
                print("Reading sub-folders in {0} ".format(subdir))
                images =  read_images(folder=os.path.join(AUG_PATH, parentdir, subdir))
                cropped_images = random_crop(images)
                print("Cropped  {}".format(len(cropped_images)))
                save_images(filepath='/'.join([AUG_PATH, 'all', parentdir, subdir]), images=cropped_images, prefix="cropped")

if __name__ == "__main__":
    add_augs()
    # create_dataset()
    
                
                
