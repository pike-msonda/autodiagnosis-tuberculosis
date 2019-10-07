import os
import cv2 as openCv
import tensorflow as tf
import numpy as np
import matplotlib.image as mpimg 
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image
IMAGE_SIZE = 256

CROP_SIZE = 227

IMAGE_PATH = ''

SEED = 1000

FOLDER = 'train'
AUG_PATH='data' # Store the transformed image into the project folder
IMAGE_PATH="E:\Pike\Data/"+FOLDER #  Folder containing all the image to augment.

def resize_images(filepath, width=256, height=256):
    resized_images = []
    images = [i for i in os.listdir(os.path.join(filepath)) if i.endswith('.png')]
    for image in images:
        img = openCv.imread(os.path.join(filepath, image))
        # imgClahe = applyClahe(img)
        resize_image = openCv.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
        resized_images.append(resize_image)
    np.array(resized_images, dtype ="float") / 255.0
    return resized_images


def save_images(filepath, images, prefix="untitled"):
    for index, image in enumerate(images):
        filename = filepath+'/'+prefix+'_'+str(index)+'.png'
        Image.fromarray(image, mode='RGB').save(filename)
        # import pdb; pdb.set_trace()
        # imageToSave = Image.fromarray(image)

def applyClahe(image):
    clahe = openCv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)  

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


def generate_images(images, total=10, save_dir=None):
    aug = ImageDataGenerator(
	# rotation_range=90,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
    count  = 0
    expanded_images = []

    for im in images:
        im = np.expand_dims(im, axis=0)
        count = 0
        for batch in aug.flow(im, batch_size=1, save_to_dir=save_dir, save_prefix="image", save_format="png"):
            count += 1
            if count > total:
                break

def random_crop(images, samples=2):
    x_random_crops = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.uint8, shape = (IMAGE_SIZE, IMAGE_SIZE,3))
    
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
            
            print("{0} Resized to ({1}, {2}".format(len(images), IMAGE_SIZE, IMAGE_SIZE))
            
            # rotated_images=rotate_images(images)
            # print("{0} Images Rotated".format(len(rotated_images)))

            # generate_images(images=images,total=10, save_dir='/'.join([AUG_PATH, FOLDER, parentdir, subdir]))
            # flipped_images = flip_images(rotated_images)
            cropped = random_crop(images, 3)
            print("Cropped {}".format(len(cropped)))
            
            # aug_images =  np.concatenate((cropped, rotated_images))
            # print("Total auged images {}".format(len(aug_images)))
            
            save_images(filepath='/'.join([AUG_PATH, FOLDER, parentdir, subdir]), images=cropped, prefix="im")
        
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
    
                
                
