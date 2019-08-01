import os
import cv2 as openCv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Resize
)
import numpy as np
IMAGE_SIZE = 256

CROP_SIZE = 227

IMAGE_PATH = ''

SEED = 1000

IMAGE_PATH='../data' # Store the transformed image into the project folder
IMAGE_PATH="D:\Data" #  Folder containing all the image to augment.
AUG_PATH = 'data/aug'

def read_images(filepath, width=256, height=256):
    X_images  = []
    images = [i for i in os.listdir(os.path.join(filepath)) if i.endswith('.png')]
    for image in images:
        img = openCv.imread(os.path.join(filepath, image))
        X_images.append(img)
    return X_images
    
def resize(images):
    X_resized=[]
    for img in images:
        X_resized.append(openCv.resize(img, (IMAGE_SIZE, IMAGE_SIZE)))
    return X_resized

def resize(images):
    X_resize = []
    resize = Resize(IMAGE_SIZE, IMAGE_SIZE, always_apply=True)
    for img in images:
        X_resize.append(resize(image=img['image']))
    return X_resize

def applyClahe(images):
    X_clahe = []
    clahe = CLAHE(clip_limit=2, always_apply=True)
    for img in images:
        X_clahe.append(clahe(image=img))
    return X_clahe
if __name__ == "__main__":

     for parentdir in os.listdir(IMAGE_PATH):
        print("Reading sub-folders in {0} ".format(parentdir))
        for subdir in os.listdir(os.path.join(IMAGE_PATH, parentdir)):
            print("Reading sub-folders in {0} ".format(subdir))
            images = read_images(os.path.join(IMAGE_PATH, parentdir, subdir))
            claheImages = applyClahe(images)
            resized_images = resize(claheImages)
            import pdb; pdb.set_trace()
           
