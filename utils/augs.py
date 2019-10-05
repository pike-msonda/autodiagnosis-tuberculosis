import cv2
import random;
random.seed(1000)
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,RandomRotate90,VerticalFlip,RandomCrop,
    ToFloat, ShiftScaleRotate, GaussianBlur
)

AUGMENTATIONS_TRAIN = Compose([
    HorizontalFlip(p=1),
    VerticalFlip(p=0.5), 
    # RandomContrast(limit=0.2, p=0.5),
    RandomRotate90(p=0.5),
    RandomGamma(gamma_limit=(80, 120), p=0.5),
    # RandomBrightness(limit=0.2, p=0.5),
    # HueSaturationValue(hue_shift_limit=5, sat_shift_limit=20,
    #                    val_shift_limit=10, p=.9),
    # CLAHE(p=1.0, clip_limit=2.0),
    ShiftScaleRotate(
        shift_limit=0.0625, scale_limit=0.1, 
        rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.5), 
    # RandomCrop(227, 227, p=1.0),
    # ToFloat(max_value=255)
])

AUGMENTATIONS_TEST = Compose([
    # CLAHE(p=1.0, clip_limit=2.0),
    # RandomCrop(227, 227, p=1.0),
    # ToFloat(max_value=255,p=1)
])