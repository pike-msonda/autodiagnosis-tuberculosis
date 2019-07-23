from __future__ import division, print_function, absolute_import
import os
import csv
import cv2
import random
import numpy as np
from PIL import Image
import pickle
import warnings
from urllib.parse import urlparse
from urllib import request
from io import BytesIO
import matplotlib.pyplot as plt
import itertools
np.random.seed(1000)
_EPSILON = 1e-8

def to_categorical(y, nb_classes=None):
    if nb_classes:
        y = np.asarray(y, dtype='int32')
        if len(y.shape) > 2:
            print("Warning: data array ndim > 2")
        if len(y.shape) > 1:
            y = y.reshape(-1)
        Y = np.zeros((len(y), nb_classes))
        Y[np.arange(len(y)), y] = 1.
        return Y
    else:
        y = np.array(y)
        return (y[:, None] == np.unique(y)).astype(np.float32)


def load_image(in_image):
    # load image
    img = Image.open(in_image)
    return img

def resize_image(in_image, new_width, new_height, out_image=None,
                 resize_mode=Image.ANTIALIAS):
    img = in_image.resize((new_width, new_height), resize_mode)
    if out_image:
        img.save(out_image)
    return img

def convert_color(in_image, mode):
    return in_image.convert(mode)

def pil_to_nparray(pil_image):
    pil_image.load()
    return np.asarray(pil_image, dtype="float32")
    
def build_image_dataset_from_dir(directory,
                                 dataset_file="my_tflearn_dataset.pkl",
                                 resize=None, convert_to_color=None,
                                 filetypes=None, shuffle_data=False,
                                 categorical_Y=False):
    try:
        X, Y = pickle.load(open(dataset_file, 'rb'))
    except Exception:
        X, Y = image_dirs_to_samples(directory, resize, convert_to_color, filetypes)
        if categorical_Y:
            Y = to_categorical(Y, np.max(Y) + 1) # First class is '0'
        if shuffle_data:
            X, Y = shuffle(X, Y)
        pickle.dump((X, Y), open(dataset_file, 'wb'), protocol=4)
    return X, Y

def image_dirs_to_samples(directory, resize=None, convert_to_color=False,
                          filetypes=None):
    print("Starting to parse images...")
    if filetypes:
        if filetypes not in [list, tuple]: filetypes = list(filetypes)
    samples, targets = directory_to_samples(directory, flags=filetypes)
    for i, s in enumerate(samples):
        samples[i] = load_image(s)
        if resize:
            samples[i] = resize_image(samples[i], resize[0], resize[1])
        if convert_to_color:
            samples[i] = convert_color(samples[i],'RGB')
        samples[i] = pil_to_nparray(samples[i])
        # import pdb; pdb.set_trace()
        samples[i] /= 255
    print("Parsing Done!")
    return samples, targets

def shuffle(*arrs):
    arrs = list(arrs)
    for i, arr in enumerate(arrs):
        assert len(arrs[0]) == len(arrs[i])
        arrs[i] = np.array(arr)
    p = np.random.permutation(len(arrs[0]))
    return tuple(arr[p] for arr in arrs)

def directory_to_samples(directory, flags=None, filter_channel=False):
    samples = []
    targets = []
    label = 0
    try: # Python 2
        classes = sorted(os.walk(directory).next()[1])
    except Exception: # Python 3
        classes = sorted(os.walk(directory).__next__()[1])
    for c in classes:
        c_dir = os.path.join(directory, c)
        try: # Python 2
            walk = os.walk(c_dir).next()
        except Exception: # Python 3
            walk = os.walk(c_dir).__next__()
        for sample in walk[2]:
            if not flags or any(flag in sample for flag in flags):
                if filter_channel:
                    if get_img_channel(os.path.join(c_dir, sample)) != 3:
                        continue
                samples.append(os.path.join(c_dir, sample))
                targets.append(label)
        label += 1
    return samples, targets


def get_img_channel(image_path):
    img = load_image(image_path)
    img = pil_to_nparray(img)
    try:
        channel = img.shape[2]
    except:
        channel = 1
    return channel

def onehot_to_cat(y):
    return np.argmax(y, axis=1)

def store_model(model, path,filename):
    json_model = model.to_json()
    with open(path+filename+".json", 'w') as file:
        file.write(json_model)
    model.save_weights(path+filename+".h5")

def get_labels(y_onehot):
    y = onehot_to_cat(y_onehot)
    labels = np.empty(len(y), dtype=object)
    labels[y == 0 ] = "N"
    labels[y == 1 ] = "P"

    return labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def write_csv_file(file, data, headers):
    if not (os.path.exists(file)):
        #write to file with headers
        with open(file,'wt',newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(h for h in headers)
            writer.writerow(data)
    else:
        with open(file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(data)

