import os
from keras.optimizers import SGD
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_curve, roc_auc_score

MODELS_PATH='../models'

MODELS_SPP = [
    'alexnet_sppchina',
    'alexnet_sppturkey',
    'alexnet_sppusa',
    'googlenet-v1-sppchina',
    'googlenet-v1-sppturkey',
    'googlenet-v1-sppusa',
    'resnet50_sppchina',
    'resnet50_sppturkey',
    'resnet50_sppusa'
]

MODELS = [
    'AlexNetchina',
    'AlexNetturkey',
    'AlexNetusa',
    'googlenet-v1china',
    'googlenet-v1turkey',
    'googlenet-v1usa',
    'resnet50china',
    'resnet50turkery'
    'resnet50usa'
]
def main():
    models =  os.listdir(MODELS_PATH);
    self.model.load_weights('../models/'+self.model.name+FOLDER+'.h5') 
if __name__ == "__main__":
    pass