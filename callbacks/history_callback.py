import sys
from keras.callbacks import Callback
sys.path.append("..") 
from data_utils import *

class HistoryCallback(Callback):

    def __init__(self, file_path):
        self.file_path = file_path
        self.headers = ["model", "loss", "acc", "val_loss", "val_acc", "epochs"]
        
    def on_epoch_end(self, epoch, logs={}):
        model_name = self.model.name
        loss = logs.get('loss')
        acc = logs.get('acc')
        val_loss = logs.get('val_loss')
        val_acc = logs.get('val_acc')
        data = [model_name, loss, acc, val_loss, val_acc, epoch + 1]
        write_csv_file(self.file_path, data, self.headers)

