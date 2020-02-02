import math
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, TensorBoard
from matplotlib import pyplot as plt
import keras
weight_path="{}_weights.best.hdf5".format('model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

def lr_decay(epoch):
  return 0.01 * math.pow(0.666, epoch)

callback_learning_rate = LearningRateScheduler(lr_decay, verbose=True)

class EarlyStop(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_loss')<0.05):
      print("\nReached 005%% value losse so cancelling training!")
      self.model.stop_training = True

early_stop = EarlyStop()

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.accuracy = []
        self.val_accuracy = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.accuracy.append(logs.get('accuracy'))
        self.val_accuracy.append(logs.get('val_accuracy'))
        self.i += 1

        clear_output(wait=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(self.x, self.accuracy, label="accuracy")
        ax1.plot(self.x, self.val_accuracy, label="val_accuracy")
        ax1.set_title('Model accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'], loc='upper left')

        ax2.plot(self.x, self.losses, label="loss")
        ax2.plot(self.x, self.val_losses, label="val_loss")
        ax2.set_title('Model loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'], loc='upper left')
        plt.show()

plot_losses = PlotLosses()

tensorboard_log = TensorBoard(log_dir="./logs")

callbacks_list = [checkpoint,callback_learning_rate, plot_losses]