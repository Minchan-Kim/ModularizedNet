import tensorflow as tf
import pandas as pd
import network as nn
import dataset as ds
import argparse
import os
import wandb


parser = argparse.ArgumentParser()
parser.add_argument('-profile', action = 'store_true')
parser.add_argument('-use_gpu', action = 'store_true')
parser.add_argument('-use_wandb', action = 'store_true')
parser.add_argument('--name', type = str, default = '')
parser.add_argument('--epochs', type = int, default = 100)
parser.add_argument('--timesteps', type = int, default = 5)
parser.add_argument('--regularization_factor', type = float, default = 0.000004)
parser.add_argument('--dropout_rate', type = float, default = 0.0)
parser.add_argument('--learning_rate', type = float, default = 0.00001)
parser.add_argument('--buffer_size', type = int, default = 20000)
parser.add_argument('--batch_size', type = int, default = 1000)
args = parser.parse_args()

profile = args.profile
use_gpu = args.use_gpu
use_wandb = args.use_wandb
name = args.name
epochs = args.epochs
timesteps = args.timesteps
regularization_factor = args.regularization_factor
dropout_rate = args.dropout_rate
learning_rate = args.learning_rate
buffer_size = args.buffer_size
batch_size = args.batch_size

if not tf.__version__.startswith('2'):
    print('Use TensorFlow 2.X!')
    sys.exit(0)

if use_gpu is True:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = nn.ModularizedNet(timesteps = timesteps, num_joint = 6, num_data_type = 3, dropout_rate = dropout_rate, regularization_factor = regularization_factor, learning_rate = learning_rate)

dataset = ds.Dataset('/home/dyros/mc_ws/data', 65, 20000, 1000, num_parallel_calls = 1, drop_remainder = True)
validation_dataset = pd.read_csv('/home/dyros/mc_ws/data/ValidationData5Cut.csv', header = None).to_numpy()
validation_dataset = (validation_dataset[:, :65], validation_dataset[:, 65:])
#validation_dataset = ds.Dataset("", 100, buffer_size, batch_size)

class MetricsLog(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = None):
        if use_wandb == True:
            wandb_dict = dict()
            wandb_dict['Training Accuracy'] = logs.get('acc')
            wandb_dict['Validation Accuracy'] = logs.get('val_acc')
            wandb_dict['Training Cost'] =  logs.get('loss')
            wandb_dict['Validation Cost'] = logs.get('val_loss')
            wandb.log(wandb_dict)

callbacks = []

model.fit(x = dataset, epochs = epochs, callbacks = callbacks, validation_data = validation_dataset)