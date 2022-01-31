from load_data import data_loader
from ConvLSTM_U_net_v1 import ConvLSTM_U_net
from Dense_U_net import Dense_U_net
from Unet import multi_unet_model
from tensorflow.python.client import device_lib
import tensorflow as tf
import numpy as np
from tensorflow.keras.metrics import MeanIoU
import keras.backend as K
import keras
import matplotlib.pyplot as plt
from evaluation import run

if __name__ == "__main__":
    # Uncomment the following lines if GPU is used
    # config = tf.compat.v1.ConfigProto()
    # config.gpu_options.allow_growth = True
    print(device_lib.list_local_devices())
    # Specify batch size and number of epochs and classes
    batch_size = 8
    nepoch = 2000
    n_class = 5
    # Get data and classes weights
    x_train, x_test , x_valid, y_train, y_test, y_valid, class_weights = data_loader(n_class)
    _, height, width, channel = x_train.shape
    print("class weights are: ", class_weights)
    # Build the model
    model_Unet = multi_unet_model(n_classes=n_class, IMG_HEIGHT=height, IMG_WIDTH= width, IMG_CHANNELS= channel)
    model_Dense_Unet = Dense_U_net(n_class=n_class, img_height=height, img_width= width, img_ch= channel)
    # Run the training and evaluate the performances
    Acc_test_Unet, Acc_valid_Unet, mean_IOU_Unet, IOU_Unet = run(model_Unet, 'Unet', x_train, x_test , x_valid, y_train, y_test, y_valid, batch_size, nepoch, class_weights, n_class)
    Acc_test_Dense_Unet, Acc_valid_Dense_Unet, mean_IOU_Dense_Unet, IOU_Dense_Unet = run(model_Dense_Unet, 'Dense_Unet', x_train, x_test , x_valid, y_train, y_test, y_valid, batch_size, nepoch, class_weights, n_class)
