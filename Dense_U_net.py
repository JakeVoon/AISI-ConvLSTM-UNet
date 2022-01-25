from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Dropout, UpSampling2D
from tensorflow.keras.optimizers import Adam
from keras.utils.vis_utils import plot_model as plot
from keras.layers import *
import numpy as np


class Params():
    def __init__(self) -> None:
        self.filter_size = 64
        self.img_row = 360
        self.img_cols = 640
        self.img_channel = (3,3)
        self.lr = 1e-4

Params = Params()

def Dense_U_net(n_class = 5, img_height = Params.img_row, img_width = Params.img_cols, img_ch = 1):
    N = Params.img_row
    # Converting input to readable type
    inputs = Input((img_height, img_width, img_ch)) 

    # Contracting Path (with max pooling)#
    # Dense block 1
    conv1_1 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1_2 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1_1) 
    drop1_2 = Dropout(0.5)(conv1_2)
    Merge1 = concatenate([conv1_1,drop1_2], axis = 3) # Saved for passing to the skipping connections
    pool1 = MaxPooling2D(pool_size=(2, 2))(Merge1) # Passed to the next dense block

    # Dense block 2
    conv2_1 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2_2 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2_1)
    drop2_2 = Dropout(0.5)(conv2_2)
    Merge2 = concatenate([conv2_1,drop2_2], axis = 3) # Saved for passing to the skipping connections
    pool2 = MaxPooling2D(pool_size=(2, 2))(Merge2) # Passed to the next dense block

    # Dense block 3
    conv3_1 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3_2 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3_1)
    drop3_2 = Dropout(0.5)(conv3_2)
    Merge3 = concatenate([conv3_1,drop3_2], axis = 3) # Saved for passing to the skipping connections
    pool3 = MaxPooling2D(pool_size=(2, 2))(Merge3) # Passed to the next dense block

    # Dense block 4
    conv4_1 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4_2 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_1)
    drop4_2 = Dropout(0.5)(conv4_2)
    Merge4 = concatenate([conv4_1,drop4_2], axis = 3) # Saved for passing to the skipping connections
    pool4 = MaxPooling2D(pool_size=(2, 2))(Merge4) # Passed to the next dense block

    # Final Dense block 5 with no max pooling, ready to feed to upsampling path
    conv5_1 = Conv2D(Params.filter_size * 16, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5_2 = Conv2D(Params.filter_size * 16, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5_1)
    drop5_2 = Dropout(0.5)(conv5_2)
    Merge5 = concatenate([conv5_1,drop5_2], axis = 3) # Saved for passing to the skipping connections

    # UpSampling Path #

    # Dense Up 1
    up6 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge5))
    Merge6 = concatenate([up6, Merge4], axis = 3)
    Merge6 = BatchNormalization(axis=3)(Merge6)
    Merge6 = Activation('relu')(Merge6)
    conv6_1 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge6)
    conv6_2 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge6)
    drop6_2 = Dropout(0.5)(conv6_2)
    Merge6 = concatenate([conv6_1, drop6_2], axis = 3)

    # Dense Up 2
    up7 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge6))
    Merge7 = concatenate([up7, Merge3], axis = 3)
    Merge7 = BatchNormalization(axis=3)(Merge7)
    Merge7 = Activation('relu')(Merge7)
    conv7_1 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge7)
    conv7_2 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge7)
    drop7_2 = Dropout(0.5)(conv7_2)
    Merge7 = concatenate([conv7_1, drop7_2], axis = 3)

    # Dense Up 3
    up8 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge7))
    Merge8 = concatenate([up8, Merge2], axis = 3)
    Merge8 = BatchNormalization(axis=3)(Merge8)
    Merge8 = Activation('relu')(Merge8)
    conv8_1 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge8)
    conv8_2 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge8)
    drop8_2 = Dropout(0.5)(conv8_2)
    Merge8 = concatenate([conv8_1, drop8_2], axis = 3)

    # Dense Up 4
    up9 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(Merge8))
    Merge9 = concatenate([up9, Merge1], axis = 3)
    Merge9 = BatchNormalization(axis=3)(Merge9)
    Merge9 = Activation('relu')(Merge9)
    conv9_1 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge9)
    conv9_2 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge9)
    drop9_2 = Dropout(0.5)(conv9_2)
    Merge9 = concatenate([conv9_1, drop9_2], axis = 3)

    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Merge9)
    conv10 = Conv2D(n_class, 1, activation = 'softmax')(conv9)

    model = Model(inputs, conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
            
    return model

    # Example of how to use returned model in the main file
    """
    history = model.fit(X_train, y_train_cat, 
                    batch_size = 16, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test_cat), 
                    #class_weight=class_weights,
                    shuffle=False)

    model.save('test.hdf5')
    #model.save('sandstone_50_epochs_catXentropy_acc_with_weights.hdf5')
    ############################################################
    #Evaluate the model
    _, acc = model.evaluate(X_test, y_test_cat)
    print("Accuracy is = ", (acc * 100.0), "%")
    """
