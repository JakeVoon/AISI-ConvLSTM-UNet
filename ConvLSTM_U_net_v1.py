from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Reshape, Dropout
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *
import numpy as np
import SimpleITK as sitk
import nibabel as nib

Params = {
    'img_row': 256,
    'img_cols': 256,
    'img_channel': 3,
    'lr': 1e-4,
    'filter_size': 64
}

def ConvLSTM_U_net(n_class = 4, input_size = (Params.row, Params.cols, Params.img_channel)):
    N = Params.row
    # Converting input to readable type
    inputs = Input(input_size) 

    ### DownSampling Path ###

    # 1st dense block with 2 layers of convolution followed by maxpooling
    conv1 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1) 
    conv1 = Dropout(0.5)(conv1) # passing to the skipping connection
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # passing to the next block

    # 2nd dense block with 2 layers of convolution followed by maxpooling
    conv2 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2) 
    conv2 = Dropout(0.5)(conv2) # passing to the skipping connection
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # 3rd dense block with 2 layers of convolution followed by maxpooling
    conv3 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3) # passing to the skipping connection
    conv3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    ### Dense Net (No max pooling in this block) ###
    
    # Dense block 1
    conv4 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    conv4_1 = Dropout(0.5)(conv4_1)
    # Dense block 2
    conv4_2 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_1)     
    conv4_2 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    # Dense block 3
    merge_dense = concatenate([conv4_2,conv4_1], axis = 3)
    conv4_3 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = Conv2D(Params.filter_size * 8, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    conv4_3 = Dropout(0.5)(conv4_3)
    
    ### UpSampling Path ###

    # Upsample 1 with Conv2DTranspose
    up1 = Conv2DTranspose(Params.filter_size * 4, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv4_3)
    up1 = BatchNormalization(axis=3)(up1)
    up1 = Activation('relu')(up1)
    
    # Concatenate corresponding layer with upsampling (up 1 -> drop3 last block at contracting path)
    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), Params.filter_size * 4))(conv3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), Params.filter_size * 4))(up1)
    merge1  = concatenate([x1,x2], axis = 1) 
    merge1 = ConvLSTM2D(filters = Params.filter_size * 2, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge1)

    # Dense block 1 in upsampling path        
    conv6 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    conv6 = Conv2D(Params.filter_size * 4, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    # Upsample 2 with Conv2DTranspose
    up2 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up2 = BatchNormalization(axis=3)(up2)
    up2 = Activation('relu')(up2)

    # Concatenate corresponding layer with upsampling (up 2 -> conv2 2nd last block at contracting path)
    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), Params.filter_size * 2))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), Params.filter_size * 2))(up2)
    merge2  = concatenate([x1,x2], axis = 1) 
    merge2 = ConvLSTM2D(filters = Params.filter_size, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge2)

    # Dense block 2 in upsampling pat    
    conv7 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    conv7 = Conv2D(Params.filter_size * 2, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    # Upsample 3 with Conv2DTranspose
    up3 = Conv2DTranspose(Params.filter_size, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up3 = BatchNormalization(axis=3)(up3)
    up3 = Activation('relu')(up3)    

    # Concatenate corresponding layer with upsampling (up 3 -> conv1 1st block at contracting path)
    x1 = Reshape(target_shape=(1, N, N, Params.filter_size))(conv1)
    x2 = Reshape(target_shape=(1, N, N, Params.filter_size))(up3)
    merge3  = concatenate([x1,x2], axis = 1) 
    merge3 = ConvLSTM2D(filters = Params.filter_size* 0.5, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge3)    
    
    # Dense block 3 in upsampling pat
    conv8 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    conv8 = Conv2D(Params.filter_size, Params.img_channel, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    
    # Instead of sigmoid in the basic U-Net, as this is multi-level, it should be activated with softmax with n_classes output
    conv9 = Conv2D(n_class, 1, activation = 'softmax')(conv8)

    model = Model(inputs, conv9)
    model.compile(optimizer = Adam(lr = Params.lr), loss = 'binary_crossentropy', metrics = ['accuracy'])    

    return model
