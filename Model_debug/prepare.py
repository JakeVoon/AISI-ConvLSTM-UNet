from __future__ import division
import numpy as np
import scipy.io as sio
import scipy.misc as sc
import glob
import imageio
from skimage.transform import resize


# Parameters
height = 256
width  = 256
channels = 3

############################################################# Prepare ISIC 2018 data set #################################################
Dataset_add = 'Glenda_v1.5_classes/'
Tr_add = 'frames'

Tr_list = glob.glob(Dataset_add+ Tr_add+'/*.jpg')
# It contains 2594 training samples
Data_train    = np.zeros([373, height, width, channels])
Label_train   = np.zeros([373, height, width, channels])

print('Reading Glenda_v1.5')
for idx in range(len(Tr_list)):
    print(idx+1)
    img = imageio.imread(Tr_list[idx])
    img = np.double(resize(img, output_shape=(height, width, channels)))
    Data_train[idx, :,:,:] = img

    b = Tr_list[idx]    
    a = b[0:len(Dataset_add)]
    b = b[27: len(b)-4] 
    add = (a+ 'annots/' + b +'.png')
    img2 = imageio.imread(add)
    # img2 = np.double(resize(img2, output_shape=(height, width)).reshape(1,(256*256)))
    img2 = np.double(resize(img2, output_shape=(height, width, channels)))
    # img2 = sc.imread(add)
    # img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
    Label_train[idx, :, :, :] = img2    
         
print('Reading ISIC 2018 finished')

################################################################ Make the train and test sets ########################################    
# We consider 1815 samples for training, 259 samples for validation and 520 samples for testing

Train_img      = Data_train[:150,:,:,:]
Validation_img = Data_train[150:150+73,:,:,:]
Test_img       = Data_train[150+73:,:,:,:]

Train_mask      = Label_train[:150,:,:]
Validation_mask = Label_train[150:150+73,:,:]
Test_mask       = Label_train[150+73,:,:]


np.save('data_train', Train_img)
np.save('data_test' , Test_img)
np.save('data_val'  , Validation_img)

np.save('mask_train', Train_mask)
np.save('mask_test' , Test_mask)
np.save('mask_val'  , Validation_mask)
