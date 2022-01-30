from cmath import inf
from zipfile import ZipFile
import wget
import os
import json
import glob
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os 


def download_zip():
    print('Downloading zip files...')

    url1 = 'https://zenodo.org/record/4570965/files/Glenda_v1.5_classes.zip?download=1'
    url2 = 'https://zenodo.org/record/4570965/files/GLENDA_v1.5_no_pathology.zip?download=1'

    wget.download(url1)
    wget.download(url2)

    print('Finished downaloading!')

def extract_zip():
    file_name = "Glenda_v1.5_classes.zip"

    with ZipFile(file_name,'r') as zip:
        zip.printdir()

        print('\nExtracting all files from', file_name,'...')
        zip.extractall()
        print('Finished extracting!\n')
    
    file_name = "GLENDA_v1.5_no_pathology.zip"

    with ZipFile(file_name,'r') as zip:
        zip.printdir()

        print('\nExtracting all files from', file_name,'...')
        zip.extractall()
        print('Finished extracting!\n')

def delete_zip():
    print('Deleting .zip files...')

    os.remove('Glenda_v1.5_classes.zip')
    os.remove('GLENDA_v1.5_no_pathology.zip')

    print('Finished deleting!')

def find_common_image_dim():
    coco = json.load(open('Glenda_v1.5_classes/coco.json'))
    wxh_list = []
    label = []
    # record all image dimensions and find the most common image dimension
    for dic in coco["annotations"]:
        width = dic["width"]
        height = dic["height"]
        label.append(int(dic["category_id"])-1) # converting to label starting from 0
        wxh_list.append('{0:d}x{1:d}'.format(width,height))
    wxh = max(set(wxh_list), key=wxh_list.count).split('x')
    width, height = wxh[0], wxh[1]

    return int(width), int(height)


# def get_color():
#     for directory_path in glob.glob('Glenda_v1.5_classes/annots/'):
#         for img_path in glob.glob(os.path.join(directory_path, "*.png")):
#             print(img_path)


# def extract_img_masks(width, height):
#     # height = 64
#     # width = 64
#     # Define the output size
#     size = 64
#     # The step length of scanning 
#     step = 16
#     # Threshold for the proportion of background
#     thresh = 0.5
#     #Capture images 
#     all_image = []
#     image = []
#     mask = []
#     """
#     Images and masks are aligned in order so we do not have to use coco.json to find the corresponding annotations
#     """
#     # Extract all images, read it wwith colour scale and resize it -> convert to np.array
#     for directory_path in glob.glob('Glenda_v1.5_classes/frames/'):
#         for img_path in glob.glob(os.path.join(directory_path, "*.jpg")):
#             img = cv2.imread(img_path, 0) # pass 0 for grayscale, in this case 1 for colour images       
#             img = cv2.resize(img, (height,width)) # Resizing all images to the most common dimension
#             all_image.append(img)
#     all_image = np.array(all_image)

#     # Extract all masks, read it wwith colour scale and resize it -> convert to np.array
#     for directory_path in glob.glob('Glenda_v1.5_classes/annots/'):
#         idx = 0
#         for img_path in glob.glob(os.path.join(directory_path, "*.png")):
#             msk = cv2.imread(img_path, 0) # pass 0 for grayscale, in this case 1 for colour images       
#             msk = cv2.resize(msk, (height,width)) # Resizing all images to the most common dimension
#             #msk = cv2.cvtColor(msk, cv2.COLOR_BGR2RGB)
#             # Use a size * size kernel to scann the image with the step length
#             for i in range((width-size)//step):
#                 for j in range((height-size)//step):
#                     # Count the number of the pixel with 0 value
#                     count_0 = 0
#                     for m in range(i*step,(i*step+size)):
#                         for n in range(j*step,(j*step+size)):
#                             if msk[m,n] == 0:
#                                 count_0+=1
#                     # If the 0 color ranks less than threshold, save the area
#                     if count_0/(size*size) <= thresh:
#                         mask.append(msk[i*step:i*step+size, j*step:j*step+size])
#                         image.append(all_image[idx, i*step:i*step+size, j*step:j*step+size])
#             # mask.append(msk)
#             idx += 1
#     mask = np.array(mask)
#     image = np.array(image)
#     return image, mask




def extract_img_masks(width, height):
    height = 128
    width = 128
    #Capture images 
    image = []
    mask = []
    """
    Images and masks are aligned in order so we do not have to use coco.json to find the corresponding annotations
    """
    # Extract all images, read it wwith colour scale and resize it -> convert to np.array

    for directory_path in glob.glob('Glenda_v1.5_classes/frames/'):
        path = glob.glob(os.path.join(directory_path, "*.jpg"))
        path.sort()
        id = path
        for img_path in path:
            img = cv2.imread(img_path, 0) # pass 0 for grayscale, in this case 1 for colour images       
            img = cv2.resize(img, (height, width)) # Resizing all images to the most common dimension
            # Enhance contrast
            img = cv2.equalizeHist(img)
            image.append(img)
    image = np.array(image)

    # Extract all masks, read it wwith colour scale and resize it -> convert to np.array
    for directory_path in glob.glob('Glenda_v1.5_classes/annots/'):
        path = glob.glob(os.path.join(directory_path, "*.png"))
        path.sort()
        for img_path in path:
            msk = cv2.imread(img_path, 0) # pass 0 for grayscale, in this case 1 for colour images       
            msk = cv2.resize(msk, (height, width)) # Resizing all images to the most common dimension
            mask.append(msk)
    mask = np.array(mask)

    return image, mask, id

def label_msk(mask, n_classes, id):
    # mask have different color in the boundary making the pixel label not consistent, thus, by selecting the top n_classes
    # frequent color as the standard and eliminates all other color in the mask. 
    color = [0]
    mask_ID = ['Glenda_v1.5_classes/frames\\c_1_v_(video_17.mp4)_f_925.jpg', 'Glenda_v1.5_classes/frames\\c_1_v_(video_17.mp4)_f_1232.jpg', 'Glenda_v1.5_classes/frames\\c_7_v_(video_130.mp4)_f_1233.jpg', 'Glenda_v1.5_classes/frames\\c_8_v_(video_145.mp4)_f_496.jpg']
    for idx in mask_ID:
        i = id.index(idx)
        plt.imshow(mask[i])
        plt.title('{0:s}'.format(idx)) #Give this plot a title, 
                                #so I know it's from matplotlib and not cv2
        plt.show()
        unique, counts = np.unique(mask[i], return_counts=True)
        unique = np.delete(unique, 0)
        counts = np.delete(counts, 0)
        print(unique, counts)
        i = np.argmax(counts)
        color.append(unique[i])
        

    num, height, width = mask.shape
    for n in range(num):
        for h in range(height):
            for w in range(width):
                if mask[n,h,w] not in color:
                    mask[n,h,w] = 0
    labelencoder = LabelEncoder()
    mask = mask.reshape(-1,1)
    mask = labelencoder.fit_transform(mask)
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight('balanced',
                                                    np.unique(mask),
                                                    mask)
    #class_weights = {i : class_weights[i] for i in range(5)}
    print("Class weights are...:", class_weights)
    mask = mask.reshape(num, height, width)

    return mask, class_weights

def split_data(image, mask):
    x1, x_test, y1, y_test = train_test_split(image, mask, test_size = 0.10, random_state = 0)

    #Further split training data t a smaller subset for quick testing of models
    x_train, x_valid, y_train, y_valid = train_test_split(x1, y1, test_size = 0.2, random_state = 0)

    return x_train, x_test , x_valid, y_train, y_test, y_valid

def data_loader(n_classes = 5):
    # check if directory exists
    path_Glenda = os.path.isdir('Glenda_v1.5_classes')
    path_pathology = os.path.isdir('no_pathology')
    # if either one is not existed, dowwnload the file
    if (not path_Glenda) or (not path_pathology):
        download_zip()
        extract_zip()
        delete_zip()
    else:
        print('file exists')
    
    # find most common image dimensions
    width, height= find_common_image_dim()
    image, mask, id = extract_img_masks(width, height)
    mask, class_weights = label_msk(mask, n_classes, id)
    image = np.expand_dims(image, axis=3) # expanding to fit the input of model
    image = image/255 # making the value within the image from 0~1
    mask = np.expand_dims(mask, axis=3)
    x_train, x_test , x_valid, y_train, y_test, y_valid = split_data(image, mask)
    # Before this, each pixel has a label of 0~4/ after to categorical, it would be come one-hot vector with length of n_classes
    y_train = to_categorical(y_train, n_classes).reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
    y_test = to_categorical(y_test, n_classes).reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
    y_valid = to_categorical(y_valid, n_classes).reshape((y_valid.shape[0], y_valid.shape[1], y_valid.shape[2], n_classes))
    
    return x_train, x_test , x_valid, y_train, y_test, y_valid, class_weights

if __name__ == "__main__":
    data_loader()
    
