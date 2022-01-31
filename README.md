# ConvLSTM_UNet

First, the `wget` library needs to be installed using a bash terminal:
```
pip install wget
```
To download and extract the dataset, in a bash terminal do:
```
python load_data.py
```
To run the whole segmentation, create a bash terminal, go to the folder and run the command:
```
python main.py
```
'n_class', 'nepoch' and 'batch' in mian.py can be adjusted.
'size', 'step', 'thresh' of extract_img_masks() function in load_data.py can be adjusted.
If GPU is used, uncomment line 16 and line 17 in main.py.
The 'color' in label_msk() function and the 'height' and 'weight' in extract_img_masks() function might be manually defined due to the device limit. 
