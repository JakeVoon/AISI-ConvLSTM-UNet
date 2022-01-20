"""
This file contains the methods used to download and process the data before using 

Before importing, in a bash terminal do : conda install cython
"""

from zipfile import ZipFile
import json

def load_zip():
    file_name = "Glenda_v1.5_classes.zip"

    with ZipFile(file_name,'r') as zip:
        zip.printdir()

        print('\nExtracting all files from', file_name,'...')
        zip.extractall()
        print('Finished extracting!\n')

def load_json():
    
    
