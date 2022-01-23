from zipfile import ZipFile
import wget
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


if __name__ == "__main__":
    download_zip()
    extract_zip()
    delete_zip()
    
