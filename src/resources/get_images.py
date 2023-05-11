from wand.image import Image
import gdown
import zipfile
import glob
import cv2
import numpy as np
import os
from numpy import asarray

os.mkdir("images_txt")
#download files from google drive
url = "https://drive.google.com/file/d/10x89SV_YlatdYSmY8DKSGAecRu55rfsW/view?usp=sharing"
output = 'images_files.zip'
gdown.download(url,output,quiet=False, fuzzy=True)

with zipfile.ZipFile('images_files.zip', 'r') as zip_ref:
    zip_ref.extractall("images_txt")

arr = []
arr = np.array(arr)
files = os.listdir('images_txt')
file_num = 1
for name in files:
    file = open("images_txt/images{0}.txt".format(file_num), "r")
    content = file.read()
    file.close()
    arr = np.append(arr,content)
    print(content)
    file_num += 1
print(arr.shape)


