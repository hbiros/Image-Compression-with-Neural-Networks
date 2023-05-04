from wand.image import Image
import gdown
import zipfile
import glob
import cv2
import numpy as np
import os
from numpy import asarray

#download files from google drive
url = "https://drive.google.com/file/d/16250k3Ju0Eu14ZcCP3QrswK5shNDcCng/view?usp=sharing"
output = 'images.zip'
gdown.download(url,output,quiet=False, fuzzy=True)

with zipfile.ZipFile('images.zip', 'r') as zip_ref:
    zip_ref.extractall("")

#split images into 64x64 tiles
os.mkdir("output")
print(os.listdir('images'))
files = os.listdir('images')
file_num = 0
for name in files:
    file_num +=1
    os.mkdir("output/image{0}".format(file_num))
    with Image(filename="images/{0}".format(name)) as image:
        num = 0
        chunk_size = 64
        for i in range(0,image.width,chunk_size):
            for j in range(0,image.height,chunk_size):
                if (i+chunk_size>image.width):
                    continue
                if (j+chunk_size>image.height):
                    continue
                num += 1
                with image[i:i + chunk_size, j:j + chunk_size] as chunk:
                    chunk.save(filename='output/image{0}/image_{1}.jpg'.format(file_num,num))

#images to array
files = os.listdir('images')
output = []
dir_num = 0
for name in files:
    dir_num += 1
    file = 'output/image{0}/*.jpg'.format(dir_num)
    glob.glob(file)
    # Using List Comprehension to read all images
    images = [asarray(cv2.imread(image)) for image in glob.glob(file)]
    images = np.array(images)
    print(images.shape)
    output.append(images)

arr = np.asarray(output, dtype=object)
print(arr.shape)


