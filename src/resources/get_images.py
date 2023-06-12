from PIL import Image
import gdown
import zipfile
import numpy as np
import os
import click 

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def get_data():
    #download files from google drive
    url = "https://drive.google.com/file/d/16250k3Ju0Eu14ZcCP3QrswK5shNDcCng/view?usp=sharing"
    output = 'images.zip'
    

    if not os.path.isdir(output.replace(".zip", "")):
        print("Downloading images from google drive")
        gdown.download(url, output, quiet=False, fuzzy=True)
        with zipfile.ZipFile('images.zip', 'r') as zip_ref:
            zip_ref.extractall("")
    
    chunk_size = 64
    fragments = []
    images = os.listdir('images')
    length = len(images)

    print("Fragmenting images...")
    for i, image in enumerate(images):
        printProgressBar(i+1, length, prefix="Progress", suffix="Complete", length=100)
        img = Image.open(os.path.join(output.replace(".zip", "") ,image))
        width, height = img.size

        chunks_per_width = (width // chunk_size)
        crop_width = chunks_per_width * chunk_size
        
        chunks_per_height = (height // chunk_size)
        crop_height = chunks_per_height * chunk_size
        
        cropped_image = img.crop((0, 0, crop_width, crop_height))
        

        for i in range(0, cropped_image.width, chunk_size):
            for j in range(0, cropped_image.height, chunk_size):
                fragment = cropped_image.crop((i, j, i + chunk_size, j + chunk_size))
                fragments.append(np.array(fragment))
        
    fragments = np.array(fragments)
    fragments = fragments / 255.0
    
    print("Number of fragments in the train dataset: {}".format(fragments.shape[0]))

    if not os.path.isdir('data'):
        os.mkdir('data')

    fragments.tofile(os.path.join(os.getcwd(), 'data/train_fragments'))
    print("Dataset saved in the {}".format(os.path.join(os.getcwd(), 'data')))

if __name__ == "__main__":
    get_data()