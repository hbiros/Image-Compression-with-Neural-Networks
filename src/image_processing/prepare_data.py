import numpy as np
from PIL import Image
import os

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

def load_data(train_data, shape=(64,64,3)):
  x_train = np.fromfile(train_data).reshape(-1, shape[0], shape[1], shape[2])
  return x_train

if __name__ == "__main__":
  
  path = './cats'

  cats = []

  length = len(os.listdir(path))

  for i, img in enumerate(os.listdir(path)):
    printProgressBar(i+1, length, prefix="Progress", suffix="Complete", length=100)
    arr = np.array(Image.open(os.path.join(path, img)))
    if arr is None:
      continue
    cats.append(arr)

  cats = np.array(cats)# (1, 64, 64, 3)

  cats_normalized = cats / 255.0
  test_ratio = 0.1
  x_train = cats_normalized[:int(cats_normalized.shape[0]*(1-test_ratio))]
  x_test = cats_normalized[int(cats_normalized.shape[0]*(1-test_ratio)):]

  print(x_train.shape)
  print(x_test.shape)

  if not os.path.isdir('data'):
    os.mkdir('data')

  x_train.tofile(os.path.join(os.getcwd(), 'data/train_fragments'))
  x_test.tofile(os.path.join(os.getcwd(), 'data/test_fragments'))