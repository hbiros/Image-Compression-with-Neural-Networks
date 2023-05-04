from models.model_1 import model, encoder
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def show_data(x, n=5, height=64, width=64, title=""):
  plt.figure(figsize=(10,3))
  for i in range(n):
    ax = plt.subplot(2,n,i+1)
    plt.imshow(x[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
  plt.suptitle(title, fontsize=16)


path = './cats'

cats = []

for img in os.listdir(path):
  arr = cv2.imread(os.path.join(path, img)) 
  if arr is None:
    continue
  # if arr.shape != (64, 64, 3):
  #   continue
  cats.append(arr)

cats = np.array(cats)# (1, 64, 64, 3)

cats_normalized = cats / 255.0
test_ratio = 0.1
x_train = cats_normalized[:int(cats_normalized.shape[0]*(1-test_ratio))]
x_test = cats_normalized[int(cats_normalized.shape[0]*(1-test_ratio)):]

print(x_train.shape)
print(x_test.shape)

# plt.imshow(cats[0])
# show_data(x_train, title="train cats")
# show_data(x_test, title="test cats")
# plt.show()

model.fit(x_train, x_train, epochs=20, batch_size=32, validation_data=(x_test, x_test))
