import cv2
import numpy as np
from PIL import Image
import os

file_name = "../data/train_dome_data/images/raccoon-128.jpg"
assert os.path.isfile(file_name) == True and os.path.isfile("../data/train_dome_data/new_labels.txt")
data = open("../data/train_dome_data/new_labels.txt").readlines()
for i in range(len(data)):
    image_info = data[i].split()
    if image_info[0] == file_name: break

image = cv2.imread(image_info[0])
n_box = len(image_info[1:]) // 5  # xmin, ymin, xmax, ymax, id
for i in range(n_box):
    image = cv2.rectangle(image, (int(float(image_info[1 + i * 5])),
                                  int(float(image_info[2 + i * 5]))),
                          (int(float(image_info[3 + i * 5])),
                           int(float(image_info[4 + i * 5]))), (255, 0, 0), 2)

image = Image.fromarray(np.uint8(image))
image.show()
