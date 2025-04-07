import os
import time
import uuid
import cv2

IMAGES_PATH = os.path.join('data', 'images')
# number_images = 30

# cap = cv2.VideoCapture(0)
# for imgnum in range(number_images):
#     print('Collecting image {}'.format(imgnum))
#     ret, frame = cap.read()
#     imgname = os.path.join(IMAGES_PATH, f'{str(uuid.uuid1())}.jpg')
#     cv2.imwrite(imgname, frame)
#     cv2.imshow('frame', frame)
#     time.sleep(0.5)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

import tensorflow as tf 
import cv2 
import json 
import numpy as np 
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

print(tf.config.list_physical_devices('GPU'))

images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=True)

print(images.as_numpy_iterator().next())

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)
print(images.as_numpy_iterator().next())

print(type(images))

image_generator = images.batch(4).as_numpy_iterator()
plot_images = image_generator.next()

# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx, image in enumerate(plot_images):
#     ax[idx].imshow(image)
# plt.show()

# 63 to train (70%)
# 14 to test (15%)
# 13 to val (15%)

for folder in ['train', 'test', 'val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        filename = file.split('.')[0]+'.json'
        existing_filepath = os.path.join('data', 'labels', filename)
        if os.path.exists(existing_filepath):
            new_filepath = os.path.join('data',folder,'labels',filename)
            os.replace(existing_filepath, new_filepath)

import albumentations as alb

augmentor = alb.Compose([alb.RandomCrop(width=950, height=650),
alb.HorizontalFlip(p=0.5),
alb.RandomBrightnessContrast(p=0.2),
alb.RandomGamma(p=0.2),
alb.RGBShift(p=0.2),
alb.VerticalFlip(p=0.5)],
bbox_params=alb.BboxParams(format='albumentations',
label_fields=['class_labels']))


img = cv2.imread(os.path.join('data','train','images','e4ed0e66-1345-11f0-a59b-acde48001122.jpg'))
# print(img)
# Image shape: (720, 1280 ,3)
print("Image shape: ", img.shape)

with open(os.path.join('data', 'train', 'labels', 'e4ed0e66-1345-11f0-a59b-acde48001122.json'), 'r') as f:
    label = json.load(f)

print(label['shapes'][0]['points'])

coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

print(coords)

coords = list(np.divide(coords, [1280, 720, 1280, 720]))
print(coords)

augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
print(augmented['image'])

cv2.rectangle(augmented['image'],
tuple(np.multiply(augmented['bboxes'][0][:2], [950,650]).astype(int)),
tuple(np.multiply(augmented['bboxes'][0][2:], [950,650]).astype(int)),
(255,0,0), 2)

plt.imshow(augmented['image'])
plt.show()