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

# print(tf.config.list_physical_devices('GPU'))

images = tf.data.Dataset.list_files('data/images/*.jpg', shuffle=True)

# print(images.as_numpy_iterator().next())

def load_image(x):
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

images = images.map(load_image)


# image_generator = images.batch(4).as_numpy_iterator()
# plot_images = image_generator.next()

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
print("Image shape: ", img.shape)

with open(os.path.join('data', 'train', 'labels', 'e4ed0e66-1345-11f0-a59b-acde48001122.json'), 'r') as f:
    label = json.load(f)

# print(label['shapes'][0]['points'])

coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]

# print(coords)

coords = list(np.divide(coords, [1280, 720, 1280, 720]))
# print(coords)

augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
# print(augmented['image'])

cv2.rectangle(augmented['image'],
tuple(np.multiply(augmented['bboxes'][0][:2], [950,650]).astype(int)),
tuple(np.multiply(augmented['bboxes'][0][2:], [950,650]).astype(int)),
(255,0,0), 2)

plt.imshow(augmented['image'])
plt.show()

# for partition in ['train', 'test', 'val']:
#     for image in os.listdir(os.path.join('data', partition, 'images')):
#         img = cv2.imread(os.path.join('data', partition, 'images', image))

#         coords = [0,0,0.00001,0.00001]
#         label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
#         if os.path.exists(label_path):
#             with open(label_path, 'r') as f:
#                 label = json.load(f)

#             coords[0] = label['shapes'][0]['points'][0][0]
#             coords[1] = label['shapes'][0]['points'][0][1]
#             coords[2] = label['shapes'][0]['points'][1][0]
#             coords[3] = label['shapes'][0]['points'][1][1]
#             coords = list(np.divide(coords, [1280,720,1280,720]))

#         try:
#             for x in range(60):
#                 augmented = augmentor(image=img, bboxes=[coords], class_labels=['face'])
#                 cv2.imwrite(os.path.join('aug_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

#                 annotation = {}
#                 annotation['image'] = image

#                 if os.path.exists(label_path):
#                     if len(augmented['bboxes']) == 0:
#                         annotation['bbox'] = [0,0,0,0]
#                         annotation['class'] = 0
#                     else:
#                         bbox = augmented['bboxes'][0]
#                         bbox = [bbox[0] / 950, bbox[1] / 650, bbox[2] / 950, bbox[3] / 650]
#                         annotation['bbox'] = bbox
#                         annotation['class'] = 1
#                 else:
#                     annotation['bbox'] = [0,0,0,0]
#                     annotation['class'] = 0
                
#                 with open(os.path.join('aug_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as f:
#                     json.dump(annotation, f)

#         except Exception as e:
#             print(e)

# 
train_images = tf.data.Dataset.list_files('aug_data/train/images/*.jpg', shuffle=False)
train_images = train_images.map(load_image)
train_images = train_images.map(lambda x: tf.image.resize(x, (120,120)))
train_images = train_images.map(lambda x: x/255)

test_images = tf.data.Dataset.list_files('aug_data/test/images/*.jpg', shuffle=False)
test_images = test_images.map(load_image)
test_images = test_images.map(lambda x: tf.image.resize(x, (120,120)))
test_images = test_images.map(lambda x: x/255)

val_images = tf.data.Dataset.list_files('aug_data/val/images/*.jpg', shuffle=False)
val_images = val_images.map(load_image)
val_images = val_images.map(lambda x: tf.image.resize(x, (120,120)))
val_images = val_images.map(lambda x: x/255)


def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as f:
        label = json.load(f)

    return [label['class']], label['bbox']

def tf_load_labels(x):
    clas, bbox = tf.py_function(load_labels, [x], [tf.uint8, tf.float32])
    clas.set_shape([1])         # Class is a list with 1 element
    bbox.set_shape([4])        # Assuming bbox is [x, y, width, height]
    return clas, bbox

train_labels = tf.data.Dataset.list_files('aug_data/train/labels/*.json', shuffle=False)
train_labels = train_labels.map(tf_load_labels)

test_labels = tf.data.Dataset.list_files('aug_data/test/labels/*.json', shuffle=False)
test_labels = test_labels.map(tf_load_labels)

val_labels = tf.data.Dataset.list_files('aug_data/val/labels/*.json', shuffle=False)
val_labels = val_labels.map(tf_load_labels)

print('train_labels', train_labels.as_numpy_iterator().next())

# print(len(train_images))

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(5000)
train = train.batch(8)
train = train.prefetch(4)

test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(5000)
test = test.batch(8)
test = test.prefetch(4)

val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(5000)
val = val.batch(8)
val = val.prefetch(4)

data_samples = train.as_numpy_iterator()
res = data_samples.next()

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
print("for looping")
for idx in range(4):
    sample_image = res[0][idx].copy()
    sample_coords = res[1][1][idx]
    cv2.rectangle(sample_image,
            tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
            tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)),
                    (255,0,0), 2)

                    

    ax[idx].imshow(sample_image)

plt.show()