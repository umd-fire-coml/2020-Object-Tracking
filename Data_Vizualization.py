#!/usr/bin/env python
# coding: utf-8

# In[80]:


import json
import cv2
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np


# In[81]:


annotation_file='data/annotations/annotations/instances_val2017.json'


# In[110]:


def parse_data(annotation_file):
    print('loading annotations into memory...')
    dataset = json.load(open(annotation_file, 'r'))
    print('annotations loaded!')
    print('creating index...')
    imgToAnns = {ann['image_id']: [] for ann in dataset['annotations']}
    anns = {ann['id']: [] for ann in dataset['annotations']}
    for ann in dataset['annotations']:
        imgToAnns[ann['image_id']] += [ann]
        anns[ann['id']] = ann
    imgs  = {im['id']: {} for im in dataset['images']}
    for img in dataset['images']:
        imgs[img['id']] = img
    categories = {category['id']: category for category in dataset['categories']}
    cats = []
    catToImgs = []
 
    print('index created!')
    return imgToAnns,categories


# In[112]:


imgToAnns, categories = parse_data(annotation_file)


# In[108]:


imgId=37777
anns = sum([imgToAnns[imgId]],[])
ids = [ann['id'] for ann in anns]
img = imgs[imgId]
img_filename = img['file_name']
img = np.array(cv2.imread(os.path.join('data/val/val2017',img_filename)))


# In[113]:


imgToAnns_items = imgToAnns.items()
first_three = list(imgToAnns_items )[:3]
print(first_three )


# In[114]:


bboxs=[ann['bbox'] for ann in anns]
category_names=[categories[ann['category_id']]['name'] for ann in anns]


# In[115]:


plt.figure(figsize = (15, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img)
img_bbox = img.copy()
for bbox, category_name in zip(bboxs, category_names):
    x, y, w, h = bbox
    cv2.rectangle(img_bbox,(int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_bbox, category_name, (int(x), int(y) - 10), font, 0.5, (0, 255, 0), 2)
plt.subplot(1, 2, 2)
plt.title('Image with Bounding Box')
plt.imshow(img_bbox)
plt.show()

