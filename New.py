#!/usr/bin/env python
# coding: utf-8

# In[50]:


import json
import cv2
import os
import datetime
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from skimage.draw import polygon
from PIL import Image


# In[43]:


annotation_file='data/annotations/annotations/instances_val2017.json'


# In[44]:


print('loading annotations into memory...')
dataset = json.load(open(annotation_file, 'r'))
print('annotations loaded!')


# In[45]:


print('creating index...')
imgToAnns = {ann['image_id']: [] for ann in dataset['annotations']}
anns = {ann['id']: [] for ann in dataset['annotations']}
for ann in dataset['annotations']:
   imgToAnns[ann['image_id']] += [ann]
   anns[ann['id']] = ann
imgs  = {im['id']: {} for im in dataset['images']}
for img in dataset['images']:
   imgs[img['id']] = img

cats = []
catToImgs = []

# cats = {cat['id']: [] for cat in dataset['categories']}
# for cat in dataset['categories']:
#     cats[cat['id']] = cat
# catToImgs = {cat['id']: [] for cat in dataset['categories']}
# for ann in dataset['annotations']:
#     catToImgs[ann['category_id']] += [ann['image_id']]


print('index created!')


# In[46]:


imgId=397133
anns = sum([imgToAnns[imgId]],[])
ids = [ann['id'] for ann in anns]
img = imgs[imgId]
img_filename = img['file_name']
img = np.array(Image.open(os.path.join('data/val/val2017',img_filename)))


# In[ ]:





# In[47]:


bboxs=[ann['bbox'] for ann in anns]


# In[49]:


plt.figure(figsize = (15, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img)
img_bbox = img.copy()
for bbox in bboxs:
    x, y, w, h = bbox
    cv2.rectangle(img_bbox, (x,y), (x+w, y+h), (0, 255, 0), 2)
plt.subplot(1, 2, 2)
plt.title('Image with Bounding Box')
plt.imshow(img_bbox)
plt.show()


# In[13]:





# In[ ]:





# In[ ]:




