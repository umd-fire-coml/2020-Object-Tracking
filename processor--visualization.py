import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import Sequential
from data_generator import DataGenerator

imagecount = 0

generator = DataGenerator('data/annotations/annotations/instances_val2017.json')

# call time it, start time here
start = time.time() 
for x in range(100):
    imgs,labels = next(generator)
    imagecount += imgs.size[0]

# record end time
end = time.time()
# then calculate it
time = end - start
print('Time per batch: ' + time/imagecount)
print('Batch per second: ' + imagecount/time)
imgs,labels = next(generator)
img = imgs[0]
label = labels[0]
plt.figure(figsize = (15, 10))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img)
img_bbox = img.copy()
bboxs=[ann['bbox'] for ann in label]

for bbox in bboxs:
    x, y, w, h = bbox
    cv2.rectangle(img_bbox,(int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
    
plt.subplot(1, 2, 2)
plt.title('Image with Bounding Box')
plt.imshow(img_bbox)
plt.show()

print(imagecount)