import cv2
import numpy as np
import matplotlib.pyplot as plt

file_path = "data/test/airplane-20/img/reference.jpg"
bounding_boxes = [261,345,343,167]

img = np.array(cv2.imread(file_path))

pad = 5000
color = np.mean(img, (0, 1))

img = cv2.copyMakeBorder( img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value = color)
bbox = bounding_boxes
x, y, w, h = bbox
x = x + pad
y = y + pad

x1, x2, y1, y2 = x, x + w, y, y+h
if x2 - x1 > y2 - y1:
    
    dist = x2 - x1
    add = int(dist/5 * 6)
    x1 -= add 
    x2 += add

    total_width = x2 - x1
    add = total_width - (y2 - y1)
    add = add // 2
    y1 -= add 
    y2 += add 

else:
    dist = y2 - y1
    add = int(dist/5 * 6)

    y1 -= add 
    y2 += add

    total_width = y2 - y1
    add = total_width - (x2 - x1)
    add = add // 2
    x1 -= add 
    x2 += add 


img = img[y1:y2, x1:x2, :]


if sum(img.shape) - 3 > 127 * 2:
    img = cv2.resize(img, (127, 127), interpolation=cv2.INTER_AREA)
else:
    img = cv2.resize(img, (127, 127), interpolation=cv2.INTER_LINEAR)

cv2.imwrite("data/test/airplane-20/img/reference_good.jpg", img)
