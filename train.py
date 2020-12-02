from model import make_model
from data_generator import DataGenerator
import numpy as np
from math import ceil, sqrt
import cv2, os

dg = DataGenerator(batch_size=8, n_channels=3, shuffle=True, limit=500)

# yolo = YoloV3(size=(416, 416), channels=3)
m = make_model((255, 255, 3), (127, 127, 3))

dg.set_return()

test = dg[0]
test_x, test_y, cats = test
test_x_base, test_x_search = test_x[0][0], test_x[1][0]


dg.set_return()
=
try:
    print ("starting")
    H = m.fit(dg, epochs=500)

except KeyboardInterrupt as e:
    pass

if os.path.exists("saved_model/") == False:
    os.mkdir("saved_model")
m.save("saved_model")
x = m.predict([np.expand_dims(test_x_base, 0), np.expand_dims(test_x_search, 0)])
print (cats[0])
import matplotlib.pyplot as plt 
fig=plt.figure(figsize=(14, 8))
columns = 3
rows = 1

label = x[0]


from skimage.transform import resize
label = resize(label, (test_x_base.shape[0], test_x_base.shape[1]))
label = np.ma.masked_where(label == 0, label)

fig.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(np.array(test_x_base).astype(np.uint8), cv2.COLOR_BGR2RGB))
fig.add_subplot(rows, columns, 2)
plt.imshow(label)
fig.add_subplot(rows, columns, 3)
plt.imshow(cv2.cvtColor(np.array(test_x_base).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.imshow(label, alpha=0.4, interpolation = 'none', vmin = 0)
plt.show()