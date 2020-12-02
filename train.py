from model import make_model
from data_generator import DataGenerator
import numpy as np
from math import ceil, sqrt
image_size = (416, 416, 3)
import cv2

dg = DataGenerator(batch_size=16, dim=(416, 416), n_channels=3, n_classes=80, shuffle=True, limit=1000)

# yolo = YoloV3(size=(416, 416), channels=3)
m = make_model((255, 255, 3), (127, 127, 3))

dg.set_return()

test = dg[0]
test_x, test_y, cats = test
test_x_base, test_x_search = test_x[0][0], test_x[1][0]


dg.set_return()
# print (label.shape)
# import matplotlib.pyplot as plt
 
# plt.imshow(label)
# plt.show()
try:

    H = m.fit(dg, epochs=500)

except KeyboardInterrupt as e:
    pass

m.save("saved_model")
x = m.predict([np.expand_dims(test_x_base, 0), np.expand_dims(test_x_search, 0)])
print (cats[0])
import matplotlib.pyplot as plt 
fig=plt.figure(figsize=(14, 8))
columns = 3
rows = 1
# for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
# plt.show()
# norm label
label = x[0]
label = np.array(label)
label -= np.min(label)
label /= np.max(label)

# plt.imshow(cv2.cvtColor(np.array(test_x_base).astype(np.uint8), cv2.COLOR_BGR2RGB))
# fig.add_subplot(rows, columns, 2)
# plt.imshow(cv2.cvtColor(np.array(test_x_search).astype(np.uint8), cv2.COLOR_BGR2RGB))
# fig.add_subplot(rows, columns, 3)
# plt.imshow(x[0])
# plt.show()


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