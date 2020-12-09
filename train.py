from model import make_model
from data_generator import DataGenerator
import numpy as np
from math import ceil, sqrt
import cv2, os
import tensorflow as tf
import matplotlib.pyplot as plt 
from skimage.transform import resize
from tensorflow.keras import backend as K


dg = DataGenerator(batch_size=8, n_channels=3, shuffle=True, multi=5)
decay_steps = len(dg) * 1#lr_config['num_epochs_per_decay']
test_dg = DataGenerator(batch_size=1, n_channels=3, shuffle=True, test=True)
class PlotCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_dg):
        super(PlotCallback, self).__init__()
        self.test_dg = test_dg
        
        if os.path.exists("samples") == False:
            os.mkdir("samples")
        else:
            os.rmdir("samples")
            os.mkdir("samples")

        # best_weights to store the weights at which the minimum loss occurs.
    def on_epoch_end(self, epoch, logs=None):
        fig=plt.figure(figsize=(14, 8))
        columns = 3
        rows = 3
        c = 1
        for i in range(3):
            test = self.test_dg[i]
            test_x, test_y, cats = test
            test_x_base, test_x_search = test_x[0][0], test_x[1][0]

            label = self.model.predict([np.expand_dims(test_x_base, 0), np.expand_dims(test_x_search, 0)])[0]
            label = resize(label, (test_x_base.shape[0], test_x_base.shape[1]))
            label = np.ma.masked_where(label == 0, label)

            fig.add_subplot(rows, columns, c)
            plt.imshow(cv2.cvtColor(np.array(test_x_base).astype(np.uint8), cv2.COLOR_BGR2RGB))
            c += 1
            fig.add_subplot(rows, columns, c)
            plt.imshow(label)
            c += 1
            fig.add_subplot(rows, columns, c)
            plt.imshow(cv2.cvtColor(np.array(test_x_base).astype(np.uint8), cv2.COLOR_BGR2RGB))
            plt.imshow(label, alpha=0.7, interpolation = 'none', vmin = 0)

            c += 1

        fig.savefig("samples/%s.png" % epoch)
        plt.close()    
        test_dg.on_epoch_end()

        lr = self.model.optimizer.lr
        print(K.eval(lr))


# yolo = YoloV3(size=(416, 416), channels=3)
m = make_model((255, 255, 3), (127, 127, 3), decay_steps=decay_steps)

dg.set_return()
test_dg.set_return()

import time 
t = time.time()
test = dg[0]
print (time.time() - t)
# exit()
test_x, test_y, cats = test
test_x_base, test_x_search = test_x[0][0], test_x[1][0]


dg.set_return()

try:
    pc = PlotCallback(test_dg)
    print ("starting")
    H = m.fit(dg, workers=8, epochs=50, callbacks=[pc], max_queue_size=20)

except KeyboardInterrupt as e:
    pass

if os.path.exists("saved_model/") == False:
    os.mkdir("saved_model")
m.save("saved_model")
x = m.predict([np.expand_dims(test_x_base, 0), np.expand_dims(test_x_search, 0)])
print (cats[0])
fig=plt.figure(figsize=(14, 8))
columns = 3
rows = 1

label = x[0]


label = resize(label, (test_x_base.shape[0], test_x_base.shape[1]))
label = np.ma.masked_where(label == 0, label)



fig.add_subplot(rows, columns, 1)
plt.imshow(cv2.cvtColor(np.array(test_x_base).astype(np.uint8), cv2.COLOR_BGR2RGB))
fig.add_subplot(rows, columns, 2)
plt.imshow(label)
fig.add_subplot(rows, columns, 3)
plt.imshow(cv2.cvtColor(np.array(test_x_base).astype(np.uint8), cv2.COLOR_BGR2RGB))
plt.imshow(label, alpha=0.7, interpolation = 'none', vmin = 0)
plt.show()
