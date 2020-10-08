import numpy as np
from keras.models import Sequential
from my_classes import DataGenerator

imagecount = 0
generator = DataGenerator( list_IDS, labels)

for x in range(100):
    imgs,label = next(generator)
    imagecount += 1

print(imagecount)