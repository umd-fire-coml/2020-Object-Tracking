import numpy as np
import time
from keras.models import Sequential
from data_generator import DataGenerator

annotation_file='data/annotations/annotations/instances_val2017.json'
list_IDs, labels = parse_data(annotation_file)
imagecount = 0
# ignore self, 
generator = DataGenerator(list_IDS, labels)

# call time it, start time here
start = time.time() 
for x in range(100):
    imgs,label = next(generator)
    imagecount += imgs.size[0]

# record end time
end = time.time()
# then calculate it
time = end - start
print('Time per batch: ' + time/imagecount)
print('Time per second: ' + imagecount/time)
plt.imshow(img)
plt.show()
print(imagecount)