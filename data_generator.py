import numpy as np
import cv2
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    
    annotation_file='data/annotations/annotations/instances_val2017.json'
    
    def __init__(self, annotation_file, data_path='data/val/val2017', batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10, shuffle=True):

        self.dim = dim
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end();
        imgsToAnns, categories, imgs = parse_data(annotation_file)
        self.imgsToAnns = imgsToAnns
        self.categories = categories
        self.imgs = imgs
 
    def __len__(self):

        return int(np.floor(len(self.imgsToAnns) / self.batch_size))
    
    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.imgsToAnns.keys()[k] for k in indexes]

        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.imgsToAnns))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs):

        dtype=np.dtype(int)
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, dtype)

        for i, ID in enumerate(list_IDs):

            img = self.imgs[ID]
            img_filename = img['file_name']
            img = np.array(cv2.imread(os.path.join(self.data_path,img_filename)))
                     
            X[i, ] = img

            y[i] = self.imgsToAnns[ID]

        return X, y



    
