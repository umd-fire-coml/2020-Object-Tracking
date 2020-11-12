import numpy as np
import cv2
import os
from tensorflow import keras
from got10k.datasets import OTB

class trackingGenerator(keras.utils.Sequence):
    
    #GOT-10k, **OTB, VOT
    def __init__(self, dataset, batch_size=1):
        self.dataset = OTB(root_dir=os.path.join('data', dataset))
        self.batch_size = batch_size
 
    def __len__(self):

        return len(self.dataset)//self.batch_size
    
    def __getitem__(self, index):
        
        img_files, annos = self.dataset[index:index + 2]
        img1 = np.array(cv2.imread(img_files[0]))
        img2 = np.array(cv2.imread(img_files[1])) 
        box1 = annos[0]
        box2 = annos[1]
        return img1, img2, box1, box2
