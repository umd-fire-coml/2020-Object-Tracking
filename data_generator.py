import numpy as np
import cv2
from tensorflow import keras
import random, os
# from data_viz import parse_data
import json
import matplotlib.pyplot as plt
from math import ceil, sqrt 
from skimage.transform import resize


def euclidean_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def make_label(dim, radius):
    label = np.full((dim, dim), -1)
    center = int(dim / 2.0)
    start = center - ceil(radius)
    end = center + ceil(radius)
    for i in range(start, end + 1):
        for j in range(start, end + 1):
            if euclidean_distance(i, j, center, center) <= radius:
                label[i,j] = 1
    return label



image_size = (127, 127)

class DataGenerator(keras.utils.Sequence):
    
    annotation_file='data/annotations/annotations/instances_val2017.json'
    
    def __init__(self, annotation_file=annotation_file, data_path='data/val/val2017', batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10, shuffle=True, limit=-1):

        self.dim = dim
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        imgsToAnns, categories, imgs = parse_data(annotation_file)
        if limit != -1:
            choices = random.sample(list(imgsToAnns.keys()), limit)
            new = {}
            for c in choices:
                new[c] = imgsToAnns[c]
            imgsToAnns = new
        self.imgsToAnns = imgsToAnns
        self.categories = categories
        self.imgs = imgs
        self.r = False
        self.limit = limit
        self.indexes = []
        self.on_epoch_end()

    def set_return(self):
        self.r = not self.r

    def __len__(self):

        return int(np.floor(len(self.imgsToAnns) / self.batch_size))
    
    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [list(self.imgsToAnns.keys())[k] for k in indexes]

        X, y, c = self.__data_generation(list_IDs_temp)
        if self.r:
            return X, y[0], c
        return X, y[0]

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.imgsToAnns))
        # if self.limie != -1:
        #     self.indexes = self.indexes[:self.limit]
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # def _rescale_image(image, bounding_box):
    #     transform = A.Compose([
    #         A.resize(image_size[0], image_size[1])
    #     ], bbox_params=A.BboxParams(format='coco'))

    def _image_augment(img, bboxes):
        '''
        image_augment
            ARGS: img - numpy array of image
                bboxes - array of bounding boxes in COCO format
        '''
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Resize(image_size[0], image_size[1])

            ], bbox_params=A.BboxParams(format='coco'))

        transformed = transform(
            image=img,
            bboxes=bboxes
        )
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        return transformed_image, transformed_bboxes


    def __data_generation(self, list_IDs):


        exemplars = np.empty((self.batch_size, 127, 127, self.n_channels))
        instances = np.empty((self.batch_size, 255, 255, self.n_channels))
        bboxes = []
        cats = []

        for i, ID in enumerate(list_IDs):
            img = self.imgs[ID]
            cat = random.choice(self.imgsToAnns[ID])
            
            img_filename = img['file_name']
            instance = np.array(cv2.imread(os.path.join(self.data_path,img_filename)))
            instance = cv2.resize(instance, (255, 255))


            ex_choice = self.categories[cat["category_id"]]["imgs"]

            cats.append(self.categories[cat["category_id"]]["cat"]["name"])

            ex = random.choice(ex_choice)
            ex_img = self.imgs[ex["image_id"]]
            ex_fname = ex_img["file_name"]
            exemplar = np.array(cv2.imread(os.path.join(self.data_path,ex_fname)))
            e = list(np.array(exemplar))
            e = np.array(e)
            e = cv2.cvtColor(np.array(e).astype(np.uint8), cv2.COLOR_BGR2RGB)
            pad = 1000
            color = np.mean(e, (0, 1))

            exemplar = cv2.copyMakeBorder( exemplar, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value = color)

            bbox = ex["bbox"]
            bbox = [int(b) for b in bbox]
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

            if x1 < 0:
                x1 = 0
            if y1 < 0:
                y1 = 0
            
            exemplar = exemplar[y1:y2, x1:x2, :]
            if 0 in exemplar.shape:
                print ("OH HELL NAH")
                print (x1, x2, y1, y2)
                print (exemplar.shape)
            if sum(exemplar.shape) - 3 > 127 * 2:
                exemplar = cv2.resize(exemplar, (127, 127), interpolation=cv2.INTER_AREA)
            else:
                exemplar = cv2.resize(exemplar, (127, 127), interpolation=cv2.INTER_LINEAR)


            anns = []
            for a in self.imgsToAnns[ID]:
                if a["category_id"] == cat["category_id"]:
                    anns.append(a)
            
            instances[i] = instance 
            exemplars[i] = exemplar

        positive_label_pixel_radius = 16 # distance from center of target patch still considered a 'positive' match
        response_size = 17
        response_stride = 6.0
        data_size = len(instances)
        label = make_label(response_size, positive_label_pixel_radius / response_stride)
        labels = np.empty((data_size,) + label.shape)
        labels[:] = label
        # plt.imshow(label)
        # plt.show()
        # exit()

        return [instances, exemplars], [labels], cats

        # for i, ID in enumerate(list_IDs):

        #     img = self.imgs[ID]
        #     img_filename = img['file_name']
        #     img = np.array(cv2.imread(os.path.join(self.data_path,img_filename)))

        #     img, bbx = _image_augment(img, self.imgsToAnns)
                     
        #     X[i, ] = img

        #     y[i] = bbx

        # return X, y


def parse_data(annotation_file):
    print('loading annotations into memory...')
    dataset = json.load(open(annotation_file, 'r'))
    print('annotations loaded!')
    print('creating index...')
    imgToAnns = {ann['image_id']: [] for ann in dataset['annotations']}
    anns = {ann['id']: [] for ann in dataset['annotations']}
    for ann in dataset['annotations']:
        imgToAnns[ann['image_id']] += [ann]
        anns[ann['id']] = ann
    categories = {category['id']: {"cat": category, "imgs": []} for category in dataset['categories']}
    cats = []
    catToImgs = []
    imgs  = {im['id']: {} for im in dataset['images']}
    out_anns = {}
    for img in dataset['images']:
        if img["id"] in imgToAnns:
            for seg in imgToAnns[img["id"]]:
                categories[seg["category_id"]]["imgs"].append(seg)
                # exit()
            imgs[img['id']] = img
            out_anns[img['id']] = imgToAnns[img["id"]]
     
    print('index created!')
    return out_anns,categories,imgs

    
