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
    
    
    def __init__(self, annotation_file=annotation_file, data_path='data/', batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10, shuffle=True, limit=-1):

        self.dim = dim
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.limit = limit

        self.data, self.imgs_to_anns = parse_data(data_path)
        if self.limit != -1:
            choices = random.sample(list(self.imgs_to_anns.keys()), self.limit)
            new = {}
            for c in choices:
                new[c] = self.imgs_to_anns[c]
            self.imgs_to_anns = new
        
        self.r = False
        self.limit = limit
        self.indexes = []
        self.on_epoch_end()

    def set_return(self):
        self.r = not self.r

    def __len__(self):

        return int(np.floor(len(self.imgs_to_anns) / self.batch_size))
    
    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [list(self.imgs_to_anns.keys())[k] for k in indexes]

        X, y, c = self.__data_generation(list_IDs_temp)
        if self.r:
            return X, y[0], c
        return X, y[0]

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.imgs_to_anns))
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
            # print (list_IDs)
            # exit()
            instance_img_data = self.imgs_to_anns[ID]
            exemplar_img_data = random.choice(self.data[instance_img_data["category_id"]][instance_img_data["video_id"]])

            # print (instance_img_data)
            # print (exemplar_img_data)

            cats.append(instance_img_data["category_name"])
            
            instance_img = np.array(cv2.imread(instance_img_data["file_path"]))
            exemplar_img = np.array(cv2.imread(exemplar_img_data["file_path"]))

            instance_img = cv2.resize(instance_img, (255, 255))
            import matplotlib.pyplot as plt 

            pad = 5000
            color = np.mean(exemplar_img, (0, 1))

            exemplar_img = cv2.copyMakeBorder( exemplar_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value = color)
            bbox = exemplar_img_data["bbox"]
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


            exemplar_img = exemplar_img[y1:y2, x1:x2, :]


            if sum(exemplar_img.shape) - 3 > 127 * 2:
                exemplar_img = cv2.resize(exemplar_img, (127, 127), interpolation=cv2.INTER_AREA)
            else:
                exemplar_img = cv2.resize(exemplar_img, (127, 127), interpolation=cv2.INTER_LINEAR)

            
            instances[i] = instance_img
            exemplars[i] = exemplar_img

        positive_label_pixel_radius = 16 # distance from center of target patch still considered a 'positive' match
        response_size = 17
        response_stride = 6.0
        data_size = len(instances)
        label = make_label(response_size, positive_label_pixel_radius / response_stride)
        labels = np.empty((data_size,) + label.shape)
        labels[:] = label
   
        return [instances, exemplars], [labels], cats


from glob import glob
def parse_data(top_folder):
    imgs_to_anns = {}
    categories = glob(top_folder + "*/")
    categories = [os.path.basename(os.path.normpath(c)) for c in categories]
    data = {}
    cat_count = 0
    for c in categories:
        data[cat_count] = {}
        path = os.path.join(top_folder, c + "/")
        
        videos = list(glob(path + "*"))
        vid_count = 0
        for v in videos:
            vid_id = os.path.basename(os.path.normpath(v))
            data[cat_count][vid_count] = []
            gt_path = os.path.join(v, "groundtruth.txt")
            with open(gt_path, "r") as f:
                ground_truth = f.read().split("\n")
            count = 0
            for image in glob(os.path.join(v, "img/") + "*.jpg"):
                bbox = ground_truth[count].split(",")

                # print (bbox, image)
                bbox = [int(b) for b in bbox]
                if bbox[2] == bbox[3]:
                    continue
                data[cat_count][vid_count].append({"file_path": image, "bbox": bbox, "video_folder": vid_id, "category_name": c, "category_id": cat_count, "video_id": vid_count})
                imgs_to_anns[image] = {"file_path": image, "bbox": bbox, "video_folder": vid_id, "category_name": c, "category_id": cat_count, "video_id": vid_count}
                count += 1
            vid_count += 1
          
        cat_count += 1
    
    return data, imgs_to_anns
    # print('loading annotations into memory...')
    # dataset = json.load(open(annotation_file, 'r'))
    # print('annotations loaded!')
    # print('creating index...')
    # imgToAnns = {ann['image_id']: [] for ann in dataset['annotations']}
    # anns = {ann['id']: [] for ann in dataset['annotations']}
    # for ann in dataset['annotations']:
    #     imgToAnns[ann['image_id']] += [ann]
    #     anns[ann['id']] = ann
    # categories = {category['id']: {"cat": category, "imgs": []} for category in dataset['categories']}
    # cats = []
    # catToImgs = []
    # imgs  = {im['id']: {} for im in dataset['images']}
    # out_anns = {}
    # for img in dataset['images']:
    #     if img["id"] in imgToAnns:
    #         for seg in imgToAnns[img["id"]]:
    #             categories[seg["category_id"]]["imgs"].append(seg)
    #             # exit()
    #         imgs[img['id']] = img
    #         out_anns[img['id']] = imgToAnns[img["id"]]
     
    # print('index created!')
    # return out_anns,categories,imgs

    
