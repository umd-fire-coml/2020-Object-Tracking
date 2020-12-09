import numpy as np
import cv2
from tensorflow import keras
import random, os
# from data_viz import parse_data
import json
import matplotlib.pyplot as plt
from math import ceil, sqrt 
from skimage.transform import resize
from utils import crop_and_resize, calculate_x_z_sz
import albumentations as A

def get_center(x):
    return (x - 1.) / 2.

def construct_batch_gt_score_maps(response_size, stride, batch_size):

    gt = construct_single_gt_score_map(response_size, stride)
    gt_expand = np.expand_dims(gt, 0)
    gt = np.tile(gt_expand, [batch_size, 1, 1])
    return gt

def construct_single_gt_score_map(response_size, stride):
    """Construct a batch of groundtruth score maps
    Args:
        response_size: A list or tuple with two elements [ho, wo]
        batch_size: An integer e.g., 16
        stride: Embedding stride e.g., 8
        gt_config: Configurations for groundtruth generation
    Return:
        A float tensor of shape [batch_size] + response_size
    """
    ho = response_size[0]
    wo = response_size[1]
    y = np.arange(0, ho, dtype=np.float32) - get_center(ho)
    x = np.arange(0, wo, dtype=np.float32) - get_center(ho)
    # x = tf.cast(tf.range(0, wo), dtype=tf.float32) - get_center(wo)
    [Y, X] = np.meshgrid(y, x)

    def _logistic_label(X, Y, rPos, rNeg):
      # dist_to_center = tf.sqrt(tf.square(X) + tf.square(Y))  # L2 metric
      dist_to_center = np.abs(X) + np.abs(Y)  # Block metric
      Z = np.where(dist_to_center <= rPos,
                   np.ones_like(X),
                   np.where(dist_to_center < rNeg,
                            0.5 * np.ones_like(X),
                            np.zeros_like(X)))
      return Z

    rPos = 16 / stride
    rNeg = 0 / stride
    gt = _logistic_label(X, Y, rPos, rNeg)

    # Duplicate a batch of maps
    return gt

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
    
    
    def __init__(self, data_path='data/', batch_size=32, dim=(32, 32, 32), n_channels=1, n_classes=10, shuffle=True, limit=-1, test=False, multi=1):

        self.dim = dim
        self.data_path = data_path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.limit = limit
        self.test = test
        self.c_labels = construct_batch_gt_score_maps((17, 17), 6, self.batch_size)
        self.multi = multi 

        

        self.data, self.imgs_to_anns = parse_data(data_path, test=self.test)
        if self.limit != -1:
            choices = random.sample(list(self.imgs_to_anns.keys()), self.limit)
            new = {}
            for c in choices:
                new[c] = self.imgs_to_anns[c]
            self.imgs_to_anns = new

        self.imgs_choices = []
        for i in range(self.multi):
            self.imgs_choices.extend(list(self.imgs_to_anns.keys()))
        
        self.r = False
        self.limit = limit
        self.indexes = []
        self.on_epoch_end()

    def set_return(self):
        self.r = not self.r

    def __len__(self):

        return int(np.floor(len(self.imgs_choices) / self.batch_size))
    
    def __getitem__(self, index):
        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_IDs_temp = [self.imgs_choices[k] for k in indexes]

        X, y, c = self.__data_generation(list_IDs_temp)
        if self.r:
            return X, y[0], c
        return X, y[0]

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.imgs_choices))
        # if self.limie != -1:
        #     self.indexes = self.indexes[:self.limit]
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    # def _rescale_image(image, bounding_box):
    #     transform = A.Compose([
    #         A.resize(image_size[0], image_size[1])
    #     ], bbox_params=A.BboxParams(format='coco'))

    def _image_augment(self, img):#, bboxes):
        '''
        image_augment
            ARGS: img - numpy array of image
                bboxes - array of bounding boxes in COCO format
        '''
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Rotate(p=0.5),
            A.RandomBrightnessContrast(p=0.3),

            ])#, bbox_params=A.BboxParams(format='coco'))

        transformed = transform(
            image=img,
            # bboxes=bboxes
        )
        transformed_image = transformed['image']
        # transformed_bboxes = transformed['bboxes']
        return transformed_image#, transformed_bboxes


    def __data_generation(self, list_IDs):


        exemplars = np.empty((self.batch_size, 127, 127, self.n_channels))
        instances = np.empty((self.batch_size, 255, 255, self.n_channels))
        bboxes = []
        cats = []

        for i, ID in enumerate(list_IDs):
            # print (list_IDs)
            # exit()
            instance_img_data = self.imgs_to_anns[ID]
            base_name = os.path.basename(os.path.normpath(instance_img_data["file_path"]))
            base_name = int(base_name.split(".")[0])
            min_ = base_name - 100
            max_ = base_name + 100 

            if min_ < 0:
                min_ = 0
            if max_ > instance_img_data["total"]:
                max_ = instance_img_data["total"]
            ex_file_name = ""
            while ex_file_name not in self.imgs_to_anns:
                ex_idx = random.randint(min_, max_)#random.choice(self.data[instance_img_data["category_id"]][instance_img_data["video_id"]])
                ex_idx = str(ex_idx).zfill(8) + ".jpg"
                ex_file_name = instance_img_data["file_path"]
                ex_file_name = ex_file_name[:-12]
                ex_file_name += ex_idx 
                
                # print (self.data[instance_img_data["category_id"]][instance_img_data["video_id"]])
                # exit(
            # print (self.imgs_to_anns.keys())
            exemplar_img_data = self.imgs_to_anns[ex_file_name]

            bbox = exemplar_img_data["bbox"]
            x, y, w, h = bbox
            box = [y, x, h, w]

            # print (instance_img_data)
            # print (exemplar_img_data)

            cats.append(instance_img_data["category_name"])
            
            instance_img = np.array(cv2.imread(instance_img_data["file_path"]))
            exemplar_img = np.array(cv2.imread(exemplar_img_data["file_path"]))

            x_sz, z_sz = calculate_x_z_sz(box)

            instance_img = cv2.resize(instance_img, (255, 255))

            pad = 3000
            color = [0, 0, 0]#np.mean(exemplar_img, (0, 1))

            # exemplar_img = cv2.copyMakeBorder( exemplar_img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value = color)
            # exemplar_img = np.pad(exemplar_img, [(pad, pad), (pad, pad), (0,0)], mode="constant", constant_values=[color, color])
            
            exemplar_img = crop_and_resize(
                    exemplar_img, box, x_sz,
                    out_size=127,
                    border_value=[0,0,0])

            # plt.imshow(exemplar_img)
            # plt.show()
            instance_img = self._image_augment(instance_img)
            exemplar_img = self._image_augment(exemplar_img)
      
            instances[i] = instance_img
            exemplars[i] = exemplar_img

        # positive_label_pixel_radius = 16 # distance from center of target patch still considered a 'positive' match
        # response_size = 17
        # response_stride = 6.0
        # data_size = len(instances)
        # label = make_label(response_size, positive_label_pixel_radius / response_stride)
        # labels = np.empty((data_size,) + label.shape)
        # labels[:] = label
        labels = np.array(list(self.c_labels))#construct_batch_gt_score_maps((17, 17), 6, len(instances))
   
        return [instances, exemplars], [labels], cats


from glob import glob
def parse_data(top_folder, test=False):
    imgs_to_anns = {}
    categories = glob(top_folder + "*/")
    categories = [os.path.basename(os.path.normpath(c)) for c in categories]
    data = {}
    cat_count = 0
    if test:
        categories = ["test"]
    for c in categories:
        if c == "test" and test == False:
            continue
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
            imgs = sorted(list(glob(os.path.join(v, "img/") + "*.jpg")))
            for image in imgs:
                bbox = ground_truth[count].split(",")
                bbox = [int(b) for b in bbox]
                x, y, w, h = bbox

                # instance_img = np.array(cv2.imread(image))
                # instance_img = cv2.rectangle(instance_img, (x, y), (x+w, y+h), (255,0,0), 2)
     
                # print (bbox, image)
                bbox = [int(b) for b in bbox]
                if bbox[2] == bbox[3]:
                    continue
                data[cat_count][vid_count].append({"file_path": image, "bbox": bbox, "video_folder": vid_id, "category_name": c, "category_id": cat_count, "video_id": vid_count, "total": len(imgs)})
                imgs_to_anns[image] = {"file_path": image, "bbox": bbox, "video_folder": vid_id, "category_name": c, "category_id": cat_count, "video_id": vid_count, "total": len(imgs)}
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

    
