from glob import glob 
from tensorflow import keras
import tensorflow as tf
import cv2
import numpy as np
from model import *
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import exposure 
from utils import crop_and_resize, calculate_x_z_sz

def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y


def normalize_map(score_map):
    score_map = (score_map - np.min(score_map))/(np.max(score_map) - np.min(score_map))
    return score_map

m = keras.models.load_model('saved_model/', custom_objects={"loss_fn": loss_fn, "aucMetric": aucMetric})

images = list(glob("data/test/airplane-20/img/*.jpg"))
images = sorted(images)

# exemplar = np.array(cv2.imread("data/test/airplane-20/img/reference_good.jpg"))
exemplar = np.array(cv2.imread("data/test/airplane-20/img/00000001.jpg"))
exemplar_base = np.array(list(exemplar))


height, width, layers = exemplar.shape
size = (width,height)

out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

with open("data/test/airplane-20/groundtruth.txt", "r") as f:
    box = [int(b) for b in f.read().split("\n")[0].split(",")]

x, y, w, h = box

# x = cv2.rectangle(exemplar, (x, y), (x+w, y+h), (255, 0, 0), 2)
# plt.imshow(x)
# plt.show()


hp = {
	"response_up": 8,
	"window_influence": 0.175,
	"z_lr": 0.03,
	"scale_num": 3,
	"scale_step": 1.047,
	"scale_penalty": 0.9825,
	"scale_lr": 0.68,
	"scale_min": 0.2,
	"scale_max": 5
}


scale_factors = hp["scale_step"]**np.linspace(-np.ceil(hp["scale_num"]/2), np.ceil(hp['scale_num']/2), hp['scale_num'])

hann_1d = np.expand_dims(np.hanning(17), axis=0)
penalty = np.transpose(hann_1d) * hann_1d
penalty = penalty / np.sum(penalty)

x_sz, z_sz = calculate_x_z_sz(box)

min_z = hp['scale_min'] * z_sz
max_z = hp['scale_max'] * z_sz
min_x = hp['scale_min'] * x_sz
max_x = hp['scale_max'] * x_sz

# images = images[210:]
co = 0
for image in images[1:]:
    if "reference" in image:
        continue

    with open("data/test/airplane-20/groundtruth.txt", "r") as f:
        box = [int(b) for b in f.read().split("\n")[co].split(",")]

    scaled_exemplar = z_sz * scale_factors
    scaled_search_area = x_sz * scale_factors
    scaled_target_w = box[-2] * scale_factors
    scaled_target_h = box[-1] * scale_factors

    target_w = box[-2]
    target_h = box[-1]

    x, y, w, h = box
    box2 = [y, x, h, w]
    box2 = np.array([
            box2[1] + (box2[3]) / 2,
            box2[0] + (box2[2]) / 2,
            box2[3], box2[2]], dtype=np.float32)
    pos_x, pos_y = box2[:2]


    print (image)
    image = np.array(cv2.imread(image))
    copy_img = np.array(image)
    copy_img2 = np.array(image)
    copy_img3 = np.array(image)
    copy_img = cv2.resize(copy_img, (255, 255))



    exemplar = [
        crop_and_resize(
            exemplar, [y, x, h, w], x_sz * f,
            out_size=127,
            border_value=[0,0,0]) for f in scale_factors
    ]

    # for e in exemplar:
    # plt.imshow(np.array(exemplar[0]).astype(np.uint8))
    # plt.show()

    exemplar  = np.stack(exemplar, axis=0)

    print (exemplar.shape)

    scores = m.predict([np.array([copy_img] * 3), exemplar])

    print (scores.shape)

    scores[0,:,:] = hp['scale_penalty']*scores[0,:,:]
    scores[2,:,:] = hp['scale_penalty']*scores[2,:,:]

    new_scale_id = np.argmax(np.amax(scores, axis=(1,2)))
            # update scaled sizes
    x_sz = (1-hp["scale_lr"])*x_sz + hp["scale_lr"]*scaled_search_area[new_scale_id]        
    target_w = (1-hp["scale_lr"])*target_w + hp["scale_lr"]*scaled_target_w[new_scale_id]
    target_h = (1-hp["scale_lr"])*target_h + hp["scale_lr"]*scaled_target_h[new_scale_id]

    score = scores[new_scale_id,:,:]
    score = score - np.min(score)
    score = score/np.sum(score)

    score = (1-hp["window_influence"])*score + hp["window_influence"]*penalty
    pos_x, pos_y = _update_target_position(pos_x, pos_y, score, 17, 6, 127, 8, x_sz)

    box = [pos_x-target_w/2, pos_y-target_h/2, target_w, target_h] #[x, y, w, h]
    box = [int(b) for b in box]

    c = cv2.rectangle(copy_img3, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
    # plt.imshow(c)
    # plt.show()

    # update the target representation with a rolling average
    if hp['z_lr']>0:
        new_exemplar_img = copy_img2

        exemplar_base = (1-hp['z_lr'])*np.asarray(exemplar_base) + hp['z_lr']*np.asarray(new_exemplar_img)
        exemplar = np.array(list(exemplar_base))


    #exemplar = copy_img2#np.array(list(exemplar_base))

    # print (new_scale_id)
    # print (target_w, target_h)

    # exit()

    out.write(c)

    
    z_sz = (1-hp['scale_lr'])*z_sz + hp["scale_lr"]*scaled_exemplar[new_scale_id]
    



