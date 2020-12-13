import numpy as np 
import cv2 


def calculate_x_z_sz(box):
    context = 0.5*(box[-2]+box[-1])
    z_sz = np.sqrt(np.prod((box[-2]+context)*(box[-1]+context)))
    x_sz = float(255) / 127 * z_sz
    return x_sz, z_sz


def crop_and_resize(img, box, size, out_size, border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    size = round(size)
    box = [box[1], box[0], box[3], box[2]] #[y, x, h, w]
    box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
    center = box[:2]

    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    pads = np.concatenate((
        -corners[:2], corners[2:] - img.shape[:2]))
    npad = max(0, int(pads.max()))
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # crop image patch
    corners = (corners + npad).astype(int)
    patch = img[corners[0]:corners[2], corners[1]:corners[3]]

    # resize to out_size
    patch = cv2.resize(patch, (out_size, out_size))
    return patch