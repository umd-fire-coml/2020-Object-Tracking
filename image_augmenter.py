import albumentations as A

def image_augment(img, bboxes):
    '''
    image_augment
        ARGS: img - numpy array of image
              bboxes - array of bounding boxes in COCO format
    '''
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        ], bbox_params=A.BboxParams(format='coco'))

    transformed = transform(
        image=img,
        bboxes=bboxes
    )
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    return transformed_image, transformed_bboxes

img = np.random.uniform(0, 1, (100, 100))
