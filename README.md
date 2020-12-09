# Object Tracking with Siamese Networks implemented in Tensorflow

## Description of Product and Model Architecture
Object tracking is a specific field within computer vision that aims to track objects as they move across a series of video frames. The goal of object tracking is to train a model and estimate the target object present in the scene from previous frames. We completed this by taking a starting bounding box coordinate and creating a unique ID for each of the initial detections of the object and tracking them as they move while keeping the unique ID of each object. The specific learning model we used is a Siam-FC model with an AlexNet backend. The model takes in an instance image, and a search image, and uses the backend to process each image into a embedding, then uses cross-correlation to find the search image in the instance image. This project uses the [Lasot](http://vision.cs.stonybrook.edu/~lasot/) dataset.

## Samples
![Alt text](samples/1.png "Sample Image")
![Alt text](samples/video.gif "Sample Video")

## Youtube Video

## Directory Guide 

## How to get the training started
First, you need to install the environment.yaml and activate it with anaconda.
Then, go to the [Lasot](http://vision.cs.stonybrook.edu/~lasot/) dataset, and download whatever image type you want (airplane, dog, cat, etc) and extract them into a folder called data. Then, you can add a test folder, and copy individual videos into the test folder for testing. To run the training, just run python train.py

## Testing and Visualization Notebook
Run the python create_video.py. Just make sure you go into the create_video.py file and modify what folder you want to use in the test folder.