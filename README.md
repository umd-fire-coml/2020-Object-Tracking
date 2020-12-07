# Object Tracking with Siamese Networks implemented in Tensorflow

## Description of Product and Model Architecture
Object tracking is a specific field within computer vision that aims to track objects as they move across a series of video frames. The goal of object tracking is to train a model and estimate the target object present in the scene from previous frames. We completed this by taking a starting bounding box coordinate and creating a unique ID for each of the initial detections of the object and tracking them as they move while keeping the unique ID of each object. The specific learning model we used is a Siamese network, which differentiates two neural networks. This is done by creating two identical neural networks and feeding a different image from an image pair into each of them. These networks are then fed into a contrastive loss function that calculates the similarity of the two images.

## Youtube Video

## Directory Guide 

## How to get the training started

## Testing and Visualization Notebook
