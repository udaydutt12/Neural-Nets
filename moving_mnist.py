#!/usr/bin/env python3
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import cv2 as cv

mnist_builder = tfds.builder("moving_mnist")
datasets = mnist_builder.as_dataset()

# tensorflow hasn't split up the test and train,
# they only provided 10,000 test data,
# so we need to manually shuffle the data
# and split it into train and test manually.
# use a 70:30 train:test split.
dataset = datasets["test"] 
assert isinstance(dataset, tf.data.Dataset)

# And then the rest of your input pipeline
#dataset = dataset.repeat().shuffle(1024).batch(128)
#dataset = dataset.prefetch(2)

#iterator = dataset.make_one_shot_iterator()
#features = iterator.get_next()
#image_sequence = features['image_sequence']
# currently, the npimage has a shape of (20,64,64,1)  
# looks like the first video is 20 frames 
# of a 64x64 prixel image, 
# with a 1 dimensional intensity(aka grayscale)
iter = dataset.as_numpy_iterator()
cur = iter.next()
x = cur['image_sequence']
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter('test.avi',fourcc,fps = 1.0,frameSize = (64,64),isColor = False)
for frame in x:
    #cv.imshow("yo",frame)
    out.write(frame)
    #cv.waitKey()
#cv.destroyAllWindows()
out.release()
cap = cv.VideoCapture('test.avi')
while(cap.isOpened()):
    _, frame = cap.read()
    if frame is None:
        break
    cv.imshow('result',frame)
    if cv.waitKey(1) == ord('q'): # wait 1ms
        break
cv.destroyAllWindows()
'''
for batch in dataset:
    image_sequence = batch['image_sequence']
    npimage = image_sequence.numpy()
    for vid in npimage:
        for frame in vid:
            cv.imshow("yo",frame)
            if cv.waitKey(1) == ord('q'):
                break
        break
    break
cv.destroyAllWindows()
'''
