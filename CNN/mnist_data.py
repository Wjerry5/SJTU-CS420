
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import gzip
import os

import numpy
from scipy import ndimage

from six.moves import urllib

import tensorflow as tf

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
DATA_DIRECTORY = "data"

# Params for MNIST
IMAGE_SIZE = 45
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.

# Download MNIST data

# Extract the images
def extract_data(filename, num_images):

    data = np.fromfile("../data/"+filename,dtype=np.uint8)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images,IMAGE_SIZE,IMAGE_SIZE, NUM_CHANNELS)
    data = numpy.reshape(data, [num_images, -1])
    return data
# Extract the labels
def extract_labels(filename, num_images):

    labels=np.fromfile("../data/"+filename,dtype=np.uint8)
    num_labels_data = len(labels)
    one_hot_encoding = numpy.zeros((num_labels_data,NUM_LABELS))
    one_hot_encoding[numpy.arange(num_labels_data),labels] = 1
    one_hot_encoding = numpy.reshape(one_hot_encoding, [-1, NUM_LABELS])
    return one_hot_encoding
# Augment training data
def expend_training_data(images, labels):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('expanding data : %03d / %03d' % (j,numpy.size(images,0)))

        # register original data
        expanded_images.append(x)
        expanded_labels.append(y)

        # get a value for the background
        # zero is the expected value, but median() is used to estimate background's value 
        bg_value = numpy.median(x) # this is regarded as background's value        
        image = numpy.reshape(x, (-1, 45))

        for i in range(4):
            # rotate the image with random degree
            angle = numpy.random.randint(-30,30,1)
            new_img = ndimage.rotate(image,angle,reshape=False, cval=bg_value)

            # shift the image with random distance
            shift = numpy.random.randint(-10, 10, 2)
            new_img_ = ndimage.shift(new_img,shift, cval=bg_value)

            # register new training data
            expanded_images.append(numpy.reshape(new_img_, 45*45))
            expanded_labels.append(y)

    # images and labels are concatenated for random-shuffle at each epoch
    # notice that pair of image and label should not be broken
    expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
    numpy.random.shuffle(expanded_train_total_data)

    return expanded_train_total_data

# Prepare MNISt data
def prepare_MNIST_data(use_data_augmentation=True):
    # Get the data.
    #train_data_filename = maybe_download('train-images-idx3-ubyte.gz')
    #train_labels_filename = maybe_download('train-labels-idx1-ubyte.gz')
    #test_data_filename = maybe_download('t10k-images-idx3-ubyte.gz')
    #test_labels_filename = maybe_download('t10k-labels-idx1-ubyte.gz')

    # Extract it into numpy arrays.
    train_data = extract_data('mnist_train_data', 60000)
    train_labels = extract_labels('mnist_train_label', 60000)
    test_data = extract_data('mnist_test_data', 10000)
    test_labels = extract_labels('mnist_test_label', 10000)

    # Generate a validation set.
    validation_data = train_data[:VALIDATION_SIZE, :]
    validation_labels = train_labels[:VALIDATION_SIZE,:]
    train_data = train_data[VALIDATION_SIZE:, :]
    train_labels = train_labels[VALIDATION_SIZE:,:]

    # Concatenate train_data & train_labels for random shuffle
    if use_data_augmentation:
        train_total_data = expend_training_data(train_data, train_labels)
    else:
        print(train_data.shape,train_labels.shape)
        train_total_data = numpy.concatenate((train_data, train_labels), axis=1)

    train_size = train_total_data.shape[0]

    return train_total_data, train_size, validation_data, validation_labels, test_data, test_labels


