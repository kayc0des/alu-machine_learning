#!/usr/bin/env python3

''' a function that performs a valid grayscale convolution'''

#import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import cv2

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

def convolve_grayscale_valid(images, kernel):
    ''' This function performs a vlaid convolution on 
    grayscale images (one channel)'''

    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh, kw = kernel.shape[0], kernel.shape[1]

    #dimenshions of conv image
    conv_height = h - kh + 1
    conv_width = w - kw + 1

    #initialize the convolved image
    convolved_image = np.zeros((m, conv_height, conv_width), dtype=int)
    
    #convolve iteration
    for i in range(conv_height):
        for j in range(conv_width):
            #extract patches
            patches = images[:, i:i+kh, j:j+kw]
            convolved_image[:, i, j] = np.sum(patches * kernel.reshape(1, kh, kw), axis=(1, 2))
    return convolved_image

# Run and Debug
if __name__ == '__main__': 
    dataset = np.load('/Users/kayc0des/dev/learning/Machine Learning/alu-machine_learning/math/convolutions_and_pooling/convolutions_playground/data/mnist.npz')
    images = dataset['x_train']
    print(images.shape)
    images_conv = convolve_grayscale_valid(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()