#!/usr/bin/env python3

''' a function that performs a valid grayscale convolution'''

#import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import cv2

# image = cv2.imread('/Users/kayc0des/dev/learning/Machine Learning/alu-machine_learning/math/convolutions_and_pooling/convolutions_playground/images/dog.png')
# image_np = np.array(image)
# print(f'the numpy image has shape {image_np.shape}')

# #define function
# def convolve_grayscale_valid(image, kernel):
#     m = image.shape[2]
#     h = image.shape[0]
#     w = image.shape[1]

#     kh, kw = kernel.shape

#     # dimensions of convolved image
#     conv_height = h - kh + 1
#     conv_width = w - kw + 1
    
#     #initialize the convolved image
#     convolved_image = np.zeros((conv_height, conv_width, m), dtype=int)
#     print(f'The shape of the convolved image is {convolved_image.shape}')

#     #iterate over the image input
#     for i in range(conv_height):
#         for j in range (conv_width):
#             shadow_area = image[i:i+kh+1, j:j+kw]
#             convolved_image[i][j] = np.sum(kernel * shadow_area, axis=(1, 2))
    
#     return convolved_image

# print(convolve_grayscale_valid(image, kernel))

kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

def convolve_grayscale(images, kernel):
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
    print(convolve_grayscale(images, kernel))