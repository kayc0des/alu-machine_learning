#!/usr/bin/env python3

''' A function that perfroms same convolution on grayscale images'''

import numpy as np
import matplotlib.pyplot as plt


def convolve_grayscale_same(images, kernel):
    pass

if __name__ == '__main__':
    dataset = np.load('/Users/kayc0des/dev/learning/Machine Learning/alu-machine_learning/math/convolutions_and_pooling/convolutions_playground/data/mnist.npz')
    images = dataset['X_train']
    print(images.shape)
    kernel = np.array([[1 ,0, -1], [1, 0, -1], [1, 0, -1]])
    images_conv = convolve_grayscale_same(images, kernel)
    print(images_conv.shape)

    plt.imshow(images[0], cmap='gray')
    plt.show()
    plt.imshow(images_conv[0], cmap='gray')
    plt.show()