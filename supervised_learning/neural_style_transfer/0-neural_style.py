#!/usr/bin/env python3
'''Create a class NST that performs tasks for neural style transfer'''


import numpy as np
import tensorflow as tf


class NST:
    '''
    This class performs neural style transfer
    '''

    # Declare public class attributes
    # Accessible from outside the class
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    # Class constructor
    def __init__(self, style_image,
                 content_image, alpha=1e4, beta=1):
        '''
        Creates an instance of the NST class

        Args:
            style_image: image np array used as style reference
            content_image: image np.array used as content reference
            alpha: the weight of the content cost
            beta: the weight of the style cost

        Returns:
            An instance of the NST Class
        '''

        # Check style_image type and shape
        if not (isinstance(style_image, np.ndarray) and
            len(np.shape(style_image)) == 3 and
            np.shape(style_image)[2] == 3):
            raise TypeError('style_image must be a numpy.ndarray with shape (h, w, 3)')

        # Check content_image type and shape
        if not (isinstance(content_image, np.ndarray) and
            len(np.shape(content_image)) == 3 and
            np.shape(content_image)[2] == 3):
            raise TypeError(
                    'content_image must be a numpy.ndarray with shape (h, w, 3)')

        # Ensure alpha and beta are non-negative numbers
        if (type(alpha) is not float or
            type(alpha) is not int or
            alpha < 0):
            raise TypeError('alpha must be a non-negative number')

        if (type(beta) is not float or
                type(beta) is not int or
                beta < 0):
            raise TypeError('beta must be a non-negative number')

        tf.enable_eager_execution()

        # Instance attributes
        self.style_image = style_image
        self.content_image = content_image
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        '''
        Rescales an image such that
        values are between 0 and 1 and largest
        side is 512 pixels

        Args:
            image - a np.ndarray with shape (h, w, 3)

        Returns:
            the scaled image    
        '''
        if not (isinstance(image, np.ndarray) and
            len(np.shape(image)) == 3 and
            np.shape(image)[2] == 3):
            raise TypeError('image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape
        
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize_bicubic(np.expand_dims(image, axis=0),
                                          size=(h_new, w_new))
        rescaled = resized / 255
        rescaled = tf.clip_by_value(rescaled, 0, 1)
        
        return (rescaled)
