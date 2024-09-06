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
    def __init__(self,
                 style_image,
                 content_image,
                 alpha=1e4,
                 beta=1):
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
        if not (isinstance(
            style_image, np.ndarray) and len(
                np.shape(style_image)) == 3 and np.shape(
                    style_image)[2] == 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')

        # Check content_image type and shape
        if not (isinstance(
            content_image, np.ndarray) and len(
                np.shape(content_image)) == 3 and np.shape(
                    content_image)[2] == 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')

        # Ensure alpha and beta are non-negative numbers
        if (
            type(alpha) is not float and type(
                alpha) is not int) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')

        if (
            type(beta) is not float and type(
                beta) is not int) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        # disable lazy execution tf v1.12
        tf.enable_eager_execution()

        # Instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

        self.load_model()
        self.generate_features()

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
        if not (
            isinstance(image, np.ndarray) and len(
                np.shape(image)) == 3 and np.shape(
                    image)[2] == 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape

        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        resized = tf.image.resize_bicubic(
            np.expand_dims(image, axis=0), size=(h_new, w_new))

        rescaled = resized / 255
        rescaled = tf.clip_by_value(rescaled, 0, 1)

        return rescaled

    # Public Instance Method
    def load_model(self):
        '''
        Creates the model used to calculate the loss
        '''

        # load vgg model
        vgg_model = tf.keras.applications.VGG19(
            include_top=False, weights='imagenet')

        # MaxPooling2D - AveragePooling 2D
        vgg_model.save('base')
        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg = tf.keras.models.load_model(
            'base', custom_objects=custom_objects)

        style_outputs = [
            vgg.get_layer(name).output for name in self.style_layers]
        content_outputs = [
            vgg.get_layer(self.content_layer).output]
        model_outputs = style_outputs + content_outputs

        model = tf.keras.models.Model(
            vgg.input, model_outputs, name="model")

        # Freeze weights
        model.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        '''
        Calculates gram matrices

        Args:
            input_layer: an instance of tf.tensor or
            tf.Variable of shape (1, h, w, c) containing
            the output whose gram matrix should be calculated

        Returns:
            A tf.Tensor of shape (1, c, c) containing
            the gram matrix of input layer
        '''

        # Run checks
        if not (isinstance(input_layer, tf.Tensor) or
                isinstance(input_layer, tf.Variable)) or len(
                    input_layer.shape) != 4:
            raise TypeError('input_layer must be a tensor of rank 4')

        channels = int(input_layer.shape[-1])
        a = tf.reshape(input_layer, [1, -1, channels])
        n = tf.shape(a)[1]

        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def generate_features(self):
        '''
        Extract features used to calculate neural style cost
        '''
        self.gram_style_features = [
            self.gram_matrix(
                self.model.get_layer(name).output) for name in self.style_layers
        ]
        self.content_feature = self.model.get_layer(self.content_layer).output
