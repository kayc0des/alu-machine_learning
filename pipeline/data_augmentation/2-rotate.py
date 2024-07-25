# Rotates and image by 90 degrees counter-clockwise

# Docs -> https://www.tensorflow.org/api_docs/python/tf/image/rot90

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt


def rotate_image(image):
    '''
    Rotates as image 90 degress counter-clockwise
    
    Params:
        - image: 3d tf.tensor
    
    Returns:
        - rotated image
    '''
    return tf.image.rot90(
        image, k=1, name=None
    )
    
if __name__ == '__main__':
    doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)
    for image, _ in doggies.shuffle(10).take(1):
        plt.imshow(rotate_image(image))
        plt.show()
