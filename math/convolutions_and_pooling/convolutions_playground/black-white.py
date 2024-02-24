import numpy as np
import cv2
from PIL import Image

file = cv2.imread('/Users/kayc0des/dev/learning/Machine Learning/alu-machine_learning/math/convolutions_and_pooling/convolutions_playground/images/laughter.jpeg')
image = np.array(file)

#alternative 
main_kernel = np.full((3, 3, 1), 1/9)

kernel = np.array([[0.299, 0.587, 0.114],
                   [0.299, 0.587, 0.114],
                   [0.299, 0.587, 0.114]])

def black_white(image, kernel):
    ''' Returns a black and white photo'''
    
    #Validate params
    if not isinstance(image, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise ValueError('The image and kernel have to be ndarrays')
    # if image.shape[2] != kernel.shape[2]:
    #     raise ValueError('Both image and kernel should have equal numbers of channels')
    
    # Dimensions of the image
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Dimensions of the kernel
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # Number of channels of convolved image
    num_channels = image.shape[2]

    #dimensions of output 
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # initialize convolved channels
    convolved_image = np.zeros((output_height, output_width, num_channels))
    
    for channel in range(num_channels):
        for i in range(output_height):
            for j in range(output_width):
                patch = image[i:i+kernel_height, j:j+kernel_width, channel]
                value = np.sum(patch * kernel[:, :, 0])
                convolved_image[i, j, channel] = value

    try: 
        # cv2.imwrite('output.jpeg', convolved_image)
        convolved_image_pil = Image.fromarray(convolved_image.astype(np.uint8))
        convolved_image_pil.save('output12.jpg')
        print('Image saved succesfully')
    except Exception as e:
        print(f'An error occurred: {e}')

black_white(image, main_kernel)
