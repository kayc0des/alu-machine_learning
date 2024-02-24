import numpy as np
import cv2

file = cv2.imread('/Users/kayc0des/dev/learning/Machine Learning/alu-machine_learning/math/convolutions_and_pooling/convolutions_playground/images/laughter.jpeg')
image = np.array(file)

# black and white filter 
kernel = np.full((3, 3, 3), 1/3)

#alternative 
main_kernel = np.full((3, 3, 1), 1/3)

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

    # separating the channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # initialize convolved channels
    red_output = np.zeros((output_height, output_width, num_channels))
    green_output = np.zeros((output_height, output_width, num_channels))
    blue_output = np.zeros((output_height, output_width, num_channels))

    for i in range(output_height):
        for j in range(output_width):
            patch = red_channel[i:i+kernel_height, j:j+kernel_width]
            red_output[i,j] = np.sum(patch * kernel.reshape(1, kernel_height, kernel_width), axis=(1, 2))

    for i in range(output_height):
        for j in range(output_width):
            patch = green_channel[i:i+kernel_height, j:j+kernel_width]
            green_output[i,j] = np.sum(patch * kernel.reshape(1, kernel_height, kernel_width), axis=(1, 2))

    for i in range(output_height):
        for j in range(output_width):
            patch = blue_channel[i:i+kernel_height, j:j+kernel_width]
            blue_output[i,j] = np.sum(patch * kernel.reshape(1, kernel_height, kernel_width), axis=(1, 2))
    
    convolved_image = np.stack((red_output, green_output, blue_output), axis=2)
    print(convolved_image)

    try: 
        cv2.imwrite('output.jpeg', convolved_image)
        print('Image saved succesfully')
    except Exception as e:
        print(f'An error occurred: {e}')

black_white(image, main_kernel)
