import numpy as np
import cv2

file = cv2.imread('/Users/kayc0des/dev/learning/Machine Learning/alu-machine_learning/math/convolutions_and_pooling/convolutions_playground/images/laughter.jpeg')
image = np.array(file)

# Define a simple averaging kernel for black and white conversion
kernel = np.full((3, 3), 1/9)  # 3x3 averaging kernel

def black_white(image, kernel):
    ''' Returns a black and white photo'''
    
    #Validate params
    if not isinstance(image, np.ndarray) or not isinstance(kernel, np.ndarray):
        raise ValueError('The image and kernel have to be ndarrays')

    # Dimensions of the image
    image_height = image.shape[0]
    image_width = image.shape[1]

    # Dimensions of the kernel
    kernel_height = kernel.shape[0]
    kernel_width = kernel.shape[1]

    # Number of channels of the image
    num_channels = image.shape[2]

    #dimensions of output 
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Initialize convolved channels
    convolved_image = np.zeros((output_height, output_width, num_channels))

    # Convolve each channel separately
    for channel in range(num_channels):
        for i in range(output_height):
            for j in range(output_width):
                patch = image[i:i+kernel_height, j:j+kernel_width, channel]
                convolved_value = np.sum(patch * kernel)
                convolved_image[i, j, channel] = convolved_value
    
    return convolved_image

# Perform black and white conversion
convolved_image = black_white(image, kernel)

# Save the image
cv2.imwrite('output2.jpg', convolved_image)
print('Image saved successfully')
