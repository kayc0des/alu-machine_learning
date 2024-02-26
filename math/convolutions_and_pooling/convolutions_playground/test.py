import numpy as np

dataset = np.load('/Users/kayc0des/dev/learning/Machine Learning/alu-machine_learning/math/convolutions_and_pooling/convolutions_playground/data/mnist.npz')
images = dataset['x_train']
#image = images[0]

#image = np.pad(image, (1, 1), 'constant', constant_values=(0, 0))
placeholder = np.zeros((60000, 30, 30), dtype=int)

for i in range(len(images)):
    for image in images:
        placeholder[i] = np.pad(image, (1, 1), 'constant', constant_values=(0, 0))

print(placeholder.shape)
# print(image[0].shape)