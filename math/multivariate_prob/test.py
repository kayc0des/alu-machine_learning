import numpy as np 

array = np.array([[90, 80, 5],
                 [60, 30, 9],
                 [100, 70, -1]])
sum_array = np.sum(array, axis=0)
mean = sum_array/3

num_colums = array.shape[1]

variance = []

difference_array = np.zeros((3,3))

for i in range(num_colums):
    column = array[:, i]
    squared_difference = np.sum((column - mean[i]) **2)
    value = squared_difference/(array.shape[0] - 1)
    variance.append(value)

    # step 7
    difference = column - mean[i]
    difference_array[i] = difference

print(f'mean = {mean}')
print(f'variance = {variance}')
print(difference_array)