import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

a_sum = np.sum(a, axis=0) / a.shape[0]
a_mean = np.mean(a, axis=0)
stddev = np.std(a, axis=0)

print(a_sum)
print(a_mean)
print(stddev)

