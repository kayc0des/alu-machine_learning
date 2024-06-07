import numpy as np

# Define data and labels
data = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
labels = np.array([0, 1, 0, 1])

len(data) == data.shape[0]
print(len(data))

# Generate a permutation of indices
permuted_indices = np.random.permutation(len(data))

print(permuted_indices)

# Shuffle data and labels
shuffled_data = data[permuted_indices]
shuffled_labels = labels[permuted_indices]

print(shuffled_data)
print(shuffled_labels)
