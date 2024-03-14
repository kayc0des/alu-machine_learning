import numpy as np

# Example dataset
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

# Calculate the correlation matrix
correlation_matrix = np.corrcoef(data, rowvar=False)

print("Correlation Matrix:")
print(correlation_matrix)