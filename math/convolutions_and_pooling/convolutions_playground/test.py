import numpy as np

array = np.array([[[1, 2, 3, 5, -1, 0],
                   [4, 5, 6, 3, 2, 1],
                   [7, 8, 9, 4, -5, 6],
                   [1, 2, 3, 5, -1, 0],
                   [4, 5, 6, 3, 2, 1],
                   [7, 8, 9, 4, -5, 6]],
                   [[7, 8, 9, 2, 3, 1],
                    [1, 2, 3, 4, 6, 8],
                    [4, 5, 6, 3, 2, 0],
                    [7, 8, 9, 2, 3, 1],
                    [1, 2, 3, 4, 6, 8],
                    [4, 5, 6, 3, 2, 0]]])

print(array.shape)

for i in range(4):
    for j in range(4):
        area = array[:, i:i+3, j:j+3]
        print(f"when i = {i} and j = {j} then the 3x3 matrix is\n {area}\n")

