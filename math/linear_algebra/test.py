# import numpy as np

# mat1 = [[1, 2],
#         [3, 4],
#         [5, 6]]
# mat2 = [[1, 2, 3, 4],
#         [5, 6, 7, 8]]

# A = np.array(mat1)
# B = np.array(mat2)

# C = A @ B
# print(C)

# #!/usr/bin/env python3
# def mat_mul(mat1, mat2):
#     C = [[0] * len(mat2) for _ in range(len(mat1))]
#     #loop through the rows of matrix B
#     # loop through the colums of matrix B
#     for row_a in range(len(mat1)):
#         for column_b in range(len(mat2[0])):
#             for column_a in range(len(mat1[0])):
#                 C[row_a][column_b] += mat1[row_a][column_a] * mat2[column_a][column_b]
#     return C

# # mat1 = [[1, 2],
# #         [3, 4]]
# # mat2 = [[1, 0],
# #         [0, 1]]
# # print(mat_mul(mat1, mat2))

#!/usr/bin/env python3

#!/usr/bin/env python3

def matrix_shape(matrix):
    shape = []

    def nested_shape(sub_matrix):
        shape.append(len(sub_matrix))
        if isinstance(sub_matrix[0], list):
            for sublist in sub_matrix:
                nested_shape(sublist)

    nested_shape(matrix)
    return list(set(shape))

mat1 = [[1, 2], [3, 4]]
print(matrix_shape(mat1))

mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        [[16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
print(matrix_shape(mat2))


