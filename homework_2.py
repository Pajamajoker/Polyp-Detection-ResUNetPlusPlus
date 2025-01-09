import numpy as np

# Given matrices A and B
A = np.array([
    [3, 8, 1],
    [4, 6, 2],
    [7, 5, 9]
])

B = np.array([
    [1, 2, 3],
    [8, 9, 10],
    [11, 23, 30]
])

# 1. Transpose of A and determinant of B
transpose_A = A.T
print(" A transpose is :\n", transpose_A)

det_B = np.linalg.det(B)
print(" B daeterminant:", det_B)

# 2. Proving that det(A * B) = det(A) * det(B)
det_A = np.linalg.det(A)
product_AB = np.dot(A, B)
det_AB = np.linalg.det(product_AB)
print("Determinant of A:", det_A)
print("Determinant of A * B:", det_AB)
print("Product of determinants det(A) * det(B):", det_A * det_B)
print("Property holds:", np.isclose(det_AB, det_A * det_B))

# 3. Solve AX = B (AX = Y where Y is the product of A and X)
# Here I have added the try catch block because it is possible
# that X does not exist and in that case np will throu erro
# so I catch error with except block
try:
    # Compute the solution X by solving the equation AX = B
    X = np.linalg.solve(A, B)
    print("Solution matrix X:\n", X)
except np.linalg.LinAlgError:
    print("Cannot solve AX = B, X Not found")
