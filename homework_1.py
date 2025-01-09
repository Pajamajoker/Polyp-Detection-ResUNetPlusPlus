import numpy as np

# Given arrays
arr1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr2 = np.array([21, 22, 23, 34, 25, 26, 27, 28, 29, 30])

# 1. Compute the dot product
dot_product = np.dot(arr1, arr2)
print("dot product is:", dot_product)

# 2. Concatenate and calculate mean and standard deviation
concat_arr = np.concatenate((arr1, arr2))
mean_concat = np.mean(concat_arr)
std_dev_concat = np.std(concat_arr)
print("mean concatenated array is :", mean_concat)
print("Standard deviation of concatenated array is :", std_dev_concat)

# 3. Sort and slice the array from index 2 to 8
sorted_arr = np.sort(concat_arr)
print("Elements from index 2 to 8 in sorted array:", sorted_arr[2:9])
