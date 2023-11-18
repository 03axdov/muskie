import numpy as np

arr1 = np.array([[1,2,3],[4,5,6], [7,8,9]])
arr2 = np.array([[10,11,12],[13,14,15], [16,17,18]])

print(np.dstack((arr1,arr2)))