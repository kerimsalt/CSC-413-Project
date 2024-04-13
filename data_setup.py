import numpy as np
import os
import matplotlib.pyplot as plt

# 128 x 128 x 4
file_path = os.path.join('numpy_matrix_data', 'ordered_data.npy')
loaded_array = np.load(file_path, allow_pickle=True)
# Input data is 6171 by 2
# loaded_array[i][0] = 128 by 128 by 4 np matrix that represents the image
# loaded_array[i][1] = label = {buy:1, sel: -1, hold:0}

print("len of input data set")
print(loaded_array.shape)
print("first matrix")
print(loaded_array[0][0])
print("first label")
print(loaded_array[0][1])

print("Shape of first input image in the data set")
print(loaded_array[1][0].shape)
print(type(loaded_array[1][0]))
print("5311th matrix")
plt.imshow(loaded_array[5311][0])
print("label of 5311th")
print(loaded_array[5311][1])
plt.show()
