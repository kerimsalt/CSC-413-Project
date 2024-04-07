import numpy as np

# Load the NPZ file
data = np.load('data5.npz')

# Print the contents of the NPZ file
print("Keys in the NPZ file:", data.files)

# Access and print individual arrays
for key in data.files:
    print("Data in", key, ":", data[key])
