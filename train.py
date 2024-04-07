import Model
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt


# Initialize the CNN
model = Model.CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Get the input
file_path = os.path.join('numpy_matrix_data', 'shuffled_data.npy')
input_data = np.load(file_path, allow_pickle=True)

i = 0
while i < 3:
    plt.imshow(input_data[i][0])
    plt.show()
    i += 1
