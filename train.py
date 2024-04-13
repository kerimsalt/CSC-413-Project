import Model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
import matplotlib.pyplot as plt


# Initialize the CNN
model = Model.CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Get the input
file_path = os.path.join('numpy_matrix_data', 'ordered_data.npy')
input_data = np.load(file_path, allow_pickle=True)

total_size = len(input_data)
split1_size = int(total_size * 0.6)
split2_size = int(total_size * 0.2)

# Split the input data
training_set = input_data[:split1_size]
test_set = input_data[split1_size:split1_size + split2_size]
validation_set = input_data[split1_size + split2_size:]

# Split the training data into batches of size batch_size
batch_size = 300
batches = Model.split_to_batches(training_set=training_set, batch_size=batch_size)

# example training with 1 datapoint
print("Small example")
first_image = input_data[0][0]
print(first_image.shape)
first_tensor = torch.tensor(first_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
raw_output = model(first_tensor)
output = torch.softmax(raw_output, dim=1)
output_label = Model.convert_raw_output_to_label(raw_output=raw_output)
print("raw")
print(output)
print("predicted label")
print(output_label)
print("actual label")
print(input_data[0][1])

# Actual training

# optimizer.zero_grad()
# loss.backward()
# optimizer.step()

training_loss = []
x_axis = []
# training loop
index = 0
optimizer.zero_grad()  # Zero the gradients

while index < len():
    current_tensor = torch.tensor(input_data[index][0], dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    raw_output = model(current_tensor)
    output = torch.softmax(raw_output, dim=1)
    label = torch.zeros(output.shape)
    label[0, input_data[index][1] + 1] = 1
    # print("output")
    # print(output.shape)
    # print(output)
    # print("label")
    # print(label.shape)
    # print(label)
    loss = criterion(output, label)  # Compute the loss
    training_loss.append(loss.item())

    loss.backward()  # Backward pass: compute gradients
    optimizer.step()  # Update model parameters based on gradients

    index += 1
    x_axis.append(index)

for i in range(len(batches)):
    current_batch = batches[i]

plt.plot(x_axis, training_loss)
plt.show()


# 60 train, 20 validation, 20 test
