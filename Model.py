import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)  # Adjusted input size to match the output of conv3
        self.fc2 = nn.Linear(512, 3)  # 3 classes for classification

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten the output from convolutional layers
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def convert_raw_output_to_label(raw_output):
    labels = {0: -1, 1: 0, 2: 1}
    predicted_label = torch.argmax(raw_output)  # the raw output is in tensor form
    return labels[predicted_label.item()]


def split_to_batches(training_set, batch_size):
    """
    split the training set into batches of batch_size
    """
    np.random.shuffle(training_set)  # Shuffle the array randomly
    batches = []
    num_batches = len(training_set) // batch_size

    for i in range(num_batches):
        batch = training_set[i * batch_size: (i + 1) * batch_size]  # Extract a batch of size k
        batches.append(batch)

    # If there are remaining elements, create a last batch with fewer elements
    if len(training_set) % batch_size != 0:
        last_batch = training_set[num_batches * batch_size:]
        batches.append(last_batch)
    return batches


# Create an instance of the CNN
model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming you have a numpy array img with shape (128, 128, 4) representing your input image
# You would need to transpose it to get it in the right shape (batch size, channels, height, width)
# Also, normalizing the image is usually necessary
img = np.random.randn(128, 128, 4)  # Example random input image with 4 channels
img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
img_tensor /= 255.0  # Normalize the image

# Forward pass
outputs = model(img_tensor)

# print(outputs.shape)  # Check the shape of the output
print("Model is initiated")

# Example of training the model:
# loss = criterion(outputs, labels)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
