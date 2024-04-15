import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, in1=4, out1=32, out2=64, out3=128, fcb1=512):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in1, out_channels=out1, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(out3 * 16 * 16, fcb1)  # Adjusted input size to match the output of conv3
        self.fc2 = nn.Linear(fcb1, 1)  # Changed to 1 as per lab03 guidelines. (prev: 2 classes for classification)

    def forward(self, x):
        conv1_out = self.pool(torch.relu(self.conv1(x)))
        conv2_out = self.pool(torch.relu(self.conv2(conv1_out)))
        conv3_out = self.pool(torch.relu(self.conv3(conv2_out)))
        conv3_out = conv3_out.reshape(conv3_out.size(0), -1)  # Flatten the output from convolutional layers
        fc1_out = torch.relu(self.fc1(conv3_out))
        out = self.fc2(fc1_out)
        return out
