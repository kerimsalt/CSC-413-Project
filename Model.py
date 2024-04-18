import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import models, transforms
import data_setup


alexnet = models.alexnet(pretrained=True)
def compute_features(data):
    fets = []
    for img, t in data:
        features = alexnet.features(img.unsqueeze(0)).detach().squeeze()  # TODO
        fets.append((features, t),)
    return fets

train_data_s = data_setup.train_data
validation_data_s = data_setup.val_data
test_data_s = data_setup.test_data

train_data_fets = compute_features(train_data_s)
valid_data_fets = compute_features(validation_data_s)
test_data_fets = compute_features(test_data_s)

img, label = train_data_s[0]
features = alexnet.features(img.unsqueeze(0)).detach()

print(features.shape)

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fc1 = nn.Linear(256*6*6, 1)
        # TODO: What layer need to be initialized?

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6) # flatten the input
        z = self.fc1(x) # TODO: What computation needs to be performed?
        return z



# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self, in1=4, out1=64, out2=128, out3=256, out4=512, fcb1=25088, fcb2=2048, fcb3=100, fcb4=1):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in1, out_channels=out1, kernel_size=7, stride=1, padding=4)
        self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out1, kernel_size=7, stride=1, padding=4)
        self.conv3 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=7, stride=1, padding=4)
        self.conv4 = nn.Conv2d(in_channels=out2, out_channels=out2, kernel_size=7, stride=1, padding=4)
        self.conv5 = nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=7, stride=1, padding=4)
        self.conv6 = nn.Conv2d(in_channels=out3, out_channels=out3, kernel_size=7, stride=1, padding=4)
        self.conv7 = nn.Conv2d(in_channels=out3, out_channels=out4, kernel_size=7, stride=1, padding=4)
        self.conv8 = nn.Conv2d(in_channels=out4, out_channels=out4, kernel_size=7, stride=1, padding=4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avg = nn.AdaptiveAvgPool2d((7,7))
        self.fc1 = nn.Linear(fcb1, fcb2)  # Adjusted input size to match the output of conv3
        self.fc2 = nn.Linear(fcb2, fcb3)
        self.fc3 = nn.Linear(fcb3, fcb3)
        self.fc4 = nn.Linear(fcb3, fcb4) # Changed to 1 as per lab03 guidelines. (prev: 2 classes for classification)
        self.dropout = nn.Dropout(0.5, inplace=False)

    def forward(self, x):
        # print(x.shape)
        conv1_out = torch.relu(self.conv1(x))
        # print(conv1_out.shape)
        conv2_out = self.pool(torch.relu(self.conv2(conv1_out)))
        # print(conv2_out.shape)
        conv3_out = torch.relu(self.conv3(conv2_out))
        # print(conv3_out.shape)
        conv4_out = self.pool(torch.relu(self.conv4(conv3_out)))
        # print(conv4_out.shape)
        conv5_out = torch.relu(self.conv5(conv4_out))
        conv6_out = torch.relu(self.conv6(conv5_out))
        conv7_out = self.pool(torch.relu(self.conv7(conv6_out)))
        conv8_out = self.avg(torch.relu(self.conv8(conv7_out)))
        # print(conv8_out.shape)
        conv8_out = conv8_out.reshape(conv8_out.size(0), -1)  # Flatten the output from convolutional layers
        # print(conv8_out.shape)
        # conv3_out = torch.flatten(conv3_out)
        # print(conv3_out.shape)
        x = torch.relu(self.fc1(conv8_out))
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = torch.relu(self.fc2(x))
        # print(x.shape)
        x = torch.relu(self.fc3(x))
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        out = self.fc4(x)
        # print(out.shape)
        return out


# class CNN(nn.Module):
#     def __init__(self, in1=4, out1=32, out2=64, out3=128, fcb1=512):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in1, out_channels=out1, kernel_size=7, stride=1, padding=4)
#         self.conv2 = nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=7, stride=1, padding=4)
#         self.conv3 = nn.Conv2d(in_channels=out2, out_channels=out3, kernel_size=7, stride=1, padding=4)
#         self.conv4 = nn.Conv2d(in_channels=out3, out_channels=out3, kernel_size=7, stride=1, padding=4)
#         self.avg = nn.AdaptiveAvgPool2d((7,7))
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(out3*7*7, fcb1)  # Adjusted input size to match the output of conv3
#         self.fc2 = nn.Linear(fcb1, 1)  # Changed to 1 as per lab03 guidelines. (prev: 2 classes for classification)

#     def forward(self, x):
#         conv1_out = self.pool(torch.relu(self.conv1(x)))
#         conv2_out = self.pool(torch.relu(self.conv2(conv1_out)))
#         conv3_out = self.pool(torch.relu(self.conv3(conv2_out)))
#         avg = self.avg(torch.relu(self.conv4(conv3_out)))
#         conv3_out = avg.reshape(avg.size(0), -1)  # Flatten the output from convolutional layers
#         # print(conv3_out.shape)
#         fc1_out = torch.relu(self.fc1(conv3_out))
#         out = self.fc2(fc1_out)
#         return out
