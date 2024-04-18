import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

# the class which helps create dataset objects
class ImageLabelDataset(Dataset):
    def __init__(self, images, labels):
        """
        images: A tensor containing the image data.
        labels: A tensor containing the corresponding labels.
        """
        self.images = images
        self.labels = labels

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Fetches the sample at index `idx` from the dataset."""
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

# 128 x 128 x 4
file_path = os.path.join('numpy_matrix_data', 'ordered_data.npy')
loaded_array = np.load(file_path, allow_pickle=True)
# Input data is 6171 by 2
# loaded_array[i][0] = 128 by 128 by 4 np matrix that represents the image
# loaded_array[i][1] = label = {buy:1, sel: -1, hold:0}

total_size = len(loaded_array)
split1_size = int(total_size * 0.6)
split2_size = int(total_size * 0.2)

# seperating raw data into images and labels
raw_images = np.array([item[0] for item in loaded_array], dtype=np.float32)
raw_labels = np.array([item[1] for item in loaded_array], dtype=np.int64)   # labels as integers

# print(raw_images[1])

# replace all instnaces of -1 in the original list with 0
raw_labels[raw_labels==-1] = 0

print(np.count_nonzero(raw_labels==1))
print(np.count_nonzero(raw_labels==0))
print(len(raw_labels))

# converting them to torch tensors
images = torch.from_numpy(raw_images).transpose(3,1)
# print(images.shape)
labels = torch.from_numpy(raw_labels)

labels = labels.reshape(6171,1)
# print(labels.shape)

# splitting the data
train_images = images[:split1_size]
test_images = images[split1_size:split1_size + split2_size]
val_images = images[split1_size + split2_size:]

train_labels = labels[:split1_size]
test_labels = labels[split1_size:split1_size + split2_size]
val_labels = labels[split1_size + split2_size:]

# print(train_images.shape)
# print(test_images.shape)
# print(val_images.shape)

# print(train_labels.shape)
# print(test_labels.shape)
# print(val_labels.shape)

# converting data into dataset objects to pass into Dataloader
train_data = ImageLabelDataset(train_images, train_labels)
val_data = ImageLabelDataset(val_images, val_labels)
test_data = ImageLabelDataset(test_images, test_labels)


train_dataloader = DataLoader(train_data, batch_size=10)
val_dataloader = DataLoader(val_data, batch_size=10)
test_dataloader = DataLoader(test_data, batch_size=10)

# print(len(train_dataloader))
# print(len(val_dataloader))
# print(len(test_dataloader))

# for x, t in train_data:
#     # print(t)
#     print(t)
#     print(x)
#     break