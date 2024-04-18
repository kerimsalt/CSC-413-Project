import numpy as np
import matplotlib.pyplot as plt

# import medmnist
from medmnist import PneumoniaMNIST
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms  # contains a collection of transformations
import Model
# import train
import data_setup

train_data_imgs = PneumoniaMNIST(split='train', download=True)

train_data = PneumoniaMNIST(split='train', download=True, transform=transforms.ToTensor())
val_data = PneumoniaMNIST(split='val', download=True, transform=transforms.ToTensor())
test_data = PneumoniaMNIST(split='test', download=True, transform=transforms.ToTensor())


# class MLPModel(nn.Module):
#     """A three-layer MLP model for binary classification"""
#     def __init__(self, input_dim=28*28, num_hidden=100):
#         super(MLPModel, self).__init__()
#         self.fc1 = nn.Linear(input_dim, num_hidden)
#         self.fc2 = nn.Linear(num_hidden, num_hidden)
#         self.fc3 = nn.Linear(num_hidden, 1)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         out = self.relu(out)
#         out = self.fc3(out)
#         return out

def accuracy(model, dataset, device):
    """
    Compute the accuracy of `model` over the `dataset`.
    We will take the **most probable class**
    as the class predicted by the model.

    Parameters:
        `model` - A PyTorch MLPModel
        `dataset` - A data structure that acts like a list of 2-tuples of
                  the form (x, t), where `x` is a PyTorch tensor of shape
                  [1, 28, 28] representing an MedMNIST image,
                  and `t` is the corresponding binary target label

    Returns: a floating-point value between 0 and 1.
    """

    correct, total = 0, 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=100)
    for img, t in loader:
        # X = img.reshape(-1, 784)
        img = img.to(device)
        t = t.to(device)
        z = model(img)
        y = torch.sigmoid(z)
        pred = (y >= 0.5).int()
        # pred should be a [N, 1] tensor with binary
        # predictions, (0 or 1 in each entry)

        correct += int(torch.sum(t == pred))
        total += t.shape[0]
    return correct / total


criterion = nn.BCEWithLogitsLoss()


def train_model1(model,  # an instance of MLPModel
                 train_data,  # training data
                 val_data,  # validation data
                 learning_rate=0.001,
                 batch_size=100,
                 num_epochs=10,
                 plot_every=2,  # how often (in # iterations) to track metrics
                 plot=True,
                 device=torch.device(
                     "cuda" if torch.cuda.is_available() else "cpu")):  # whether to plot the training curve
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               shuffle=True)  # reshuffle minibatches every epoch
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model = model.to(device)

    # these lists will be used to track the training progress
    # and to plot the training curve
    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0  # count the number of iterations that has passed

    try:
        for e in range(num_epochs):
            # print(1)
            for i, (images, labels) in enumerate(train_loader):

                images = images.to(device)
                labels = labels.to(device)

                z = model(images).float()  # TODO
                # print(z.shape)
                # print(labels.shape)
                # break
                loss = criterion(z, labels.float())  # TODO

                loss.backward()  # propagate the gradients
                optimizer.step()  # update the parameters
                optimizer.zero_grad()  # clean up accumualted gradients

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = accuracy(model, train_data, device)
                    va = accuracy(model, val_data, device)
                    train_loss.append(float(loss))
                    train_acc.append(ta)
                    val_acc.append(va)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
    finally:
        # This try/finally block is to display the training curve
        # even if training is interrupted
        if plot:
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Accuracy")
            plt.legend(["Train", "Validation"])


# Please include the output of this cell for grading
if torch.cuda.is_available():
    device = torch.device("cuda")

print(torch.cuda.is_available())

train_data_s = data_setup.train_data
validation_data_s = data_setup.val_data
test_data_s = data_setup.test_data

model = Model.CNN(in1=4, out1=64, out2=128, out3=256, out4=512, fcb1=25088, fcb2=2048, fcb3=100, fcb4=1)
# model = Model.CNN(in1=1, out1=32, out2=64, out3=32, fcb1=32)
train_model1(model, train_data_s, validation_data_s)
# print(model.parameters)