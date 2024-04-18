import Model
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import data_setup
import time
import mnist_model
from torch.utils.data import DataLoader
import os


def get_accuracy(model, data, device="cpu"):
    loader = torch.utils.data.DataLoader(data, batch_size=10)
    model.to(device)
    model.eval()  # annotate model for evaluation (important for batch normalization)
    correct = 0
    total = 0
    for imgs, labels in loader:
        labels = labels.to(device)
        logits = model(imgs.to(device))
        prob_output = torch.sigmoid(logits)
        pred = (prob_output >= 0.5).int()
        correct += int(torch.sum(labels == pred))
        total += labels.shape[0]
    return correct / total


def train_model(model,
                train_data,
                validation_data,
                learning_rate=0.1,
                batch_size=10,
                num_epochs=10,
                plot_every=10,
                plot=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size, )
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    iters, train_loss, train_acc, val_acc = [], [], [], []
    iter_count = 0
    try:
        for e in range(num_epochs):
            print(1)
            # for param in model.parameters():
            #         # print(param.data.shape)
            #         # print(param.data[0][0])
            #         break
            for imgs, labels in iter(train_loader):
                start = time.time()
                labels = labels.to(device)
                imgs = imgs.to(device)
                # print(imgs.shape)
                model.train()

                out = model(imgs).float()
                # print(out)
                # print(out.shape)
                # print(labels.shape)
                # break
                loss = criterion(out, labels.float())
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    t_acc = get_accuracy(model, train_data, device)
                    v_acc = get_accuracy(model, validation_data, device)
                    train_loss.append(float(loss))
                    train_acc.append(t_acc)
                    val_acc.append(v_acc)
                    end = time.time()
                    time_taken = round(end - start, 3)
                    print(iter_count, "Loss:", float(loss), "Train Acc:", round(t_acc, 3), "Val Acc:", round(v_acc, 3),
                          'Time taken:', time_taken)
    finally:
        plt.figure()
        plt.scatter(iters, train_loss, color='red', s=15)
        plt.xlabel('Iteration')
        plt.ylabel('Training Loss')
        plt.show()

        plt.figure()
        plt.plot(iters, train_acc, color='green', label='Training accuracy')
        plt.plot(iters, val_acc, color='blue', label='Validation accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.show()


train_data = data_setup.train_data
validation_data = data_setup.val_data
test_data = data_setup.test_data
# for x, t in train_data:
#     print(x, t)
#     print(t.shape)
#     break
if torch.cuda.is_available():
    device = torch.device("cuda")

print(torch.cuda.is_available())

# model = Model.CNN(in1=4, out1=64, out2=128, out3=256, out4=512, fcb1=25088, fcb2=2048, fcb3=100, fcb4=1)
model = Model.CNN(in1=4, out1=64, out2=64, out3=32, fcb1=32)

mnist_model.train_model1(model, train_data, validation_data, batch_size=100, num_epochs=10)


