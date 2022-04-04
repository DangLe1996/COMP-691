#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from sys import argv
from torchvision import datasets, transforms

local = True

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()

        self.layers += [nn.Conv2d(1, 64, kernel_size=3),
                       nn.ReLU(inplace=True)]
        self.layers += [nn.BatchNorm2d(64)]
        self.layers += [nn.MaxPool2d(2)]
        self.layers += [nn.Conv2d(64, 32, kernel_size=3, stride=2),
                        nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(32, 32, kernel_size=3),
                        nn.ReLU(inplace=True)]

        self.fc = nn.Linear(32 * 4 * 4, 2)

        # self.layers += [nn.Conv2d(1, 128, kernel_size=3),
        #                 nn.ReLU(inplace=True)]
        # self.layers += [nn.BatchNorm2d(128)]
        # self.layers += [nn.MaxPool2d(2)]
        #
        # self.layers += [nn.Conv2d(128, 64, kernel_size=2),
        #                 nn.ReLU(inplace=True)]
        # self.layers += [nn.MaxPool2d(2)]
        # self.layers += [nn.BatchNorm2d(64)]
        #
        # self.layers += [nn.Conv2d(64, 32, kernel_size=2),
        #                 nn.ReLU(inplace=True)]
        # self.layers += [nn.MaxPool2d(2)]
        # self.layers += [nn.BatchNorm2d(32)]
        # self.layers += [nn.Dropout(0.5)]
        #
        # self.layers += [nn.Flatten()]
        # self.layers += [nn.Linear(128, 32)]
        # self.layers += [nn.Dropout(0.5)]
        # self.layers += [nn.Linear(32, 1)]

    def forward(self, x):
        for i in range(len(self.layers)):
            #print(self.layers[i])
            #print(x.size())

            x = self.layers[i](x)
            # print(x.size())
        x = x.view(-1, 32 * 4 * 4)
        x = self.fc(x)
        return x

def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    if display:
        print('Train: Epoch {} [{}/{} ({:.0f}%)],\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, i):
    # Do our own testing on the training...
    x0_test = torch.Tensor()
    x1_test = torch.Tensor()
    for j in range(1, 6):
        if i != j:
            x0 = torch.load(
                input_dir + '/data/train_data/train_{}/class_0/image_tensors.pt'.format(
                    j))
            x1 = torch.load(
                input_dir + '/data/train_data/train_{}/class_1/image_tensors.pt'.format(
                    j))
            x0_test = torch.cat([x0, x0_test])
            x1_test = torch.cat([x1, x1_test])

    test_data = torch.utils.data.TensorDataset(
        transform_test(torch.cat([x0_test, x1_test])), torch.cat(
            [torch.zeros((x0_test.shape[0])),
             torch.ones((x1_test.shape[0]))]).long())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128,
                                               shuffle=False)
    correct = 0
    model.eval()
    for (data, target) in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    print('Accuracy is ', accuracy)
    return accuracy

if __name__ == "__main__":
    if len(argv) == 1:
        input_dir = os.getcwd()
        output_dir = os.getcwd()
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])

    print("Using input_dir: " + input_dir)
    print("Using output_dir: " + output_dir)

    ### Preparation
    print("[Preparation] Start...")
    # select device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataset: normalize and convert to tensor
    transform_train = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
        #transforms.RandomHorizontalFlip(0.25),
        #transforms.RandomVerticalFlip(0.25)
    ])

    transform_test = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # dataset: load mednist data
    dir_path = os.path.dirname(os.path.realpath(__file__))
    accuracies = []
    r=26
    if local:
        r = 6
    for i in range(1, r): # To change to 26 when submitting

        ##################### YOUR CODE GOES HERE
        x0 = torch.load(
            input_dir + '/data/train_data/train_{}/class_0/image_tensors.pt'.format(
                i))
        x1 = torch.load(
            input_dir + '/data/train_data/train_{}/class_1/image_tensors.pt'.format(
                i))
        train_data = torch.utils.data.TensorDataset(
            transform_train(torch.cat([x0, x1])), torch.cat(
                [torch.zeros((x0.shape[0])),
                 torch.ones((x1.shape[0]))]).long())

        x = torch.load(input_dir + '/data/val/image_tensors.pt')
        val_data = torch.utils.data.TensorDataset(transform_test(x))

        # dataset: initialize dataloaders for train and validation set
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128,
                                                   shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=128,
                                                 shuffle=False)

        # model: initialize model
        model = Net()
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.01, momentum=0.9,
                                    weight_decay=0.0005)

        print("[Preparation] Done")

        ### Training
        # model: training loop
        print("[Training] Start...\n")
        for epoch in range(50):
            train(model, device, train_loader, optimizer, epoch,
                  display=epoch % 5 == 0)
        print("\n[Training] Done")
        ##################### END OF YOUR CODE

        ### Saving Outputs
        print("[Saving Outputs] Start...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Do our own testing
        if local:
            acc = test(model, device, i)
            accuracies.append(acc)



        # test evaluation: make predictions
        print("[Saving Outputs] Test set...")

        test_loader = torch.utils.data.DataLoader(val_data, batch_size=128,
                                                  shuffle=False)
        test_predictions = []
        model.eval()
        with torch.no_grad():
            for (data,) in test_loader:
                data = data.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                test_predictions.extend(pred.squeeze().cpu().tolist())

        # test evaluation: save predictions
        test_str = '\n'.join(list(map(str, test_predictions)))

        with open(os.path.join(output_dir, 'answer_test_{}.txt'.format(i)),
                  'w') as result_file:
            result_file.write(test_str)
        print("[Saving Outputs] Done")

    if local:
        print(sum(accuracies)/len(accuracies))
    print("All done!")