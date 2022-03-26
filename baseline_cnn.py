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
import torchvision.transforms as T
import matplotlib.pyplot as plt
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers+=[nn.Conv2d(1, 16,  kernel_size=3) , 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(16, 16,  kernel_size=3, stride=2), 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(16, 32,  kernel_size=3), 
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(32, 32,  kernel_size=3, stride=2), 
                      nn.ReLU(inplace=True)]
        self.fc = nn.Linear(32*4*4, 2)
        
    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
        x = x.view(-1, 32*4*4)
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

import os
def plot(imgs, with_orig=False, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):

        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img[0]), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
if __name__=="__main__":
    if len(argv)==1:
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
    transform=transforms.Compose([
    transforms.Normalize((0.5,), (0.5,)),

    ])
    # do flips to increase dataset
    transform_horizontal = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomHorizontalFlip(1)
    ])
    # do flips to increase dataset
    transform_vertical = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomVerticalFlip(1)
    ])

    # dataset: load mednist data
    dir_path = os.path.dirname(os.path.realpath(__file__))

    test_index = list(range(1,6))
    accuracies = []
    for i in range(1, 6):
    
        ##################### YOUR CODE GOES HERE
        x0 = torch.load(input_dir + '/data/train_data/train_{}/class_0/image_tensors.pt'.format(i))
        x1 = torch.load(input_dir + '/data/train_data/train_{}/class_1/image_tensors.pt'.format(i))

        blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        blurrer = T.RandomAutocontrast()
        def makeBlur(input):
            output = torch.Tensor()
            for i in input:
                temp = [blurrer(i) for _ in range(4)]
                temp.append(i)
                temp = torch.stack(temp)
                output = torch.cat([output,temp])
            return output
        x0 = makeBlur(x0)
        x1 = makeBlur(x1)
        train_data = torch.utils.data.TensorDataset(transform(torch.cat([x0,x1])), torch.cat([torch.zeros((x0.shape[0])), torch.ones((x1.shape[0]))]).long())


        x0_test = torch.Tensor()
        x1_test = torch.Tensor()
        test_cases = []
        for j in test_index:
            if i != j:
                x0 = torch.load(input_dir + '/data/train_data/train_{}/class_0/image_tensors.pt'.format(j))
                x1 = torch.load(input_dir + '/data/train_data/train_{}/class_1/image_tensors.pt'.format(j))
                x0_test = torch.cat([x0,x0_test])
                x1_test = torch.cat([x1,x1_test])

        test = torch.utils.data.TensorDataset(transform(torch.cat([x0_test, x1_test])), torch.cat(
            [torch.zeros((x0_test.shape[0])), torch.ones((x1_test.shape[0]))]).long())

        x = torch.load(input_dir + '/data/val/image_tensors.pt')
        val_data = torch.utils.data.TensorDataset(transform(x))
        
        # dataset: initialize dataloaders for train and validation set
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
        '''
        Dang: Only 10 samples, so having batch size more than 10 doesn't do anything.
         Having batch size of 2 may be better (5 train) ?
         Setting batch size at 2 perform worse than batch size 10. Overfitting? 
        '''
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)

        # model: initialize model
        model = Net()
        model.to(device)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=0.01, momentum=0.9,
                                    weight_decay=0.0005)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.01,
        #                             weight_decay=0.0005)
        print("[Preparation] Done")

        ### Training
        # model: training loop
        print("[Training] Start...\n")
        for epoch in range(50):
            train(model, device, train_loader, optimizer, epoch, display=epoch % 5 == 0)
        print("\n[Training] Done")
        ##################### END OF YOUR CODE


        ### Saving Outputs
        print("[Saving Outputs] Start...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # test evaluation: make predictions
        print("[Saving Outputs] Test set...")

        # test_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)
        test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)
        test_predictions = []
        model.eval()
        with torch.no_grad():
            for (data, target)in test_loader:
                data = data.to(device)
                output = model(data)
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                test_predictions.extend(pred.squeeze().cpu().tolist())

                correct  = (np.array(pred.squeeze().cpu().tolist()) == np.array(target.squeeze().cpu().tolist()))
        accuracy = correct.sum() / correct.size
        accuracies.append(accuracy)
        print('Accuracy is ', accuracy)
        # test evaluation: save predictions
        test_str = '\n'.join(list(map(str, test_predictions)))
        
        with open(os.path.join(output_dir, 'answer_test_{}.txt'.format(i)), 'w') as result_file:
            result_file.write(test_str)
        print("[Saving Outputs] Done")

    print("All done!")
    print("Average accuracy is ", np.mean(accuracies))