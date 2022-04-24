#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
from sys import argv
from torchvision import datasets, transforms
import torchvision.transforms as T

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers+=[nn.Conv2d(1, 128,  kernel_size=3) ,
                      nn.ReLU(inplace=True)]
        self.layers += [nn.BatchNorm2d(128)]
        self.layers += [nn.MaxPool2d(2)]

        self.layers+=[nn.Conv2d(128, 60,  kernel_size=3, stride=1),
                      nn.ReLU(inplace=True)]
        self.layers+=[nn.Conv2d(60, 32,  kernel_size=3),
                      nn.ReLU(inplace=True)]
        self.layers += [nn.BatchNorm2d(32)]
        # self.layers += [nn.Dropout()]

        self.layers+=[nn.Conv2d(32, 32,  kernel_size=3),
                      nn.ReLU()]
        self.fc = nn.Linear(32*7*7, 2)

    def forward(self, x):
        for i in range(len(self.layers)):
            # print(x.size())
            x = self.layers[i](x)
            # print(x.size())

        # x = x.view(-1, 32*4*4)
        x = x.view(-1, 32*7*7)
        x = self.fc(x)
        return x


def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    model = model.to(device)
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

# from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd
if __name__=="__main__":
    validationMode = False

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

    transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,)),
    ])

    def combineTransform(img):
        img += 2*T.functional.adjust_sharpness(img, 5)
        img += 4* T.functional.adjust_contrast(img, 3)
        return img

    # dataset: load mednist data
    dir_path = os.path.dirname(os.path.realpath(__file__))

    test_index = list(range(1,6))
    accuracies = []
    for i in range(1, 26):
    
        ##################### YOUR CODE GOES HERE
        x0 = torch.load(input_dir + '/data/train_data/train_{}/class_0/image_tensors.pt'.format(i))
        x1 = torch.load(input_dir + '/data/train_data/train_{}/class_1/image_tensors.pt'.format(i))

        # x0 = torch.stack([combineTransform(x) for x in x0])
        # x1 = torch.stack([combineTransform(x) for x in x1])

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
        # x0_test = torch.stack([combineTransform(x) for x in x0_test])
        # x1_test = torch.stack([combineTransform(x) for x in x1_test])

        test = torch.utils.data.TensorDataset(transform(torch.cat([x0_test, x1_test])), torch.cat(
            [torch.zeros((x0_test.shape[0])), torch.ones((x1_test.shape[0]))]).long())

        x = torch.load(input_dir + '/data/val/image_tensors.pt')
        x =  torch.stack([combineTransform(_x) for _x in x])
        val_data = torch.utils.data.TensorDataset(transform(x))
        
        # dataset: initialize dataloaders for train and validation set
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
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
        for epoch in range(100):
            train(model, device, train_loader, optimizer, epoch, display=epoch % 5 == 0)
        print("\n[Training] Done")
        ##################### END OF YOUR CODE


        ### Saving Outputs
        print("[Saving Outputs] Start...")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # test evaluation: make predictions
        print("[Saving Outputs] Test set...")

        test_predictions = []
        model.eval()


        if validationMode:
            import matplotlib.pyplot as plt

            test_loader = torch.utils.data.DataLoader(test, batch_size=128, shuffle=True)
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
            y_pred = []
            y_true = []

            # iterate over test data
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                output = model(inputs)  # Feed Network

                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output)  # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels)  # Save Truth
                break

            for index, (pre, target) in enumerate(zip(y_pred, y_true)):
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
                image = test_loader.dataset[index][0]
                x = image.numpy()[0]
                fig.suptitle(f'True Label: {target}, Predicted: {pre}')
                ax1.matshow(x)
                ax1.set_title('Original')
                # blur = makeBlur([image])
                # ax2.matshow(blur[0][0] )
                # ax2.set_title(f'Blur ')
                bright = T.functional.adjust_sharpness(image, 4)
                ax2.matshow(bright[0])
                ax2.set_title(f'Sharpness')

                x = T.functional.adjust_contrast(image, 2)
                ax3.matshow(x[0])
                ax3.set_title(f'Contrast')

                x = combineTransform(image)
                ax4.matshow(x[0])

                ax4.set_title(f'Combined')
                fig.tight_layout()

                plt.show()
                if index == 10:
                    exit()
        else:
            test_loader = torch.utils.data.DataLoader(val_data, batch_size=128, shuffle=False)
            with torch.no_grad():
                for (data,) in test_loader:
                    data = data.to(device)
                    output = model(data)
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    test_predictions.extend(pred.squeeze().cpu().tolist())
        # test evaluation: save predictions
        test_str = '\n'.join(list(map(str, test_predictions)))


        #
        # # constant for classes
        # classes = ('havenot','have')
        #
        # # Build confusion matrix
        # cf_matrix = confusion_matrix(y_true, y_pred)
        # df_cm = pd.DataFrame(cf_matrix , index=[i for i in classes],
        #                      columns=[i for i in classes])
        # plt.figure(figsize=(12, 7))
        # sn.heatmap(df_cm, annot=True)
        # plt.title(f'Baseline + Minor Enhancement {i}')
        # plt.ylabel('True Label')
        # plt.xlabel('Predicted Label')
        # plt.savefig(f'baseline_enhanced_model_confusionmatrix{i}.png')
        # plt.show()



        with open(os.path.join(output_dir, 'answer_test_{}.txt'.format(i)), 'w') as result_file:
            result_file.write(test_str)
        print("[Saving Outputs] Done")

    print("All done!")
    # print("Average accuracy is ", np.mean(accuracies))

