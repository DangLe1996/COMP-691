# -*- coding: utf-8 -*-
"""2022_Final_Project_Validation_Initial_Test_Bed_Ex.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c8JJKrPal3o3lYJqiXd___VFLMKe7eef

In this project, we will work with the medical mnist datasource, specifically the pneumonia dataset from https://medmnist.com/ . The problem consists of classifying chest x-ray images as having pneumonia or not. Run the below lines of code to install the appropriate dataloaders and visualize the data
"""

# !pip install -qqq medmnist

from tqdm import tqdm
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms

import medmnist
from medmnist import INFO, Evaluator
from numpy.random import RandomState
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset
import re
from torchvision import datasets, transforms
import torchvision.transforms as T

"""Now let's visualize the chest xray data"""

data_flag = 'pneumoniamnist'
info = INFO[data_flag]
task = info['task']
DataClass = getattr(medmnist, info['python_class'])

# load the data
train_dataset = DataClass(split='train', download=True)
train_dataset.montage(length=10)

# Here the goal is to train on 10 samples on the pneumonia mnist data. In this preliminary testbed, the evaluation will be done on a 1000 sample randomly sampled development set. Note in the end the final evaluation will be done on the full Pneumoniamnist test set as well as potentially a separate dataset. The development set samples here thus should not be used for training in any way, the final evaluation will provide only 10 random samples of the same distribution and as well to evaluate the generality of your algorithm from a data source that is not the Pneumoniamnist training data.
#
# Feel free to modify this testbed to your liking, including the normalization transformations, etc. Note, however, the final evaluation testbed will have a rigid set of components where you will need to place your answer. The only constraint is the data. Refer to the full project instructions for more information.
#
# Below we set up training functions. Again you are free to fully modify this testbed in your prototyping within the constraints of the data used. You can use tools outside of PyTorch for training models if desired as well although the torchvision dataloaders will still be useful for interacting with the Pneumoniamnist dataset.
# """
#
#
def train(model, device, train_loader, optimizer, epoch, display=True):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.binary_cross_entropy_with_logits(output, target.float())
        loss.backward()
        optimizer.step()
    if display:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, name="\nVal"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.binary_cross_entropy_with_logits(output, target.float(),
                                                            size_average=False).item()  # sum up batch loss
            pred = output >= 0.5
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)

#
import torch
import torch.nn as nn
import torch.nn.functional as F



import torchvision.models as models

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(0)
np.random.seed(0)

# preprocessing
data_flag = 'pneumoniamnist'
download = True

info = INFO[data_flag]
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])


data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Normalize(mean=[.5], std=[.5]),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),

])
def combineTransform(img):
    img = data_transform (img)
    img += 2 * T.functional.adjust_sharpness(img, 1)
    img += 4 * T.functional.adjust_contrast(img, 1)
    return img



#%%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
# load the data
train_dataset = DataClass(split='train',transform = transforms.ToTensor(),  download=download)
val_dataset = DataClass(split='train',transform = transforms.ToTensor(), download=download)



#%%

accs_val = []
seedNumber = 20
for seed in range(0, seedNumber):
    print(f'{seed}/{seedNumber}')
    prng = RandomState(seed)
    random_permute = prng.permutation(np.arange(0, 1000))
    train_top = 10 // n_classes
    val_top = 1000 // n_classes
    indx_train = np.concatenate(
        [np.where(train_dataset.labels == label)[0][random_permute[0:train_top]] for label in range(0, n_classes)])
    indx_val = np.concatenate(
        [np.where(train_dataset.labels == label)[0][random_permute[train_top:train_top + val_top]] for label in
         range(0, n_classes)])
    train_data = Subset(train_dataset, indx_train)
    val_data = Subset(val_dataset, indx_val)


    print(
        'Num Samples For Training %d Num Samples For Val %d' % (train_data.indices.shape[0], val_data.indices.shape[0]))
    from matplotlib import pyplot as plt

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    ax1.matshow(val_data[0][0][0])


    train_data = torch.utils.data.TensorDataset(torch.stack([combineTransform(x[0]) for x in train_data]),
                                                torch.stack([torch.Tensor(x[1]).long() for x in train_data]))
    val_data = torch.utils.data.TensorDataset(torch.stack([combineTransform(x[0]) for x in val_data]),
                                              torch.stack([torch.Tensor(x[1]).long() for x in val_data]))


    images = val_data[0][0]
    img = images.numpy().transpose(1, 2, 0)
    ax2.matshow(images[0])
    plt.show()

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=32,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=128,
                                             shuffle=False)
    model = models.alexnet(pretrained=True)
    model.classifier = nn.Linear(256 * 6 * 6, 1)

    model.to(device)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)

    for epoch in range(100):
        train(model, device, train_loader, optimizer, epoch, display=epoch % 5 == 0)
    accs_val.append(test(model, device, val_loader))


accs_val = np.array(accs_val)

print('Val acc over %.2f  instances on dataset: %s %.2f +- %.2f' % (seedNumber,data_flag, accs_val.mean(), accs_val.std()))