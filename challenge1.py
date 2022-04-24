# !pip install -qqq medmnist
import math

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
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
from torch.utils.data import Subset, DataLoader
import re
from torchvision import datasets, transforms
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sn
from matplotlib import pyplot as plt
import pandas as pd
torch.clear_autocast_cache()
def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        target = target.flatten(0)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
    if display:
      print('Train: Epoch {} [{}/{} ({:.0f}%)],\tLoss: {:.6f}'.format(
          epoch, batch_idx * len(data), len(train_loader.dataset),
          100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, name="\nVal"):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.flatten(0)
            output = model(data)
            # test_loss += F.binary_cross_entropy_with_logits(output, target.float(), size_average=False).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_true.extend(target.tolist())
            pred = pred.flatten(0)
            y_pred.extend(pred.tolist())
    # constant for classes
    classes = ('havenot','have')
    accuracy =  round(100. * correct / len(test_loader.dataset),2)
    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix , index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    label= f'Without transformation, Accuracy {accuracy}%'
    plt.title(label)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{label} .png')
    plt.show()
    # print(disp.confusion_matrix)

    # test_loss /= len(test_loader.dataset)

    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)
n = 32*5*5
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()

        self.layers += [nn.Conv2d(1, 128, kernel_size=11),
                        nn.ReLU(inplace=True)]
        self.layers += [nn.BatchNorm2d(128)]
        self.layers += [nn.MaxPool2d(2)]
        self.layers += [nn.Conv2d(128, 60, kernel_size=6),
                        nn.ReLU(inplace=True)]
        self.layers += [nn.MaxPool2d(2)]

        self.layers += [nn.Conv2d(60, 60, kernel_size=3),
                        nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(60, 32, kernel_size=2,stride= 3),
                        nn.ReLU(inplace=True)]
        self.layers += [nn.BatchNorm2d(32)]
        # self.layers += [nn.Dropout()]

        self.layers += [nn.Conv2d(32, 32, kernel_size=2),
                        nn.ReLU()]
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=n, out_features=4096),
            nn.ReLU(),
            # nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=2),
        )
    def forward(self, x):
        for i in range(len(self.layers)):
            # print(x.size())
            # print(x.size())
            x = self.layers[i](x)
            #
        # x = x.view(-1, 32*4*4)
        x = x.view(-1, n)
        x = self.classifier(x)
        return x


#%%
"""Now let's visualize the chest xray data"""
data_flag = 'pneumoniamnist'
info = INFO[data_flag]
task = info['task']
DataClass = getattr(medmnist, info['python_class'])

# load the data
train_dataset = DataClass(split='train', download=True)
train_dataset.montage(length=10)
#%%
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
    transforms.Normalize((0.5,), (0.5,)),
])
def combineTransform(img,sharpness_w = 2, contrast_w = 3):
    img = data_transform (img)
    img += sharpness_w * T.functional.adjust_sharpness(img, 4)
    img += contrast_w * T.functional.adjust_contrast(img, 2)
    img = transforms.Resize(100)(img)
    return img

#%%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)
# load the data
train_dataset = DataClass(split='train',transform = transforms.ToTensor(),  download=download)
val_dataset = DataClass(split='train',transform = transforms.ToTensor(), download=download)

#%%
seed = 10
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
#%%


#%%
train_data = torch.utils.data.TensorDataset(torch.stack([combineTransform(x[0]) for x in train_data]),
                                            torch.stack([torch.Tensor(x[1]).long() for x in train_data]))
val_data = torch.utils.data.TensorDataset(torch.stack([combineTransform(x[0]) for x in val_data]),
                                          torch.stack([torch.Tensor(x[1]).long() for x in val_data]))

dataloader = DataLoader(train_data, batch_size=100,
                        shuffle=True)
testLoader = DataLoader(val_data, batch_size=50,
                        shuffle=True)

#%%
model = Net()
# model = model_mnist.to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer = torch.optim.SGD(model.parameters(),
                            lr=0.01, momentum=0.9,
                            weight_decay=0.0005)
total_step = len(dataloader)
criterion = torch.nn.TripletMarginLoss()
model = model.to(device)
for epoch in range(700):
    train(model, device, dataloader, optimizer, epoch, display=epoch % 5 == 0)
    if epoch % 20 == 0:
        test(model,device,testLoader)

#%%
# trainLoader =  DataLoader(train_data, batch_size=100,
#                         shuffle=True)
# testLoader =  DataLoader(val_data, batch_size=100,
#                         shuffle=True)
# trainEmbedding =  extract_embeddings(trainLoader,model)
# testEmbedding =  extract_embeddings(testLoader,model)
#
# #%%
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn import svm
# # neigh = KNeighborsClassifier(n_neighbors=3)
# clf = svm.SVC()
# y = [x[1].item() for x in trainLoader.dataset]
# clf.fit(trainEmbedding,y)
# #%%
# y_pred = clf.predict(testEmbedding)
# y_test = [x[1].item() for x in testLoader.dataset]
# #%%
# from sklearn.metrics import classification_report, confusion_matrix
# print(confusion_matrix(y_test, y_pred))
# print(classification_report(y_test, y_pred))