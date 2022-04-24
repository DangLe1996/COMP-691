import random

import medmnist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from medmnist import INFO
from numpy.random import RandomState
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
random.seed(0)

# preprocessing
data_flag = 'pneumoniamnist'
download = True

info = INFO[data_flag]
n_classes = len(info['label'])
DataClass = getattr(medmnist, info['python_class'])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

train_dataset = DataClass(split='train', transform=transforms.ToTensor(), download=download)
val_dataset = DataClass(split='train', transform=transforms.ToTensor(), download=download)

# Visualize the data
train_dataset.montage(length=10)


def train(model, device, train_loader, optimizer, epoch, display=True):
    model.train()
    test_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.view(-1))
        loss.backward()
        optimizer.step()
        test_loss += loss.item()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    if display:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.item()))
    return 100. * correct / len(train_loader.dataset), test_loss


def test(model, device, test_loader, name="\nVal"):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target.view(-1), reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        name, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), test_loss


def evaluate(model, device, test_loader):
    predictions = torch.tensor([])
    targets = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = torch.max(torch.exp(output), 1)[1]
            predictions = torch.cat((predictions, pred.view(-1)))
            targets = torch.cat((targets, target.view(-1)))
    return predictions, targets


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.ModuleList()

        self.layers += [nn.Conv2d(1, 128, kernel_size=3),
                        nn.ReLU(inplace=True)]
        self.layers += [nn.BatchNorm2d(128)]
        self.layers += [nn.MaxPool2d(2)]

        self.layers += [nn.Conv2d(128, 60, kernel_size=3, stride=1),
                        nn.ReLU(inplace=True)]
        self.layers += [nn.Conv2d(60, 32, kernel_size=3),
                        nn.ReLU(inplace=True)]
        self.layers += [nn.BatchNorm2d(32)]
        # self.layers += [nn.Dropout()]

        self.layers += [nn.Conv2d(32, 32, kernel_size=3),
                        nn.ReLU()]
        self.fc = nn.Linear(32 * 7 * 7, 2)

    def forward(self, x):
        for i in range(len(self.layers)):
            # print(x.size())
            x = self.layers[i](x)
            # print(x.size())

        # x = x.view(-1, 32*4*4)
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

def combineTransform(img):
    img += 2 * transforms.functional.adjust_sharpness(img, 5)
    img += 4 * transforms.functional.adjust_contrast(img, 3)
    return img

transform_base = transforms.Compose(
    [
        transforms.Normalize(mean=0.5, std=0.5)
    ]
)


def augment_training(train_data):
    train_data = torch.utils.data.TensorDataset(torch.stack([combineTransform(transform_base(x[0])) for x in train_data]),
                                                torch.stack([torch.Tensor(x[1]).long() for x in train_data]))
    return train_data


def preprocess_val(val_data):
    val_data = torch.utils.data.TensorDataset(torch.stack([combineTransform(transform_base(x[0])) for x in val_data]),
                                              torch.stack([torch.Tensor(x[1]).long() for x in val_data]))
    return val_data


accs_val = []

accs_train = []
accs_test = []
losses_train = []
losses_test = []

models_list = []

for seed in range(0, 20):
    accs_train.append([])
    accs_test.append([])
    losses_train.append([])
    losses_test.append([])

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
    train_data = augment_training(train_data)
    val_data = preprocess_val(val_data)

    print('Num Samples For Training %d Num Samples For Val %d' % (len(train_data), len(val_data)))

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=128,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=128,
                                             shuffle=False)

    model = Net()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)

    for epoch in range(100):
        acc_train, loss_train = train(model, device, train_loader, optimizer, epoch, display=True)
        acc_test, loss_test = test(model, device, val_loader)
        accs_train[seed].append(acc_train)
        accs_test[seed].append(acc_test)
        losses_train[seed].append(loss_train)
        losses_test[seed].append(loss_test)
    accs_val.append(test(model, device, val_loader)[0])

    models_list.append(model)
accs_val = np.array(accs_val)
accs_train = np.array(accs_train)
accs_test = np.array(accs_test)
losses_train = np.array(losses_train)
losses_test = np.array(losses_test)

print('Val acc over %s instances on dataset: %s %.2f +- %.2f' % (
    len(accs_val), data_flag, accs_val.mean(), accs_val.std()))


def plot_acc(accs_train, accs_test):
    plt.plot(np.arange(0, len(accs_train[0])), accs_train.mean(0), label="Train")
    plt.plot(np.arange(0, len(accs_test[0])), accs_test.mean(0), label="Test")
    plt.title("Accuracy of train and test set over each epoch")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig("challenge1_acc_plot2.png")


def plot_loss(loss_train, loss_test):
    plt.plot(np.arange(0, len(loss_train[0])), loss_train.mean(0), label="Train")
    plt.plot(np.arange(0, len(loss_test[0])), loss_test.mean(0), label="Test")
    plt.title("Loss of train and test set over each epoch")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.show()
    plt.savefig("challenge1_loss_plot2.png")


plot_acc(accs_train, accs_test)
plot_loss(losses_train, losses_test)


best_model = models_list[np.argmax(accs_val)]

def print_confusion_matrix(model, loader):
    y_pred, y_true = evaluate(model, device, loader)
    classes = ('havenot', 'have')

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title('Confusion matrix of the best model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    plt.savefig("confusion2.png")

print_confusion_matrix(best_model, val_loader)

f1s = []
for model in models_list:
  y_pred, y_true = evaluate(model, device, val_loader)
  f1 = f1_score(y_true, y_pred)
  print(f"f1 score: {f1}")
  f1s.append(f1)

f1s = np.array(f1s)
print(f1s.mean())

