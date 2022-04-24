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
from torch.utils.data import Subset, DataLoader
import re
from torchvision import datasets, transforms
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F

from matplotlib import pyplot as plt
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
    transforms.Resize(224),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

])
def combineTransform(img,sharpness_w = 1, contrast_w = 2):
    img = data_transform (img)
    img += sharpness_w * T.functional.adjust_sharpness(img, 2)
    img += contrast_w * T.functional.adjust_contrast(img, 1.5)
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

#
# paddings = (1, 3, 10, 20)
# result = []
#
#
#
# for img,label in train_data:
#     result.append((img,label))
#     for pad in paddings:
#         augmenter = T.TrivialAugmentWide()
#         img = img.type(torch.uint8 )
#         out = augmenter(img)
#         result.append((out, label))
# train_data = torch.utils.data.TensorDataset(torch.stack([combineTransform(x[0]) for x in result]),
#                                             torch.stack([torch.Tensor(x[1]).long() for x in result]))
# for i in range(5):
#     images = result[i][0]
#     img = images.numpy().transpose(1, 2, 0)
#     plt.imshow(img)
#     plt.show()

#%%
index = []
label_to_indices = {}
label_set = range(0,2)
for i in range(0,2):
    label_to_indices[i] = []
for image,label in train_data:
    label_to_indices[label.item()].append(image)
labels = [x[1] for x in train_data]
idx_pos = []
idx_neg = []
M= []
for epoch in range(30):
    for img,lab in train_data:
        positive_index = np.random.choice(label_to_indices[lab.item()])
        negative_label = np.random.choice(list(set(label_set) - set([lab.item()])))
        negative_index = np.random.choice(label_to_indices[negative_label])

        M.append((img,positive_index,negative_index))
dataloader = DataLoader(M, batch_size=100,
                        shuffle=True)

#%%
model = models.alexnet(pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=True)
model.classifier = nn.Linear(256 * 6 * 6, 5)
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
model.eval()
total_step = len(dataloader)
criterion = torch.nn.TripletMarginLoss()
model = model.to(device)
for epoch in range(10):
    for i, img in enumerate(dataloader):
        optimizer.zero_grad()
        img = [im.to(device) for im in img]
        svnOut = [model(img[0]),model(img[1]),model(img[2])]
        loss = criterion(svnOut[0],svnOut[1],svnOut[2])
        loss.backward()
        optimizer.step()
        print(' Step [{}], Loss: {:.4f}'
              .format(i + 1, loss.item()))
#%%
def extract_embeddings(dataloader, imodel):
    with torch.no_grad():
        imodel.eval()
        embeddings = np.zeros((dataloader.dataset.__len__(), 5))
        k = 0
        for images, label in dataloader:
            images = images.to(device)
            embeddings[k:k + len(images)] = imodel(images).data.cpu().numpy()
            k += len(images)

    return embeddings

#%%
trainLoader =  DataLoader(train_data, batch_size=100,
                        shuffle=True)
testLoader =  DataLoader(val_data, batch_size=100,
                        shuffle=True)
trainEmbedding =  extract_embeddings(trainLoader,model)
testEmbedding =  extract_embeddings(testLoader,model)

#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
# neigh = KNeighborsClassifier(n_neighbors=3)
clf = svm.SVC()
y = [x[1].item() for x in trainLoader.dataset]
clf.fit(trainEmbedding,y)
#%%
y_pred = clf.predict(testEmbedding)
y_test = [x[1].item() for x in testLoader.dataset]
#%%
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))