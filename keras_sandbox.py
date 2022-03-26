import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import matplotlib.pyplot as plt
# import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
# import cv2
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from sys import argv
from torchvision import datasets, transforms
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


import os
#%%
labels = ['PNEUMONIA', 'NORMAL']
input_dir = r'C:\Users\dale\PycharmProjects\COMP-691'
output_dir = r'C:\Users\dale\PycharmProjects\COMP-691'
print("Using input_dir: " + input_dir)
print("Using output_dir: " + output_dir)
i = 1
x0 = torch.load(input_dir + '/data/train_data/train_{}/class_0/image_tensors.pt'.format(i))
x1 = torch.load(input_dir + '/data/train_data/train_{}/class_1/image_tensors.pt'.format(i))

#%%
data_flag = 'pneumoniamnist'
info = INFO[data_flag]
task = info['task']
DataClass = getattr(medmnist, info['python_class'])

# load the data
train_dataset = DataClass(split='train', download=True)
train_dataset.montage(length=10).show()

#%%
x = 1
from PIL import Image

image = Image.open(train_dataset[0][0])
image.show()
#%%
x = x0[0][0].numpy()
plt.matshow(x, )
plt.show()
#%%



transformer = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()])
transformer.transforms(x0)