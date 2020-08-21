import torch, torchvision

import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, utils


#Define transforms
#Probably need to add normalization

transform = transforms.Compose([transforms.ToTensor()])


#Train/Test paths
train_path = 'seg_train/seg_train'
test_path = 'seg_test/seg_test'


traindata = datasets.ImageFolder(train_path,transform = transform)
testdata = datasets.ImageFolder(test_path,transform = transform)

#Dataloaders
trainloader = DataLoader(traindata,batch_size = 32, shuffle = True)
testloader = DataLoader(testdata,batch_size = 32, shuffle = True)



