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



images,labels = next(iter(trainloader))

dict_labels = {0:'Buildings', 1:'Forest', 2:'Glacier', 3:'Mountain', 4:'Sea', 5:'Street'}

label_num = labels[0].item()
import matplotlib.pyplot as plt
plt.style.use('ggplot')

plt.title(dict_labels[label_num])
frame1 = plt.gca()
frame1.axes.xaxis.set_ticklabels([])
frame1.axes.yaxis.set_ticklabels([])
frame1.axes.get_xaxis().set_ticks([])
frame1.axes.get_yaxis().set_ticks([])
plt.imshow(images[0].permute(1, 2, 0))
