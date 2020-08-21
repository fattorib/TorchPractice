import torch, torchvision

import torch.nn as nn
import torch.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, utils


#Define transforms
#Probably need to add normalization

transform = transforms.Compose([transforms.Resize(150),transforms.ToTensor()])


#Train/Test paths
train_path = 'seg_train/seg_train'
test_path = 'seg_test/seg_test'


traindata = datasets.ImageFolder(train_path,transform = transform)
testdata = datasets.ImageFolder(test_path,transform = transform)

#Dataloaders
trainloader = DataLoader(traindata,batch_size = 32, shuffle = True)
testloader = DataLoader(testdata,batch_size = 32, shuffle = True)


#Checking whether we have GPU
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')


class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        #Tuple given as: input channels, feature maps, kernel size. Keeping default size of 1
        self.conv1 = nn.Conv2d(3, 64, 3,padding = 1)
        
        #Depth here is the number of feature maps we had previously passed
        self.conv2 = nn.Conv2d(64, 32, 3,padding = 1)
        
        #Pooling layer with size of 2 and stride of 2. Effectively halving the image size.
        self.pool = nn.MaxPool2d(2,2)
        
        
        #Final image size is x_final*y_final*depth. In our case this is 75*75*32
        self.fc1 = nn.Linear(75*75*32,256)
        
        self.fc2 = nn.Linear(256,10)
        
        
    def forward(x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.conv2(x))
        
        

    
    

