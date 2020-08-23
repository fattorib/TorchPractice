import torch, torchvision

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, utils


import numpy as np

#Define transforms

#Resize to power of 2 for better conv dimensions
transform = transforms.Compose([transforms.Resize((128,128)),                               
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])



#Train/Test paths
pred_path = 'seg_pred'



#In total, about 14000 training images
preddata = datasets.ImageFolder(pred_path,transform = transform)

#Dataloaders
predloader = DataLoader(preddata,batch_size = 3, shuffle = True)






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
        self.conv1 = nn.Conv2d(3, 32, 3,stride = 1, padding = 1)
        
        #Depth here is the number of feature maps we had previously passed
        self.conv2 = nn.Conv2d(32, 64, 3,stride = 1, padding = 1)
        
        self.conv3 = nn.Conv2d(64,128,3, stride = 1, padding = 1)
        
        self.conv4 = nn.Conv2d(128,256,3, stride = 1, padding = 1)
        
        #Pooling layer with size of 2 and stride of 2. Halving the xy size.
        self.pool = nn.MaxPool2d(2,2)
        
        
        #Final image size is x_final*y_final*depth
        self.fc1 = nn.Linear(8*8*256,256)
        
        self.fc2 = nn.Linear(256,64)
        
        self.fc3 = nn.Linear(64,6)
        
        
        self.dropout = nn.Dropout(p = 0.25)
        
    def forward(self, x):
        x = F.relu(self.pool(self.conv1(x)))
        x = F.relu(self.pool(self.conv2(x)))
        x = F.relu(self.pool(self.conv3(x)))
        x = F.relu(self.pool(self.conv4(x)))
        
        x = x.view(-1,8*8*256)
        
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        
        x = self.dropout(x)
        
        x = self.fc3(x)
        
        
        return F.log_softmax(x,dim = 1) 

#Reversing normaliziation to get original image to show
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
class_names = ['Buildings','Forest','Glacier',
                           'Mountain','Sea', 'Street' ]

model = Model()
model.cuda()


#Loading saved model parameters
state_dict = torch.load('FinalModel.pth')
model.load_state_dict(state_dict)   

images,labels = next(iter(predloader))



image = images.cuda()

logpred = model.forward(image)
top_p, top_class = logpred.topk(1, dim=1)


class_name = class_names[top_class[0].item()]

class_prob = top_p[0].item()
#.cpu() moves to cpu, .detach removes gradients for specific tensor

probabilities = torch.exp(logpred).cpu().detach().numpy()



import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14, 6))

plt.sca(ax1)
plt.xticks([])
plt.yticks([])

probability = 100*max(probabilities[0])

plt.title('{} \n Probability {:.2f} %'.format(class_name,probability))

ax1.imshow(unorm(images[0]).permute(1, 2, 0) )


plt.sca(ax2)
plt.xticks(np.arange(6), ['Buildings','Forest','Glacier',
                           'Mountain','Sea', 'Street' ])
ax2.bar(np.arange(6),probabilities[0], align = 'center')









