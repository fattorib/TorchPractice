import torch, torchvision

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms, utils


#Define transforms
#Probably need to add normalization

#Resize to power of 2 for better conv dimensions
transform = transforms.Compose([transforms.Resize((128,128)),                               
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])


#Train/Test paths
train_path = 'seg_train/seg_train'
test_path = 'seg_test/seg_test'

#In total, about 14000 training images
traindata = datasets.ImageFolder(train_path,transform = transform)
testdata = datasets.ImageFolder(test_path,transform = transform)

#Dataloaders
trainloader = DataLoader(traindata,batch_size = 64, shuffle = True)
testloader = DataLoader(testdata,batch_size = 64, shuffle = False)


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
    


criterion = nn.NLLLoss()

model = Model()
model.cuda()

#Loading any previously saved states
state_dict = torch.load('checkpoint.pth')
model.load_state_dict(state_dict)   



optimizer = torch.optim.SGD(model.parameters(),lr = 0.001)
epochs = 10

for e in range(1,epochs+1):
    
    running_loss = 0
    
    for images,labels in trainloader:
        
        #Pass tensors to GPU
        images,labels = images.cuda(),labels.cuda()
        
        optimizer.zero_grad()
        
        output = model(images)
        
        
        loss = criterion(output,labels)
        running_loss += loss.item()
        
        loss.backward()
        optimizer.step()
    
    
    if e%2 == 0:
        
        
        #Evaluate model on test data
        accuracy = 0
        test_loss = 0
        #Turning off gradient tracking
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                logps = model.forward(inputs)
                
                loss = criterion(logps,labels)
                test_loss += loss.item()
                
                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                
        
        print('Epoch:',e)
        print('Training Loss:', running_loss/len(trainloader))
        print('Test Loss:', test_loss/len(testloader))
        print('Test Accuracy:', 100*accuracy/len(testloader),'%')
        
        #Turning back on gradient tracking
        model.train()
        
        
        
#Saving model parameters
torch.save(model.state_dict(), 'checkpoint.pth')
        
        
