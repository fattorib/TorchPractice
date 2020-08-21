import pandas as pd
import numpy as np


stock_prices = pd.read_csv('NOK.csv')


close_prices = stock_prices['Close']


train_data = np.array(close_prices[0:4000])

from sklearn.preprocessing import MinMaxScaler
#Range of data varies a lot, need to rescale

MM = MinMaxScaler()
train_data = MM.fit_transform(train_data.reshape(-1, 1))



import torch 
from torch import nn




# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')


def get_batches(arr, batch_size, seq_length):

    '''       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    ##Get the number of batches we can make
    n_batches = len(arr)//(batch_size*seq_length)
    
    ##Keep only enough characters to make full batches
    arr = arr[0:n_batches*batch_size*seq_length]
    
    ##Reshape into batch_size rows
    
    arr = arr.reshape(batch_size,-1)
    
    ##Iterate over the batches using a window of size seq_length

    for n in range(0, arr.shape[1], seq_length):
        
        x = arr[:,n:n+seq_length]
        y = np.zeros_like(x)
        
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            
        #This is a generator, not a return, all we have to do is reference x,y now
        yield x, y


        
class Model(nn.Module):
    def __init__(self,input_size,output_size, hidden_size,n_layers):
        super().__init__()
        
        #Parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        
        #Model Architecture
        self.lstm = nn.LSTM(self.input_size,self.hidden_size, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size,self.output_size)
        

    def forward(self, x, hidden):
        
        x, hidden = self.lstm(x, hidden)
        
        #Reshaping tensor so it can be applied to fc network
        x = x.contiguous().view(-1,self.hidden_size)
        
        x = self.fc(x)
        
        return x, hidden
    
    
    
    def init_hidden(self,batch_size):
        #Initialize hidden state and cell state, originally set to 0
        if(train_on_gpu):
            weight = next(self.parameters()).data
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda(),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().cuda())
            
            
        else:
            weight = next(self.parameters()).data
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                          weight.new(self.n_layers, batch_size, self.hidden_size).zero_())
        
        return hidden
        





model = Model(1,1,64,1)
#Regression problem, MSELoss is a good choice
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

batch_size = 64
seq_length = 50
clip=5


model = model.cuda()

epochs = 300

for e in range(0,epochs):
    
    hidden = model.init_hidden(batch_size)
    running_loss = 0
    
    for x,y in get_batches(train_data, batch_size, seq_length):
        
        hidden = tuple([each.data for each in hidden])
        
        #Need to add third dimension back in
        x = torch.Tensor(x).view(batch_size, seq_length,1).cuda()
        
        
        y = torch.Tensor(y).view(batch_size*seq_length,1).cuda()
        
        optimizer.zero_grad()
        
        outputs,hidden = model.forward(x,hidden)
        
        
        loss = criterion(outputs,y)
        
        loss.backward()
        
        
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        running_loss += loss.item()
        
        
    # print(running_loss/batch_size)
        
        
#Evaluating Model on Test Data

#Turn off gradient tracking
model.eval()


#Getting test data
test_data = np.array(close_prices[4000:5000])
test_data = MM.transform(test_data.reshape(-1, 1))


predicted_vals = []


#Get first 50 in sequence to prime the LSTM
prime = test_data[0:50]

#Convert to tensor, reshape and pass to GPU
prime = torch.Tensor(prime).view(1,len(prime),1).cuda()

#Initialize null hidden state, with a batch size of 1
hidden = model.init_hidden(1)
hidden = tuple([each.data for each in hidden])




#Model forward pass
y_pred,hidden = model.forward(prime,hidden)
y_pred= y_pred.unsqueeze(0)

# Detach from GPU and pass to numpy array
pred_vals = y_pred.flatten()


pred_vals = pred_vals.cpu().detach().numpy().tolist()

predicted_vals.extend(pred_vals)




for i in range(0,50):
    #Model forward pass
    y_pred,hidden = model.forward(y_pred,hidden)
    y_pred = y_pred.unsqueeze(0)
    
    # Detach from GPU and pass to numpy array
    pred_vals = y_pred.flatten()
    pred_vals = pred_vals.cpu().detach().numpy().tolist()
    predicted_vals.append(pred_vals[-1])





import matplotlib.pyplot as plt

prime = test_data[0:100]
time = np.linspace(0,100,100)
plt.plot(time,predicted_vals, label = 'Predicted')
plt.plot(time,prime, label = 'Ground Truth')
plt.legend()





































        
        
