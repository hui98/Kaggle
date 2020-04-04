import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F

class Mymodel(nn.Module):
    def __init__(self,num= 1):
        super(Mymodel,self).__init__()
        self.linear1 = nn.Linear(num,100)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm = nn.LSTM(100,100,batch_first = True,bidirectional = True,num_layers  = 2)
        self.dropout2 = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        self.Linear3 =  nn.Linear(200,11)
    def forward(self,inputs):
        out = F.relu(self.linear1(inputs))
        out = self.dropout1(out)
        out = self.lstm(out)
        out = self.relu(out[0])
        out = self.dropout2(out)
        out = self.Linear3(out)
        return out

class Resblock(nn.Module):
    def __init__(self,channels,kernel,stride,padding):
        super(Resblock,self).__init__()
        self.conv1 = nn.Conv1d(channels,channels,kernel,stride,padding,bias = True)
        self.BN1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.dropout1= nn.Dropout(0.05)
    def forward(self,inputs):
        out = self.conv1(inputs)
        out = self.BN1(out)
        out = self.relu(out)
        out = inputs + out
        out = self.dropout1(out)
        return out

class Rorblock(nn.Module):
    def __init__(self):
        super(Rorblock,self).__init__()
        self.seq = nn.Sequential(
        Resblock(500,7,1,3),
        Resblock(500,5,1,2),
        Resblock(500,3,1,1),
        )
    def forward(self,inputs):
        out = self.seq(inputs)
        out = F.relu(out)
        out = out+inputs
        return out
    
class Convnet(nn.Module):
    def __init__(self):
        super(Convnet,self).__init__()
        self.conv1 = nn.Conv1d(1,100,3,1,1,bias = False)
        self.conv2 = nn.Conv1d(1,300,5,1,2,bias = False)
        self.conv3 = nn.Conv1d(1,50,1,1,0,bias = False)
        self.conv4 = nn.Conv1d(1,50,7,1,3,bias = False)
        
        self.seq1 = nn.Sequential(
            Resblock(500,7,1,3),
            Resblock(500,5,1,2),
            Resblock(500,3,1,1),
            Resblock(500,7,1,3),
            Resblock(500,5,1,2),
            Resblock(500,3,1,1),
            Resblock(500,7,1,3),
            Resblock(500,5,1,2),
            Resblock(500,3,1,1),
            Resblock(500,7,1,3),
            Resblock(500,5,1,2),
            Resblock(500,3,1,1),
            )
        

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.linear = nn.Linear(500,200)
        self.lstm = Mymodel(200)

    def forward(self,inputs):
        inputs = inputs.transpose(1,2)
        out =  F.relu(self.conv1(inputs))
        out2 =  F.relu(self.conv2(inputs))
        out3 =  F.relu(self.conv3(inputs))
        out4 = F.relu(self.conv4(inputs))
        out = torch.cat((out,out2),1)
        out = torch.cat((out,out3),1)
        out = torch.cat((out,out4),1)
        del out2,out3,out4
        out = self.dropout1(out)
        

        out = self.seq1(out)

        
        out = torch.transpose(out,1,2)
        out = F.relu(self.linear(out))
        out = self.dropout2(out)
        out = self.lstm(out)
        #out = self.linear2(out)
        return out


if __name__ == "__main__":     
    '''
    a = torch.randn(10,400,1)
    abc = Convnet()
    o = abc(a)
    print(o.size())
    '''
    a = Mymodel(200)
    torch.save(a,'mymodel.pth')
        
        
        
        
        
        