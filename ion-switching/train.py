import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import KFold, train_test_split

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.nn import functional as F
import models
import torch.nn.init as init

df_train = pd.read_csv("/home/aistudio/work/github/kaggle/ion-switching/train_clean.csv")
df_test = pd.read_csv("/home/aistudio/work/github/kaggle/ion-switching/test_clean.csv")


train_input = df_train["signal"].values.reshape(-1,400,1)#number_of_data:1250 x time_step:4000
train_input_mean = train_input.mean()
train_input_sigma = train_input.std()
train_input = (train_input-train_input_mean)/train_input_sigma
test_input = df_test["signal"].values.reshape(-1,400,1)#
test_input = (test_input-train_input_mean)/train_input_sigma


train_target = df_train["open_channels"].values.reshape(-1,400,1)#regression
#train_target = pd.get_dummies(df_train["open_channels"]).values.reshape(-1,4000,11)#classification

idx = np.arange(train_input.shape[0])
train_idx, val_idx = train_test_split(idx, random_state = 321,test_size = 0.2)


val_input = torch.from_numpy(train_input[val_idx]).float()
train_input = torch.from_numpy(train_input[train_idx]).float()
val_target = torch.from_numpy(train_target[val_idx]).float()
train_target = torch.from_numpy(train_target[train_idx]).float()
test_input = torch.from_numpy(test_input).float()
'''
train_input = torch.from_numpy(train_input).float()
train_target = torch.from_numpy(train_target).float()

'''


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv3d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)

class Trainset(Dataset):
    def __init__(self,transform=None):
        super(Trainset,self).__init__()
        self.train_input = train_input
        self.train_target = train_target
        self.transform = transform
    def __getitem__(self, index):
        return self.train_input[index],self.train_target[index]
    def __len__(self):
        return len(train_input)

class Valset(Dataset):
    def __init__(self,transform=None):
        super(Valset,self).__init__()
        self.val_input = val_input
        self.val_target = val_target
        self.transform = transform
    def __getitem__(self, index):
        return self.val_input[index],self.val_target[index]
    def __len__(self):
        return len(val_input)

class Testset(Dataset):
    def __init__(self,transform=None):
        super(Testset,self).__init__()
        self.test_input = test_input
        self.transform = transform
    def __getitem__(self, index):
        return self.test_input[index]
    def __len__(self):
        return len(test_input)

batch = 100
loader = DataLoader(dataset=Trainset(), batch_size=batch)

testloader = DataLoader(dataset=Testset(), batch_size=batch)

valloader = DataLoader(dataset=Valset(), batch_size=batch)
    
epoch = 100
lr = 0.0002

criterion = nn.CrossEntropyLoss()
print('是否加载现有模型?')
ifload = input()
if ifload == '1':
    model = torch.load('/home/aistudio/work/github/kaggle/ion-switching/model.pth')
else:
    model = models.Convnet()
    model.apply(weights_init)
model.cuda()
print(model.parameters)
optimer = optim.RMSprop(model.parameters(), lr=lr)
#RMSprop
loss = 0
avg_loss = 0
j = 0
#minloss = torch.load('minloss.pth')
minloss = 0.089
print('之前最好成绩：'+str(minloss))
#minloss = 10000
k = 0
for i in range(0,epoch):
    model.train()
    for data,target in loader:
        model.zero_grad()
        out = model(data.cuda())
        out = out.view(-1,11)
        target = target.view(-1).long()
        loss = criterion(out,target.cuda())
        loss.backward()
        optimer.step()
        avg_loss+=loss
        j+=1
        '''
    for m in model.parameters():
        class_name=m.__class__.__name__
        if class_name.find('Conv')!=-1:
            m.data.clamp_(-1,1)
            '''
    print('第'+str(i)+'次, train loss:'+str(avg_loss/j))
    avg_loss = 0
    j=0
    #torch.save(model,'model.pth')
    model.eval()
    with torch.no_grad():
        for data,target in valloader:
            k+=1
            out = model(data.cuda())
            out = out.view(-1,11)
            target = target.view(-1).long()
            loss = criterion(out,target.cuda())
            avg_loss += loss
    avg_loss = avg_loss/k

    print('         test loss:'+str(avg_loss))
    if avg_loss<minloss:
        minloss = avg_loss
        torch.save(model,'minconvmodel.pth')
        torch.save(minloss,'minloss.pth')
    torch.cuda.empty_cache()
    avg_loss = 0
    k = 0
torch.save(model,'model.pth')

del model

model_test = torch.load('minconvmodel.pth')
minloss = torch.load('minloss.pth')
flag = 0
out2 = 0
print(type(out2))
model_test.eval()
for data in testloader:
    out = model_test(data.cuda())
    data.cpu()
    out1 = out.view(-1,11).cpu()
    _,order = torch.sort(out1,dim = 1,descending=True)
    out1 = torch.transpose(order,0,1)[0]
    if flag == 0:
        out2 = out1
        flag = 1
    else:
        out2 = torch.cat((out2,out1),0)

print(out2.size())
df_sub = pd.read_csv("/home/aistudio/work/github/kaggle/ion-switching/sample_submission.csv", dtype={'time':str})
df_sub.open_channels = out2.int().detach().numpy()
df_sub.to_csv("/home/aistudio/work/github/kaggle/ion-switching/submission.csv",index=False)
print('最好成绩：'+str(minloss))
'''
print('sb')
res = 200
plt.figure(figsize=(20,5))
plt.plot(df_sub.time[::res],df_sub.open_channels[::res])
plt.show()

'''






