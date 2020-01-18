import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import scipy.io
import numpy as np


train_data = scipy.io.loadmat('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/data/nist36_train_set1.mat')
#test_data= scipy.io.loadmat('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/data/nist36_test_set1.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
#test_x, test_y = test_data['test_data'], test_data['test_labels']
train_x=torch.from_numpy(train_x)
train_y=torch.from_numpy(train_y)
training_data= torch.utils.data.TensorDataset(train_x,train_y)
trainloader=torch.utils.data.DataLoader(training_data, batch_size=50,shuffle=True)
input_size=1024
hidden_size=64
output_size=36
iteration=20
batch_size=50

#define the neural network model
model=nn.Sequential(nn.Linear(input_size,hidden_size),nn.Sigmoid(),nn.Linear(hidden_size,output_size),nn.Softmax(dim=1))
model=model.float()

#define the loss function 
criterion=nn.CrossEntropyLoss()

#optimizing function 
optimizer=optim.SGD(model.parameters(),lr=3e-3)

acc_list=[]
loss_list=[]
for epoch in range(iteration):
    total_loss=0
    total_acc=0
    length=0
    for data,label in trainloader:
        #print(data)
        optimizer.zero_grad()
        output=model(data)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        acc=((label==(torch.argmax(output,1))).sum().item())/batch_size
        total_loss+=loss
        total_acc+=acc
        #print(torch.is_tensor(data))
        length+=1

    total_loss=total_loss/length
    total_acc=(total_acc/length)
    acc_list.append(total_acc)
    loss_list.append(total_loss)


    if epoch % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} ".format(epoch,total_loss))


print("finished training")

import matplotlib.pyplot as plt 

plt.figure(1)
x=np.arange(iteration+1)
plt.plot(x,acc_list)
plt.legend(['accuracy train'])
plt.xlabel('epoch number')
plt.ylabel('accuracy')
plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/711_mnist_acc.png')
plt.show()

plt.figure(2)
plt.plot(x,loss_list)
plt.legend('average loss over data')
plt.xlabel('epoch number')
plt.ylabel('loss')
plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/711_mnsit_loss.png')
plt.show()