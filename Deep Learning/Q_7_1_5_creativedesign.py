import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import scipy.io
import numpy as np


batch_size=50
iteration=15

#loading my dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,shuffle=False)

print("dataset downloaded")

#define the neural network model
class Somenet(nn.Module):
    def __init__(self):
        super(Somenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5,5),stride=1)
        self.activation=nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3),stride=1)
        self.conv3 = nn.Conv2d(32,64,kernel_size=(3,3),stride=1)
        self.conv4 = nn.Conv2d(64,128, kernel_size=(4,4),stride=2)
        self.func1 = nn.Linear(128, 90)
        self.func2= nn.Linear(90,10)
        self.final= nn.Softmax(dim=-1)

    def forward(self, x):
        output = self.pool(self.activation(self.conv1(x)))
        output = self.pool(self.activation(self.conv3(self.conv2(output))))
        output = self.activation(self.conv4(output))
        output = output.view(x.size(0),-1)
        output = self.activation(self.func1(output))
        output = self.final(self.func2(output))
        return output

print("somenet defined")
net=Somenet()
#define the loss function 
criterion=nn.CrossEntropyLoss()

#optimizing function 
optimizer=optim.SGD(net.parameters(),lr=1e-1)

acc_list=[]
loss_list=[]
for epoch in range(iteration+1):
    total_loss=0
    total_acc=0
    length=0
    for data,label in trainloader:
        #print("running")
        optimizer.zero_grad()
        output=net(data)
        loss=criterion(output,label)
        loss.backward()
        optimizer.step()
        total_loss+=loss
        acc=((label==(torch.argmax(output,1))).sum().item())/batch_size
        total_acc+=acc
        length+=1

    total_loss=total_loss/length
    total_acc=(total_acc/length)
    acc_list.append(total_acc)
    loss_list.append(total_loss)

    if epoch % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,total_loss,total_acc))


print("finished training")
#torch.save(net.state_dict(),'C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/python/lnet_Emnist.pth')

import matplotlib.pyplot as plt 

plt.figure(1)
x=np.arange(iteration+1)
plt.plot(x,acc_list)
plt.legend(['accuracy train'])
plt.xlabel('epoch number')
plt.ylabel('accuracy')
#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/715_creativedesign_acc.png')
plt.show()

plt.figure(2)
plt.plot(x,loss_list)
plt.legend('average loss over data')
plt.xlabel('epoch number')
plt.ylabel('loss')
#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/714_creativedesign_loss.png')
plt.show()