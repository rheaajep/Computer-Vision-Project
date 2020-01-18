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

train_data = scipy.io.loadmat('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/data/nist36_train_set1.mat')
#test_data= scipy.io.loadmat('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/data/nist36_test_set1.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
#test_x, test_y = test_data['test_data'], test_data['test_labels']
train_x=torch.from_numpy(train_x)
train_y=torch.from_numpy(train_y)
training_data= torch.utils.data.TensorDataset(train_x,train_y)
trainloader=torch.utils.data.DataLoader(training_data, batch_size=50,shuffle=True)

#define the neural network model
class Lnet(nn.Module):
    def __init__(self):
        super(Lnet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(4,4),stride=1)
        self.activation=nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(4,4),stride=1)
        self.conv3 = nn.Conv2d(16,120,kernel_size=(4,4))
        self.func1 = nn.Linear(120, 84)
        self.full= nn.Linear(84,36)
        self.final= nn.Softmax(dim=-1)

    def forward(self, x):
        print(" in here")
        output = self.conv1(x)
        print("done 1")
        output = self.activation(output)
        print("done 2")
        output = self.pool(output)
        print("done 3")
        output = self.conv3(self.pool(self.activation(self.conv2(output))))
        output = output.view(x.size(0),-1)
        output = self.activation(self.func1(output))
        output = self.final(self.full(output))
        return output

net=Lnet()
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
        print("running")
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
#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/711_mnist_acc.png')
plt.show()

plt.figure(2)
plt.plot(x,loss_list)
plt.legend('average loss over data')
plt.xlabel('epoch number')
plt.ylabel('loss')
#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/711_mnsit_loss.png')
plt.show()