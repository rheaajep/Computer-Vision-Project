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
learning_rate=1e-1

#loading my dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,shuffle=False)

print("dataset downloaded")
#define the convolution network 
def conv3x3(input,output,stride=1):
    conv=nn.Conv2d(input,output,kernel_size=(3,3),stride=stride,padding=0,bias=False)

    return conv

print("convolution defined")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out=self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out=self.bn2(out)
        #out = self.relu(out)
        if self.downsample:
            residual = self.downsample(x)
        out = torch.cat(residual)
        out=self.relu(out)
        return out

print("residual block defined")

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(1, 16)
        self.bn = nn.BatchNorm2d(16)
        #self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[0], 2)
        self.layer3 = self.make_layer(block, 64, layers[1], 2)
        self.avg_pool = nn.AvgPool2d(8,ceil_mode=False)
        self.finalfc = nn.Linear(64, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample)) # 残差直接映射部分
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


net=ResNet(ResidualBlock,[2,2,2,2])

print("resnet defined")

#define the loss function 
criterion=nn.CrossEntropyLoss()

#optimizing function 
optimizer=optim.SGD(net.parameters(),lr=learning_rate)

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

for epoch in range(iteration+1):
    total_loss=0
    total_acc=0
    length=0
    curr_lr=learning_rate
    print("in the loop")
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

    if epoch % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(epoch,total_loss,total_acc)) 
    
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

print("finished training")