import numpy as np
import scipy.io
from nn import *
from collections import Counter

train_data = scipy.io.loadmat('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/data/nist36_train_set1.mat')
valid_data = scipy.io.loadmat('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/data/nist36_valid_set1.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
#print("dataset downloaded")

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# initialize layers here
initialize_weights(1024,32,params,'encoder1')
initialize_weights(32,32,params,'encoder2')
initialize_weights(32,32,params,'encoder3')
initialize_weights(32,1024,params,'decoder')
#print("parameters initialized")

#initializing momentum for w parameters 
params['m_Wencoder1']=np.zeros((1024,32))
params['m_Wencoder2']=np.zeros((32,32))
params['m_Wencoder3']=np.zeros((32,32))
params['m_Wdecoder']=np.zeros((32,1024))

#initializing momentum for b parameters 
params['m_bencoder1']=np.zeros((32))
params['m_bencoder2']=np.zeros((32))
params['m_bencoder3']=np.zeros((32))
params['m_bdecoder']=np.zeros((1024))

#print("momentum initialized")

#loss function 
def compute_loss (yb,xb):
    loss=np.sum(np.square(np.subtract(yb,xb)))
    return loss

total_acc_list=[]
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        h1=forward(xb,params,'encoder1',activation=relu)
        h2=forward(h1,params,'encoder2',activation=relu)
        h3=forward(h2,params,'encoder3',activation=relu)
        output=forward(h3,params,'decoder',activation=sigmoid)
        #print("forward pass done")
        #computing loss 
        loss=compute_loss(output,xb)
        #print("loss computed")
        #backpropagation 
        delta1=2*np.subtract(output,xb)
        delta2=backwards(delta1,params,'decoder',activation_deriv=sigmoid_deriv)
        delta3=backwards(delta2,params,'encoder3',activation_deriv=relu_deriv)
        delta4=backwards(delta3,params,'encoder2',activation_deriv=relu_deriv)
        backwards(delta4,params,'encoder1',activation_deriv=relu_deriv)
        #print("backpropagation done")
        #updating the gradients 
        params['m_Wdecoder'] = 0.9*params['m_Wdecoder'] - learning_rate*params['grad_W'+'decoder']
        params['Wdecoder']= params['Wdecoder'] + params['m_Wdecoder']
        params['m_Wencoder3'] = 0.9*params['m_Wencoder3'] - learning_rate*params['grad_W'+'encoder3']
        params['Wencoder3']= params['Wencoder3'] + params['m_Wencoder3']
        params['m_Wencoder2'] = 0.9*params['m_Wencoder2'] - learning_rate*params['grad_W'+'encoder2']
        params['Wencoder2']= params['Wencoder2'] + params['m_Wencoder2']
        params['m_Wencoder1'] = 0.9*params['m_Wencoder1'] - learning_rate*params['grad_W'+'encoder1']
        params['Wencoder1']= params['Wencoder1'] + params['m_Wencoder1']

        #updating the b parameters 
        params['m_bdecoder'] = 0.9*params['m_bdecoder'] - learning_rate*params['grad_b'+'decoder']
        params['bdecoder']= params['bdecoder'] + params['m_bdecoder']
        params['m_bencoder3'] = 0.9*params['m_bencoder3'] - learning_rate*params['grad_b'+'encoder3']
        params['bencoder3']= params['bencoder3'] + params['m_bencoder3']
        params['m_bencoder2'] = 0.9*params['m_bencoder2'] - learning_rate*params['grad_b'+'encoder2']
        params['bencoder2']= params['bencoder2'] + params['m_bencoder2']
        params['m_bencoder1'] = 0.9*params['m_bencoder1'] - learning_rate*params['grad_b'+'encoder1']
        params['bencoder1']= params['bencoder1'] + params['m_bencoder1']
        #print("update done")
        total_loss+=loss
    
    total_loss=total_loss/(batch_num*batch_size)
    total_acc_list.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9


#plotting 
import matplotlib.pyplot as plt 

plt.figure(1)
x=np.arange(max_iters)
plt.plot(x,total_acc_list)
plt.legend(['Training data Loss'])
plt.xlabel('epoch number')
plt.ylabel('loss')
plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/52_loss_train.png')
plt.show()


# visualize some results
# Q5.3.1
import matplotlib.pyplot as plt
classes=[2,14,6,23,8]
valid_y=valid_data['valid_labels']

for one_class in classes:
    a=np.where(valid_y[:,one_class]==1)
    a=np.reshape(a,(-1))
    indices=np.random.choice(a,size=2,replace=False)
    for index in indices:
        xb=valid_x[index,:]
        h1 = forward(xb,params,'encoder1',relu)
        h2 = forward(h1,params,'encoder2',relu)
        h3 = forward(h2,params,'encoder3',relu)
        out = forward(h3,params,'decoder',sigmoid)
        plt.subplot(2,1,1)
        plt.imshow(xb.reshape(32,32).T)
        plt.subplot(2,1,2)
        plt.imshow(out.reshape(32,32).T)
        plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/531_reconstructed_'+str(one_class)+'class_'+str(index)+'_number.png')
        plt.show()

from skimage.measure import compare_psnr as psnr
# evaluate PSNR
# Q5.3.2
#psnr value for valid data 
psnr_final=0
batches2 = get_random_batches(valid_x,np.ones((valid_x.shape[0],1)),batch_size)
batch_num2=len(batches2)
for xb,_ in batches2:
    h1 = forward(xb,params,'encoder1',relu)
    h2 = forward(h1,params,'encoder2',relu)
    h3 = forward(h2,params,'encoder3',relu)
    out = forward(h3,params,'decoder',sigmoid)
    psnr_final+=psnr(xb,out)

psnr_final/=batch_num2
print("psnr value from valid data: ",psnr_final)

#psnr value for train data
psnr_final=0
for xb,_ in batches:
    h1 = forward(xb,params,'encoder1',relu)
    h2 = forward(h1,params,'encoder2',relu)
    h3 = forward(h2,params,'encoder3',relu)
    out = forward(h3,params,'decoder',sigmoid)
    psnr_final+=psnr(xb,out)

psnr_final/=batch_num
print("psnr value from train data: ",psnr_final)