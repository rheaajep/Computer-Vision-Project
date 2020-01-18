import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('.../data/nist36_train_set1.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid_set1.mat')
test_data= scipy.io.loadmat('../data/nist36_test_set1.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

#print(train_x.shape)
#print(valid_x.shape)
#print(valid_y.shape)
x_dim=train_x.shape[1]
y_dim=train_y.shape[1]

max_iters = 90
# pick a batch size, learning rate
batch_size = 50
learning_rate = 3e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)
batches1= get_random_batches(valid_x,valid_y,batch_size)

params = {}

# initialize layers here
initialize_weights(x_dim,hidden_size,params,'layer1')
initial_weights=params['Wlayer1']
initialize_weights(hidden_size,y_dim,params,'output')

loss_list=[]
acc_list=[]
acc_list_valid=[]
# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters+1):
    total_loss = 0
    total_acc = 0
    total_acc_valid=0
    #total_loss_valid=0
    for xb,yb in batches:
        #print("iter: ",i)
        # forward
        #print("input xb shape :",xb.shape)
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        #print("done 1")
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss=total_loss+loss
        total_acc=total_acc+acc
        #print("done 2")
        # backward
        delta1 = probs-yb
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)
        #print("done 3")
        # apply gradient
        params['W'+'output']=params['W'+'output']-learning_rate*params['grad_W'+'output']
        params['W'+'layer1']=params['W'+'layer1']-learning_rate*params['grad_W'+'layer1']
        params['b'+'output']=params['b'+'output']-learning_rate*params['grad_b'+'output']
        params['b'+'layer1']=params['b'+'layer1']-learning_rate*params['grad_b'+'layer1']
        
        #i=i+1
        #print("done 4")

    total_acc=total_acc/len(batches)
    total_loss=total_loss/len(batches)
    loss_list.append(total_loss)
    acc_list.append(total_acc)
        # training loop can be exactly the same as q2!
        
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    # run on validation set and report accuracy! should be above 75%

    for xc,yc in batches1:
        h1 = forward(xc,params,'layer1')
        probs = forward(h1,params,'output',softmax)
        #print("done 1")
        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yc, probs)
        #total_loss_valid=total_loss_valid+loss
        total_acc_valid=total_acc_valid+acc
    
    total_acc_valid=total_acc_valid/len(batches1)
    acc_list_valid.append(total_acc_valid)

    if itr % 2 == 0:
        print("itr valid: {:02d} \t acc valid: {:.2f}".format(itr,total_acc_valid))



valid_acc = total_acc_valid

print('Validation accuracy: ',valid_acc)

import matplotlib.pyplot as plt 

plt.figure(1)
x=np.arange(max_iters+1)
plt.plot(x,acc_list,x,acc_list_valid)
plt.legend(['accuracy train','accuracy valid'])
plt.xlabel('epoch number')
plt.ylabel('accuracy')
#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/315_lower_acc.png')
plt.show()

plt.figure(2)
plt.plot(x,loss_list)
plt.legend('average loss over data')
plt.xlabel('epoch number')
plt.ylabel('loss')
#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/315_lower_loss.png')
plt.show()

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()
import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
saved_params['Worig']=initial_weights
with open('.../q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Q3.1.3
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid

weights=pickle.load(open('../q3_weights.pickle', 'rb'))

w1=weights['Wlayer1']
print(w1.shape)
figure3=plt.figure(3)
grid = ImageGrid(figure3, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
size1=0
for i in range(0,16):
    im=w1[size1:size1+64,:]
    grid[i].imshow(im)
    size1+=64

#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/315_learned_weights.png')
plt.show()

w2=weights['Worig']
#print(w2)
figure4=plt.figure(4)
grid = ImageGrid(figure4, 111,  # similar to subplot(111)
                 nrows_ncols=(4, 4),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )
size1=0
for i in range(0,16):
    im=w2[size1:size1+64,:]
    grid[i].imshow(im)
    size1+=64       

#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/315_initial_weights.png')
plt.show()

# Q3.1.3
import matplotlib.pyplot as plt 
import pickle
from mpl_toolkits.axes_grid1 import ImageGrid

trained_params=pickle.load(open('../q3_weights.pickle', 'rb'))

h1 = forward(test_x,trained_params,'layer1')
probs = forward(h1,trained_params,'output',softmax)

#print("test_y: ",test_y)
#print("predicted y: ",probs)
sample_size=test_y.shape[0]
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
data_index=np.argmax(test_y,axis=1).reshape(sample_size,1)
predicted_index=np.argmax(probs,axis=1).reshape(sample_size,1)
#print(data_index)
#print(predicted_index)
#indices=np.concatenate((data_index,predicted_index),axis=1)
for i in range(0,sample_size):
    confusion_matrix[data_index[i],predicted_index[i]]+=1



import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
#plt.savefig('C:/Users/rheap/Documents/CMU/Fall 2019/Computer Vision 16720-A/Homework Solutions/hw5/hw5/results/315_confusion_matrix.png')
plt.show()
