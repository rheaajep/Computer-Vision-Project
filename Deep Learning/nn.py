import numpy as np
from util import *
import random
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    #initializing bias vector b to zero - size is the output layer size 
    b=np.zeros(out_size)
    mean=0
    var=2/(in_size+out_size)
    sd=np.sqrt(var)
    W=np.random.normal(mean,sd,(in_size,out_size))
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res=1/(1+np.exp(-x))
    
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    #print("w shape in forward: ",W.shape)
    #print("shape of b is: ",b.shape)
    # your code here
    pre_act=np.dot(X,W) +b
    #print(pre_act)
    post_act=activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):

    c=-(np.max(x,axis=1))
    c=np.reshape(c,(-1,1))
    x= np.add(x,c)
    x_new=np.exp(x)
    x_sum=np.sum(x_new,axis=1)
    x_sum=np.reshape(x_sum,(-1,1))
    #print(x_sum)
    res=np.divide(x_new,x_sum)

    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss=np.log(probs)
    y=np.array(y)
    loss=(np.multiply(y,loss))
    loss=-(np.sum(loss))
    acc = np.sum(np.equal(np.argmax(y, axis=-1), np.argmax(probs, axis=-1))) / y.shape[0]
    
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    
    res=activation_deriv(post_act)
    delta=delta*res
    grad_W=np.dot(np.transpose(X),delta)
    grad_X=np.dot(delta,np.transpose(W))
    grad_b=np.ones((1,delta.shape[0]))
    grad_b=np.reshape(np.dot(grad_b,delta),(-1))

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches=[]
    size=x.shape[0]
    num_samples=np.divide(size,batch_size)
    #print(num_samples)
    batch_index=np.random.choice(int(size),size=size,replace=False).reshape(int(num_samples),int(batch_size))
    #print(batch_index)
    for i in range(0,int(num_samples)):
        batch=batch_index[i,:]
        batchx=x[batch,:]
        #print(batchx.shape)
        batchy=y[batch,:]
        #print(batchy.shape)
        batch=(batchx,batchy)
        batches.append(batch)


    return batches
