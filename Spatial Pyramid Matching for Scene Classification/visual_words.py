import numpy as np
import multiprocessing as mp
import imageio
import scipy.ndimage
import skimage.color
import skimage.io
import sklearn.cluster
import scipy.spatial.distance as sd
import os,time
import util
import random

def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    #check whether the image is in gray scale 
    if image.shape[2]==1:
        image=np.dstack((image,image,image))

    #check whether the image is rgba image 
    elif image.shape[2]==4:
        image=image[:,:,:3]

    #convert the image to lab space 
    image=skimage.color.rgb2lab(image)
    #convert the image to a floating point
    image = image.astype('float')/255
    #extract individual channels 
    image_channel1=image[:,:,0] 
    image_channel2=image[:,:,1]
    image_channel3=image[:,:,2]
    #create individual output channels 
    output1=np.zeros((image.shape[0],image.shape[1])) 
    output2=np.zeros((image.shape[0],image.shape[1]))
    output3=np.zeros((image.shape[0],image.shape[1]))
    #create a final output matrix of size mxnx3F
    filter_responses=np.zeros((image.shape[0],image.shape[1],image.shape[2]*5*4))
    num=0 #initiate the number of channels 
    

    scale=1 #define the initial scale value 
    for i in range(1,6):
        if i==5:
            #apply gaussian filter 
            scipy.ndimage.gaussian_filter(image_channel1,sigma=8*np.sqrt(2),output=output1)
            scipy.ndimage.gaussian_filter(image_channel2,sigma=8*np.sqrt(2),output=output2)
            scipy.ndimage.gaussian_filter(image_channel3,sigma=8*np.sqrt(2),output=output3)
            output=np.dstack((output1,output2,output3))
            filter_responses[:,:,num*3:num*3+3]=output
            num=num+1
            #apply gaussian laplace
            scipy.ndimage.gaussian_laplace(image_channel1,sigma=8*np.sqrt(2),output=output1)
            scipy.ndimage.gaussian_laplace(image_channel2,sigma=8*np.sqrt(2),output=output2)
            scipy.ndimage.gaussian_laplace(image_channel3,sigma=8*np.sqrt(2),output=output3)
            output=np.dstack((output1,output2,output3))
            filter_responses[:,:,num*3:num*3+3]=output
            num=num+1
            #apply derivative of gaussian in x direction 
            scipy.ndimage.gaussian_filter(image_channel1,sigma=8*np.sqrt(2),order=(0,1),output=output1)
            scipy.ndimage.gaussian_filter(image_channel2,sigma=8*np.sqrt(2),order=(0,1),output=output2)
            scipy.ndimage.gaussian_filter(image_channel3,sigma=8*np.sqrt(2),order=(0,1),output=output3)
            output=np.dstack((output1,output2,output3))
            filter_responses[:,:,num*3:num*3+3]=output
            num=num+1
            #apply derivative of gaussian i y direction 
            scipy.ndimage.gaussian_filter(image_channel1,sigma=8*np.sqrt(2),order=(1,0),output=output1)
            scipy.ndimage.gaussian_filter(image_channel2,sigma=8*np.sqrt(2),order=(1,0),output=output2)
            scipy.ndimage.gaussian_filter(image_channel3,sigma=8*np.sqrt(2),order=(1,0),output=output3)
            output=np.dstack((output1,output2,output3))
            filter_responses[:,:,num*3:num*3+3]=output
            num=num+1
            

        else:
            #apply gaussian filter 
            scipy.ndimage.gaussian_filter(image_channel1,sigma=scale,output=output1)
            scipy.ndimage.gaussian_filter(image_channel2,sigma=scale,output=output2)
            scipy.ndimage.gaussian_filter(image_channel3,sigma=scale,output=output3)
            output=np.dstack((output1,output2,output3))
            filter_responses[:,:,num*3:num*3+3]=output
            num=num+1
            #apply gaussian laplace 
            scipy.ndimage.gaussian_laplace(image_channel1,sigma=i,output=output1)
            scipy.ndimage.gaussian_laplace(image_channel2,sigma=i,output=output2)
            scipy.ndimage.gaussian_laplace(image_channel3,sigma=i,output=output3)
            output=np.dstack((output1,output2,output3))
            filter_responses[:,:,num*3:num*3+3]=output
            num=num+1
            #apply derivative of gaussian in x direction 
            scipy.ndimage.gaussian_filter(image_channel1,sigma=i,order=(0,1),output=output1)
            scipy.ndimage.gaussian_filter(image_channel2,sigma=i,order=(0,1),output=output2)
            scipy.ndimage.gaussian_filter(image_channel3,sigma=i,order=(0,1),output=output3)
            output=np.dstack((output1,output2,output3))
            filter_responses[:,:,num*3:num*3+3]=output
            num=num+1
            #apply derivative of gaussian in y direction
            scipy.ndimage.gaussian_filter(image_channel1,sigma=i,order=(1,0),output=output1)
            scipy.ndimage.gaussian_filter(image_channel2,sigma=i,order=(1,0),output=output2)
            scipy.ndimage.gaussian_filter(image_channel3,sigma=i,order=(1,0),output=output3)
            output=np.dstack((output1,output2,output3))
            filter_responses[:,:,num*3:num*3+3]=output
            num=num+1
        #increment the value of sigma
        scale=scale*2
            
    return filter_responses
    


def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    #create a wordmap of the same size as the image 
    x=image.shape[0]
    y=image.shape[1]
    wordmap=np.zeros((x,y))

    #get filter responses for an image
    filter_responses=extract_filter_responses(image)

    #create a temporary array 
    temp_array=np.zeros((y,60))

    for i in range(0,x):
        temp_array[:,:]=filter_responses[i,:,:]
        distance=sd.cdist(temp_array,dictionary,'euclidean')
    
        min_elements=np.argmin(distance,axis=1)
        wordmap[i,:]=min_elements

    return wordmap


def compute_dictionary_one_image(index,alpha,image_path):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''
    image_path=image_path+index
    image=skimage.io.imread(image_path)

    filter_responses=extract_filter_responses(image)

    x=filter_responses.shape[0]
    y=filter_responses.shape[1]
    #pick random alhpa pixels 
    new_x=np.random.choice(x,alpha)
    new_y=np.random.choice(y,alpha)
    #a new filter of size alphax3F
    new_filter_response=np.zeros((alpha,60))

    #assign random pixel values from an earlier filter response to the new filter response
    for num in range(0,alpha):
        for channel in range(0,60):
            new_filter_response[num][channel]=filter_responses[new_x[num],new_y[num],channel]
    

    np.save('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/temp_folder/'+index[:-4]+'.npy',new_filter_response)
    

def compute_dictionary(num_workers=2):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)s
    NOTE : Please save the dictionary as 'dictionary.npy' in the same dir as the code.
    '''

    #loading the training data
    train_data = np.load('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/data/train_data.npz')
    #creating the multiprocessing object pool
    pool=mp.Pool(num_workers)
    #calling the function with multiple processes and passing the argument 
    pool.starmap(compute_dictionary_one_image,[(i,300,'/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/data/') for i in train_data['files']])
    pool.close()
    pool.join()
    #creating a path for temporary folder
    temp_folder_path='/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/temp_folder/'
    #storing the path of the created temporary file for the first element
    element_path=temp_folder_path+train_data['files'][0][:-4]+'.npy'
    #loading the filter response data of the first image data
    filter_responses=np.load(element_path)
    size=len(train_data['files'])
    #for every element in train_data, recall their filter response and concatenate into a final array - filter responses
    for i in range(1,size):
        element_path=temp_folder_path+train_data['files'][i][:-4]+'.npy'
        data=np.load(element_path)
        filter_responses=np.concatenate((filter_responses,data),axis=0)
    
    #call the k-means cluster to 
    kmeans=sklearn.cluster.KMeans(n_clusters=200).fit(filter_responses)
    dictionary=kmeans.cluster_centers_

    np.save('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/code/dictionary.npy',dictionary)


