import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import multiprocessing as mp 
import matplotlib
from matplotlib import pyplot as plt

def build_recognition_system(num_workers=2):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    train_data = np.load("/data/train_data.npz")
    dictionary = np.load("/code/dictionary.npy")
    base_path="/hw2/data/"
    layer_num=3
    K=dictionary.shape[0]
    pool=mp.Pool(num_workers)
    features=pool.starmap(get_image_feature,[(base_path+file_path,dictionary,layer_num,K) for file_path in train_data['files']])
    pool.close()
    pool.join()
    np.savez('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/trained_system.npz',dictionary=dictionary,features=features,labels=train_data['labels'],SPM_layer_num=layer_num)


def evaluate_recognition_system(num_workers=2):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''


    confusion_matrix=np.zeros((8,8))
    test_data = np.load("/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/data/test_data.npz")
    trained_system = np.load("/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/trained_system.npz")
    train_data=np.load("/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/data/train_data.npz")
    dictionary=np.load("/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/code/dictionary.npy")
    histograms=trained_system['features']
    test_files=test_data['files']
    trained_labels=train_data['labels']
    test_labels=test_data['labels']
    base_path="/home/geekerlink/Desktop/Computer Vision/Homeworks/hw2/hw2/data/"
    layer_num=3
    K=dictionary.shape[0]
    histograms=histograms.reshape(len(histograms),len(histograms[0]))
    pool=mp.Pool(num_workers)
    word_hist_all=pool.starmap(get_image_feature,[(base_path+file_path,dictionary,layer_num,K) for file_path in test_files])
    sim=pool.starmap(distance_to_set,[(word_hist,histograms) for word_hist in word_hist_all ])
    pool.close()
    pool.join()
    for j in range(0,len(sim)):
        temp_array=sim[j]
        min_index=np.argmin(temp_array)
        label_training=trained_labels[min_index]
        label_testing=test_labels[j]
        confusion_matrix[label_testing][label_training]=confusion_matrix[label_testing][label_training]+1


    accuracy=np.trace(confusion_matrix)/np.sum(confusion_matrix)
    return confusion_matrix,accuracy




def get_image_feature(file_path,dictionary,layer_num,K):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^(L+1)/3))
    '''
    image=imageio.imread(file_path)
    wordmap = visual_words.get_visual_words(image,dictionary)
    dict_size=K
    feature=get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size)
    return feature
   


def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''

    min_value=np.minimum(word_hist,histograms)
    sim=np.sum(min_value,axis=1)

    return sim
    



def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    hist,edges=np.histogram(wordmap,bins=dict_size,density=True)
    return hist



def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3) (as given in the write-up)
    '''
    
    layers=np.arange(0,layer_num,1)
    hist_all=[]
    for l in layers:
        if l==0 or l==1:
            weight=np.float_power(2,(-layer_num))
        else:
            weight=np.float_power(2,(-l-(layer_num)-1))

        #declare the dimensions of the cell
        dimension_of_cells=2**l
        #splitting the rows and columns in number of cells 
        rows=np.array_split(wordmap,dimension_of_cells,axis=1)
        
        

        for i in rows:
            columns=np.array_split(i,dimension_of_cells,axis=0)
            for cell in columns:
                hist=get_feature_from_wordmap(cell,dict_size)
                hist=hist*weight
                hist_all=np.append(hist_all,hist)

    hist_all=hist_all/np.max(hist_all)  
    return hist_all






    

