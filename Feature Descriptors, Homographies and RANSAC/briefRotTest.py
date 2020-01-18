import numpy as np
import cv2
import os
from scipy.spatial.distance import cdist
from keypointDetect import DoGdetector
from BRIEF import briefLite
from BRIEF import briefMatch
import matplotlib.pyplot as plt


if __name__=='__main__':

    im3 = cv2.imread('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/data/model_chickenbroth.jpg')
    locs3, desc3 = briefLite(im3)
    (h,w)=im3.shape[:2]
    centre=(w/2,h/2)
    scale=1
    num_of_matches=[]
    rotation_angles=[]
    for angle in range(0,360,10):
        rotation=cv2.getRotationMatrix2D(centre,angle,scale)
        im3_rotated=cv2.warpAffine(im3,rotation,(w,h))
        locs4,desc4=briefLite(im3_rotated)
        matches=briefMatch(desc3,desc4)
        num_of_matches.append(matches.shape[0])
        rotation_angles.append(angle)

    plt.bar(rotation_angles,num_of_matches,width=3.5,align='center')
    plt.xlabel('angle')
    plt.ylabel('number of matches')
    plt.savefig('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/bar_graph.jpg')
    plt.show()