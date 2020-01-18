import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
import pdb
import argparse


def myHough(img_name,ce_params,hl_params): 
    canny_image=cv2.Canny(img_name,ce_params[0], ce_params[1], ce_params[2])
    hough_lines=cv2.HoughLinesP(canny_image,hl_params[0],hl_params[1],hl_params[2],(hl_params[3],hl_params[4]))
    for i in range(0,hough_lines.shape[0]):
        for [x1,y1,x2,y2] in hough_lines[i]:
            cv2.line(img_name,(x1,y1),(x2,y2),(0,0,255),2)

    cv2.imshow("final_image",img_name)
    cv2.waitKey(0)
    cv2.imwrite("/home/geekerlink/Desktop/Computer Vision/hw1/hw1/results/img01_hlines.jpg",img_name)
    
    
    



if __name__=="__main__":

    # create a list of the params 
    # for both your edge detector 
    # hough transform
    #for image01

    ce_params = [80,130,3]
    hl_params = [1,np.pi/180,5,0,0]
    img_name = cv2.imread("/home/geekerlink/Desktop/Computer Vision/hw1/hw1/data/img01.jpg")
    myHough(img_name,ce_params,hl_params)

    #for image02 - ce_params=[60,175,3],hl_params=[1,np.pi/180,2,0,0]
    #for image03 - ce_params=[70,165,3],hl_params=[1,np.pi/180,0,5,0]
    #for image04 - ce_params=[140,330,3],hl_params=[1,np.pi/180,0,0,0]
    #for image07 - ce_params=[40,330,3],hl_params=[1,np.pi/180,0,5,0]
