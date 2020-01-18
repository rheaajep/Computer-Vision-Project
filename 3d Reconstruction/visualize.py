'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''
import submission
import helper
import numpy as np
import findM2
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import cv2


if __name__=="__main__":
    pts=np.load('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/data/some_corresp.npz')
    a=pts.files
    pts1=pts[a[0]]
    pts2=pts[a[1]]
    image1=cv2.imread('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/data/im1.png')
    image2=cv2.imread('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/data/im2.png')
    im_width=image1.shape[1]
    im_height=image1.shape[0]
    #calculating the scaling factor
    M=max(im_height,im_width)
    M=float(M)
    #calculating the fundamental matrix
    F=submission.eightpoint(pts1,pts2,M)
    helper.displayEpipolarF(image1,image2,F)
    #seven point algorithm
    F_seven_temp=submission.sevenpoint(pts1,pts2,M)
    errorvalue=[]
    for i in range(0,3):
        least_error=np.sum(np.square(np.subtract(F,F_seven_temp[i])))
        errorvalue.append(least_error)
    error_index=np.argmin(errorvalue)
    F_seven=F_seven_temp[error_index]
    helper.displayEpipolarF(image1,image2,F_seven)
    #intrinsic parameters
    intrinsic=np.load('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/data/intrinsics.npz')
    b=intrinsic.files
    k1=intrinsic[b[0]]
    k2=intrinsic[b[1]]
    #calculating essential matrix
    E=submission.essentialMatrix(F,k1,k2)
    #matching epipolar correspondences 
    epipolarMatchGUI(image1,image2,F)
    #loading image 1 coordinates
    points1=np.load('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/data/templeCoords.npz')
    x1=points1['x1']
    y1=points1['y1']
    number_points=len(x1)
    x2=np.zeros((number_points,1))
    y2=np.zeros((number_points,1))
    #finding corresponding points in image 2 
    for i in range(0,number_points):
        [x2_temp,y2_temp]=submission.epipolarCorrespondence(image1,image2,F,x1[i],y1[i])
        x2[i,0]=x2_temp
        y2[i,0]=y2_temp   

    pts1_new=np.concatenate((x1,y1),axis=1)
    pts2_new=np.concatenate((x2,y2),axis=1)
    M1,M2,C1,C2,w=findM2.findM2(E,k1,k2,pts1_new,pts2_new)

    #np.savez('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/results/q4_2.npz',F=F,M1=M1,M2=M2,C1=C1,C2=C2)
    fig = plt.figure()
    wx = fig.add_subplot(111, projection='3d')
    x=w[:,0]
    y=w[:,1]
    z=w[:,2]
    wx.scatter(x, y, z, c='r', marker='o')

    wx.set_xlabel('X Label')
    wx.set_ylabel('Y Label')
    wx.set_zlabel('Z Label')

    plt.show()