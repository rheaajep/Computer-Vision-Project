"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
from numpy import matlib
import helper
from helper import _singularize
from helper import refineF
import random
from scipy.ndimage import gaussian_filter
# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    #making a scaling T matrix 
    T=np.zeros((3,3))
    T[0][0]=1/M 
    T[1][1]=1/M
    T[2][2]=1
    #number of correspondences
    n=pts1.shape[0] 
    A=np.zeros((n,9))
    #normalizing the matrix
    pts1=pts1/M
    pts2=pts2/M
    #creating the a matrix 
    A[:,0]=np.multiply(pts1[:,0],pts2[:,0])
    A[:,1]=np.multiply(pts1[:,0],pts2[:,1])
    A[:,2]=pts1[:,0]
    A[:,3]=np.multiply(pts1[:,1],pts2[:,0])
    A[:,4]=np.multiply(pts1[:,1],pts2[:,1])
    A[:,5]=pts1[:,1]
    A[:,6]=pts2[:,0]
    A[:,7]=pts2[:,1]
    A[:,8]=np.ones(n)
    #svd and reshaping fundamental matrix 
    U,S,V=np.linalg.svd(A)
    F=np.transpose(V[-1,:]).reshape(3,3)
    #singulairy function 
    F=_singularize(F)
    #refine 
    F=refineF(F,pts1,pts2)
    #unnormalize the matrix 
    mat=np.matmul(T.transpose(),F)
    F=np.matmul(mat,T)
    #np.savez('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/results/q2_1.npz',F=F,M=M)
    return F







'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    #loading the fundamental matrix found in eightpoint
    data=np.load('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/results/q2_1.npz')
    a=data.files
    FF=data[a[1]]
    #creating empty array for points 1 and 2
    pts1_temp=[]
    pts2_temp=[]
    all_value=[]

    #selecting random 7 points 
    for i in range(0,800):
        pts_index=np.array((random.sample(range(pts1.shape[0]),7)))
        points1=pts1[pts_index]
        points2=pts2[pts_index]
        pts1_temp.append(points1)
        pts2_temp.append(points2)
        points1=np.insert(points1,2,1,axis=1)
        points2=np.insert(points2,2,1,axis=1)
        value=0
        for j in range(0,7):
            pt1=points1[j,:]
            pt2=points2[j,:]
            value=value+np.matmul(np.matmul(np.transpose(pt2),FF),pt1)

        #storing the random values selected and the error value 
        all_value.append(value)

    #selecting value closest to 0 
    all_value_index=np.argmin(np.abs(all_value))
    pts1=pts1_temp[all_value_index]
    pts2=pts2_temp[all_value_index]
    
    #normalizing these points 
    pts1=pts1/M
    pts2=pts2/M
    #creating the a matrix 
    A=np.zeros((7,9))
    A[:,0]=np.multiply(pts1[:,0],pts2[:,0])
    A[:,1]=np.multiply(pts1[:,0],pts2[:,1])
    A[:,2]=pts1[:,0]
    A[:,3]=np.multiply(pts1[:,1],pts2[:,0])
    A[:,4]=np.multiply(pts1[:,1],pts2[:,1])
    A[:,5]=pts1[:,1]
    A[:,6]=pts2[:,0]
    A[:,7]=pts2[:,1]
    A[:,8]=np.ones(7)
    #svd and last 2 columns of V matrix 
    U,S,V=np.linalg.svd(A)
    f1=np.transpose(V[-1,:]).reshape(3,3)
    f2=np.transpose(V[-2,:]).reshape(3,3)
    #making the polynomial equation
    fun=lambda a:np.linalg.det(a*f1+(1-a)*f2)
    a0=fun(0)
    a1=2*(fun(1)-fun(-1))/3 - (fun(2)-fun(-2))/12
    a2=0.5*fun(1)+0.5*fun(-1)-fun(0)
    a3=fun(1)-a0-a1-a2
    solution=np.roots([a3,a2,a1,a0])
    #making a scaling T matrix 
    T=np.zeros((3,3))
    T[0][0]=1/M 
    T[1][1]=1/M
    T[2][2]=1
    F=[]
    errorvalue=[]
    for i in range(0,3):
        alpha=solution[i]
        #find the fundamental matrix through alpha value
        F_temp=np.add(np.multiply(alpha,f1),np.multiply(1-alpha,f2))
        #refine
        F_temp=refineF(F_temp,pts1,pts2)
        #unnormalize F
        mat=np.matmul(T.transpose(),F_temp)
        F_temp=np.matmul(mat,T)
        #putting it in the main F list 
        F.append(F_temp)
        least_error=np.sum(np.square(np.subtract(FF,F_temp)))
        errorvalue.append(least_error)
    
    error_index=np.argmin(errorvalue)
    F=F[error_index]
    #np.savez('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/results/q2_2.npz',F=F,M=M,pts1=pts1,pts2=pts2)

    return F



'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E=np.matmul(np.matmul(np.transpose(K2),F),K1)

    return E

'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    n=pts1.shape[0]
    A=np.zeros((4,4))
    w=[]
    for i in range(0,n):
        #make the A matrix
        A[0,:]=[(pts1[i,1]*C1[2,0])-C1[1,0],(pts1[i,1]*C1[2,1])-C1[1,1],(pts1[i,1]*C1[2,2])-C1[1,2],(pts1[i,1]*C1[2,3])-C1[1,3]]
        A[1,:]=[(pts1[i,0]*C1[2,0])-C1[0,0],(pts1[i,0]*C1[2,1])-C1[0,1],(pts1[i,0]*C1[2,2])-C1[0,2],(pts1[i,0]*C1[2,3])-C1[0,3]]
        A[2,:]=[(pts2[i,1]*C2[2,0])-C2[0,0],(pts2[i,1]*C2[2,1])-C2[1,1],(pts2[i,1]*C2[2,2])-C2[1,2],(pts2[i,1]*C2[2,3])-C2[1,3]]
        A[3,:]=[(pts2[i,0]*C2[2,0])-C2[0,0],(pts2[i,0]*C2[2,1])-C2[0,1],(pts2[i,0]*C2[2,2])-C2[0,2],(pts2[i,0]*C2[2,3])-C2[0,3]]
        #finding 3d points 
        U,S,V=np.linalg.svd(A)
        w_temp=np.transpose(V[-1,:])
        #normalizing the points
        w_temp=np.divide(w_temp[0:4],w_temp[3])
        w.append(w_temp)

    w=np.reshape(w,(n,4))
    
    #reprojecting the 3d points 
    proj1=[]
    proj2=[]
    for i in range(0,n):
        w_temp=w[i,:]
        w_temp=np.transpose(w_temp)
        proj1_temp=np.matmul(C1,w_temp)
        proj2_temp=np.matmul(C2,w_temp)
        proj1.append(proj1_temp)
        proj2.append(proj2_temp)

    proj1=np.reshape(proj1,(n,3))
    proj2=np.reshape(proj2,(n,3))
    pts1=np.insert(pts1,2,1,axis=1)
    pts2=np.insert(pts2,2,1,axis=1)
    error1=np.sum(np.square(np.subtract(pts1,proj1)),axis=1)
    error2=np.sum(np.square(np.subtract(pts2,proj2)),axis=1)
    error=np.add(error1,error2)
    error=np.sum(error)
    w=np.delete(w,3,axis=1)

    return w,error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    points=[[x1],[y1],[1]]
    #finding the epipolar line 
    epipolar_line=np.dot(F,points)
    epipolar_line=epipolar_line/np.linalg.norm(epipolar_line)
    a=epipolar_line[0][0]
    b=epipolar_line[1][0]
    c=epipolar_line[2][0]
    #window size and coordinate range 
    window_size=10
    coordinates=20
    min_error=np.inf
    x2=0
    y2=0
    #creating the patch 
    window_image1=im1[int(y1-window_size):int(y1+window_size),int(x1-window_size):int(x1+window_size)]
    #assigning x range 
    y_range=np.arange(int(y1)-coordinates,int(y1)+coordinates+1)
    for y in y_range:
        x=(-c-b*y)/a
        x=round(int(x))
        #finding error
        if x>=window_size and (x<=im2.shape[1]-window_size) and y>=window_size and (y<=im2.shape[0]-window_size):
            window_image2=im2[int(y-window_size):int(y+window_size),int(x-window_size):int(x+window_size)]
            error=window_image1-window_image2
            error=np.sqrt(np.sum(np.square(gaussian_filter(error,sigma=4))))

            if error<min_error:
                x2=x
                y2=y
                min_error=error

    return x2,y2
                

