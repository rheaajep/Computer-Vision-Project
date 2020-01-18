'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''
import numpy as np
import submission
from submission import essentialMatrix
from submission import triangulate
import helper
from helper import camera2

def findM2(E,k1,k2,pts1,pts2):
    M2s=camera2(E)
    M1=np.matrix([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    C1=np.matmul(k1,M1)
    C2=np.matmul(k2,M2s[:,:,0])
    [w,err]=triangulate(C1,pts1,C2,pts2)
    min_err=err
    final_C2=C2
    final_w=w
    final_M2=M2s[:,:,0]
    for i in range(1,4):
        C2=np.matmul(k2,M2s[:,:,i])
        [w,err]=triangulate(C1,pts1,C2,pts2)
        if err<min_err:
            final_C2=C2
            final_w=w
            final_M2=M2s[:,:,i]

    return M1,final_M2,C1,final_C2,final_w

    #np.savez('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw4/hw4/results/q3_3.npz',M2=final_M2,C2=final_C2,w=final_w)
    
    




    

    