import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...

    #print(im1.shape)
    out_size=(im1.shape[1],im1.shape[0])
    warp_im= cv2.warpPerspective(im2, H2to1, out_size)
    cv2.imwrite('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/6_1.jpg', warp_im)
    cv2.imshow("warp_im",warp_im)
    cv2.waitKey(0)
    pano_im=np.maximum(im1,warp_im)
    
    return pano_im

    


def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...
    
    pano_im=None
    #my code
    
    corners_im2=np.matrix([[0,0,1],[im2.shape[0],0,1],[0,im2.shape[1],1],[im2.shape[0],im2.shape[1],1]])
    corners_im2=np.transpose(corners_im2)
    for i in range(0,corners_im2.shape[1]-1):
        corner=corners_im2[:,i]
        corners_im2[:,i]=np.matmul(H2to1,corner)
    size_min=np.amin(corners_im2,axis=1)
    size_max=np.amax(corners_im2,axis=1)
    height2_min=size_min[1]
    height2_max=size_max[1]
    width2_min=size_min[0]
    width2_max=size_max[0]
    height1_min=0
    height1_max=im1.shape[1]
    width1_min=0
    width1_max=im1.shape[0]
    height_min=min(height1_min,height2_min)
    height_max=max(height1_max,height2_max)
    width_min=min(width1_min,width2_min)
    width_max=max(width1_max,width2_max)
    width=width_max-width_min
    height=height_max-height_min
    scale=3
    trans_x=abs(width_min)
    trans_y=scale*abs(height_min)
    width=height*2
    out_size=(width,height)
    #print("height_min",height_min)
    #print("height_max",height_max)
    scale=3 


    M=np.matrix([[scale,0,trans_x],[0,scale,trans_y+100],[0,0,3]],dtype='f')
    warp_im1=cv2.warpPerspective(im1,M,out_size)
    warp_im2=cv2.warpPerspective(im2,np.matmul(M,H2to1),out_size)
    pano_im=np.maximum(warp_im1,warp_im2)

    return pano_im
  


def generatePanorama(im1, im2):
    '''
    Returns a panorama of im1 and im2 without cliping.
    ''' 
    ######################################
    # TO DO ...

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)

    return pano_im
    


if __name__ == '__main__':
    im1 = cv2.imread('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/data/incline_L.png')
    im2 = cv2.imread('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/data/incline_R.png')
    
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    print("descriptors done")
    matches = briefMatch(desc1, desc2)
    print("matches done")

    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/q6_1.npy', H2to1)
    
    
    pano_im=imageStitching(im1, im2, H2to1)
    cv2.imwrite('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/q6_1_pan.jpg',pano_im)
    cv2.imshow("6_1_panaroma",pano_im)
    cv2.waitKey(0)

    
    print("image stitching done")

    #path='/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/q6_1.npy'
    #H2to1=np.load(path)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/q6_2_pan.jpg', pano_im)
    cv2.imshow("6_2_panaroma",pano_im)
    cv2.waitKey(0)
    print("image stiching with no clip done")

    im3 = generatePanorama(im1, im2)
    cv2.imwrite('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/q6_3.jpg', im3)
    print("final panaroma done")
    cv2.imshow('panoramas', im3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    

   
    
