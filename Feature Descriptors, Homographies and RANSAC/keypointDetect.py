import numpy as np
import cv2
import skimage
import skimage.util.shape
from skimage.util.shape import view_as_windows
import math

def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):
    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here

    for i in range(1,gaussian_pyramid.shape[2]):
        final_pyramid=gaussian_pyramid[:,:,i]-gaussian_pyramid[:,:,i-1]
        DoG_pyramid.append(final_pyramid)
    
    DoG_pyramid=np.stack(DoG_pyramid,axis=-1)
    
    DoG_levels = levels[1:]
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''
    principal_curvature = np.zeros((DoG_pyramid.shape[0],DoG_pyramid.shape[1],DoG_pyramid.shape[2]))
    ##################
    # TO DO ...
    # Compute principal curvature here

    for i in range(0,DoG_pyramid.shape[2]):
        img=DoG_pyramid[:,:,i]
        dx=cv2.Sobel(img,ddepth=-1,dx=1,dy=0)
        dy=cv2.Sobel(img,ddepth=-1,dx=0,dy=1)
        dxx=cv2.Sobel(dx,ddepth=-1,dx=1,dy=0)
        dyy=cv2.Sobel(dy,ddepth=-1,dx=0,dy=1)
        dxy=cv2.Sobel(dx,ddepth=-1,dx=0,dy=1)
        dyx=cv2.Sobel(dy,ddepth=-1,dx=1,dy=0)
        trace=np.add(dxx,dyy)
        trace=np.square(trace)
        det1=np.multiply(dxx,dyy)
        det2=np.multiply(dxy,dyx)
        det=np.subtract(det1,det2)
        R=np.divide(trace,det)
        principal_curvature[:,:,i]=R

    
    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
   
    
    #my code
    
    locsDoG = np.array([])
    
    #  TO DO ...
    # Compute locsDoG here
    ##############
    window_shape=(3,3)
    for i in range(1,len(DoG_levels)-1):  # for every level of the pyramid - from level 1 to level 4 t
        count1=0 #test
        count2=0 #test
        count3=0
        count4=0
        count5=0
        array=DoG_pyramid[:,:,i]
        windows=view_as_windows(array,window_shape)
        size=windows.shape
        for index,num in np.ndenumerate(array): #considering every element in the matrix 
            y=index[0]
            x=index[1]
            if (y<=size[0]-1) and (x<=size[1]-1):
                neighbours=windows[y][x]
            else:
                neighbours=windows[size[0]-1][size[1]-1]
                
            upper_level=DoG_pyramid[y,x,i+1]   #considering the correspoinding element from the upper level 
            lower_level=DoG_pyramid[y,x,i-1]   #considering the corresponsding element from the lower level 
            max_num=np.amax(neighbours)
            min_num=np.amin(neighbours)
            if max_num==num or min_num==num:  #if the number is extremum in space
                count1=count1+1
                if (num>lower_level and num>upper_level) or (num<lower_level and num<upper_level): #if the number is extremum in scale 
                    count2=count2+1
                    pcv=principal_curvature[y,x,i]
                    rel1=math.isnan(pcv)
                    rel2=math.isinf(pcv)
                    if (rel1!=True) and  (rel2!=True):   #if the principal curvature is not nan or infinty 
                        count3=count3+1
                        if (abs(num)>th_contrast):  #if the number clears the thresholds value 
                            count4=count4+1
                            if (pcv<th_r):
                                count5=count5+1
                                locsDoG=np.append(locsDoG,x)
                                locsDoG=np.append(locsDoG,y)
                                locsDoG=np.append(locsDoG,i)
    
        #print("count1=",count1)
        #print("count2=",count2)
        #print("count3=",count3)
        #print("count4=",count4)
        #print("count5=",count5)
        

    locsDoG=np.reshape(locsDoG,(-1,3))
    return locsDoG.astype(int)
    
    
    

    

def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4], 
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here
    gauss_pyramid=createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    #added extra part 
    '''
    green=(0,255,0)
    for x,y,level in locsDoG:
        cv2.circle(im,(int(x),int(y)),1,green,1)
    cv2.namedWindow("result",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result",600,600)
    cv2.imwrite("/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/result.jpg",im)
    cv2.imshow("result",im)
    cv2.waitKey(0)
    '''
    return locsDoG, gauss_pyramid


        

if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/data/model_chickenbroth.jpg')
    im_pyr = createGaussianPyramid(im)
    displayPyramid(im_pyr)
    
    # test DoG pyramid
    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    displayPyramid(DoG_pyr)
    
    # test compute principal curvature
    pc_curvature = computePrincipalCurvature(DoG_pyr)

    # test get local extrema
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)
    print("done1")

    # test DoG detector
    locsDoG, gaussian_pyramid = DoGdetector(im)
    print(locsDoG.shape)
    green=(0,255,0)
    for x,y,level in locsDoG:
        cv2.circle(im,(int(x),int(y)),1,green,1)
    cv2.namedWindow("result",cv2.WINDOW_NORMAL)
    cv2.resizeWindow("result",600,600)
    im = cv2.resize(im, (im.shape[1]*5, im.shape[0]*5))
    cv2.imwrite("/home/geekerlink/Desktop/Computer Vision/Homeworks/hw3/result/result.jpg",im)
    cv2.imshow("result",im)
    cv2.waitKey(0)


    
    
    
    


