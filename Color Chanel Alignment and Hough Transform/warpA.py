import numpy as np
from matplotlib import pyplot as plt 


def warp(im, A, output_shape):
    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""

    output_image=np.zeros(shape=(output_shape[0],output_shape[1]))
    p_warped=[]
    output_image=np.array(output_image)
    p_warped=np.array(p_warped)

    for i in range(0,output_shape[0]):
        for j in range(0,output_shape[1]):
            p_source=np.array([[i],[j],[1]])
            p_warped=np.linalg.inv(A).dot(p_source)
            idx=int(round(p_warped[0][0]))
            idy=int(round(p_warped[1][0]))
            if idx<=output_shape[0]-1 and idx >= 0:
                if idy<=output_shape[1]-1 and idy >= 0:            
                    output_image[i][j]=im[idx][idy]

    return output_image




















































            