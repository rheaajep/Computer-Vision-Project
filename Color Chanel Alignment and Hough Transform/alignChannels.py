import numpy as np
import cv2

def alignChannels(red, green, blue):
    """Given 3 images corresponding to different channels of a color image,
    compute the best aligned result with minimum abberations

    Args:
      red, green, blue - each is a HxW matrix corresponding to an HxW image

    Returns:
      rgb_output - HxWx3 color image output, aligned as desired"""

    blue_sum=[]
    green_sum=[]
    blue_sum_i=[]
    blue_sum_j=[]
    green_sum_i=[]
    green_sum_j=[]

    for i in range(-30,31):
      for j in range(-30,31):
        blue_new=np.roll(blue,i,axis=0)
        blue_new2=np.roll(blue_new,j,axis=1)
        sub=np.subtract(red,blue_new2)
        square=np.square(sub)
        final_sum=np.sum(square)
        blue_sum.append(final_sum)
        blue_sum_i.append(i)
        blue_sum_j.append(j)

 
    for i in range(-30,31):
      for j in range(-30,31):
        green_new=np.roll(green,i,axis=0)
        green_new2=np.roll(green_new,j,axis=1)
        sub=np.subtract(red,green_new2)
        square=np.square(sub)
        final_sum=np.sum(square)
        green_sum.append(final_sum)
        green_sum_i.append(i)
        green_sum_j.append(j)

    blue_min=np.argmin(blue_sum)
    blue_min_i=blue_sum_i[blue_min]
    blue_min_j=blue_sum_j[blue_min]

    green_min=np.argmin(green_sum)
    green_min_i=green_sum_i[green_min]
    green_min_j=green_sum_j[green_min]


    final_blue=np.roll(blue,blue_min_i,axis=0)
    final_blue2=np.roll(final_blue,blue_min_j,axis=1)

    final_green=np.roll(green,green_min_i,axis=0)
    final_green2=np.roll(final_green,green_min_j,axis=1)

    final_image = np.dstack((red,final_green2,final_blue2))

    return final_image
