"""
Coloring functions used to create a 3-channel matrix representing an RGB image
from a layer represented by a bitmask. The output matrix should support 8 bit
color depth (i.e. its data type is uint8).

@author: Sebastian Lotter <sebastian.g.lotter@fau.de>
"""
import numpy as np
import random

def color_w_constant_color(fmask, color):
    """
    Color whole layer with a constant color.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    color -- (r,g,b) tuple

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """
    

    # Initialize empty matrix
    img=np.zeros((fmask.shape[0], fmask.shape[1], 3), dtype=np.uint8)

    
    img[:,:,0]=np.array(fmask)
    img[:,:,1]=np.array(fmask)
    img[:,:,2]=np.array(fmask)
    
    
    
    # Set each channel to the value given by the input color tuple
    img[img[:,:,0]>0,0]=color[0];
    img[img[:,:,1]>0,1]=color[1];
    img[img[:,:,2]>0,2]=color[2];

    return img


def color_w_random_color(fmask, mean, range):
    """
    Colors layer with constant color, then draws random integer uniformly from
    [mean-range;mean+range] and adds it to the image.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured. Is 1 at positions which shall be coloured and 0 elsewhere.
    mean  -- mean color, (r,g,b) tuple
    range -- range within which to vary mean color

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """

    # Generate an image coloured with the 'mean' color
    Range=range
    img=np.zeros((fmask.shape[0], fmask.shape[1], 3), dtype=int)

    
    img[:,:,0]=np.array(fmask)
    img[:,:,1]=np.array(fmask)
    img[:,:,2]=np.array(fmask)
    img[img[:,:,0]>0,0]=mean[0];
    img[img[:,:,1]>0,1]=mean[1];
    img[img[:,:,2]>0,2]=mean[2];
    
    # Cast image to a data type supporting negative values and values greater 
    # than 255 to avoid overflows
    

    # Produce random integer noise uniformly drawn from [-range;range] covering
    # the whole image and add it to the image
    
    noise=np.random.randint(low=-Range,high=Range,size=img.shape)
    noise=noise*fmask[:,:, np.newaxis]

    img=img+noise
    img[img<0]=0
    img[img>255]=255

    # Cut off values exceeding the uint8 data type and cast the image back

    return np.array(img, dtype=np.uint8)


def color_w_constant_color_random_mean(fmask, mean, lb, ub):
    """
    Picks a random color from ([mean[0]-lb;mean[0]+ub],...) and colors
    layer with this color.

    Input:
    fmask -- binary mask indicating the shape that is to be coloured
    mean  -- mean color, (r,g,b) tuple
    lb    -- lower bound for the interval to draw the random mean color from
    ub    -- upper bound for the interval to draw the random mean color from

    Output:
    matrix of dimensions (x,y,3) representing 3-channel image
    """
    
    dr=random.randint(lb,ub)
    dg=random.randint(lb,ub)
    db=random.randint(lb,ub)
    color_r=mean[0]+dr
    color_r=min(255,max(0,color_r))

    color_g=mean[1]+dg
    color_g=min(255,max(0,color_g))
    color_b=mean[2]+db
    color_b=min(255,max(0,color_b))
    color=[color_r,color_g,color_b]
    
    img=color_w_constant_color(fmask, color)
    
    
    
    

    # Draw a random color from [r/g/b-lb;r/g/b+ub] and use it to colour
    # the image. Make sure the generated color is supported on [0;255]x[0;255]x[0;255]

    return img


# Public API
# Exporting a registry instead of the functions allows us to change the
# implementation whenever we want.
COLOR_FCT_REGISTRY = {
    'constant'            : color_w_constant_color,
    'random'              : color_w_random_color,
    'constant_random_mean': color_w_constant_color_random_mean
}

