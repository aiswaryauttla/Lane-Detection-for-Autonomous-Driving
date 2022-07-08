import matplotlib.pyplot as plt
from IPython.display import Markdown as md
from torchvision import transforms
import torch
from PIL import Image
import PIL
import torchvision
import os
import torch.nn.functional as F
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_probability_map(output_3d):
    '''Returns the probabilty map tensor(2d: H * W) for the given network-output tensor (3d:C* H * W)'''
    #Use softmax
    
    soft_max = F.softmax(output_3d)[1]
        
    
    return soft_max
    


def get_output_images(input_layer,output_3d):
    ''' Returns overlayed image and probabilty map in 3d: C * H * W
        Inputs:
            input_layer: 3D input image. Value range between 0 to 1.
            output_3d: 3D tensor of network outputs '''
    # Generate the probabilty map using above function get_probability_map(). Use network output tensor as input.
    probability_map=get_probability_map(output_3d)
    input_layer=input_layer.detach().cpu().numpy()
    
    probability_map=probability_map.detach().cpu().numpy()
    probability_map=np.round(probability_map)
    
    
    # create a 3 channel image with probabilty map in the red channel.
    prob_image=np.zeros((input_layer.shape))
    prob_image[0,:,:]=probability_map
    
    
    
    #Use round/threshold function to assign pixels as 0 (no detection) or 1 (detection)
    
    
    #Overlay the class image on top of the input image to have red colour
    #at pixels where class value =1
    # You may use other modules to create the images but 
    #the return value must be 3d tensors
    overlay_image=np.zeros((input_layer.shape))
    
    overlay_image[0,:,:]=input_layer[0,:,:]
    overlay_image[1,:,:]=input_layer[1,:,:]*(1-probability_map)
    overlay_image[2,:,:]=input_layer[2,:,:]*(1-probability_map)
    overlay_image=torch.from_numpy(overlay_image).float().to(device)
    prob_image=torch.from_numpy(prob_image).float().to(device)
    

    
    
    

    return overlay_image,prob_image



def play_notebook_video(folder,filename):
    '''Plays video in ipython using markdown'''
    file_path=os.path.join(folder, filename)  
    return md('<video controls src="{0}"/>'.format(file_path))




def display_output(image,prob_img,overlay_img):
    '''
    Displays the output using matplotlib subplots
    
    '''
    #Inputs are numpy array images.
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,figsize=(10,10))
    
    ax1.imshow(image)
    ax2.imshow(prob_img)
    ax3.imshow(overlay_img)
    

    
 



    
    
