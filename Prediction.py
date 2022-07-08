# System Modules
import os.path
from enum import Enum
from serde import *
from utils.augmentation import *
import matplotlib.pyplot as plt
import skvideo.io

# Deep Learning Modules
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import *

# User Defined Modules

from serde import read_config
from utils.visualization import *
import torch
from utils.visualization import get_output_images
import numpy as np
from skimage.io import imread

class Prediction:
    '''
    This class represents prediction process similar to the Training class.

    '''
    def __init__(self,cfg_path,torch_seed=None):
        self.params = read_config(cfg_path)
        self.cfg_path = cfg_path
        self.model_info=self.params['Network']
        self.model_info['seed']=torch_seed or self.model_info['seed']
        
        self.setup_cuda()
        self.writer = SummaryWriter(os.path.join(self.params['tf_logs_path']))

        
    def setup_cuda(self, cuda_device_id=0):
        '''Setup the CUDA device'''
        #Refer similar function from training
        torch.backends.cudnn.fastest = True
        torch.cuda.set_device(cuda_device_id)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.cuda.manual_seed_all(self.model_info['seed'])
        torch.manual_seed(self.model_info['seed'])



        
    def setup_model(self, model,model_file_name=None):
        '''
        Setup the model by defining the model, load the model from the pth file saved during training.
        
        '''
        
        # Use the default file from params['trained_model_name'] if 
        # user has not specified any pth file in model_file_name argument
        if model_file_name is None:
            model_file_name=self.params['trained_model_name']
            
        
        #Set model to self.device
        #self.net = model(n_in_channels=3, n_out_classes=2).to(self.device)
        net_path=self.params["network_output_path"]
        self.model=model().to(self.device)

        #Load model from model_file_name and default network_output_path
        path=os.path.join(net_path,model_file_name)
        
        self.model.load_state_dict(torch.load(path))
        
        
        
    def predict(self,predict_data_loader,visualize=True,save_video=False):
        # Read params to check if any params have been changed by user
        self.params = read_config(self.cfg_path)
        #Set model to evaluation mode
        self.model.eval()
        net_path=self.params["network_output_path"]
        model_file_name=self.params['trained_model_name']
        path=os.path.join(net_path,model_file_name)
        
        
        if save_video:
            self.create_video_writer()
            

        with torch.no_grad():
            for j, images in enumerate(predict_data_loader):
                #Batch operation: depending on batch size more than one image can be processed.
                #Set images to self.device
                images=images.to(self.device)

                #Provide the images as input to the model and save the result in outputs variable.
                outputs = self.model(images)

                
               
            #for each image in batch
                for i in range(outputs.size(0)):
                    image=images[i]/255
                    output=outputs[i]

                    
                    #Get overlay image and probability map using function from utils.visualization
                    [overlay_img,prob_img]=get_output_images(image,output)


                    #Convert image, overlay image and probability image to numpy so that it can be visualized using matplotlib functions later. Use convert_tensor_to_numpy function from below.
                    image=self.convert_tensor_to_numpy(image)
                    overlay_img=self.convert_tensor_to_numpy(overlay_img)
                    prob_img=self.convert_tensor_to_numpy(prob_img)

                    #print(image.max())
                    if save_video:
                        #Concatentate input and overlay image(along the width axis [axis=1]) to create video frame. Hint:Use concatenate function from numpy
                        #Write video frame
                        video_frame=np.concatenate((image,overlay_img),axis=1)*255
                        self.writer.writeFrame(video_frame)
                        


                    if(visualize):
                        display_output(image,prob_img,overlay_img)

            if save_video:
                self.writer.close()
                #Uncomment the below line and replace ??? with the appropriate filename
                filename="outputvideo.mp4"
                return play_notebook_video(self.params['output_data_path'],filename)
                #return play_notebook_video(path,model_file_name)
            
            
    def create_video_writer(self):
        '''Initialize the video writer'''
        filename="outputvideo.mp4"
        output_path=self.params['output_data_path']
        self.writer = skvideo.io.FFmpegWriter(os.path.join(output_path, filename).encode('ascii'),
                                              outputdict={'-vcodec':'libx264'},
                                             )
       

    def convert_tensor_to_numpy(self,tensor_img):
        '''
        Convert the tensor image to a numpy image

        '''
        #torch has numpy function but it requires the device to be cpu
        
        np_img=tensor_img.cpu().numpy()
        


        # np_img image is now in  C X H X W
        # transpose the array to H x W x C
        np_img=np.transpose(np_img,(1,2,0))



                

        return np_img
        
        
    

        


        
        
