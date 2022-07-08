import math
import numbers
import random
import numpy as np

from PIL import Image,ImageOps

from specs import *
from serde import read_config

class Augmentation(object):
    def __init__(self,cfg_path):
        self.params = read_config(cfg_path)
        
    
    def __call__(self):
        raise Exception("Unable to initalize abstract class Augmentation")


class Compose(Augmentation):
    '''
        - Combines several augmentation functions that are defined in augmentation_list.
        - cfg_path contains the path to the config.json file where the parameters for the below augmentation_functions are defined.
    '''
    def __init__(self, augmentation_list, cfg_path):
        Augmentation.__init__(self, cfg_path)
        self.augmentation_list = augmentation_list
        
    def __call__(self, image, label):
        #Convert numpy to PIL image
        image = Image.fromarray(image, mode='RGB')
        label= Image.fromarray(label, mode='L') # mode L => 8 bit black and white images


        for transform in self.augmentation_list:
            image, label = transform(image, label)

        #Convert PIL image to numpy
        image=np.array(image)
        label=np.array(label, dtype=np.uint8)

        return image,label


class LeftRightFlip(Augmentation):
    '''Mirror both image and label'''
    def __call__(self, image, label):
        #Apply augmentation only if random number generated is less than probabilty specfied in params
        if random.random() < self.params['left_right_flip_prob']: # We do not want the augmentation function to be applied to every single image in the SimulationDataset resp. TrueNegativeDataset in the Train.ipynb notebook. Instead the augmentation function shall only be applied to a certain fraction of randomly chosen images. This fraction is defined by params['left_right_flip_prob']. This parameter can be changed in the config.json file.
        
            image=ImageOps.mirror(image)
            label=ImageOps.mirror(label)
           
        return image, label
    
class Rotate(Augmentation):
    '''Rotate both image and label
            - rotation angle is randomly drawn from a uniform distribution with interval [-params['max_rot'], +params['max_rot'])'''
    def __call__(self, image, label):
        #Apply augmentation only if random number generated is less than probabilty specfied in params
        if random.random() < self.params['rotate_prob']: # We do not want the augmentation function to be applied to every single image in the SimulationDataset resp. TrueNegativeDataset in the Train.ipynb notebook. Instead the augmentation function shall only be applied to a certain fraction of randomly chosen images. This fraction is defined by params['rotate_prob']. This parameter can be changed in the config.json file.
            angle=random.uniform(-self.params['max_rot'],self.params['max_rot'])
            image=image.rotate(angle)
            label=label.rotate(angle)
           
        return image, label

    
class GaussianNoise(Augmentation):
    '''Add Gaussian noise to image only
        - Gaussian noise is added pixel- and channelwise to image
        - value added added to each channel of each pixel is drawn from a normal distribution of mean params['mean'] and width params['std']'''
    def __call__(self, image, label):
        #Apply augmentation only if random number generated is less than probabilty specfied in params
        if random.random() < self.params['gaussian_prob']: # We do not want the augmentation function to be applied to every single image in the SimulationDataset resp. TrueNegativeDataset in the Train.ipynb notebook. Instead the augmentation function shall only be applied to a certain fraction of randomly chosen images. This fraction is defined by params['gaussian_prob']. This parameter can be changed in the config.json file.
            image=np.array(image)
            gauss = np.random.normal(self.params['mean'],self.params['std'],image.shape)
            gauss = gauss.reshape(image.shape)
            #print(gauss)
            #gauss=gauss.float()
            image = image + gauss
            image = image.astype(np.uint8)
            image = Image.fromarray(image, mode='RGB')
        return image, label


class ColRec(Augmentation):
    '''Add colored rectangles of height params['y_size'] and width params['x_size'] to image only
        - number  of rectangles is specified by params['num_rec']
        - position is drawn randomly from a uniform distribution
        - value of each color channel is drawn randomly from a uniform distribution'''
    def __call__(self, image, label):
        #Apply augmentation only if random number generated is less than probabilty specfied in params
        if random.random() < self.params['colrec_prob']: # We do not want the augmentation function to be applied to every single image in the SimulationDataset resp. TrueNegativeDataset in the Train.ipynb notebook. Instead the augmentation function shall only be applied to a certain fraction of randomly chosen images. This fraction is defined by params['colrec_prob']. This parameter can be changed in the config.json file.
            np_img = np.array(image)
            pos = np.array([0,0],dtype = np.int_)
            for i in range(self.params['num_rec']):
                pos[0] = int(random.uniform(0, 1) * (image.height - self.params['y_size']))
                pos[1] = int(random.uniform(0, 1) * (image.width - self.params['x_size']))
                rand_col = np.ones((self.params['y_size'],self.params['x_size'],3))
                rand_col[:,:,0] = rand_col[:,:,0] * random.uniform(0,255)
                rand_col[:,:,1] = rand_col[:,:,0] * random.uniform(0,255)
                rand_col[:,:,2] = rand_col[:,:,0] * random.uniform(0,255)
                np_img[pos[0]:(pos[0]+self.params['y_size']),pos[1]:(pos[1]+self.params['x_size']),:] = rand_col
            np_img = np.clip(np_img,0,255)
            np_img = np.uint8(np_img)
            image = Image.fromarray(np_img, mode = "RGB")
        return image, label
    
    
class ZoomIn(Augmentation):
    ''' - from the original image and label crop a squared box
        - height and width of the box is uniformly drawn from [256 * params['box_size_min'], 256 * params['box_size_max'])
        - position of the box is drawn randomly from a uniform distribution
        - cropped is resized to PIL image of size 256x256'''
    def __call__(self, image, label):
        #Apply augmentation only if random number generated is less than probabilty specfied in params
        if random.random() < self.params['zoomin_prob']: # We do not want the augmentation function to be applied to every single image in the SimulationDataset resp. TrueNegativeDataset in the Train.ipynb notebook. Instead the augmentation function shall only be applied to a certain fraction of randomly chosen images. This fraction is defined by params['zoomin_prob']. This parameter can be changed in the config.json file.
            pos = np.array([0,0],dtype = np.int_)
            box_size = random.uniform(self.params['box_size_min'],self.params['box_size_max'])
            pos[0] = np.int_(random.uniform(0, 1) * int(256 - 256 * box_size))
            pos[1] = np.int_(random.uniform(0, 1) * int(256 - 256 * box_size))
            box_ = (0+pos[0],0+pos[1],int(256 * box_size)+pos[0],int(256 * box_size)+pos[1])
            image = image.crop(box_)
            image = image.resize(size=(256,256))
            label = label.crop(box_)
            label = label.resize(size=(256,256))
        return image, label
    
    
class ZoomOut(Augmentation):
    ''' - reduce size of image and label by the same factor that is randomly drawn from a uniform distribution with interval [params['zoomfac_min'],params['zoomfac_max'])
        - put the zoomed out image and label at the same position randomly drawn from a uniform distribution in a black image of size 256x256'''
    def __call__(self, image, label):
        #Apply augmentation only if random number generated is less than probabilty specfied in params
        if random.random() < self.params['zoomout_prob']: # We do not want the augmentation function to be applied to every single image in the SimulationDataset resp. TrueNegativeDataset in the Train.ipynb notebook. Instead the augmentation function shall only be applied to a certain fraction of randomly chosen images. This fraction is defined by params['zoomout_prob']. This parameter can be changed in the config.json file.
            pos = np.array([0,0],dtype = np.int_)
            zoomfac = random.uniform(self.params['zoomfac_min'],self.params['zoomfac_max'])
            pos[0] = np.int_(random.uniform(0, 1) * (image.height*zoomfac - image.height))
            pos[1] = np.int_(random.uniform(0, 1) * (image.width*zoomfac - image.width))
            
            np_img = np.array(image)
            black_img = np.zeros((int(image.height*zoomfac), int(image.width*zoomfac),3))
            black_img[pos[0]:(pos[0]+image.height),pos[1]:(pos[1]+image.width),:] = np_img
            black_img = np.clip(black_img,0,255)
            black_img = np.uint8(black_img)
            image = Image.fromarray(black_img, mode = "RGB")
            image = image.resize(size=(256,256))
            
            np_label = np.array(label)
            black_img = np.zeros((int(label.height*zoomfac), int(label.width*zoomfac)))
            black_img[pos[0]:(pos[0]+label.height),pos[1]:(pos[1]+label.width)] = np_label
            black_img = np.clip(black_img,0,255)
            black_img = np.uint8(black_img)
            label = Image.fromarray(black_img, mode = "L")
            label = label.resize(size=(256,256))
        return image, label

class Invert(Augmentation):
    ''' - invert colors of image'''
    def __call__(self, image, label):
        #Apply augmentation only if random number generated is less than probabilty specfied in params
        if random.random() < self.params['invert_prob']: # We do not want the augmentation function to be applied to every single image in the SimulationDataset resp. TrueNegativeDataset in the Train.ipynb notebook. Instead the augmentation function shall only be applied to a certain fraction of randomly chosen images. This fraction is defined by params['invert_prob']. This parameter can be changed in the config.json file.
            image = ImageOps.invert(image)
        return image, label
