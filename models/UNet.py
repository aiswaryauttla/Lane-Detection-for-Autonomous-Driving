#UNet Implementation
#Refer to the block diagram and build the UNet Model
#You would notice many blocks are repeating with only changes in paramaters like input and output channels
#Make use of the remaining classes to construct these repeating blocks. You may follow the order of ToDo specified
#above each class while writing the code.


#Additional Task: 
#How are wieghts inintialized in this model?
#Read about the various inintializers available in pytorch
#Define a function to inintialize weights in your model
#and create experiments using different initializers.
#Set the name of your experiment accordingly as 
#this initializer information will not be available
#in config file for later reference.
#You can also implement some parts of this task in other
#scripts like Training.py


import torch
import torch.nn as nn
import torch.nn.functional as F

#ToDo 5
class UNet(nn.Module):
    def __init__(self, n_in_channels=3, n_out_classes=2):
        super(UNet, self).__init__()
        #Create object for the components of the network. You may also make use of inconv,up,down and outconv classes defined below.
        self.m1=inconv(n_in_channels,64)
        self.m2=down(64,128)
        self.m3=down(128,256)
        self.m4=down(256,512)
        self.m5=down(512,512)
        
        self.u1=up(1024,256)
        self.u2=up(512,128)
        self.u3=up(256,64)
        self.u4=up(128,64)
        self.out=outconv(64,n_out_classes)
        
        

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        out_1=self.m1(input_tensor)
        out_2=self.m2(out_1)
        out_3=self.m3(out_2)
        out_4=self.m4(out_3)
        out_5=self.m5(out_4)
        output_tensor=self.u1(out_5,out_4)
        output_tensor=self.u2(output_tensor,out_3)
        output_tensor=self.u3(output_tensor,out_2)
        output_tensor=self.u4(output_tensor,out_1)
        output_tensor=self.out(output_tensor)

        return output_tensor
#ToDo 1: Implement double convolution components that you see repeating throughout the architecture.
class double_conv(nn.Module):
    #(conv => Batch Normalization => ReLU) * 2
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        #Create object for the components of the block
        self.model = nn.Sequential(
          nn.Conv2d(in_ch,out_ch,3,padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(),
          nn.Conv2d(out_ch,out_ch,3,padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU()
        )
        



    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor=self.model(input_tensor)
        
        
        

        return output_tensor

#ToDo 2: Implement input block
class inconv(nn.Module):
    #Input Block
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        #Create object for the components of the block. You may also make use of double_conv defined above.
        self.model_2=double_conv(in_ch,out_ch)


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor=self.model_2(input_tensor)

        
        return output_tensor

#ToDo 2: Implement generic down block
class down(nn.Module):
    #Down Block
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        #Create object for the components of the block.You may also make use of double_conv defined above.
        self.pool=nn.MaxPool2d(2)
        self.down_model=double_conv(in_ch,out_ch)  


    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        pool_tensor=self.pool(input_tensor)
        output_tensor=self.down_model(pool_tensor)


        return output_tensor

#ToDo 3: Implement generic up block
class up(nn.Module):
    #Up Block
    def __init__(self, in_ch, out_ch):
        super(up, self).__init__()
        # Create an object for the upsampling operation
        self.upsample=nn.Upsample(scale_factor=2) 
        
        #Create an object for the remaining components of the block.You may also make use of double_conv defined above.
        self.model_3=double_conv(in_ch,out_ch)
        
        



    def forward(self, input_tensor_1, input_tensor_2):
        #Upsample the input_tensor_1
        input_tensor_1=self.upsample(input_tensor_1)
        
        dif_height=input_tensor_1.shape[2]-input_tensor_2.shape[2]
        dif_width=input_tensor_1.shape[3]-input_tensor_2.shape[3]
        k=dif_height//2
        l=dif_width//2
        p2d = (k, dif_height-k, l, dif_width-l) 

        input_tensor_2=F.pad(input_tensor_2,p2d)

        
        #Make sure that upsampled tensor and input_tensor_2 have same size for all dimensions that are not concatenated in next step. You may use the method pad() from torch.nn.functional.
        
        #Concatenation of the  upsampled result and input_tensor_2
        input_tensor=torch.cat((input_tensor_1, input_tensor_2), 1)
       
        
        #Apply concatenated result to the object containing remaining components of the block and return result
        output_tensor=self.model_3(input_tensor)


        return output_tensor

#ToDo 4: Implement out block
class outconv(nn.Module):
    #Out Block
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        #Create object for the components of the block
        self.model_4=nn.Conv2d(in_ch,out_ch,1)

    def forward(self, input_tensor):
        #Apply input_tensor on object from init function and return the output_tensor 
        output_tensor=self.model_4(input_tensor)

        return output_tensor

# class Weights(nn.Module)

#       def __init__(self,in_ch,out_ch):
#         self.in_ch=in_ch
#         self.out_ch=out_ch
        
    
#       def Uniform(self)
#         w = torch.empty(self.in_ch,self.out_ch)
#         w=nn.init.uniform_(w)
#         return w
    
#        def Constant(self)
#             w = torch.empty(self.in_ch,self.out_ch)
#             w=nn.init.constant_(w)
#             return w
        
#         def Xavier(self)
#             w = torch.empty(self.in_ch,self.out_ch)
#             w=nn.init.xavier_normal_(w)
#             return w
      
        
        
        
        