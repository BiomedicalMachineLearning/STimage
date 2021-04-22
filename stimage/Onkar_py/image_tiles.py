import os; import pandas as pd; import numpy as np
import PIL; from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def Image_tiles_classification(image, Vals):
    
    Vals = Vals.iloc[:,:]
    Vals['left']=Vals['x']-145
    Vals['up']=Vals['y']-145
    Vals['right']=Vals['x']+145
    Vals['down']=Vals['y']+145
    Vals=Vals.values
    x_offset = Vals[:,4] 
    Y_offset = Vals[:,5]
    width = Vals[:,6] 
    height = Vals[:,7]
    box = (Y_offset, x_offset, height, width)
        
    numpy_array = np.array(box)
    transpose = numpy_array.T
    box = transpose.tolist()
    box
    
    imgno=1
    for i in range(0,len(box)):
        crop = image.crop(box[i])
        crop.save(str(imgno).zfill(4)+'img.tif')
        imgno=imgno+1
    
#image=Image.open('D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/block1/V1_Breast_Cancer_Block_A_Section_1_image.tif')
#Vals=pd.read_csv('link_to_img_trial.csv')
#Image_tiles_classification(image, Vals)

#%%