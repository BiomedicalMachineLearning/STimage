import os; import pandas as pd; import numpy as np
import PIL; from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import os; import glob
from PIL import Image; import matplotlib.pyplot as plt; 
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16, ResNet50
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import warnings
warnings.filterwarnings('ignore')
#%%


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
    
image=Image.open('D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/block1/V1_Breast_Cancer_Block_A_Section_1_image.tif')
Vals=pd.read_csv('link_to_img_trial.csv')
Image_tiles_classification(image, Vals)

#%%


def ResNet50_features_test(dataset, pre_model):
 
    x_scratch = []
    # loop over the images
    for imagePath in dataset:
        # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(290,290))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
       # image = imagenet_utils.preprocess_input(image)
        # add the image to the batch
        x_scratch.append(image)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 2048))
    #return x, features, features_flatten
    features = pd.DataFrame(features)
    features.to_csv('ResNet50_Trainig_Breast_2A.csv')
    
test = sorted(glob.glob("UntitledFolder/Trainimg_breast_test_299/*.tif"))
model = ResNet50(weights="imagenet", include_top=False, input_shape=(290,290, 3), pooling="avg")
ResNet50_features_test(test, model)
#%%

def ResNet50_features_train(dataset, pre_model):
 
    x_scratch = []
    # loop over the images
    for imagePath in dataset:
        # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(290,290))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
       # image = imagenet_utils.preprocess_input(image)
        # add the image to the batch
        x_scratch.append(image)
 
    x = np.vstack(x_scratch)
    features = pre_model.predict(x, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 2048))
    #return x, features, features_flatten
    features = pd.DataFrame(features)
    features.to_csv('ResNet50_Trainig_Breast_1A.csv')
    
train = sorted(glob.glob("UntitledFolder/Trainimg_breast_2_299/*.tif"))
model = ResNet50(weights="imagenet", include_top=False, input_shape=(290,290, 3), pooling="avg")
ResNet50_features_train(train, model)

