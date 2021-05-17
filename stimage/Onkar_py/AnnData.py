import stlearn as st
st.settings.set_figure_params(dpi=300)
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
from typing import Optional, Union
from anndata import AnnData
import pandas as pd
import stlearn
from typing import Optional, Union
from anndata import AnnData
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import os


def tiling(
        adata: AnnData,
        out_path: Union[Path, str] = "./tiling",
        library_id: str = None,
        crop_size: int = 40,
        target_size: int = 299,
        verbose: bool = False,
        copy: bool = False,
) -> Optional[AnnData]:
    

    if library_id is None:
        library_id = list(adata.uns["spatial"].keys())[0]

    # Check the exist of out_path
    if not os.path.isdir(out_path):
        os.mkdir(out_path)

    image = adata.uns["spatial"][library_id]["images"][adata.uns["spatial"]["use_quality"]]
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)
    tile_names = []

    with tqdm(
            total=len(adata),
            desc="Tiling image",
            bar_format="{l_bar}{bar} [ time left: {remaining} ]",
    ) as pbar:
        for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
            imagerow_down = imagerow - crop_size / 2
            imagerow_up = imagerow + crop_size / 2
            imagecol_left = imagecol - crop_size / 2
            imagecol_right = imagecol + crop_size / 2
            tile = img_pillow.crop(
                (imagecol_left, imagerow_down, imagecol_right, imagerow_up)
            )
            # tile.thumbnail((target_size, target_size), Image.ANTIALIAS)
            tile = tile.resize((target_size, target_size))
            tile_name = library_id + "-" + str(imagecol) + "-" + str(imagerow) + "-" + str(crop_size)#np.arange(len(pd.Series(adata))+1).astype(str).str.zfill(4)+1 + "-" +
            out_tile = Path(out_path) / (tile_name + ".jpeg")
            tile_names.append(str(out_tile))
            if verbose:
                print(
                    "generate tile at location ({}, {})".format(
                        str(imagecol), str(imagerow)
                    )
                )
            tile.save(out_tile, "JPEG")

            pbar.update(1)

    adata.obs["tile_path"] = tile_names
    return adata if copy else None


#%%

BASE_PATH = Path("D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files")
TILE_PATH = BASE_PATH / "tiles"
TILE_PATH.mkdir(parents=True, exist_ok=True)

SAMPLE = "block1"
Sample1 = st.Read10X(BASE_PATH / SAMPLE, 
                  library_id=SAMPLE, 
                  count_file="V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5",
                  quality="fulres",)
img = plt.imread(BASE_PATH / SAMPLE /"V1_Breast_Cancer_Block_A_Section_1_image.tif", 0)
Sample1.uns["spatial"][SAMPLE]['images']["fulres"] = img


SAMPLE = "block2"
Sample2 = st.Read10X(BASE_PATH / SAMPLE, 
                  library_id=SAMPLE, 
                  count_file="V1_Breast_Cancer_Block_A_Section_2_filtered_feature_bc_matrix.h5",
                  quality="fulres",)
                  #source_image_path=BASE_PATH / SAMPLE /"V1_Breast_Cancer_Block_A_Section_1_image.tif")
img = plt.imread(BASE_PATH / SAMPLE /"V1_Breast_Cancer_Block_A_Section_2_image.tif", 0)
Sample2.uns["spatial"][SAMPLE]['images']["fulres"] = img


for adata in [Sample1,Sample2,]:

    st.pp.normalize_total(adata)
    TILE_PATH_ = TILE_PATH / list(adata.uns["spatial"].keys())[0]
    TILE_PATH_.mkdir(parents=True, exist_ok=True)
    tiling(adata, TILE_PATH_, crop_size=299)
    #st.pp.extract_feature(adata)
    
#%%

"""Sample1.obs.iloc[:,4:] #Link_to_img
Sample2.obs.iloc[:,4:] #Link_to_img

Sample1.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] #top-500-genes 
Sample2.to_df()[Sample2.to_df().sum().sort_values(ascending=False).index[:500]] #top-500-genes"""

#%%

import os; import pandas as pd; import numpy as np
import PIL; from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
import os; import glob
from PIL import Image; import matplotlib.pyplot as plt; 
from keras.utils import np_utils
from keras.models import Sequential
from keras.applications import VGG16, ResNet50, inception_v3, DenseNet121
from keras.applications import imagenet_utils
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, GlobalAveragePooling2D
import warnings

wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files"
def ResNet50_features(train, test, pre_model):
 
    x_scratch_train = []
    x_scratch_test = []
    # loop over the images
    for imagePath in train:
        # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(299,299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        x_scratch_train.append(image)
        
    x_train = np.vstack(x_scratch_train)
    features_train = pre_model.predict(x_train, batch_size=32)
    features_flatten_train = features_train.reshape((features_train.shape[0], 2048))
    features_flatten_train = pd.DataFrame(features_flatten_train)
    #Sample1.obsm["ResNet50_features"] = features_flatten_train
    
    #pd.concat([Sample1.obsm.to_df().reset_index(),features_flatten_train],axis=1) # ResNet50 Features is like dataframe how to add it to AnnData?
    
    for imagePath in test:
        # load the input image and image is resized to 224x224 pixels
        image = load_img(imagePath, target_size=(299,299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        x_scratch_test.append(image)

    x_test = np.vstack(x_scratch_test)
    features_test = pre_model.predict(x_test, batch_size=32)
    features_flatten_test = features_test.reshape((features_test.shape[0], 2048))
    features_flatten_test = pd.DataFrame(features_flatten_test)
    #Sample2.obsm["ResNet50_features"] = features_flatten_test
    
    #pd.concat([Sample2.obsm.to_df().reset_index(),features_flatten_test],axis=1) # ResNet50 Features is like dataframe how to add it to AnnData?
    
train = Sample1.obs["tile_path"]
test = Sample2.obs["tile_path"]
model = ResNet50(weights="imagenet", include_top=False, input_shape=(299,299, 3), pooling="avg")

ResNet50_features(train, test, model)


#%%

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray
from os import listdir
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def Clusters(Sample1_tiles, Sample2_tiles, model):
    
    Sample1_to_df = Sample1.to_df().reset_index(drop=True)
    Sample1_to_df.drop([col for col, val in Sample1_to_df.sum().iteritems() if val < 20000], axis=1, inplace=True)

    Sample2_to_df = Sample2.to_df().reset_index(drop=True)
    Sample2_to_df.drop([col for col, val in Sample2_to_df.sum().iteritems() if val < 20000], axis=1, inplace=True)
    
    photos_Sample1, photos_Sample2 =list(), list()
    for filename in Sample1_tiles:
            # load image
            photo = load_img(filename, target_size=(140,140))
            # convert to numpy array
            photo = img_to_array(photo, dtype='uint8')
            # store
            photos_Sample1.append(photo)
    
    Img_Sample1 = asarray(photos_Sample1, dtype='uint8')
    Img_Sample1 = pd.DataFrame(Img_Sample1.reshape(Img_Sample1.shape[0],58800))
    result_Sample1 = pd.concat([Img_Sample1, Sample1_to_df], axis=1)
    y_hc_Sample1 = pd.DataFrame(index = Sample1.obs.index)
    y_hc_Sample1["Cluster"] = model.fit_predict(result_Sample1)
    Sample1.obs["Cluster"] = y_hc_Sample1["Cluster"]

    for filename in Sample2_tiles:
            # load image
            photo = load_img(
                filename, target_size=(140,140))
            # convert to numpy array
            photo = img_to_array(photo, dtype='uint8')
            # store
            photos_Sample2.append(photo)

    Img_Sample2 = asarray(photos_Sample2, dtype='uint8')
    Img_Sample2 =  pd.DataFrame(Img_Sample2.reshape(Img_Sample2.shape[0],58800))
    result_Sample2 =  pd.concat([Img_Sample2, Sample2_to_df], axis=1)
    y_hc_Sample2 = pd.DataFrame(index = Sample2.obs.index)
    y_hc_Sample2["Cluster"] = model.fit_predict(result_Sample2)
    Sample2.obs["Cluster"] = y_hc_Sample2["Cluster"]
    
model = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
Sample1_tiles = Sample1.obs["tile_path"]
Sample2_tiles = Sample2.obs["tile_path"]

Clusters(Sample1_tiles, Sample2_tiles, model)

#%%

import os; import sys; import pandas as pd; import numpy as np; from numpy import array; from numpy import argmax

from keras.applications.imagenet_utils import decode_predictions
from keras.utils import to_categorical; from sklearn.preprocessing import LabelEncoder; from sklearn.preprocessing import OneHotEncoder; from sklearn.model_selection import train_test_split

import tensorflow as tf; from matplotlib import pyplot as plt
from keras import backend; from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential; from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD; from keras.models import Model; from tensorflow.keras import regularizers; from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input as pi; from keras.callbacks import ModelCheckpoint

import tensorflow.keras.backend as K; from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions; import cv2

import lime; from lime import lime_image
from keras.preprocessing import image; import numpy as np; import h5py; from keras.models import load_model 

#%%
wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/"

def train_interpretation_model(Gene_exp_train, Gene_exp_test, gene_name, dir_train, dir_test, transfer_model):
    
    Biomarker_train = Gene_exp_train[[gene_name]]
    Biomarker_train = pd.DataFrame(data=Biomarker_train)
    Biomarker_train = Biomarker_train.apply(lambda x: pd.qcut(x, 3,duplicates='drop', labels=False))
    Biomarker_train["Image"] = Sample1.obs["tile_path"]
    Biomarker_train = Biomarker_train.rename(columns={Biomarker_train.columns[0]: 'Gene_cluster'})
    Biomarker_train["Gene_cluster"] = Biomarker_train["Gene_cluster"].astype('str') 
    
    Biomarker_test = Gene_exp_test[[gene_name]]
    Biomarker_test = pd.DataFrame(data=Biomarker_test)
    Biomarker_test = Biomarker_test.apply(lambda x: pd.qcut(x, 3,duplicates='drop', labels=False))
    Biomarker_test["Image"] = Sample2.obs["tile_path"]
    Biomarker_test = Biomarker_test.rename(columns={Biomarker_test.columns[0]: 'Gene_cluster'})
    Biomarker_test["Gene_cluster"] = Biomarker_test["Gene_cluster"].astype('str') 

    datagen=ImageDataGenerator(rescale=1./255, validation_split = 0.2, featurewise_center=True,
            featurewise_std_normalization=False,rotation_range=90,
            width_shift_range=0.2,height_shift_range=0.2,
            horizontal_flip=True,vertical_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255,featurewise_center=True)
    
    train_generator=datagen.flow_from_dataframe(dataframe=Biomarker_train, directory=dir_train, 
                                                x_col="Image", y_col="Gene_cluster", class_mode="categorical", 
                                                target_size=(299,299), batch_size=32, subset="training")
    valid_generator = datagen.flow_from_dataframe(dataframe=Biomarker_train, directory=dir_train, 
                                                  x_col="Image", y_col="Gene_cluster", class_mode="categorical", 
                                                  target_size=(299,299), batch_size=32, subset="validation")
    test_generator = test_datagen.flow_from_dataframe(dataframe=Biomarker_test, directory=dir_test, 
                                                      x_col="Image", y_col="Gene_cluster", class_mode="categorical", 
                                                      target_size=(299,299), batch_size=32)
    model = transfer_model
    flat1 = Flatten()(model.layers[-1].output)
    dense = Dense(256, activation='relu')(flat1)
    drop = Dropout(0.5)(dense)
    output = Dense(3, activation='softmax')(drop)
    model = Model(inputs=model.inputs, outputs=output)
    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='loss', verbose=1,
                             save_best_only=True, mode='auto', period=1)


    model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=['AUC'])
    history = model.fit(train_generator,validation_data=valid_generator,epochs=1, 
                        callbacks=[checkpoint])
    
    #model = load_model("Inception_model_COX6C_18") 
    return model


def LIME_heatmaps(im, sav):
    
    def transform_img_fn(path_list):
        out = []
        for img_path in path_list:
            img = image.load_img(img_path, target_size=(299, 299))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = pi(x)
            out.append(x)
        return np.vstack(out)
    
    pred_val1 = []; pred_val2 = []; pred_val3 = [] 
    for i in range(0,len(im)):
        images = transform_img_fn([os.path.join(wd+"tiles/block1",im[i])])
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(images[0].astype('double'), train_interpretation_model(Gene_exp_train, Gene_exp_test, gene_name, dir_train, dir_test, transfer_model).predict, top_labels=3, num_samples=8)
        dict_heatmap1 = dict(explanation.local_exp[explanation.top_labels[0]])
        dict_heatmap2 = dict(explanation.local_exp[explanation.top_labels[1]])
        dict_heatmap3 = dict(explanation.local_exp[explanation.top_labels[2]])
    
        pred_val1.append(explanation.top_labels[0])
        pred_val2.append(explanation.top_labels[1])
        pred_val3.append(explanation.top_labels[2])
    
        heatmap1 = np.vectorize(dict_heatmap1.get)(explanation.segments)
        heatmap1 = np.maximum(heatmap1, 0)
        heatmap1 /= np.max(heatmap1)
    
        heatmap2 = np.vectorize(dict_heatmap2.get)(explanation.segments) 
        heatmap2 = np.maximum(heatmap2, 0)
        heatmap2 /= np.max(heatmap2)
        
        heatmap3 = np.vectorize(dict_heatmap3.get)(explanation.segments) 
        heatmap3 = np.maximum(heatmap3, 0)
        heatmap3 /= np.max(heatmap3)
    
        Lime1 = plt.imsave(str(explanation.top_labels[0])+"_"+sav[i], heatmap1, cmap = 'RdBu', vmin  = -heatmap1.max(), vmax = heatmap1.max())
        Lime2 = plt.imsave(str(explanation.top_labels[1])+"_"+sav[i], heatmap2, cmap = 'RdBu', vmin  = -heatmap2.max(), vmax = heatmap2.max())
        Lime3 = plt.imsave(str(explanation.top_labels[2])+"_"+sav[i], heatmap3, cmap = 'RdBu', vmin  = -heatmap3.max(), vmax = heatmap3.max())
        
        return Lime1, Lime2, Lime3
    
Gene_exp_train = Sample1.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] 
Gene_exp_test = Sample2.to_df()[Sample2.to_df().sum().sort_values(ascending=False).index[:500]]
gene_name = 'COX6C'
dir_train = wd+"tiles/block1"
dir_test = wd+"tiles/block2"
transfer_model = ResNet50(include_top=False, input_shape=(299,299,3), weights = "imagenet")
im = ['block1-4176-20901-299.jpeg']; sav = ['block1-4176-20901-299.png']
LIME_heatmaps(im,sav)
#%%
def activation_maps(image_loc, model, layer):
    
    img = image.load_img(image_loc, target_size=(299,299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    #print(decode_predictions(predicted_vals[0,:]))
    
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer(layer)
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(x)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((10,10))
    act_map = plt.matshow(heatmap)
    plt.show()
    return act_map

Gene_exp_train = Sample1.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] 
Gene_exp_test = Sample2.to_df()[Sample2.to_df().sum().sort_values(ascending=False).index[:500]]
gene_name = 'COX6C'
dir_train = wd+"tiles/block1"
dir_test = wd+"tiles/block2"
transfer_model = ResNet50(include_top=False, input_shape=(299,299,3), weights = "imagenet")

image_loc = wd+"tiles/block2/block1-4176-20901-299.jpeg"
model = train_interpretation_model(Gene_exp_train, Gene_exp_test, gene_name, dir_train, dir_test, transfer_model)
layer = 'conv5_block3_2_relu'
activation_maps(image_loc, model, layer)