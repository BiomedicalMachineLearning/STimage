#%%
#Import Data and define Tiling Function


import stlearn as st
st.settings.set_figure_params(dpi=200)
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
import seaborn as sns
import numpy as np
import os
from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
import os
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing import image
from sklearn.metrics import roc_auc_score
from skimage.color import rgb2hed
import pandas as pd
from keras.utils import to_categorical
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import sys
from numpy import load
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras import backend
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.optimizers import SGD, Adam
from keras.models import Model
from tensorflow.keras import regularizers
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import lime
from sklearn.preprocessing import MinMaxScaler
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import watershed
import glob
import os
from tensorflow.keras.preprocessing import image as image_fun
from sklearn.preprocessing import OneHotEncoder
import skimage
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label
import scipy as sp
from scipy import ndimage as ndi
from skimage.morphology import area_opening
import math


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
#Generating Tiles
#Normalize adata


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


for adata_unnormalised in [Sample1, Sample2,]:
    
    Sample1_unnormalised = Sample1
    Sample2_unnormalised = Sample2

for adata in [Sample1,Sample2,]:
    
    st.pp.normalize_total(adata)
    TILE_PATH_ = TILE_PATH / list(adata.uns["spatial"].keys())[0]
    TILE_PATH_.mkdir(parents=True, exist_ok=True)
    tiling(adata, TILE_PATH_, crop_size=299)
    #st.pp.extract_feature(adata)
    
"""Sample1.obs.iloc[:,4:] #Link_to_img
Sample2.obs.iloc[:,4:] #Link_to_img

Sample1.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] #top-500-genes 
Sample2.to_df()[Sample2.to_df().sum().sort_values(ascending=False).index[:500]] #top-500-genes"""


#%%
#ResNet50 Features saved in obsm


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
def ResNet50_features_train(train, pre_model):
 
    x_scratch_train = []
    for imagePath in train:
        image = load_img(imagePath, target_size=(299,299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        x_scratch_train.append(image)
        
    x_train = np.vstack(x_scratch_train)
    features_train = pre_model.predict(x_train, batch_size=32)
    features_flatten_train = features_train.reshape((features_train.shape[0], 2048))
    features_flatten_train = pd.DataFrame(features_flatten_train)
    features_flatten_train.index = Sample1.obsm.to_df().index
    Sample1.obsm["Resnet50_Train_Features"] = features_flatten_train

def ResNet50_features_test(test, pre_model):
    
    x_scratch_test = []
    for imagePath in test:
        image = load_img(imagePath, target_size=(299,299))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        x_scratch_test.append(image)

    x_test = np.vstack(x_scratch_test)
    features_test = pre_model.predict(x_test, batch_size=32)
    features_flatten_test = features_test.reshape((features_test.shape[0], 2048))
    features_flatten_test = pd.DataFrame(features_flatten_test)
    features_flatten_test.index = Sample2.obsm.to_df().index
    Sample2.obsm["Resnet50_Test_Features"] = features_flatten_test
    
train = Sample1.obs["tile_path"]
test = Sample2.obs["tile_path"]
model = ResNet50(weights="imagenet", include_top=False, input_shape=(299,299, 3), pooling="avg")
ResNet50_features_train(train, model)
ResNet50_features_test(test, model)


#%%
#Clusters for Image Tiles save in Sample.obs


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from numpy import asarray
from os import listdir
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering


def Clusters(Sample1_tiles, Sample1, Sample2_tiles, Sample2, model):
    
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
    #y_hc_Sample1["Cluster"].to_csv('C:/Users/Onkar/Cluster_train.csv')

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
    #y_hc_Sample2["Cluster"].to_csv('C:/Users/Onkar/Cluster_test.csv')
    
model = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
Sample1_tiles = Sample1.obs["tile_path"]
Sample1 = Sample1
Sample2_tiles = Sample2.obs["tile_path"]
Sample2 = Sample2

Clusters(Sample1_tiles, Sample1, Sample2_tiles, Sample2, model)


#%%
#Visualise Clusters


import cv2
import matplotlib.pyplot as plt
import numpy as np

wd = 'D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/'
def Visualise(image, Sample1):
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    Spot_vals0=Sample1.obs[Sample1.obs['Cluster'] == 0]
    Spot_vals0=Spot_vals0.values
    Spot_vals1=Sample1.obs[Sample1.obs['Cluster'] == 1]
    Spot_vals1=Spot_vals1.values

    x = Spot_vals0[:,4].astype('int64')
    y = Spot_vals0[:,5].astype('int64')
    box = (x,y)
    numpy_array = np.array(box)
    transpose = numpy_array.T
    box = transpose.tolist()
    
    x1 = Spot_vals1[:,4].astype('int64')
    y1 = Spot_vals1[:,5].astype('int64')
    box1 = (x1,y1)
    numpy_array1 = np.array(box1)
    transpose1 = numpy_array1.T
    box1 = transpose1.tolist()

    for i in range(0,len(box)):
        image=cv2.circle(image, tuple(box[i]), 50,(255,0,0), -1)
    for i in range(0,len(box1)):
        image=cv2.circle(image, tuple(box1[i]), 50,(0,255,0), -1)
    cv2.imwrite(wd+"Cancer_vs_Non-Cancer_New_trial_train.png",image)

image = cv2.imread("tiles/block1/V1_Breast_Cancer_Block_A_Section_1_image.tif") 
Sample1 = Sample1
Visualise(image, Sample1)


#%%
#UMAP Features for Cancer and Non-Cancer Image Tiles' ResNet50 Features


import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import datashader.bundling as bd
import matplotlib.pyplot as plt
import colorcet
import matplotlib.colors
import matplotlib.cm
import bokeh.plotting as bpl
import bokeh.transform as btr
import holoviews as hv
import holoviews.operation.datashader as hd
import umap.plot

def Umap_points(resnet_features, label):
    
    mapper = umap.UMAP().fit(resnet_features)
    return umap.plot.points(mapper, labels=label, theme='fire', background='black')

resnet_features =Sample1.obsm["Resnet50_Train_Features"].values
label = Sample1.obs["Cluster"].values
Umap_points(resnet_features, label)


#%%
#Cancer vs Non-Cancer Prediction from Biomarkers Expression


import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import shap
import numpy as np
import lime
import lime.lime_tabular
import joblib

wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/"

def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average=average)

def Can_pred_Biomarker(Biomarkers_train, Cluster_train, Biomarkers_test, tree_model):
    X_train, X_test, y_train, y_test = train_test_split(Biomarkers_train, Cluster_train, test_size = 0.15, random_state = 0, stratify=Cluster_train)
    clf = tree_model
    clf.fit(X_train, y_train)
    return clf, clf.predict(Biomarkers_test), joblib.dump(clf, wd+'Cancer_vs_Non-Cancer_clf.pkl')

def Shapley_plot(Biomarkers_train, Biomarkers_test, clf):
    shap.initjs()
    explainer = shap.TreeExplainer(clf, Biomarkers_train)
    shap_values = explainer.shap_values(Biomarkers_test)
    return shap.summary_plot(shap_values, Biomarkers_test)

def Lime_plot(Biomarkers_train):

    explainer = lime.lime_tabular.LimeTabularExplainer(np.array(Biomarkers_train),
                        feature_names=Biomarkers_train.columns, 
                        class_names=['0','1'],                            
                        verbose=True, mode='classification')
    return explainer


    
Biomarkers_train = Sample1.to_df()[['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']]
Cluster_train = Sample1.obs["Cluster"] #pd.read_csv('C:/Users/Onkar/Cluster_train.csv').iloc[:,1:] #
Biomarkers_test = Sample2.to_df()[['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']]
Cluster_test = Sample2.obs["Cluster"] #pd.read_csv('C:/Users/Onkar/Cluster_test.csv').iloc[:,1:] #
tree_model = lgb.LGBMClassifier()

clf, y_pred_test, saved_model = Can_pred_Biomarker(Biomarkers_train, Cluster_train, Biomarkers_test, tree_model)
cm = confusion_matrix(Cluster_test, y_pred_test)
print('Confusion matrix\n\n', cm)
multiclass_roc_score = multiclass_roc_auc_score(Cluster_test, y_pred_test, average="weighted")
print('AUROC-Score:', multiclass_roc_score)

Shapley_plot(Biomarkers_train, Biomarkers_test, clf)
explainer = Lime_plot(Biomarkers_train)
exp = explainer.explain_instance(Biomarkers_test.iloc[2], clf.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True) 


#%%
#Biomarker Identicals

import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import shap
import numpy as np
import lime
import lime.lime_tabular
import joblib

wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/"

def Biomarker_Identicals(train_X, test_X, train_Y, test_Y, tree_model):

    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size = 0.20, random_state = 0)
    clf = tree_model
    clf.fit(X_train, y_train)
    explainer = shap.TreeExplainer(clf, X_train)
    shap_values = explainer.shap_values(test_X)
    return shap.summary_plot(shap_values, test_X, max_display=10), joblib.dump(clf, wd+'Biomarker_identicals.pkl')

train_X = Sample1.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] 
test_X = Sample2.to_df()[Sample2.to_df().sum().sort_values(ascending=False).index[:500]] 
train_Y = Sample1.obs["Cluster"] 
test_Y = Sample2.obs["Cluster"] 
tree_model = lgb.LGBMClassifier()

Shap, Model = Biomarker_Identicals(train_X, test_X, train_Y, test_Y, tree_model)


#%%
#3 Class AUROC Score


from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder; from sklearn.model_selection import train_test_split
from sklearn import preprocessing; from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score; from sklearn.neighbors import KNeighborsClassifier
import pandas as pd; import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder; from sklearn.model_selection import train_test_split
from sklearn import preprocessing; from sklearn.linear_model import LogisticRegression
import shap; import numpy as np; shap.initjs(); import joblib

wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/"
    
def three_class_auroc(X, test_X, Y, test_Y, number_of_genes, model):
    
    def multiclass_roc_auc_score(truth, pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(truth)
        truth = lb.transform(truth)
        pred = lb.transform(pred)
        return roc_auc_score(truth, pred, average=average)

    Y=Y.iloc[:,:number_of_genes]
    MinMax_scaler_y = preprocessing.MinMaxScaler(feature_range =(0, 1))
    Y = MinMax_scaler_y.fit_transform(Y) 
    Y = pd.DataFrame(data=Y)
    Y=Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))
    
    #test_Y.drop(['Sno'],axis=1,inplace=True)
    test_Y=test_Y.iloc[:,:number_of_genes]
    test_Y=MinMax_scaler_y.transform(test_Y)
    test_Y=pd.DataFrame(data=test_Y)
    test_Y=test_Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15, random_state = 0)
    clf = MultiOutputClassifier(model).fit(X_train, y_train)
    y_pred_test=clf.predict(test_X)
    y_pred_test = pd.DataFrame(y_pred_test)
    
    
    result = []
    for col in test_Y:
        score = multiclass_roc_auc_score(y_pred_test[col],test_Y[col])
        result.append(score)
        
    res = pd.DataFrame(index=Y.columns)
    res["Gene"] = Y.columns
    res["AUC"] = result
    return res.to_csv('AUROC_3_Class_LGBM.csv'), joblib.dump(clf, 'ResNet50-LGBM_200.pkl')

X = Sample1.obsm["Resnet50_Train_Features"]#pd.read_csv('D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/features_flatten_train.csv')
#X = X.iloc[:,1:]
test_X = Sample2.obsm["Resnet50_Test_Features"]#pd.read_csv('D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/features_flatten_test.csv')#
#test_X = test_X.iloc[:,1:]
Y = Sample1.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] 
test_Y = Sample2.to_df()[Sample2.to_df().sum().sort_values(ascending=False).index[:500]] 
number_of_top_genes = 200
model = lgb.LGBMClassifier()

three_class_auroc(X, test_X, Y, test_Y, number_of_top_genes, model)


#%%
#LIME Plots for Image Tiles


def LGBM(train_X, train_Y): 
    train_Y = train_Y.iloc[:,:2]
    Standard_scaler_y = preprocessing.MinMaxScaler(feature_range =(0, 1))
    train_Y = pd.DataFrame(data = Standard_scaler_y.fit_transform(train_Y))
    train_Y=train_Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size = 0.15, random_state = 0)
    clf = MultiOutputClassifier(lgb.LGBMClassifier()).fit(X_train, y_train) #LGBM model is fit for X and Y
    return clf

train_X = Sample1.obsm["Resnet50_Train_Features"] #pd.read_csv('D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/features_flatten_train.csv') #Train_Features 
train_X = train_X.iloc[:,1:]
train_Y = Sample1.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] #pd.read_csv('Breast_1A_500_top.csv') #
#train_Y = train_Y.drop(['Sno'], axis=1)
Model_LGBM = LGBM(train_X, train_Y)

resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")
gene_list = train_Y.columns.tolist()
    
def model_predict_gene(gene):
    i = gene_list.index(gene)
    def combine_model_predict(tile):
        feature = resnet_model.predict(tile)
        feature = feature.reshape((10, 2048))
        prediction = Model_LGBM.predict_proba(feature)
        return prediction[i]
    return combine_model_predict

def pred_label(tile):
    feature = resnet_model.predict(tile)
    feature = feature.reshape((1, 2048))
    prediction = Model_LGBM.predict_proba(feature)
    return prediction

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image_fun.load_img(img_path, target_size=(299, 299))
        x = image_fun.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        out.append(x)
    return np.vstack(out)

def watershed_segment(image):
    annotation_hed = rgb2hed(image)
    annotation_h = annotation_hed[:,:,0]
    annotation_h *= 255.0 / np.percentile(annotation_h, q=80)
#     annotation_h = np.clip(annotation_h, a_min=0, a_max=255)
    thresh = skimage.filters.threshold_otsu(annotation_h)
    im_fgnd_mask = sp.ndimage.morphology.binary_fill_holes(
        annotation_h < thresh
    )
    distance = ndi.distance_transform_edt(im_fgnd_mask)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=im_fgnd_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(annotation_h, markers, mask=im_fgnd_mask)
    im_nuclei_seg_mask = area_opening(labels, area_threshold=80).astype(np.int)
    return im_nuclei_seg_mask


def pred_label(tile):
    feature = resnet_model.predict(tile)
    feature = feature.reshape((1, 2048))
    prediction = Model_LGBM.predict_proba(feature)
    return prediction

gene = "COX6C"
#i = gene_list.get_loc(input('Enter Gene Name :'))
images = transform_img_fn([os.path.join('D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/tiles/block2/block2-7831-11564-299.jpeg')])
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[0].astype('double'), model_predict_gene(gene), segmentation_fn= None, top_labels=3, num_samples=100)
dict_heatmap = dict(explanation.local_exp[explanation.top_labels[0]])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -1, vmax = 1)
plt.colorbar()
print(pred_label(images))
#%%%
import joblib
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import pysal
from pysal.explore import esda
import pysal.lib as lps
from esda.moran import Moran, Moran_Local, Moran_BV, Moran_Local_BV
import splot
from splot.esda import moran_scatterplot, plot_moran, lisa_cluster, plot_moran_bv_simulation, plot_moran_bv, plot_local_autocorrelation
from libpysal.weights.contiguity import Queen
from libpysal import examples
import numpy as np
import os


def Spatial_AutoCorr(Sample1, Sample2, Model, test_X, gene, wd):
    Sample2.obsm["gpd"] = gpd.GeoDataFrame(Sample2.obs,
                                                 geometry=gpd.points_from_xy(
                                                     Sample2.obs.imagecol, 
                                                     Sample2.obs.imagerow))

    test_Y = Sample2.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] 
    Y = Sample1.to_df()[Sample1.to_df().sum().sort_values(ascending=False).index[:500]] 
    gene_list = Y.columns.tolist()

    Y=Y.iloc[:,:200]
    MinMax_scaler_y = preprocessing.MinMaxScaler(feature_range =(0, 1))
    Y = MinMax_scaler_y.fit_transform(Y) 
    Y = pd.DataFrame(data=Y)
    Y=Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))

    test_Y=test_Y.iloc[:,:200]
    test_Y=MinMax_scaler_y.transform(test_Y)
    test_Y=pd.DataFrame(data=test_Y)
    test_Y=test_Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))

    w = Queen.from_dataframe(Sample2.obsm["gpd"])

    y = Model.predict(test_X)
    i = gene_list.index(gene)+1
    y = pd.DataFrame(y[:,:i])

    x = test_Y[[0]].values
    Sample2.obsm["gpd"]["gc_{}".format(gene)] = x
    Sample2.obsm["gpd"]["pred_{}".format(gene)] = y.values
    tissue_image = Sample2.uns["spatial"]["block2"]["images"]["fulres"]
    
    
    moran = Moran(y,w)
    moran_bv = Moran_BV(y, x, w)
    moran_loc = Moran_Local(y, w)
    moran_loc_bv = Moran_Local_BV(y, x, w)

    fig, ax = plt.subplots(figsize=(5,5))
    moran_plot = moran_scatterplot(moran_bv, ax=ax)
    ax.set_xlabel('prediction of gene {}'.format(gene))
    ax.set_ylabel('Spatial lag of ground truth of gene {}'.format(gene))
    plt.tight_layout()
    plt.show()


    def plot_choropleth(gdf, 
                        attribute_1,
                        attribute_2,
                        bg_img,
                        alpha=0.5,
                        scheme='Quantiles', 
                        cmap='YlGnBu', 
                        legend=True):

        fig, axs = plt.subplots(2,1, figsize=(5, 8),
                                subplot_kw={'adjustable':'datalim'})

        # Choropleth for attribute_1
        gdf.plot(column=attribute_1, scheme=scheme, cmap=cmap,
                 legend=legend, legend_kwds={'loc': 'upper left',
                                             'bbox_to_anchor': (0.92, 0.8)},
                 ax=axs[0], alpha=alpha, markersize=2)

        axs[0].imshow(bg_img)
        axs[0].set_title('choropleth plot for {}'.format(attribute_1), y=0.8)
        axs[0].set_axis_off()

        # Choropleth for attribute_2
        gdf.plot(column=attribute_2, scheme=scheme, cmap=cmap,
                 legend=legend, legend_kwds={'loc': 'upper left',
                                             'bbox_to_anchor': (0.92, 0.8)},
                 ax=axs[1], alpha=alpha, markersize=2)

        axs[1].imshow(bg_img)
        axs[1].set_title('choropleth plot for {}'.format(attribute_2), y=0.8)
        axs[1].set_axis_off()

        plt.tight_layout()

        return fig, ax

    choropleth_plot = plot_choropleth(Sample2.obsm["gpd"], "gc_{}".format(gene),"pred_{}".format(gene),tissue_image)
    plt.show()

    lisa_cluster(moran_loc_bv, Sample2.obsm["gpd"], p=0.05, 
                 figsize = (9,9), markersize=12, **{"alpha":0.8})
    lisa_plot = plt.imshow(Sample2.uns["spatial"]["block2"]["images"]["fulres"])
    plt.show()
    return moran_plot, choropleth_plot, lisa_plot



gene = "COX6C"
Sample1 = Sample1
Sample2 = Sample2
wd = 'D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/'
test_X = Sample2.obsm["Resnet50_Test_Features"]#pd.read_csv(wd+"Resnet_unnorm_test.csv").iloc[:,1:] 
Model = joblib.load(wd+"ResNet50-LGBM_200_unnorm_updated.pkl")

a,b,c = Spatial_AutoCorr(Sample1, Sample2, Model, test_X, gene, wd)