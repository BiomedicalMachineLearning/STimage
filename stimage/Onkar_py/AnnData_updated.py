#%%
#To save AnnData before Normalization
#copy the AnnData fro Train and Test

#%%Save in preprocessing.py
#ResNet50 Features saved in obsm

import os; import pandas as pd; import numpy as np; import glob; import warnings
import PIL; from PIL import Image; PIL.Image.MAX_IMAGE_PIXELS = 933120000
import matplotlib.pyplot as plt; 
from keras.utils import np_utils; from keras.models import Sequential
from keras.applications import VGG16, ResNet50, inception_v3, DenseNet121, imagenet_utils
from keras.callbacks import ModelCheckpoint; from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, GlobalAveragePooling2D

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
    features_flatten_train.index = train_adata.obsm.to_df().index
    train_adata.obsm["Resnet50_Train_Features"] = features_flatten_train

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
    features_flatten_test.index = test_adata.obsm.to_df().index
    test_adata.obsm["Resnet50_Test_Features"] = features_flatten_test
#----------------------------------------------------------------------------------------------------------- 
train = train_adata.obs["tile_path"]
test = test_adata.obs["tile_path"]
model = ResNet50(weights="imagenet", include_top=False, input_shape=(299,299, 3), pooling="avg")
ResNet50_features_train(train, model)
ResNet50_features_test(test, model)


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#%% Save in Preprocessing.py
#Train and Test Set Cancer vs Non-Cancer spot Labelling done by Clustering; Clusters for Image Tiles save in Sample.obs

from keras.preprocessing.image import load_img, img_to_array
import numpy as np; from numpy import asarray
import pandas as pd; from os import listdir
from sklearn.cluster import AgglomerativeClustering

def Clusters(train_tiles, train_adata, test_tiles, test_adata, model):
    
    train_to_df = train_adata.to_df().reset_index(drop=True)
    train_to_df.drop([col for col, val in train_to_df.sum().iteritems() if val < 20000], axis=1, inplace=True)

    test_to_df = test_adata.to_df().reset_index(drop=True)
    test_to_df.drop([col for col, val in test_to_df.sum().iteritems() if val < 20000], axis=1, inplace=True)
    
    photos_train, photos_test =list(), list()
    for filename in train_tiles:
            photo = load_img(filename, target_size=(140,140))
            photo = img_to_array(photo, dtype='uint8')
            photos_train.append(photo)
    
    Img_train = asarray(photos_train, dtype='uint8')
    Img_train = pd.DataFrame(Img_train.reshape(Img_train.shape[0],58800))
    result_train = pd.concat([Img_train, train_to_df], axis=1)
    y_hc_train = pd.DataFrame(index = train_adata.obs.index)
    y_hc_train["Cluster"] = model.fit_predict(result_train)
    train_adata.obs["Cluster"] = y_hc_train["Cluster"]

    for filename in test_tiles:
            photo = load_img(filename, target_size=(140,140))
            photo = img_to_array(photo, dtype='uint8')
            photos_test.append(photo)

    Img_test = asarray(photos_test, dtype='uint8')
    Img_test =  pd.DataFrame(Img_test.reshape(Img_test.shape[0],58800))
    result_test =  pd.concat([Img_test, test_to_df], axis=1)
    y_hc_test = pd.DataFrame(index = test_adata.obs.index)
    y_hc_test["Cluster"] = model.fit_predict(result_test)
    test_adata.obs["Cluster"] = y_hc_test["Cluster"]
    
#-----------------------------------------------------------------------------------------------------------
model = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
train_tiles = train_adata.obs["tile_path"]
test_tiles = test_adata.obs["tile_path"]
train_adata = train_adata
test_adata = test_adata

Clusters(train_tiles, train_adata, test_tiles, test_tiles, model)

#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------

#%% Save in Training.py
#Cancer vs Non-Cancer Prediction from Biomarkers Expression

import pandas as pd; import numpy as np
import lightgbm as lgb; import joblib
from sklearn.preprocessing import LabelEncoder; from sklearn.model_selection import train_test_split
from sklearn import preprocessing; from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, recall_score, precision_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, roc_auc_score
from sklearn.preprocessing import LabelBinarizer; from sklearn.model_selection import train_test_split
import shap; import lime; import lime.lime_tabular


def Can_pred_Biomarker(Biomarkers_train, Cluster_train, Biomarkers_test, tree_model):
    X_train, X_test, y_train, y_test = train_test_split(Biomarkers_train, Cluster_train, test_size = 0.15, random_state = 0, stratify=Cluster_train)
    clf = MultiOutputClassifier(tree_model).fit(X_train, y_train)
    return clf.predict(Biomarkers_test), joblib.dump(clf, OUTPATH/'Cancer_vs_Non-Cancer_clf.pkl')

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

#-----------------------------------------------------------------------------------------------------------
biomarker_list = ['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']
Biomarkers_train = train_adata.to_df()[biomarker_list]
Biomarkers_test = test_adata.to_df()[biomarker_list]
Cluster_train = train_adata.obs["Cluster"]
Cluster_test = test_adata.obs["Cluster"]
tree_model = lgb.LGBMClassifier()

y_pred_test, clf = Can_pred_Biomarker(Biomarkers_train, Cluster_train, Biomarkers_test, tree_model)
cm = confusion_matrix(Cluster_test, y_pred_test)
print('Confusion matrix\n\n', cm)
roc_score = roc_auc_score(Cluster_test, y_pred_test, average="weighted")
print('AUROC-Score:', roc_score)

Shapley_plot(Biomarkers_train, Biomarkers_test, clf)
explainer = Lime_plot(Biomarkers_train)
exp = explainer.explain_instance(Biomarkers_test.iloc[2], clf.predict_proba, num_features=5)
exp.show_in_notebook(show_table=True) 


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#%%Save in training.py
#This has AUROC scores and LGBM model saved as pickle file

from sklearn.multioutput import MultiOutputClassifier; import lightgbm as lgb
import pandas as pd; import shap; import numpy as np; import joblib
from sklearn import preprocessing; from sklearn.preprocessing import LabelEncoder;
from sklearn import preprocessing; from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import roc_auc_score; from sklearn.model_selection import train_test_split

#wd = "D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files"
    
def multiclass_roc_auc_score(truth, pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(truth)
    truth = lb.transform(truth)
    pred = lb.transform(pred)
    return roc_auc_score(truth, pred, average=average)

def three_class_auroc(X, test_X, Y, test_Y, comm_genes, model):
    
    MinMax_scaler_y = preprocessing.MinMaxScaler(feature_range =(0, 1))
    
    Y = Y[comm_genes]
    Y = MinMax_scaler_y.fit_transform(Y) 
    Y = pd.DataFrame(data=Y)
    Y = Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))
    
    test_Y = test_Y[comm_genes]
    test_Y = MinMax_scaler_y.transform(test_Y)
    test_Y = pd.DataFrame(data=test_Y)
    test_Y = test_Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))
    
    clf = MultiOutputClassifier(model).fit(X, Y)
    joblib.dump(clf, OUT_PATH/'ResNet50-LGBM_comm_gene.pkl')
    
    y_pred_test = clf.predict_proba(test_X)
    y_pred_test = np.array(y_pred_test)
    
    result_ovr =[]; multi_auroc = []
    for col in test_Y:
        score_ovr =  roc_auc_score(test_Y[col], y_pred_test[col], multi_class='ovr', average='weighted')
        score_multi = multiclass_roc_auc_score(test_Y[col], y_pred_test[col], average='weighted')
        result_ovr.append(score_ovr)
        multi_auroc.append(score_multi)
        
    result_ovr = pd.DataFrame()
    result_ovr["Multi_class_defined_auroc"] = multi_auroc
    result_ovr.index = Y.columns
    return result_ovr

#-----------------------------------------------------------------------------------------------------------
comm_genes = ["PABPC1", "GNAS", "HSP90AB1", "TFF3",
                      "ATP1A1", "COX6C", "B2M", "FASN",
                      "ACTG1", "HLA-B"]
X = train_adata.obsm["Resnet50_Train_Features"]
test_X = test_adata.obsm["Resnet50_Train_Features"]
Y = train_adata.to_df()
test_Y = test_adata.to_df()
model = lgb.LGBMClassifier()

result = three_class_auroc(X, test_X, Y, test_Y, comm_genes, model)


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#%%Visualization.py
#LIME Plots for Image Tiles for Classification Model

import pandas as pd; import numpy as np; import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier; from sklearn import preprocessing
import os; from matplotlib import pyplot as plt; import joblib
from tensorflow.keras.preprocessing import image as image_fun
from keras.applications import VGG16, ResNet50
import scipy as sp; from scipy import ndimage as ndi; import lime; from lime import lime_image
from skimage.feature import peak_local_max; from skimage.segmentation import watershed; from skimage.measure import label
import skimage; from skimage.color import rgb2hed; from skimage.morphology import area_opening


resnet_model = ResNet50(weights="imagenet", include_top=False, input_shape=(299, 299, 3), pooling="avg")
#gene_list = ['COX6C','MALAT1','TTLL12','PGM5','KRT5','LINC00645','SLITRK6', 'CPB1']
comm_genes = ["PABPC1", "GNAS", "HSP90AB1", "TFF3",
                      "ATP1A1", "COX6C", "B2M", "FASN",
                      "ACTG1", "HLA-B"]

Model_LGBM = joblib.load(OUT_PATH/'ResNet50-LGBM_comm_gene.pkl')
    
def model_predict_gene(gene):
    i = comm_genes.index(gene)
    def combine_model_predict(tile):
        feature = resnet_model.predict(tile)
        feature = feature.reshape((10, 2048))
        prediction = Model_LGBM.predict_proba(feature)
        return prediction[i]
    return combine_model_predict

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

#-----------------------------------------------------------------------------------------------------------
gene = "COX6C"
images = transform_img_fn([os.path.join('D:/onkar/Projects/Project_Spt.Transcriptomics/Output_files/tiles/block2/block2-7831-11564-299.jpeg')])
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(images[0].astype('double'), model_predict_gene(gene), segmentation_fn= None, top_labels=3, num_samples=100)
dict_heatmap = dict(explanation.local_exp[explanation.top_labels[0]])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
plt.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
plt.colorbar()
#print(pred_label(images))


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#%%Save in utils.py
#UMAP Features for Cancer and Non-Cancer Image Tiles' ResNet50 Features

import pandas as pd
import datashader as ds; import datashader.transfer_functions as tf; import datashader.bundling as bd
import colorcet; import bokeh.plotting as bpl; import holoviews as hv
import matplotlib.colors; import matplotlib.cm; import matplotlib.pyplot as plt
import bokeh.transform as btr; import holoviews.operation.datashader as hd; import umap.plot

def Umap_points(resnet_features, label):
    
    mapper = umap.UMAP().fit(resnet_features)
    return umap.plot.points(mapper, labels=label, theme='fire', background='black')

#-----------------------------------------------------------------------------------------------------------
resnet_features = train_adata.obsm["Resnet50_Train_Features"].values
label = train_adata.obs["Cluster"].values
Umap_points(resnet_features, label)


#-----------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------
#%% Visualization.py Spatial Correlation

