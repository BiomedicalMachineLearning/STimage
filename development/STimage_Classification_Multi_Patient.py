import pandas as pd
from PIL import Image
from io import BytesIO
import numpy as np

import scanorama, warnings, cv2
import glob, os, sys, math, copy, joblib, shap, NaiveDE, SpatialDE, lime, cv2, zipfile, skimage, warnings, json, re, glob, shutil, anndata 

from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split, KFold
from sklearn import preprocessing
MinMax_scaler = MinMaxScaler()
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, jaccard_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image as image_fun
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, Lambda


 
warnings.filterwarnings("ignore")

import scanpy as sc

import scipy as sp
from scipy import ndimage as ndi
from scipy.stats import fisher_exact, nbinom
from statsmodels.stats.contingency_tables import mcnemar

from pathlib import Path
import pandas as pd
from matplotlib import cm as cm

from tqdm import tqdm
from typing import Optional, Union
from numpy import array, argmax, load

from libpysal.weights.contiguity import Queen
from libpysal import examples
from esda.moran import Moran
import geopandas as gpd
import splot
from splot.esda import moran_scatterplot, lisa_cluster
from esda.moran import Moran, Moran_Local
from esda.moran import Moran_BV, Moran_Local_BV
from splot.esda import plot_moran_bv_simulation, plot_moran_bv, plot_local_autocorrelation

##############################################################################################################################################

# Batch-effect Correction

##############################################################################################################################################

#Computing ResNet50 features
def ResNet50_features(pre_model, anndata):
    resnet_features = []
    for imagePath in anndata.obs["tile_path"]:
        image = plt.imread(imagePath).astype('float32')
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        resnet_features.append(pre_model.predict(image, batch_size=1))
        
    resnet_features = np.asarray(resnet_features)
    anndata.obsm["resnet50_features"] = resnet_features.reshape(resnet_features.shape[0],resnet_features.shape[2])
    
##############################################################################################################################################
    
#Training Pre-Processing
def classification_preprocessing(anndata):
    gene_exp = anndata.to_df() 
    gene_exp = pd.DataFrame(MinMax_scaler.fit_transform(gene_exp),columns=gene_exp.columns,index=gene_exp.index)
    gene_exp['library_id'] = anndata.obs['library_id']
    gene_exp_zscore = gene_exp.groupby('library_id')[list(gene_exp.iloc[:,:-1].columns)].apply(lambda x: (x-x.mean())/(x.std()))
    anndata.obsm["true_gene_expression"] = pd.DataFrame(gene_exp_zscore.apply(lambda x: [0 if y <= 0 else 1 for y in x]),
                                                       columns = anndata.to_df().columns, index = anndata.obs.index)
 ##############################################################################################################################################

#Logistic Regression Classifier
def LR_model(train_adata, iteration=200, penalty_option="elasticnet", regularization_strength=0.1, optimization="saga", l1_l2_ratio=0.5):
    model_c = LogisticRegression(max_iter=iteration, penalty=penalty_option, C=regularization_strength,solver=optimization,l1_ratio=l1_l2_ratio)
    clf_resnet = MultiOutputClassifier(model_c).fit(train_adata.obsm["resnet50_features"], train_adata.obsm["true_gene_expression"])
    joblib.dump(clf_resnet, Path+'pickle/STimage_LR.pkl')

##############################################################################################################################################        
def performance_metrics(test_adata, gene_list):
    AUROC = []; F1 = []; Moran_Index = []; Moran_p_sig = []
    for i in set(test_adata.obs["library_id"]):
        anndata_adata = test_adata[test_adata.obs["library_id"]==i]
        for j in gene_list:
            score_auroc = roc_auc_score(anndata_adata.obsm["predicted_gene_expression"][j],anndata_adata.obsm["true_gene_expression"][j])
            AUROC.append(score_auroc)

            score_f1 =  f1_score(anndata_adata.obsm["predicted_gene_expression"][j],anndata_adata.obsm["true_gene_expression"][j])
            F1.append(score_f1)
            
            anndata_adata.obsm["gpd"] = gpd.GeoDataFrame(anndata_adata.obs,
                                             geometry=gpd.points_from_xy(
                                                 anndata_adata.obs.imagecol, 
                                                 anndata_adata.obs.imagerow))
            w = Queen.from_dataframe(anndata_adata.obsm["gpd"])
            
            moran = Moran(anndata_adata.to_df()[j].values,w)
            moran_bv = Moran_BV(anndata_adata.obsm["true_gene_expression"][j].values, anndata_adata.obsm["predicted_gene_expression"][j].values, w)
            Moran_Index.append(moran_bv.I)
            Moran_p_sig.append(moran_bv.p_z_sim)


    Performance_metrics_1 = pd.concat([pd.DataFrame(AUROC), pd.DataFrame(F1)])
    Performance_metrics_1['Patient'] = list(np.repeat(list(set(test_adata.obs["library_id"])),len(gene_list)))*2
    Performance_metrics_1['Genes'] = gene_list*len(set(Performance_metrics_1['Patient']))*2
    Performance_metrics_1['Metrics'] = ['AUROC']*len(AUROC)+['F1']*len(F1)
    
    Performance_metrics_2 = pd.DataFrame(Moran_Index)
    Performance_metrics_2['Patient'] = list(np.repeat(list(set(test_adata.obs["library_id"])),len(gene_list)))
    Performance_metrics_2['Genes'] = gene_list*len(set(Performance_metrics_2['Patient']))
    Performance_metrics_2['Metrics'] = ['Spatial_Autocorrelation']*len(Moran_Index)

    sns.set(font_scale = 2, style="whitegrid")
    plt.figure(figsize=(22.50,12.50))
    plt.ylim(-0.1, 1.10)
    im = sns.boxplot(x="Patient", y=0, hue="Metrics", data=Performance_metrics_1,linewidth=3.5,width=0.6)
    im.set_xticklabels(im.get_xticklabels(),rotation = 30)
    plt.legend(loc="lower right", frameon=True, fontsize=20)
    im.axhline(0.5, linewidth=2, color='b')

    sns.set(font_scale = 2, style="whitegrid")
    plt.figure(figsize=(25,12.50))
    plt.ylim(-0.1, 1.10)
    im2 = sns.boxplot(x="Genes", y=0, hue="Metrics", data=Performance_metrics_1,linewidth=3.5,width=0.6)
    im2.set_xticklabels(im2.get_xticklabels(),rotation = 30)
    plt.legend(loc="lower right", frameon=True, fontsize=20)
    im2.axhline(0.5, linewidth=2, color='b')
    
    sns.set(font_scale = 2, style="whitegrid")
    plt.figure(figsize=(22.50,12.50))
    plt.ylim(-0.1, 1.10)
    im3 = sns.boxplot(x="Patient", y=0, hue="Metrics", data=Performance_metrics_2,linewidth=3.5,width=0.6)
    im3.set_xticklabels(im3.get_xticklabels(),rotation = 30)
    plt.legend(loc="lower right", frameon=True, fontsize=20)
    im3.axhline(0.0, linewidth=2, color='b')

    sns.set(font_scale = 2, style="whitegrid")
    plt.figure(figsize=(25,12.50))
    plt.ylim(-0.1, 1.10)
    im4 = sns.boxplot(x="Genes", y=0, hue="Metrics", data=Performance_metrics_2,linewidth=3.5,width=0.6)
    im4.set_xticklabels(im4.get_xticklabels(),rotation = 30)
    plt.legend(loc="lower right", frameon=True, fontsize=20)
    im4.axhline(0.0, linewidth=2, color='b')

    return Performance_metrics_1.to_csv(Path+'Performance_metrics_1.csv'), Performance_metrics_2.to_csv(Path+'Performance_metrics_2.csv'), im.figure.savefig(Path+'Classification_Performance.png'),im2.figure.savefig(Path+'Classification_Performance_per_gene.png'),im3.figure.savefig(Path+'Spatial_Autocorrelation.png'), im4.figure.savefig(Path+'Spatial_Autocorrelation_per_gene.png')

##############################################################################################################################################

# Clustering followed by Classification
def clustering(train_adata):
    clustering = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
    model_c = LogisticRegression(max_iter=10000,penalty='elasticnet',C=0.1,solver='saga',l1_ratio=0.5)
    clf_can_v_non_can = model_c.fit(train_adata.obsm["true_gene_expression"],clustering.fit_predict(train_adata.obsm["true_gene_expression"]))
    joblib.dump(clf_can_v_non_can, Path+'pickle/STimage_LR_cluster.pkl')
    
##############################################################################################################################################


Path = "/home/uqomulay/90days/STimage_outputs/"
adata = anndata.read_h5ad(Path+"all_adata.h5ad")
gene_list = ['CD74', 'CD24', 'CD63', 'CD81', 'CD151', 'C3',
             'COX6C', 'TP53', 'PABPC1', 'GNAS', 'B2M', 'SPARC', 'HSP90AB1', 'TFF3', 'ATP1A1', 'FASN']
adata = adata[:,gene_list]

train_data_library_id = ["block1", "block2", "1142243F","CID4290","CID4465","CID44971","CID4535"]
train_adata = adata[adata.obs["library_id"].isin(train_data_library_id)]
test_adata = adata[~adata.obs["library_id"].isin(train_data_library_id)]

train_adata.obs["tile_path"] = Path+"tiles/tiles/"+train_adata.obs["tile_path"].str.split('/', expand=True)[6]
test_adata.obs["tile_path"] = Path+"tiles/tiles/"+test_adata.obs["tile_path"].str.split('/', expand=True)[6]

model = ResNet50(weights='imagenet', pooling="avg", include_top = False)

iteration = 200
penalty_option = 'elasticnet'
regularization_strength = 0.1
optimization = 'saga'
l1_l2_ratio = 0.5

ResNet50_features(model, train_adata)
ResNet50_features(model, test_adata)
print("Done-ResNet50-features-computed")

classification_preprocessing(train_adata)
classification_preprocessing(test_adata)
print("Done-2-Preprocessing-of-adata")

LR_model(train_adata)
print("Done-3-training")

clf_resnet = joblib.load(Path+'pickle/STimage_LR.pkl')
test_adata.obsm["predicted_gene_expression"] = pd.DataFrame(clf_resnet.predict(test_adata.obsm["resnet50_features"]),columns=test_adata.to_df().columns,index=test_adata.obs.index)
performance_metrics(test_adata, gene_list)
print("Done-4-Performance-metrics")

clustering(train_adata)
#clf_can_v_non_can = joblib.load(Path+'pickle/resnet_block1_log_scaled_relu_clustering_logistic_NB.pkl')
#can_v_non_can_spot = pd.DataFrame(clf_can_v_non_can.predict(test_adata.obsm["predicted_gene_expression"]),index=test_adata.obs.index)
#test_adata.obsm["clusters_can_v_non_can"] = can_v_non_can_spot
print("Done-5")

train_adata.write_h5ad("/scratch/90days/uqomulay/STimage_outputs/pickle/train_anndata_norm.h5ad") 
test_adata.write_h5ad("/scratch/90days/uqomulay/STimage_outputs/pickle/test_anndata_norm.h5ad")


