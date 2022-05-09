import os
import pandas as pd
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
import zipfile
import anndata
import joblib
from matplotlib import pyplot as plt
import scipy as sp
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


#Computing ResNet50 features
def ResNet50_features(unzipped_file, pre_model):
    files = pd.DataFrame(unzipped_file.namelist()).iloc[1:,:][0]
    x_scratch_train = []
    for imagePath in files:
        image = plt.imread(unzipped_file.open(imagePath)).astype('float32')
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        x_scratch_train.append(image)
    x_train = np.vstack(x_scratch_train)
    resnet_features = pd.DataFrame(pre_model.predict(x_train, batch_size=1))
    resnet_features.index = files.str.split('/', expand=True)[1].str[:-5]
    resnet_features = resnet_features.sort_index()
    return resnet_features


#Training Pre-Processing
def LR_model(Path, anndata, gene_list, train_data_library_id, resnet_features):
    h5ad = anndata.obs
    gene_exp = anndata.to_df()

    h5ad['tile_tissue_mask_path'] = h5ad['tile_tissue_mask_path'].str.split('/', expand=True)[3].str[:-5]
    h5ad = h5ad.set_index(['tile_tissue_mask_path'])

    gene_exp.index = h5ad.index

    h5ad = h5ad.sort_index()
    gene_exp = gene_exp.sort_index()

    resnet_features['dataset'] = h5ad['library_id']
    gene_exp['dataset'] = h5ad['library_id']

    #Training Data Split, Binarization and Prediction
    gene_exp_zscore = gene_exp.groupby('dataset')[gene_list].apply(lambda x: (x-x.mean())/(x.std()))
    gene_exp_binary = gene_exp_zscore.apply(lambda x: [0 if y <= 0 else 1 for y in x])
    gene_exp_binary['dataset'] = gene_exp['dataset']

    model_c = LogisticRegression(max_iter=10000,penalty='elasticnet',C=0.1,solver='saga',l1_ratio=0.5)
    clf_resnet = MultiOutputClassifier(model_c).fit(resnet_features.loc[(resnet_features['dataset'] == train_data_library_id)].iloc[:,:-1], gene_exp_binary.loc[(gene_exp_binary['dataset'] == train_data_library_id)].iloc[:,:-1])
    joblib.dump(clf_resnet, Path+'LRmodel.pkl')
    clf_resnet = joblib.load(Path+'LRmodel.pkl')
    pred_gexp = pd.DataFrame(clf_resnet.predict(resnet_features.loc[(resnet_features['dataset'] != train_data_library_id)].iloc[:,:-1]),columns=gene_exp.iloc[:,:-1].columns,index=resnet_features.loc[(resnet_features['dataset'] != train_data_library_id)].index)
    pred_gexp['dataset'] = resnet_features.loc[(resnet_features['dataset'] != train_data_library_id)]['dataset']

    performance_metrics_all_predictions = [x for _, x in pred_gexp.groupby('dataset')]
    gene_exp_binary_all_true = [x for _, x in gene_exp_binary.loc[(gene_exp_binary['dataset'] != train_data_library_id)].groupby('dataset')]
    return gene_exp_binary, pred_gexp, performance_metrics_all_predictions, gene_exp_binary_all_true


#Boxplot of Performance Metrics
def performance_metrics(Path, gene_exp_binary_all_true, performance_metrics_all_predictions, gene_list):
    AUROC_genes = []; F1_genes = []; Precision_genes = []; Recall_genes = []
    for dataset in range(len(gene_exp_binary_all_true)):
        for gene in range(len(gene_list)):
            score =  roc_auc_score(performance_metrics_all_predictions[dataset].iloc[:,gene],
                                   gene_exp_binary_all_true[dataset].iloc[:,gene])
            AUROC_genes.append(score)

    for dataset in range(len(gene_exp_binary_all_true)):
        for gene in range(len(gene_list)):
            score =  f1_score(performance_metrics_all_predictions[dataset].iloc[:,gene],
                                   gene_exp_binary_all_true[dataset].iloc[:,gene])
            F1_genes.append(score)

    for dataset in range(len(gene_exp_binary_all_true)):
        for gene in range(len(gene_list)):
            score =  precision_score(performance_metrics_all_predictions[dataset].iloc[:,gene],
                                   gene_exp_binary_all_true[dataset].iloc[:,gene])
            Precision_genes.append(score)

    for dataset in range(len(gene_exp_binary_all_true)):
        for gene in range(len(gene_list)):
            score =  recall_score(performance_metrics_all_predictions[dataset].iloc[:,gene],
                                   gene_exp_binary_all_true[dataset].iloc[:,gene])
            Recall_genes.append(score)

    AUROC_genes = pd.DataFrame(AUROC_genes); F1_genes = pd.DataFrame(F1_genes); Precision_genes = pd.DataFrame(Precision_genes); Recall_genes = pd.DataFrame(Recall_genes)
    Performance_metrics = pd.concat([AUROC_genes,F1_genes,Precision_genes,Recall_genes])
    Performance_metrics['patient'] = list(np.repeat(pd.concat(gene_exp_binary_all_true).loc[(pd.concat(gene_exp_binary_all_true)['dataset'] != train_data_library_id)].drop_duplicates('dataset', keep='first')['dataset'].to_list(),len(gene_list)))*4
    Performance_metrics['genes'] = gene_list*len(set(Performance_metrics['patient']))*4
    Performance_metrics['measure'] = ['AUROC']*len(AUROC_genes)+['F1']*len(F1_genes)+['Precision']*len(Precision_genes)+['Recall']*len(Recall_genes)

    plt.figure(figsize=(19.20,10.80))
    im = sns.boxplot(x="patient", y=0, hue="measure", data=Performance_metrics,linewidth=4)
    im.axhline(0.5, linewidth=2, color='r')
    return im.figure.savefig(Path+'Classification_boxplot.png')


# Clustering followed by Classification
def clustering(Path,gene_exp_binary,train_data_library_id,pred_gexp):
    clustering = AgglomerativeClustering(n_clusters = 2, affinity = 'euclidean', linkage = 'ward')
    model_c = LogisticRegression(max_iter=10000,penalty='elasticnet',C=0.1,solver='saga',l1_ratio=0.5)
    clf_can_v_non_can = model_c.fit(gene_exp_binary.loc[(gene_exp_binary['dataset'] == train_data_library_id)].iloc[:,:-1],
                                                          clustering.fit_predict(gene_exp_binary.loc[(gene_exp_binary['dataset'] == train_data_library_id)].iloc[:,:-1]))
    joblib.dump(clf_can_v_non_can, Path+'resnet_block1_log_scaled_relu_clustering_logistic.pkl')
    clf_can_v_non_can = joblib.load(Path+'resnet_block1_log_scaled_relu_clustering_logistic.pkl')
    #can_v_non_can_spot = pd.DataFrame(clf_can_v_non_can.predict(pred_gexp.iloc[:,:-1]))
    Binarized_gene_exp = pd.DataFrame(gene_exp_binary.loc[(gene_exp_binary['dataset'] == train_data_library_id)]).to_csv(Path+"Binarized_gene_exp_train.csv")




Path = "/home/uqomulay/90days/"
anndata = anndata.read_h5ad(Path+"all_adata.h5ad")
unzipped_file = zipfile.ZipFile(Path+"tiles-wiener-Nov-03.zip", "r")
model = ResNet50(weights='imagenet', pooling="avg", include_top = False)
gene_list = ['CD74', 'CD24', 'CD63', 'CD81', 'CD151', 'COX6C', 'TP53', 'PABPC1',
             'GNAS', 'B2M', 'SPARC', 'HSP90AB1', 'TFF3', 'ATP1A1', 'FASN']
train_data_library_id = "block1"



os.chdir(Path)
resnet_features = ResNet50_features(unzipped_file, model)
print("1")
gene_exp_binary, pred_gexp, performance_metrics_all_predictions, gene_exp_binary_all_true = LR_model(Path, anndata, gene_list, train_data_library_id, resnet_features)
print("2")
performance_metrics(Path,gene_exp_binary_all_true, performance_metrics_all_predictions, gene_list)
print("3")
can_v_non_can_spot = clustering(Path, gene_exp_binary, train_data_library_id, pred_gexp)
print("4")