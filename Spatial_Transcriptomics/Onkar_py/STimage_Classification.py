import os
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
from io import BytesIO
import numpy as np
import zipfile
import anndata
import joblib
import scipy as sp
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier

from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

Path = "/QRISdata/Q2051/STimage_project/STimage_dataset/PROCESSED/dataset_breast_cancer_9visium/"

model = ResNet50(weights='imagenet', pooling="avg", include_top = False)


def ResNet50_features(files, pre_model):
    x_scratch_train = []
    for imagePath in files:
        image = plt.imread(unzipped_file.open(imagePath)).astype('float32')
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)
        x_scratch_train.append(image)
    x_train = np.vstack(x_scratch_train)
    features = pre_model.predict(x_train, batch_size=1)
    return features

unzipped_file = zipfile.ZipFile(Path+"tiles-wiener-Nov-03.zip", "r")
tiles = pd.DataFrame(unzipped_file.namelist()).iloc[1:,:]

os.chdir(Path)
resnet_features = pd.DataFrame(ResNet50_features(tiles[0],model))
resnet_features.index = tiles[0].str.split('/', expand=True)[1].str[:-5]
resnet_features = resnet_features.sort_index()
resnet_features.to_csv(Path+"resnet_features.csv")


#Training Pre-Processing
anndata = anndata.read_h5ad(Path+"all_adata.h5ad")
h5ad = anndata.obs
gene_exp = anndata.to_df()

#h5ad = pd.read_csv(Path+"h5ad_obs.csv", index_col=0)
#gene_exp = pd.read_csv(Path+"gene_expression_visium.csv", index_col=0)

h5ad['tile_tissue_mask_path'] = h5ad['tile_tissue_mask_path'].str.split('/', expand=True)[3].str[:-5]
h5ad = h5ad.set_index(['tile_tissue_mask_path'])

gene_exp.index = h5ad.index

h5ad = h5ad.sort_index()
gene_exp = gene_exp.sort_index()

resnet_features['dataset'] = h5ad['library_id']
gene_exp['dataset'] = h5ad['library_id']

#Training Data Split, Binarization and #Prediction
gene_list = ['CD74', 'CD24', 'CD63', 'CD81', 'CD151', 'COX6C', 'TP53', 'PABPC1',
             'GNAS', 'B2M', 'SPARC', 'HSP90AB1', 'TFF3', 'ATP1A1', 'FASN']

train_data_library_id = "block1"

gene_exp_log = gene_exp.groupby('dataset')[gene_list].apply(lambda x: np.log((2*x)+1))
gene_exp_log['dataset'] = gene_exp['dataset']
gene_exp_zscore = gene_exp_log.groupby('dataset')[gene_list].apply(lambda x: (x-x.mean())/(x.std()))
gene_exp_binary = gene_exp_zscore.apply(lambda x: [0 if y <= 0 else 1 for y in x])
gene_exp_binary['dataset'] = gene_exp['dataset']

model_c = MLPClassifier(max_iter=10000, hidden_layer_sizes=(150,100,50), early_stopping=True, random_state=0, activation = 'relu')


#clf_resnet = MultiOutputClassifier(model_c).fit(resnet_features.loc[(resnet_features['dataset'] == train_data_library_id)].iloc[:,:-1], gene_exp_binary.loc[(gene_exp_binary['dataset'] == train_data_library_id)].iloc[:,:-1])
#joblib.dump(clf_resnet, Path+'pickle/resnet_block1_log_scaled_relu_2.pkl')
clf_resnet = joblib.load(Path+'pickle/resnet_block1_log_scaled_relu_2.pkl')
pred_gexp = pd.DataFrame(clf_resnet.predict(resnet_features.loc[(resnet_features['dataset'] != train_data_library_id)].iloc[:,:-1]),columns=gene_exp.iloc[:,:-1].columns,index=resnet_features.loc[(resnet_features['dataset'] != train_data_library_id)].index)
pred_gexp['dataset'] = resnet_features.loc[(resnet_features['dataset'] != train_data_library_id)]['dataset']

performance_metrics_all_predictions = [x for _, x in pred_gexp.groupby('dataset')]
gene_exp_binary_all_true = [x for _, x in gene_exp_binary.loc[(gene_exp_binary['dataset'] != train_data_library_id)].groupby('dataset')]

#Performance Metrics
AUROC_genes = []; F1_genes = []; Precision_genes = []; Recall_genes = []
for dataset in range(len(gene_exp_binary_all_true)):
    for gene in range(len(pred_gexp.iloc[:,:-1].T)):
        score =  roc_auc_score(performance_metrics_all_predictions[dataset].iloc[:,gene],
                               gene_exp_binary_all_true[dataset].iloc[:,gene])
        AUROC_genes.append(score)

for dataset in range(len(gene_exp_binary_all_true)):
    for gene in range(len(pred_gexp.iloc[:,:-1].T)):
        score =  f1_score(performance_metrics_all_predictions[dataset].iloc[:,gene],
                               gene_exp_binary_all_true[dataset].iloc[:,gene])
        F1_genes.append(score)

for dataset in range(len(gene_exp_binary_all_true)):
    for gene in range(len(pred_gexp.iloc[:,:-1].T)):
        score =  precision_score(performance_metrics_all_predictions[dataset].iloc[:,gene],
                               gene_exp_binary_all_true[dataset].iloc[:,gene])
        Precision_genes.append(score)

for dataset in range(len(gene_exp_binary_all_true)):
    for gene in range(len(pred_gexp.iloc[:,:-1].T)):
        score =  recall_score(performance_metrics_all_predictions[dataset].iloc[:,gene],
                               gene_exp_binary_all_true[dataset].iloc[:,gene])
        Recall_genes.append(score)

AUROC_genes = pd.DataFrame(AUROC_genes); F1_genes = pd.DataFrame(F1_genes); Precision_genes = pd.DataFrame(Precision_genes); Recall_genes = pd.DataFrame(Recall_genes)
Performance_metrics = pd.concat([AUROC_genes,F1_genes,Precision_genes,Recall_genes])
Performance_metrics['genes'] = list(gene_exp.iloc[:,:-1].columns)*8*4
Performance_metrics['patient'] = list(np.repeat(pd.concat(gene_exp_binary_all_true).loc[(pd.concat(gene_exp_binary_all_true)['dataset'] != train_data_library_id)].drop_duplicates('dataset', keep='first')['dataset'].to_list(),15))*4
Performance_metrics['measure'] = ['AUROC']*len(AUROC_genes)+['F1']*len(F1_genes)+['Precision']*len(Precision_genes)+['Recall']*len(Recall_genes)

#Boxplot
plt.figure(figsize=(19.20,10.80))
im = sns.boxplot(x="patient", y=0, hue="measure", data=Performance_metrics,linewidth=4)
im.axhline(0.5, linewidth=2, color='r')
im.figure.savefig(Path+'Classification_boxplot.png')


#Gene Plotting on Tissue
def gene_plot(Path_of_the_tissue, gene, patient):
    image = cv2.imread(Path_of_the_tissue)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    patients_list = pd.concat(gene_exp_binary_all_true).loc[(pd.concat(gene_exp_binary_all_true)['dataset'] != train_data_library_id)].drop_duplicates('dataset', keep='first')['dataset'].to_list()
    Spots_Cluster_patient = pd.DataFrame(performance_metrics_all_predictions[patients_list.index(patient)][gene])
    Spots_Cluster = pd.concat([h5ad.loc[((h5ad['library_id'] == patient))][["imagerow","imagecol"]], Spots_Cluster_patient], axis=1)


    Spot_vals0=Spots_Cluster[Spots_Cluster[gene] == 0]
    Spot_vals0=Spot_vals0.values
    Spot_vals1=Spots_Cluster[Spots_Cluster[gene] == 1]
    Spot_vals1=Spot_vals1.values


    x = Spot_vals0[:,0].astype('int64')
    y = Spot_vals0[:,1].astype('int64')
    box = (y,x)
    numpy_array = np.array(box)
    transpose = numpy_array.T
    box = transpose.tolist()

    x1 = Spot_vals1[:,0].astype('int64')
    y1 = Spot_vals1[:,1].astype('int64')
    box1 = (y1,x1)
    numpy_array1 = np.array(box1)
    transpose1 = numpy_array1.T
    box1 = transpose1.tolist()


    for i in range(0,len(box)):
        image=cv2.circle(image, tuple(box[i]), 60,(139,0,0), -1)
    for i in range(0,len(box1)):
        image=cv2.circle(image, tuple(box1[i]), 60,(0,255,0), -1)
    return cv2.imwrite(Path+gene+"_pred_ffpe_gMLP.png",image)


Path_of_the_tissue = Path+"Other/FFPE/Visium_FFPE_Human_Breast_Cancer_image.tif" #V1_Breast_Cancer_Block_A_Section_2_image
gene = "COX6C"
patient = "FFPE"
gene_plot(Path_of_the_tissue, gene, patient)
