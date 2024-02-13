import sys
import stlearn as st
st.settings.set_figure_params(dpi=300)
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import sys
file = Path("../stimage").resolve()
parent= file.parent
sys.path.append(str(parent))
from PIL import Image
from stimage._utils import gene_plot, Read10X, ReadOldST, tiling, ensembl_to_id
from stimage._model import CNN_NB_multiple_genes, negative_binomial_layer, negative_binomial_loss
from stimage._data_generator import DataGenerator
import tensorflow as tf
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
# import geopandas as gpd
from sklearn.neighbors import KDTree
from anndata import read_h5ad
from tensorflow.keras import backend as K
import scanpy as sc
import pickle
import matplotlib.pyplot as plt
from libpysal.weights.contiguity import Queen
from libpysal import examples
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import splot
from splot.esda import moran_scatterplot, lisa_cluster
from esda.moran import Moran, Moran_Local
from esda.moran import Moran_BV, Moran_Local_BV
from splot.esda import plot_moran_bv_simulation, plot_moran_bv, plot_local_autocorrelation

from scipy import stats
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input as preprocess_densenet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.tf_utils import is_tensor_or_variable



def STNet(tile_shape, output_shape, mean_exp_tf):
    tile_input = Input(shape=tile_shape, name = "tile_input")
    DenseNet121_base = DenseNet121(input_tensor=tile_input, weights='imagenet', include_top=False)
    for layer in DenseNet121_base.layers:
        layer.trainable = False
    
    cnn = DenseNet121_base.output
    cnn = GlobalAveragePooling2D()(cnn)
#     cnn = Dropout(0.5)(cnn)
#     cnn = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.01),
#                 activity_regularizer=tf.keras.regularizers.l2(0.01))(cnn)
    # cnn = Dense(256, activation='relu')(cnn)
    
    outputs = Dense(output_shape, activation='linear', bias_initializer=mean_exp_tf)(cnn)
    model = Model(inputs=tile_input, outputs=outputs)

#     optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum=0.9)

    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=[tf.keras.metrics.MeanSquaredError()])    
    return model



def plot_correlation(df, attr_1, attr_2):
    r = stats.pearsonr(df[attr_1], 
                       df[attr_2])[0] **2

    g = sns.lmplot(data=df,
        x=attr_1, y=attr_2,
        height=5, legend=True
    )
    # g.set(ylim=(0, 360), xlim=(0,360))

    g.set_axis_labels(attr_1, attr_2)
    plt.annotate(r'$R^2:{0:.2f}$'.format(r),
                (max(df[attr_1])*0.9, max(df[attr_2])*0.9))
    return g


def calculate_correlation(attr_1, attr_2):
    r = stats.pearsonr(attr_1, 
                       attr_2)[0]
    return r

def calculate_correlation_2(attr_1, attr_2):
    r = stats.spearmanr(attr_1, 
                       attr_2)[0]
    return r




DATA_PATH = Path("/clusterdata/uqxtan9/Xiao/dataset_3_224_no_norm")


adata_all = read_h5ad(DATA_PATH / "all_adata.h5ad")

#adata_all.obs["tile_path"] = adata_all.obs.tile_path.map(lambda x:x.replace("/clusterdata/uqxtan9/Xiao/Q1851/Xiao/STimage_dataset/breast_cancer_10x_visium/dataset_3_224_no_norm/",
#                                               "/clusterdata/uqxtan9/Xiao/dataset_3_224_no_norm/"))

sample_to_keep = ['block1', 'block2', 'FFPE'] # '1160920F' 

adata_all = adata_all[adata_all.obs["library_id"].isin(sample_to_keep)].copy()

gene_list_path = "/scratch/imb/Xiao/STimage/development/stimage_compare_histogene_1000hvg/gene_list_OOD.pkl"
with open(gene_list_path, 'rb') as f:
    gene_list = pickle.load(f)

df = pd.DataFrame()

test_sample = 'FFPE'
n_genes = len(gene_list)

adata_all_train_valid = adata_all[adata_all.obs["library_id"].isin(
    adata_all.obs.library_id.cat.remove_categories(test_sample).unique())]

training_index = adata_all_train_valid.obs.sample(frac=0.7, random_state=1).index
training_dataset = adata_all_train_valid[training_index,].copy()

valid_index = adata_all_train_valid.obs.index.isin(training_index)
valid_dataset = adata_all_train_valid[~valid_index,].copy()

test_index = adata_all.obs.library_id == test_sample
test_dataset_1 = adata_all[test_index,].copy()


train_gen = tf.data.Dataset.from_generator(
        lambda:DataGenerator(adata=training_dataset, 
                      genes=gene_list, dim=(224, 224), aug=False),
        output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
        output_shapes=([224,224,3], tuple([1]*n_genes))
)
train_gen_ = train_gen.shuffle(buffer_size=500).batch(32).repeat(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
valid_gen = tf.data.Dataset.from_generator(
        lambda:DataGenerator(adata=valid_dataset, 
                      genes=gene_list, dim=(224, 224)), 
        output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
        output_shapes=([224,224,3], tuple([1]*n_genes))
)
valid_gen_ = valid_gen.shuffle(buffer_size=500).batch(32).repeat(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
test_gen_1 = tf.data.Dataset.from_generator(
        lambda:DataGenerator(adata=test_dataset_1, 
                      genes=gene_list, dim=(224, 224)), 
        output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
        output_shapes=([224,224,3], tuple([1]*n_genes))
)
test_gen__1 = test_gen_1.batch(1)

K.clear_session()
mean_exp = training_dataset[:,gene_list].to_df().mean()
mean_exp_tf = tf.keras.initializers.RandomUniform(minval=mean_exp, 
                                                  maxval=mean_exp)
model = STNet((224, 224, 3), n_genes, mean_exp_tf)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                        restore_best_weights=False)

train_history = model.fit(train_gen_,
                      epochs=100,
                      validation_data=valid_gen_,
                      callbacks=[callback]
                      )

test_predictions = model.predict(test_gen__1)

test_dataset_1.obsm["predicted_gene"] = test_predictions
test_dataset_1_ = test_dataset_1[:,gene_list].copy()
test_dataset_1_.X = test_dataset_1_.obsm["predicted_gene"]

pred_adata = test_dataset_1_
test_dataset = test_dataset_1
pred_adata.write_h5ad("/clusterdata/uqxtan9/Xiao/STimage/development/stimage_benchmarking_1000hvg_OOD/stnet_pred_adata.h5ad")
test_dataset.write_h5ad("/clusterdata/uqxtan9/Xiao/STimage/development/stimage_benchmarking_1000hvg_OOD/stnet_test_adata.h5ad")
for gene in pred_adata.var_names:
    cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val, test_sample, "STnet"], 
                         index=["Gene", "Pearson correlation", "Slide", "Method"]),
                  ignore_index=True)

df.to_csv("./stimage_benchmarking_1000hvg_OOD/stnet_cor_{}.csv".format(test_sample))

