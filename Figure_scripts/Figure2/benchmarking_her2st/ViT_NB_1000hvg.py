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

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
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

DATA_PATH = Path("/scratch/imb/Xiao/STimage/dataset/her2st")

adata_all = read_h5ad(DATA_PATH / "all_adata.h5ad")

# adata_all = ensembl_to_id(adata_all)

samples = adata_all.obs["library_id"].unique().tolist()

#gene_list=["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1",
#           "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

gene_list_path = "/scratch/imb/Xiao/STimage/development/stimage_compare_histogene_1000hvg/gene_list.pkl"
with open(gene_list_path, 'rb') as f:
    gene_list = pickle.load(f)


adata_all.obs["X"] = [x for x,y in adata_all.obs.index.str.split("x")]
adata_all.obs["Y"] = [y.split("-")[0] for x,y in adata_all.obs.index.str.split("x")]


import numpy as np
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet
from tensorflow.keras.preprocessing import image
from stimage._imgaug import seq_aug
from sklearn.neighbors import KDTree
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, adata, dim=(299, 299), n_channels=3, genes=None, aug=False, tile_path="tile_path"):
        'Initialization'
        self.dim = dim
        self.adata = adata
        self.n_channels = n_channels
        self.genes = genes
        self.num_genes = len(genes)
        self.aug = aug
        self.tile_path = tile_path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(self.adata.n_obs)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Find list of IDs
        obs_temp = self.adata.obs_names[index]

        # Generate data
        X_img = self._load_img(obs_temp)
        y = self._load_label(obs_temp)
        pos_x,pos_y = self._load_position(obs_temp)
        
        return {"input_img":X_img, "x":pos_x, "y":pos_y}, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.adata.n_obs)

    def _load_img(self, obs):
        img_path = self.adata.obs.loc[obs, 'tile_path']
        X_img = image.load_img(img_path, target_size=self.dim)
        X_img = image.img_to_array(X_img).astype('uint8')
        #         X_img = np.expand_dims(X_img, axis=0)
        #         n_rotate = np.random.randint(0, 4)
        #         X_img = np.rot90(X_img, k=n_rotate, axes=(1, 2))
        if self.aug:
            X_img = seq_aug(image=X_img)
#         X_img = preprocess_resnet(X_img)
        return X_img

    def _load_label(self, obs):
        batch_adata = self.adata[obs, self.genes].copy()

        return tuple([batch_adata.to_df()[i].values for i in self.genes])
    
    def _load_position(self, obs):
        return int(self.adata.obs.loc[obs, "X"]), int(self.adata.obs.loc[obs, "Y"])

    def get_classes(self):
        return self.adata.to_df().loc[:, self.genes]

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

df = pd.DataFrame()

i = int(sys.argv[1])
test_sample = samples[i]
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
                      genes=gene_list, aug=False),
        output_types=({"input_img":tf.float32, "x":tf.float32, "y":tf.float32}, tuple([tf.float32]*n_genes)),
#         output_shapes=([299,299,3],1,1, tuple([1]*n_genes))
)
train_gen_ = train_gen.shuffle(buffer_size=10).batch(32).repeat(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
valid_gen = tf.data.Dataset.from_generator(
        lambda:DataGenerator(adata=valid_dataset,
                      genes=gene_list),
        output_types=({"input_img":tf.float32, "x":tf.float32, "y":tf.float32}, tuple([tf.float32]*n_genes)),
#         output_shapes=(([299,299,3],1,1), tuple([1]*n_genes))
)
valid_gen_ = valid_gen.shuffle(buffer_size=10).batch(32).repeat(1).cache().prefetch(tf.data.experimental.AUTOTUNE)
test_gen_1 = tf.data.Dataset.from_generator(
        lambda:DataGenerator(adata=test_dataset_1,
                      genes=gene_list),
        output_types=({"input_img":tf.float32, "x":tf.float32, "y":tf.float32}, tuple([tf.float32]*n_genes)),
#         output_shapes=(([299,299,3],1,1), tuple([1]*n_genes))
)
test_gen__1 = test_gen_1.batch(1)


from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
def ViT_NB(tile_shape, pos_shape, n_genes=None, projection_dim=64, num_patches=24, transformer_layers=8,
           mlp_head_units = [256, 128], num_heads = 4, patch_size=13):
    tile_inputs = layers.Input(shape=tile_shape,name="input_img")
    x = layers.Input(shape=pos_shape, name="x")
    y = layers.Input(shape=pos_shape, name="y")
#     tile_emb = layers.Flatten()(tile_inputs)
#     resnet_base = ResNet50(weights='imagenet', include_top=False, pooling="max")
#     for i in range(len(resnet_base.layers)):
#         resnet_base.layers[i].trainable = False
#     tile_features = resnet_base(tile_inputs)
    
    num_patches = (299 // patch_size) ** 2
    patches = Patches(patch_size)(tile_inputs)
    
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    
    transformer_units = [
        projection_dim*2,
        projection_dim,
    ]

    for _ in range(transformer_layers):

        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1,x1)

        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)

        encoded_patches = layers.Add()([x3, x2])


    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
#     representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Flatten()(representation)
#     representation = layers.Dropout(0.5)(representation)

    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1)

    output_layers = []
    for i in range(n_genes):
        output = layers.Dense(2)(features)
        output_layers.append(layers.Lambda(negative_binomial_layer, name="gene_{}".format(i))(output))

    model = Model(inputs=[tile_inputs, x, y], outputs=output_layers)
    #     losses={}
    #     for i in range(8):
    #         losses["gene_{}".format(i)] = negative_binomial_loss(i)
    #     optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.Adam(1e-5)
    model.compile(loss=negative_binomial_loss,
                  optimizer=optimizer)
    return model


K.clear_session()
model = ViT_NB(tile_shape=(299,299,3), pos_shape=1, n_genes=n_genes)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                        restore_best_weights=False)

train_history = model.fit(train_gen_,
                      epochs=100,
                      validation_data=valid_gen_,
                      callbacks=[callback]
                      )
test_predictions = model.predict(test_gen__1)
from scipy.stats import nbinom
y_preds = []
for i in range(n_genes):
    n = test_predictions[i][:, 0]
    p = test_predictions[i][:, 1]
    y_pred = nbinom.mean(n, p)
    y_preds.append(y_pred)
test_dataset_1.obsm["predicted_gene"] = np.array(y_preds).transpose()
test_dataset_1_ = test_dataset_1[:,gene_list].copy()
test_dataset_1_.X = test_dataset_1_.obsm["predicted_gene"]

pred_adata = test_dataset_1_
test_dataset = test_dataset_1

for gene in pred_adata.var_names:
    cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val, test_sample, "ViT_NB"], 
                         index=["Gene", "Pearson correlation", "Slide", "Method"]),
                  ignore_index=True)

df.to_csv("./stimage_compare_histogene_1000hvg/ViT_NB_cor_{}.csv".format(test_sample))







