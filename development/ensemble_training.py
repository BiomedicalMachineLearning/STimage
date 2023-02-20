from pathlib import Path
from anndata import read_h5ad
import sys

job_id = int(sys.argv[1])

import scanpy

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import nbinom, pearsonr

import tensorflow as tf

from tensorflow.keras import backend as K

# set random seed 
tf.random.set_seed(job_id)
# check GPU is connected
devices = tf.config.list_physical_devices("GPU")
if not devices[0].device_type == 'GPU':
    raise Exception("Failed to connect to GPU")

# stimage custom 
file = Path("../stimage").resolve() # path to src code
parent = file.parent
sys.path.append(str(parent))

from stimage._utils import gene_plot, Read10X, ReadOldST, tiling
from stimage._model import CNN_NB_multiple_genes
from stimage._data_generator import DataGenerator

DATA_PATH = Path("/scratch/smp/uqsmac12/dataset_breast_cancer_9visium")
OUT_PATH = Path("/scratch/smp/uqsmac12/results")
OUT_PATH.mkdir(parents=True, exist_ok=True)

adata_all = read_h5ad(DATA_PATH / "all_adata.h5ad")

# update metadata for annadata such that it maps to the correct location
adata_all.obs["tile_path"] = adata_all.obs.tile_path.map(
    lambda x: x.replace("/clusterdata/uqxtan9/Xiao/breast_cancer_9visium",
                        "/scratch/smp/uqsmac12/dataset_breast_cancer_9visium"))


gene_list=["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1", 
           "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]

# remove FFPE and 1160920F --- the test set
adata_all_train_valid = adata_all[adata_all.obs["library_id"].isin(
    adata_all.obs.library_id.cat.remove_categories(["FFPE", "1160920F"]).unique())]


# training data, validation data, test data
n_genes = len(gene_list)

training_index = adata_all_train_valid.obs.sample(frac=0.7, random_state=1).index
training_dataset = adata_all_train_valid[training_index,].copy()

valid_index = adata_all_train_valid.obs.index.isin(training_index)
valid_dataset = adata_all_train_valid[~valid_index,].copy()

test_index = adata_all.obs.library_id == "FFPE"
test_dataset_1 = adata_all[test_index,].copy()

test_index = adata_all.obs.library_id == "1160920F"
test_dataset_2 = adata_all[test_index,].copy()

train_gen = tf.data.Dataset.from_generator(
            lambda: DataGenerator(adata=training_dataset, 
                          genes=gene_list, aug=False),
            output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
            output_shapes=([299,299,3], tuple([1]*n_genes))
)
train_gen_ = train_gen.shuffle(buffer_size=500).batch(128).repeat(3).cache().prefetch(tf.data.experimental.AUTOTUNE)
valid_gen = tf.data.Dataset.from_generator(
            lambda: DataGenerator(adata=valid_dataset, 
                          genes=gene_list), 
            output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
            output_shapes=([299,299,3], tuple([1]*n_genes))
)
valid_gen_ = valid_gen.shuffle(buffer_size=500).batch(128).repeat(3).cache().prefetch(tf.data.experimental.AUTOTUNE)
test_gen_1 = tf.data.Dataset.from_generator(
            lambda:DataGenerator(adata=test_dataset_1, 
                          genes=gene_list), 
            output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
            output_shapes=([299,299,3], tuple([1]*n_genes))
)
test_gen__1 = test_gen_1.batch(1)
test_gen_2 = tf.data.Dataset.from_generator(
            lambda: DataGenerator(adata=test_dataset_2, 
                          genes=gene_list), 
            output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
            output_shapes=([299,299,3], tuple([1]*n_genes))
)
test_gen__2 = test_gen_2.batch(1)

# instantiate model
K.clear_session()
model = CNN_NB_multiple_genes((299, 299, 3), n_genes, cnnbase="resnet50")
callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10,
    restore_best_weights=True
)
lr_ = 1e-4 # change the learning rate to improve results
model.optimizer.learning_rate.assign(lr_)
# train model
with tf.device("GPU:0"):
    train_history = model.fit(train_gen_,
                              epochs=100,
                              validation_data=valid_gen_,
                              callbacks=[callback]
                              )

model.save(OUT_PATH / f"resnet50_{job_id}-rev2.h5")