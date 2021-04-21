import sys
import stlearn as st
from pathlib import Path
import pandas as pd
import sys
from .utils import gene_plot, Read10X, ReadOldST, tiling, ensembl_to_id
from .model import CNN_NB_trainable, CNN_NB_multiple_genes, PrinterCallback
from .data_generator import DataGenerator
import tensorflow as tf
from scipy import stats
import numpy as np
from anndata import AnnData, read_h5ad
import configparser
import argparse
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STimage software")
    parser.add_argument('--config', dest='config', type=Path,
                        help='Path to config file')
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read(args.config)
    
    model_name = config["TRAINING"]["model_name"]
    gene_selection = config["DATASET"]["gene_selection"]
    training_ratio = float(config["DATASET"]["training_ratio"])
    valid_ratio = float(config["DATASET"]["valid_ratio"])
    batch_size = int(config["TRAINING"]["batch_size"])
    early_stop = config["TRAINING"].getboolean("early_stop")
    epochs = int(config["TRAINING"]["epochs"])
    OUT_PATH = Path(config["PATH"]["OUT_PATH"])
    DATA_PATH = Path(config["PATH"]["DATA_PATH"])
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    
    save_train_history = config["RESULTS"].getboolean("save_train_history")
    save_model_weights = config["RESULTS"].getboolean("save_model_weights")
    
    adata_all = read_h5ad(DATA_PATH /"all_adata.h5ad") 
    if gene_selection == "tumour":
        comm_genes = ["PABPC1", "GNAS", "HSP90AB1", "TFF3", 
                      "ATP1A1", "COX6C", "B2M", "FASN", 
                      "ACTG1", "HLA-B"]
    adata_all = adata_all[:,comm_genes].copy()
    
    All_sample = pd.Series(adata_all.obs["library_id"].unique())
    training_valid_sample = All_sample.sample(frac=training_ratio, random_state=1)
    training_valid_index_ = All_sample.index.isin(training_valid_sample.index)

    valid_sample = training_valid_sample.sample(frac=valid_ratio, random_state=1)
    valid_index_ = training_valid_sample.index.isin(valid_sample.index)
    
    training_sample = training_valid_sample[~valid_index_]
    
    test_Sample = All_sample[~training_valid_index_].copy()
    
    train_adata = adata_all[adata_all.obs["library_id"].isin(training_sample)].copy()
    valid_adata = adata_all[adata_all.obs["library_id"].isin(valid_sample)].copy()
    test_adata = adata_all[adata_all.obs["library_id"].isin(test_Sample)].copy()
    
    n_genes = len(comm_genes)
    train_gen = tf.data.Dataset.from_generator(
                                               lambda:DataGenerator(adata=train_adata, 
                                                                    genes=comm_genes, aug=True),
                                               output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
                                               output_shapes=([299,299,3], tuple([1]*n_genes))
                                              )
    train_gen_ = train_gen.shuffle(buffer_size=300).batch(batch_size).repeat(3).cache().prefetch(tf.data.experimental.AUTOTUNE)
    
    valid_gen = tf.data.Dataset.from_generator(
                                               lambda:DataGenerator(adata=valid_adata, 
                                                                    genes=comm_genes, aug=True),
                                               output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
                                               output_shapes=([299,299,3], tuple([1]*n_genes))
                                              )
    valid_gen_ = valid_gen.shuffle(buffer_size=300).batch(batch_size).repeat(3).cache().prefetch(tf.data.experimental.AUTOTUNE)
    
    test_gen = tf.data.Dataset.from_generator(
                                              lambda:DataGenerator(adata=test_adata, 
                                              genes=comm_genes), 
                                              output_types=(tf.float32, tuple([tf.float32]*n_genes)), 
                                              output_shapes=([299,299,3], tuple([1]*n_genes))
                                             )
    test_gen_ = test_gen.batch(1)


