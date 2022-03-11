import argparse
import configparser
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from _data_generator import DataGenerator
from _model import CNN_NB_multiple_genes, PrinterCallback
from anndata import read_h5ad
import scanpy as sc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STimage software --- Training")
    parser.add_argument('--config', dest='config', type=Path,
                        help='Path to config file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    tile_size = int(config["DATASET"]["tile_size"])
    gene_selection = config["DATASET"]["gene_selection"]
    training_ratio = float(config["DATASET"]["training_ratio"])
    valid_ratio = float(config["DATASET"]["valid_ratio"])

    model_name = config["TRAINING"]["model_name"]
    batch_size = int(config["TRAINING"]["batch_size"])
    early_stop = config["TRAINING"].getboolean("early_stop")
    epochs = int(config["TRAINING"]["epochs"])
    cnn_base = config["TRAINING"]["cnn_base"]
    fine_tuning = config["TRAINING"].getboolean("fine_tuning")

    save_train_history = config["RESULTS"].getboolean("save_train_history")
    save_model_weights = config["RESULTS"].getboolean("save_model_weights")

    OUT_PATH = Path(config["PATH"]["OUT_PATH"])
    DATA_PATH = Path(config["PATH"]["DATA_PATH"])
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    if (DATA_PATH / "train_adata.h5ad").exists() and \
            (DATA_PATH / "valid_adata.h5ad").exists() and \
            (DATA_PATH / "test_adata.h5ad").exists():
        train_adata = read_h5ad(DATA_PATH / "train_adata.h5ad")
        valid_adata = read_h5ad(DATA_PATH / "valid_adata.h5ad")
        test_adata = read_h5ad(DATA_PATH / "test_adata.h5ad")
        comm_genes = train_adata.var_names
    else:
        adata_all = read_h5ad(DATA_PATH / "all_adata.h5ad")
        if gene_selection == "tumour":
            comm_genes = ["PABPC1", "GNAS", "HSP90AB1", "TFF3",
                          "ATP1A1", "COX6C", "B2M", "FASN",
                          "ACTG1", "HLA-B"]

        if gene_selection == "markers":
            comm_genes = ["SLITRK6", "PGM5", "LINC00645",
                          "TTLL12", "COX6C", "CPB1",
                          "KRT5", "MALAT1"]

        if gene_selection == "top250":
            adata_all.var["mean_expression"] = np.mean(adata_all.X, axis=0)
            comm_genes = adata_all.var["mean_expression"].sort_values(ascending=False
                                                                      ).index[0:250]

        if gene_selection == "HVG1000":
            sc.pp.highly_variable_genes(adata_all, n_top_genes=1000)
            comm_genes = adata_all.var_names[adata_all.var.highly_variable]
        else:
            comm_genes = gene_selection

        adata_all = adata_all[:, comm_genes].copy()
        if model_name.split("_")[-1] == "classification":
            adata_all.X = adata_all.to_df().apply(lambda x: pd.qcut(x, 3, duplicates='drop', labels=False))
        All_sample = pd.Series(adata_all.obs["library_id"].unique())
        training_valid_sample = All_sample.sample(frac=training_ratio, random_state=1)
        training_valid_index_ = All_sample.index.isin(training_valid_sample.index)

        if len(training_valid_sample) > 1:
            valid_sample = training_valid_sample.sample(frac=valid_ratio, random_state=1)
            valid_index_ = training_valid_sample.index.isin(valid_sample.index)
            training_sample = training_valid_sample[~valid_index_]
        else:
            valid_sample = training_valid_sample
            training_sample = training_valid_sample

        test_Sample = All_sample[~training_valid_index_].copy()

        train_adata = adata_all[adata_all.obs["library_id"].isin(training_sample)].copy()
        train_adata.write(DATA_PATH / "train_adata.h5ad")
        valid_adata = adata_all[adata_all.obs["library_id"].isin(valid_sample)].copy()
        valid_adata.write(DATA_PATH / "valid_adata.h5ad")
        test_adata = adata_all[adata_all.obs["library_id"].isin(test_Sample)].copy()
        test_adata.write(DATA_PATH / "test_adata.h5ad")

    n_genes = len(comm_genes)
    train_gen = tf.data.Dataset.from_generator(
        lambda: DataGenerator(adata=train_adata,
                              genes=comm_genes, aug=True),
        output_types=(tf.float32, tuple([tf.float32] * n_genes)),
        output_shapes=([tile_size, tile_size, 3], tuple([1] * n_genes))
    )
    train_gen_ = train_gen.shuffle(buffer_size=200).batch(batch_size).repeat(1).cache().prefetch(
        tf.data.experimental.AUTOTUNE)

    valid_gen = tf.data.Dataset.from_generator(
        lambda: DataGenerator(adata=valid_adata,
                              genes=comm_genes, aug=True),
        output_types=(tf.float32, tuple([tf.float32] * n_genes)),
        output_shapes=([tile_size, tile_size, 3], tuple([1] * n_genes))
    )
    valid_gen_ = valid_gen.shuffle(buffer_size=200).batch(batch_size).repeat(1).cache().prefetch(
        tf.data.experimental.AUTOTUNE)

    model = None
    if model_name == "NB_regression":
        model = CNN_NB_multiple_genes((tile_size, tile_size, 3), n_genes, cnnbase=cnn_base, ft=fine_tuning)

    callbacks = [PrinterCallback()]
    # callbacks = []
    if early_stop:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                                          restore_best_weights=True))

    train_history = model.fit(train_gen_,
                              validation_data=valid_gen_,
                              epochs=epochs,
                              callbacks=callbacks,
                              verbose=0)
    if save_train_history:
        with open(OUT_PATH / "training_history.pkl", "wb") as file:
            pickle.dump(train_history.history, file)
    if save_model_weights:
        model.save(OUT_PATH / "model_weights.h5")
