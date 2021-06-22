import argparse
import configparser
from pathlib import Path
import numpy as np
import tensorflow as tf
from anndata import read_h5ad

from ._data_generator import DataGenerator
from ._model import CNN_NB_multiple_genes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STimage software --- Prediction")
    parser.add_argument('--config', dest='config', type=Path,
                        help='Path to config file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    model_name = config["TRAINING"]["model_name"]
    gene_selection = config["DATASET"]["gene_selection"]
    OUT_PATH = Path(config["PATH"]["OUT_PATH"])
    DATA_PATH = Path(config["PATH"]["DATA_PATH"])
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    if gene_selection == "tumour":
        comm_genes = ["PABPC1", "GNAS", "HSP90AB1", "TFF3",
                      "ATP1A1", "COX6C", "B2M", "FASN",
                      "ACTG1", "HLA-B"]
    n_genes = len(comm_genes)

    test_adata = read_h5ad(DATA_PATH / "test_adata.h5ad")
    test_gen = tf.data.Dataset.from_generator(
        lambda: DataGenerator(adata=test_adata,
                              genes=comm_genes),
        output_types=(tf.float32, tuple([tf.float32] * n_genes)),
        output_shapes=([299, 299, 3], tuple([1] * n_genes))
    )
    test_gen_ = test_gen.batch(1)

    model = None
    if model_name == "NB":
        model = CNN_NB_multiple_genes((299, 299, 3), n_genes)

    model.load_weights(OUT_PATH / "model_weights.h5")
    test_predictions = model.predict(test_gen_)

    if model_name == "NB":
        from scipy.stats import nbinom
        y_preds = []
        for i in range(n_genes):
            n = test_predictions[i][:, 0]
            p = test_predictions[i][:, 1]
            y_pred = nbinom.mean(n, p)
            y_preds.append(y_pred)
        test_adata.obsm["predicted_gene"] = np.array(y_preds).transpose()

    test_adata_ = test_adata.copy()
    test_adata_.X = test_adata_.obsm["predicted_gene"]
    test_adata_.write(DATA_PATH / "prediction.h5ad")