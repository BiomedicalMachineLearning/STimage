import argparse
import configparser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import read_h5ad
from scipy import stats

from _utils import gene_plot


def calculate_r(test_adata, prediction_adata, genes):
    r_list = []
    genes_list = []
    sample_list = []
    for i, library in enumerate(
            test_library
    ):
        ad_ = prediction_adata[prediction_adata.obs.library_id == library, :].copy()
        ad = test_adata[test_adata.obs.library_id == library, :].copy()
        for gene in genes:
            r = stats.pearsonr(ad[:, gene],
                               ad_[:, gene])[0]
            r_list.append(r)
            genes_list.append(gene)
            sample_list.append(library)
            print("finished calculate r for sample: {}".format(library))
    df = pd.DataFrame({"r": r_list,
                       "genes": genes_list,
                       "sample": sample_list})
    return df


def spatial_plot(adata, name, libraries, genes):
    for i, library in enumerate(
            libraries
    ):
        for gene in genes:
            ad = adata[adata.obs.library_id == library, :].copy()
            gene_plot(ad, genes=gene, spot_size=60, library_id=library,
                      output=str(OUT_PATH), name="{}_{}_{}.png".format(name, library, gene))


def plot_r(r_df, out_path):
    f, ax = plt.subplots()
    sns.boxplot(x="genes", y="r", data=r_df, palette="vlag")
    sns.swarmplot(x="genes", y="r", data=r_df, hue="sample")
    plt.savefig(out_path / "correlation_gene.png", dpi=300)

    f, ax = plt.subplots()
    sns.boxplot(x="sample", y="r", data=r_df, palette="vlag")
    sns.swarmplot(x="sample", y="r", data=r_df, hue="genes")
    plt.savefig(out_path / "correlation_sample.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STimage software --- Visualisation")
    parser.add_argument('--config', dest='config', type=Path,
                        help='Path to config file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)

    model_name = config["TRAINING"]["model_name"]
    gene_selection = config["DATASET"]["gene_selection"]
    correlation_plot = config["RESULTS"].getboolean("correlation_plot")
    spatial_expression_plot = config["RESULTS"].getboolean("spatial_expression_plot")
    OUT_PATH = Path(config["PATH"]["OUT_PATH"])
    DATA_PATH = Path(config["PATH"]["DATA_PATH"])
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    if gene_selection == "tumour":
        comm_genes = ["PABPC1", "GNAS", "HSP90AB1", "TFF3",
                      "ATP1A1", "COX6C", "B2M", "FASN",
                      "ACTG1", "HLA-B"]
    n_genes = len(comm_genes)

    test_adata = read_h5ad(DATA_PATH / "prediction.h5ad")
    prediction_adata = test_adata.copy()
    prediction_adata.X = prediction_adata.obsm["predicted_gene"]
    test_library = prediction_adata.obs["library_id"].unique()

    if correlation_plot:
        r_df = calculate_r(test_adata, prediction_adata, comm_genes)
        plot_r(r_df, OUT_PATH)

    if spatial_expression_plot:
        spatial_plot(prediction_adata, "prediction", test_library, comm_genes)
        spatial_plot(test_adata, "ground_truth", test_library, comm_genes)
