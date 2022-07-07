import argparse
import configparser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from anndata import read_h5ad
from scipy import stats
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
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


def performance_metrics(test_adata, gene_list, path=None):
    AUROC = []; F1 = []; Precision = []; Recall = []
    for i in set(test_adata.obs["library_id"]):
        anndata_adata = test_adata[test_adata.obs["library_id"]==i]
        for j in gene_list:
            score_auroc = roc_auc_score(anndata_adata.obsm["predicted_gene_expression"][j],anndata_adata.obsm["true_gene_expression"][j])
            AUROC.append(score_auroc)

            score_f1 = f1_score(anndata_adata.obsm["predicted_gene_expression"][j],anndata_adata.obsm["true_gene_expression"][j])
            F1.append(score_f1)

            score_precision = precision_score(anndata_adata.obsm["predicted_gene_expression"][j],anndata_adata.obsm["true_gene_expression"][j])
            Precision.append(score_precision)

            score_recall = recall_score(anndata_adata.obsm["predicted_gene_expression"][j],anndata_adata.obsm["true_gene_expression"][j])
            Recall.append(score_recall)

    Performance_metrics = pd.concat([pd.DataFrame(AUROC), pd.DataFrame(F1), pd.DataFrame(Precision), pd.DataFrame(Recall)])
    Performance_metrics['Patient'] = list(np.repeat(list(set(test_adata.obs["library_id"])),len(gene_list)))*4
    Performance_metrics['Genes'] = gene_list*len(set(Performance_metrics['Patient']))*4
    Performance_metrics['Metrics'] = ['AUROC']*len(AUROC)+['F1']*len(F1)+['Precision']*len(Precision)+['Recall']*len(Recall)

    sns.set(font_scale = 2, style="whitegrid",)
    plt.figure(figsize=(22.50,12.50))
    plt.ylim(-0.1, 1.10)
    im = sns.boxplot(x="Patient", y=0, hue="Metrics", data=Performance_metrics,linewidth=3.5,width=0.6)
    im.set_xticklabels(im.get_xticklabels(),rotation = 30)
    plt.legend(loc="lower right", frameon=True, fontsize=20)
    im.axhline(0.5, linewidth=2, color='r')

    sns.set(font_scale = 2, style="whitegrid",)
    plt.figure(figsize=(22.50,12.50))
    plt.ylim(-0.1, 1.10)
    im2 = sns.boxplot(x="Genes", y=0, hue="Metrics", data=Performance_metrics,linewidth=3.5,width=0.6)
    im2.set_xticklabels(im2.get_xticklabels(),rotation = 30)
    plt.legend(loc="lower right", frameon=True, fontsize=20)
    im2.axhline(0.5, linewidth=2, color='r')

    return im.figure.savefig(path+'Classification_boxplot_cancer_immune_controls.png'), im2.figure.savefig(path+'Classification_boxplot_cancer_immune_controls_genes.png')


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

    if model_name == "classification":
        performance_metrics(test_adata)
