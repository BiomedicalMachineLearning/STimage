import argparse
import configparser
import joblib
import slideio
from pathlib import Path
import pandas as pd
from anndata import AnnData, read_h5ad
import stlearn as st
st.settings.set_figure_params(dpi=300)
from PIL import Image
from stimage._img_normaliser import IterativeNormaliser
from stimage._utils import gene_plot,  tiling, scale_img, calculate_bg
from stimage._model import CNN_NB_multiple_genes
from stimage._data_generator import DataGenerator
import tensorflow as tf
import seaborn as sns
sns.set_style("white")
import numpy as np
from anndata import read_h5ad
from scipy.stats import zscore


if __name__ == "__main__":

    # Read config file
    parser = argparse.ArgumentParser(description="STimage software --- Preprocessing")
    parser.add_argument('--config', dest='config', type=Path,
                        help='Path to config file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
    model_name = config["TRAINING"]["model_name"]
    normalization = config["DATASET"]["normalization"]
    META_PATH = Path(config["PATH"]["METADATA_PATH"])
    DATA_PATH = Path(config["PATH"]["DATA_PATH"])
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    TILING_PATH = Path(config["PATH"]["TILING_PATH"])
    convert_ensembl = config["DATASET"].getboolean("ensembl_to_id")
    platform = config["DATASET"]["platform"]
    tile_size = int(config["DATASET"]["tile_size"])
    stain_normalization = config["DATASET"].getboolean("stain_normalization")
    template_sample = config["DATASET"]["template_sample"]
    tile_filtering_threshold = float(config["DATASET"]["tile_filtering_threshold"])
    TILING_PATH.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META_PATH)
    external_data_path = config["EXTERNAL_DATASET"]["external_data_path"]
    img_format = str(config["EXTERNAL_DATASET"]["format"])
    cnn_base = config["TRAINING"]["cnn_base"]
    fine_tuning = config["TRAINING"].getboolean("fine_tuning")

    OUT_PATH = Path(config["PATH"]["OUT_PATH"])
    DATA_PATH = Path(config["PATH"]["DATA_PATH"])
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Read the external H&E image
    slide = slideio.open_slide(str(external_data_path), img_format)
    scene = slide.get_scene(0)
    image = scene.read_block()

    scale_size = (image.shape[1] * 0.5, image.shape[0]* 0.5)
    image_pil = Image.fromarray(image)
    image_pil.thumbnail(scale_size, Image.ANTIALIAS)
    image = np.array(image_pil)

    train_adata = read_h5ad(DATA_PATH / "train_adata.h5ad")
    comm_genes = train_adata.var_names
    n_genes = len(comm_genes)
    n_array_row = image.shape[0] // tile_size
    n_array_col = image.shape[1] // tile_size
    array_row = np.array(np.meshgrid(np.arange(1, n_array_row),
                                     np.arange(1, n_array_col))
                         ).T.reshape(-1, 2)[:, 0].astype(int)
    array_col = np.array(np.meshgrid(np.arange(1, n_array_row),
                                     np.arange(1, n_array_col))
                         ).T.reshape(-1, 2)[:, 1].astype(int)
    spot_index = [f"{array_row[i]}x{array_col[i]}" for i in range(array_row.shape[0])]
    adata_obs = pd.DataFrame({"array_row": array_row,
                              "array_col": array_col},
                             index=spot_index)
    adata_obs["imagerow"] = adata_obs["array_row"] * tile_size
    adata_obs["imagecol"] = adata_obs["array_col"] * tile_size
    library_id = Path(external_data_path).stem
    adata_obs["library_id"] = library_id
    adata_df = pd.DataFrame(0, index=adata_obs.index, columns=comm_genes)
    adata = AnnData(adata_df, obs=adata_obs)
    adata.obsm["spatial"] = adata_obs[["imagerow", "imagecol"]].values
    adata.uns["spatial"] = dict()
    adata.uns["spatial"][library_id] = dict()
    adata.uns["spatial"][library_id]["images"] = dict()
    adata.uns["spatial"][library_id]["images"]["fulres"] = image
    adata.uns["spatial"][library_id]["use_quality"] = "fulres"
    adata.uns["spatial"][library_id]["scalefactors"] = dict()

    # stain_normalization
    template_img = train_adata.uns["spatial"][template_sample]['images']["fulres"]
    template_img = Image.fromarray(template_img.astype("uint8"))

    normaliser = IterativeNormaliser(normalisation_method='vahadane', standardise_luminosity=True)
    normaliser.fit_target(scale_img(template_img))

    TILING_PATH = "/clusterdata/uqxtan9/Xiao/STimage/dataset/TCIA/tiles"
    calculate_bg(adata, crop_size=tile_size, stain_normaliser=normaliser)
    tile_to_remove = sum(adata.obs["tissue_area"] < tile_filtering_threshold)
    adata = adata[adata.obs["tissue_area"] >= tile_filtering_threshold].copy()
    print("{} tiles with low tissue coverage are removed".format(tile_to_remove))

    tiling(adata, out_path=TILING_PATH, crop_size=tile_size, stain_normaliser=normaliser)

    adata.write(Path(DATA_PATH / "external_adata.h5ad"))

    test_dataset = adata.copy()
    test_gen = tf.data.Dataset.from_generator(
            lambda:DataGenerator(adata=test_dataset,
                                 genes=comm_genes),
            output_types=(tf.float32, tuple([tf.float32]*n_genes)),
            output_shapes=([tile_size, tile_size, 3], tuple([1]*n_genes))
    )
    test_gen_ = test_gen.batch(1)

    model = None
    if model_name == "NB_regression":
        model = CNN_NB_multiple_genes((tile_size, tile_size, 3), n_genes, cnnbase=cnn_base, ft=fine_tuning)

    model.load_weights(OUT_PATH / "model_weights.h5")
    test_predictions = model.predict(test_gen_)

    if model_name == "NB_regression":
        from scipy.stats import nbinom

        y_preds = []
        for i in range(n_genes):
            n = test_predictions[i][:, 0]
            p = test_predictions[i][:, 1]
            y_pred = nbinom.mean(n, p)
            y_preds.append(y_pred)
        test_dataset.obsm["predicted_gene"] = np.array(y_preds).transpose()
    elif model_name == "classification":
        clf_resnet = joblib.load(OUT_PATH / 'pickle/LRmodel.pkl')
        test_dataset.obsm["predicted_gene_expression"] = clf_resnet.predict(test_dataset.obsm["resnet50_features"])

    test_dataset_ = test_dataset[:, comm_genes].copy()
    test_dataset_.X = test_dataset_.obsm["predicted_gene"]
    test_dataset_z_ = test_dataset_.copy()
    test_dataset_z_.X = zscore(test_dataset_z_.X, axis=0)
    for gene in comm_genes:
        gene_plot(test_dataset_, genes=gene, spot_size=6, image_scale=10,
                  output=str(OUT_PATH), name="Predicted_{}_external.png".format(gene))
        gene_plot(test_dataset_z_, genes="COX6C", spot_size=6, vmin=-1, vmax=1, image_scale=10,
                  output=str(OUT_PATH), name="Predicted_{}_external.png".format(gene))
