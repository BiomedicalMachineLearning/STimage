import argparse
import configparser
import sys
from pathlib import Path

import pandas as pd
import stlearn as st
from PIL import Image
# file=Path("./stimage").resolve()
# parent=file.parent
# sys.path.append(str(parent))
# print(parent)
from _img_normaliser import IterativeNormaliser
from _utils import tiling, ensembl_to_id, ReadOldST, Read10X, scale_img, calculate_bg

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STimage software --- Preprocessing")
    parser.add_argument('--config', dest='config', type=Path,
                        help='Path to config file')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config)
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

    output_file = Path(DATA_PATH / "all_adata.h5ad")
    if output_file.exists():
        print("{} already exists".format(str(output_file)))
        sys.exit()
    else:
        adata_list = []
        normaliser = None
        for index, row in meta.iterrows():
            img_path = row["histology_image"]
            Sample = row["sample"]

            if platform == "Old_ST":
                cm_path = row["count_matrix"]
                spot_path = row["spot_coordinates"]

                adata = ReadOldST(count_matrix_file=cm_path,
                                  spatial_file=spot_path,
                                  image_file=img_path,
                                  library_id=Sample,
                                  quality="fulres")

            elif platform == "Visium":
                cm_file = row["count_matrix_h5_file"]
                visium_path = row["Visium_folder"]
                adata = Read10X(path=visium_path,
                                count_file=cm_file,
                                library_id=Sample,
                                source_image_path=img_path,
                                quality="fulres")

            if stain_normalization and Sample == template_sample:
                print("fitting template image for stain normalizer")
                template_img = adata.uns["spatial"][Sample]['images']["fulres"]
                template_img = Image.fromarray(template_img.astype("uint8"))

                normaliser = IterativeNormaliser(normalisation_method='vahadane', standardise_luminosity=True)
                normaliser.fit_target(scale_img(template_img))

            if normalization == "log":
                st.pp.log1p(adata)

            adata_list.append(adata)

        for i, adata in enumerate(adata_list):
            if tile_filtering_threshold < 1:
                print("filtering out tiles where tissue area less than {} of total tile area".format(
                    tile_filtering_threshold))
                calculate_bg(adata, crop_size=tile_size, stain_normaliser=normaliser)
                tile_to_remove = sum(adata.obs["tissue_area"] < tile_filtering_threshold)
                adata = adata[adata.obs["tissue_area"] >= tile_filtering_threshold].copy()
                print("{} tiles with low tissue coverage are removed".format(tile_to_remove))
                adata_list[i] = adata

            tiling(adata, out_path=TILING_PATH, crop_size=tile_size, stain_normaliser=normaliser)

        adata_all = adata_list[0].concatenate(
            adata_list[1:],
            batch_key="library_id",
            uns_merge="unique",
            batch_categories=[list(d.keys())[0]
                              for d in [adata_list[i].uns["spatial"]
                                        for i in range(len(adata_list))
                                        ]
                              ],
        )
        if convert_ensembl:
            adata_all = ensembl_to_id(adata_all)
        adata_all.write(output_file)
