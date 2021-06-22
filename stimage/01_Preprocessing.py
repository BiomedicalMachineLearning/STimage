import argparse
import configparser
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import stlearn as st

from ._utils import tiling, ensembl_to_id

st.settings.set_figure_params(dpi=300)

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
    TILING_PATH = Path(config["PATH"]["TILING_PATH"])
    convert_ensembl = config["DATASET"].getboolean("ensembl_to_id")
    platform = config["DATASET"]["platform"]
    TILING_PATH.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(META_PATH)

    output_file = Path(DATA_PATH / "all_adata.h5ad")
    if output_file.exists():
        sys.exit()
    else:
        adata_list = []
        for index, row in meta.iterrows():
            cm_path = row["count_matrix"]
            spot_path = row["spot_coordinates"]
            img_path = row["histology_image"]
            Sample = row["sample"]
            if platform == "Old_ST":
                adata = st.read.file_table(cm_path)
                spot_df = pd.read_csv(spot_path, index_col=0)
                comm_index = pd.Series(list(set(spot_df.index).intersection(set(adata.obs_names))))
                adata = adata[comm_index]
                adata.obs["imagecol"] = spot_df["X"]
                adata.obs["imagerow"] = spot_df["Y"]
                st.add.image(adata, img_path, library_id=Sample)
                # adata.obs["type"] = row["type"]
            elif platform == "Visium":
                adata = st.Read10X(cm_path,
                                     library_id=Sample,
                                     quality="fulres", )
                # source_image_path=BASE_PATH / SAMPLE /"V1_Breast_Cancer_Block_A_Section_1_image.tif")
                adata.uns["spatial"][Sample]['images']["fulres"] = plt.imread(img_path, 0)
            if normalization == "log":
                st.pp.log1p(adata)
            tiling(adata, out_path=TILING_PATH, crop_size=299)
            adata_list.append(adata)

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
