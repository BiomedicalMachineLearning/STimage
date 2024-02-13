
import anndata as ad
import numpy as np
from anndata import read_h5ad




andata_dir = '/clusterdata/uqxtan9/Xiao/dataset_3_224_no_norm/all_adata.h5ad'



gene_list = list(np.load('/scratch/imb/Xiao/STimage/development/stimage_compare_histogene_1000hvg/gene_list_OOD.pkl',allow_pickle=True))





anndata = read_h5ad(andata_dir)[:,gene_list]





anndata.obs["sub_sample"] = "not_assigned"





window = 25
sample = anndata.obs.library_id.unique()
for k in sample:
    adata = anndata[anndata.obs.library_id == k].copy()
#     print(adata.obs["array_row"].max())
#     print(adata.obs["array_col"].max())
#     anndata.obs["sub_sample"] = "{}_{}_{}".format(k,0,0)
    for i in range(adata.obs["array_row"].max()//window +1):
        for j in range((adata.obs["array_col"].max()//window)+1):
            cond = '@i*@window <= array_row <= (@i+1)*@window & @j*@window <= array_col <= (@j+1)*@window'
            print(len(adata.obs.query(cond).index))
            if len(adata.obs.query(cond).index) >0:
                anndata.obs.loc[adata.obs.query(cond).index, "array_col_min"] = anndata.obs.loc[adata.obs.query(cond).index, "array_col"].min()
                anndata.obs.loc[adata.obs.query(cond).index, "array_row_min"] = anndata.obs.loc[adata.obs.query(cond).index, "array_row"].min()
                
#                 anndata.uns["spatial"]["{}_{}_{}".format(k,i,j)] = {}
#                 anndata.uns["spatial"]["{}_{}_{}".format(k,i,j)]['images'] = {}
#                 x1 = anndata.obs.loc[adata.obs.query(cond).index,"imagecol"].min() - 300 
#                 x2 = anndata.obs.loc[adata.obs.query(cond).index,"imagecol"].max() + 300
#                 y1 = anndata.obs.loc[adata.obs.query(cond).index,"imagecol"].min() - 300
#                 y2 = anndata.obs.loc[adata.obs.query(cond).index,"imagecol"].max() + 300
#                 anndata.uns["spatial"]["{}_{}_{}".format(k,i,j)]['images']["fulres"] = anndata.uns["spatial"][k]['images']["fulres"][y1:y2,x1:x2,:]
#                 print(anndata.uns["spatial"]["{}_{}_{}".format(k,i,j)]['images']["fulres"].shape)
                anndata.obs.loc[adata.obs.query(cond).index, "sub_sample"] = "{}_{}_{}".format(k,i,j)
        





# anndata.obs["sub_sample"].unique().tolist()





anndata.obs["array_row_"] = anndata.obs["array_row"] - anndata.obs["array_row_min"]
anndata.obs["array_col_"] = anndata.obs["array_col"] - anndata.obs["array_col_min"]





anndata.write_h5ad("/clusterdata/uqxtan9/Xiao/dataset_3_224_no_norm/all_adata_window_25.h5ad")





anndata.obs["sub_sample"][anndata.obs["library_id"]=="FFPE"].unique().to_list()







