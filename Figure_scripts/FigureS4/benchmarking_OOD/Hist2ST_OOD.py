

import torch
from pathlib import Path
from anndata import read_h5ad
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
from tqdm import tqdm
from predict_OOD import *
from HIST2ST import *
from dataset import ViT_HER2ST, ViT_SKIN
from dataset_OOD import ViT_OOD
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score



import torch
torch.cuda.empty_cache()



def calculate_correlation(attr_1, attr_2):
    r = pearsonr(attr_1, 
                       attr_2)[0]
    return r




device='cuda'
# device='cpu'
tag='5-7-2-8-4-16-32'
k,p,d1,d2,d3,h,c=map(lambda x:int(x),tag.split('-'))
dropout=0.2
random.seed(12000)
np.random.seed(12000)
torch.manual_seed(12000)
torch.cuda.manual_seed(12000)
torch.cuda.manual_seed_all(12000)  
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



fold=2
data='OOD'
prune='NA'
genes=171 if data=='cscc' else 982
trainset = pk_load(fold,'train',dataset=data,flatten=False,adj=True,ori=True,
                   prune=prune)
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)




model=Hist2ST(
    depth1=d1, depth2=d2,depth3=d3,n_genes=genes,
    kernel_size=k, patch_size=p, n_pos=64,
    heads=h, channel=c, dropout=0.2, learning_rate=1e-8,
    zinb=0.25, nb=False,
    bake=5, lamb=0.5, 
)

logger=None
trainer = pl.Trainer(
    gpus=[0], max_epochs=350,
    logger=logger,
)
trainer.fit(model, train_loader)




import os
if not os.path.isdir("./model_OOD/"):
    os.mkdir("./model_OOD/")

torch.save(model.state_dict(),f"./model_OOD/{fold}-Hist2ST_1000HVG.ckpt")




model.load_state_dict(torch.load(f'./model_OOD/{fold}-Hist2ST_1000HVG.ckpt'))




testset = pk_load(fold,'test',dataset=data,flatten=False,adj=True,ori=True,prune=prune)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
# adata_pred, adata_truth = test(model, test_loader,'cuda')




adata_pred_list = []
adata_truth_list = []
for data in test_loader:
    
    adata_pred, adata_truth = test(model, [data],'cuda')
    adata_pred_list.append(adata_pred)
    adata_truth_list.append(adata_truth)




df_truth_all = pd.DataFrame()
df_truth_all_obs = pd.DataFrame()
for i in adata_truth_list:
    df_truth_all = df_truth_all.append(i.to_df())
    df_truth_all_obs = df_truth_all_obs.append(i.obs)
adata_truth_all = AnnData(df_truth_all, obs=df_truth_all_obs)



df_pred_all = pd.DataFrame()
df_pred_all_obs = pd.DataFrame()
for i in adata_pred_list:
    df_pred_all = df_pred_all.append(i.to_df())
    df_pred_all_obs = df_pred_all_obs.append(i.obs)
adata_pred_all = AnnData(df_pred_all, obs=df_pred_all_obs)




gene_list = list(np.load('/scratch/imb/Xiao/STimage/development/stimage_compare_histogene_1000hvg/gene_list_OOD.pkl',allow_pickle=True))
n_genes = len(gene_list)
adata_pred_all.var_names = gene_list
adata_truth_all.var_names = gene_list
df = pd.DataFrame()
pred_adata = adata_pred_all.copy()
test_dataset = adata_truth_all.copy()
for gene in pred_adata.var_names:
    cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val, "FFPE", "Hist2ST"], 
                         index=["Gene", "Pearson correlation", "Slide", "Method"]),
              ignore_index=True)




pred_adata.write_h5ad("/clusterdata/uqxtan9/Xiao/STimage/development/stimage_benchmarking_1000hvg_OOD/Hist2ST_pred_adata.h5ad")
test_dataset.write_h5ad("/clusterdata/uqxtan9/Xiao/STimage/development/stimage_benchmarking_1000hvg_OOD/Hist2ST_test_adata.h5ad")




df.to_csv("../stimage_benchmarking_1000hvg_OOD/hist2ST_cor_{}.csv".format("FFPE"))

