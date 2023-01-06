import torch
import numpy as np
import pytorch_lightning as pl
import torchvision.transforms as tf
from tqdm import tqdm
from predict import *
from HIST2ST import *
from dataset import ViT_HER2ST, ViT_SKIN
from scipy.stats import pearsonr,spearmanr
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from copy import deepcopy as dcp
import pickle
from collections import defaultdict as dfd
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score

def calculate_correlation(attr_1, attr_2):
    r = pearsonr(attr_1, 
                       attr_2)[0]
    return r

name=[*[f'A{i}' for i in range(2,7)],*[f'B{i}' for i in range(1,7)],
      *[f'C{i}' for i in range(1,7)],*[f'D{i}' for i in range(1,7)],
      *[f'E{i}' for i in range(1,4)],*[f'F{i}' for i in range(1,4)],*[f'G{i}' for i in range(1,4)]]
patients = ['P2', 'P5', 'P9', 'P10']
reps = ['rep1', 'rep2', 'rep3']
skinname = []
for i in patients:
    for j in reps:
        skinname.append(i+'_ST_'+j)
device='cuda'
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

cnt_dir = 'data/her2st/data/ST-cnts'
names = os.listdir(cnt_dir)
names.sort()
names = [i[:2] for i in names]

df = pd.DataFrame()
i = int(sys.argv[1])
# i=1
fold = i
test_sample = names[fold]

data='her2st'
prune='Grid' if data=='her2st' else 'NA'
genes=171 if data=='cscc' else 785
trainset = pk_load(fold,'train',dataset=data,flatten=False,adj=True,ori=True,prune=prune)
train_loader = DataLoader(trainset, batch_size=1, num_workers=0, shuffle=True)

model=Hist2ST(
    depth1=d1, depth2=d2,depth3=d3,n_genes=genes,
    kernel_size=k, patch_size=p,
    heads=h, channel=c, dropout=0.2,
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
if not os.path.isdir("./model/"):
    os.mkdir("./model/")

torch.save(model.state_dict(),f"./model/{fold}-Hist2ST.ckpt")
testset = pk_load(fold,'test',dataset=data,flatten=False,adj=True,ori=True,prune=prune)
test_loader = DataLoader(testset, batch_size=1, num_workers=0, shuffle=False)
adata_pred, adata_truth = test(model, test_loader,'cuda')
gene_list = list(np.load('data/her_hvg_cut_1000.npy',allow_pickle=True))
n_genes = len(gene_list)

adata_pred.var_names = gene_list
adata_truth.var_names = gene_list

pred_adata = adata_pred.copy()
test_dataset = adata_truth.copy()

for gene in pred_adata.var_names:
    cor_val = calculate_correlation(pred_adata.to_df().loc[:,gene], test_dataset.to_df().loc[:,gene])
    df = df.append(pd.Series([gene, cor_val, test_sample, "Hist2ST"], 
                         index=["Gene", "Pearson correlation", "Slide", "Method"]),
              ignore_index=True)

del model
torch.cuda.empty_cache()

df.to_csv("../stimage_compare_histogene_1000hvg/hist2ST_cor_{}.csv".format(test_sample))
