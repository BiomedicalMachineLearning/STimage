from pathlib import Path
import sys
sys.path.append("/clusterdata/uqxtan9/Xiao/HisToGene")


import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from vis_model import HisToGene
from utils import *
from predict import model_predict, sr_predict
from dataset import ViT_HER2ST, ViT_SKIN

import pickle
from anndata import read_h5ad
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import read_tiff
import numpy as np
import torchvision
import torchvision.transforms as transforms
import scanpy as sc
from utils import get_data
import os
import glob
from PIL import Image
import pandas as pd 
import scprep as scp
from PIL import ImageFile
import seaborn as sns
import matplotlib.pyplot as plt
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ViT_HER2ST(torch.utils.data.Dataset):
    """Some Information about HER2ST"""
    def __init__(self,train=True,gene_list=None,ds=None,sr=False,fold=0):
        super(ViT_HER2ST, self).__init__()
        self.andata_dir = '/clusterdata/uqxtan9/Xiao/dataset_3_224_no_norm/all_adata_window_51.h5ad'
        
#         self.cnt_dir = 'data/her2st/data/ST-cnts'
#         self.img_dir = 'data/her2st/data/ST-imgs'
#         self.pos_dir = 'data/her2st/data/ST-spotfiles'
#         self.lbl_dir = 'data/her2st/data/ST-pat/lbl'
        self.r = 224//4

        # gene_list = list(np.load('data/her_hvg.npy',allow_pickle=True))
        # gene_list=["COX6C","TTLL12", "PABPC1", "GNAS", "HSP90AB1", 
        #    "TFF3", "ATP1A1", "B2M", "FASN", "SPARC", "CD74", "CD63", "CD24", "CD81"]
        gene_list = list(np.load('/scratch/imb/Xiao/STimage/development/stimage_compare_histogene_1000hvg/gene_list_OOD.pkl',allow_pickle=True))
        self.gene_list = gene_list
        
        self.anndata = read_h5ad(self.andata_dir)[:,self.gene_list]
        names = self.anndata.obs["sub_sample"].unique().to_list()
        
#         names = os.listdir(self.cnt_dir)
#         names.sort()
#         names = [i[:2] for i in names]
        self.train = train
        self.sr = sr
        
        # samples = ['A1','B1','C1','D1','E1','F1','G2','H1']
        samples = names

        lib_id = self.anndata.obs["library_id"].unique().to_list()[fold]
        te_names = self.anndata.obs["sub_sample"][self.anndata.obs["library_id"]==lib_id].unique().to_list()
        print(te_names)
        tr_names = list(set(samples)-set(te_names))

        if train:
            self.names = tr_names
        else:
            self.names = te_names

        print('Loading imgs...')
        self.img_dict = {i:torch.Tensor(np.array(self.get_img(i))) for i in self.names}
        self.label={i:None for i in self.names}
        self.gene_set = list(gene_list)
        self.exp_dict = {i:self.get_cnt(i) for i in self.names}
        self.pos_dict = {i:self.get_pos(i) for i in self.names}
        
        self.center_dict = {
            i:np.floor(m[['imagecol','imagerow']].values).astype(int) 
            for i,m in self.pos_dict.items()
        }
        self.loc_dict = {i:m[['array_col_','array_row_']].values for i,m in self.pos_dict.items()}

        self.lengths = [len(i) for i in self.pos_dict.values()]
        self.cumlen = np.cumsum(self.lengths)
        self.id2name = dict(enumerate(self.names))

        self.transforms = transforms.Compose([
            transforms.ColorJitter(0.5,0.5,0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=180),
            transforms.ToTensor()
        ])

    def filter_helper(self):
        a = np.zeros(len(self.gene_list))
        n = 0
        for i,exp in self.exp_dict.items():
            n += exp.shape[0]
            exp[exp>0] = 1
            for j in range((len(self.gene_list))):
                a[j] += np.sum(exp[:,j])


    def __getitem__(self, index):
        i = index
        ID=self.id2name[index]
        im = self.img_dict[self.id2name[i]]
        im = im.permute(1,0,2)
        # im = torch.Tensor(np.array(self.im))
        exps = self.exp_dict[ID].to_numpy()
        centers = self.center_dict[ID]
        loc = self.loc_dict[ID]
        positions = torch.LongTensor(loc)
        patch_dim = 3 * self.r * self.r * 4

        if self.sr:
            centers = torch.LongTensor(centers)
            max_x = centers[:,0].max().item()
            max_y = centers[:,1].max().item()
            min_x = centers[:,0].min().item()
            min_y = centers[:,1].min().item()
            r_x = (max_x - min_x)//30
            r_y = (max_y - min_y)//30

            centers = torch.LongTensor([min_x,min_y]).view(1,-1)
            positions = torch.LongTensor([0,0]).view(1,-1)
            x = min_x
            y = min_y

            while y < max_y:  
                x = min_x            
                while x < max_x:
                    centers = torch.cat((centers,torch.LongTensor([x,y]).view(1,-1)),dim=0)
                    positions = torch.cat((positions,torch.LongTensor([x//r_x,y//r_y]).view(1,-1)),dim=0)
                    x += 56                
                y += 56
            
            centers = centers[1:,:]
            positions = positions[1:,:]

            n_patches = len(centers)
            patches = torch.zeros((n_patches,patch_dim))
            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()


            return patches, positions, centers

        else:    
            n_patches = len(centers)
            
            patches = torch.zeros((n_patches,patch_dim))
            exps = torch.Tensor(exps)


            for i in range(n_patches):
                center = centers[i]
                x, y = center
                patch = im[(x-self.r):(x+self.r),(y-self.r):(y+self.r),:]
                patches[i] = patch.flatten()

            if self.train:
                return patches, positions, exps
            else: 
                return patches, positions, exps, torch.Tensor(centers)
        
    def __len__(self):
        return len(self.exp_dict)

    def get_img(self,name):
        name_ = name.split("_")[0]
        im = Image.fromarray(self.anndata.uns["spatial"][name_]['images']["fulres"])
        return im

    def get_cnt(self,name):
        df = self.anndata[self.anndata.obs.sub_sample == name].to_df()

        return df

    def get_pos(self,name):
        adata = self.anndata[self.anndata.obs.sub_sample == name]
        df = adata.obs[['array_row_', 'array_col_', "imagecol", "imagerow"]]

        return df


import os
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from transformer import ViT
from torch.optim.lr_scheduler import ReduceLROnPlateau

class FeatureExtractor(nn.Module):
    """Some Information about FeatureExtractor"""
    def __init__(self, backbone='resnet101'):
        super(FeatureExtractor, self).__init__()
        backbone = torchvision.models.resnet101(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*layers)
        # self.backbone = backbone
    def forward(self, x):
        x = self.backbone(x)
        return x

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes=4, backbone='resnet50', learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        backbone = torchvision.models.resnet50(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = num_classes
        self.classifier = nn.Linear(num_filters, num_target_classes)
        # self.valid_acc = torchmetrics.Accuracy()
        self.learning_rate = learning_rate

    def forward(self, x):
        # use forward for inference/predictions
        embedding = self.feature_extractor(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log('valid_loss', loss)
        self.log('valid_acc', acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        h = self.feature_extractor(x).flatten(1)
        h = self.classifier(h)
        logits = F.log_softmax(h, dim=1)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        
        self.log('test_loss', loss)
        self.log('test_acc', acc)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        return parser


class STModel(pl.LightningModule):
    def __init__(self, feature_model=None, n_genes=1000, hidden_dim=2048, learning_rate=1e-5, use_mask=False, use_pos=False, cls=False):
        super().__init__()
        self.save_hyperparameters()
        # self.feature_model = None
        if feature_model:
            # self.feature_model = ImageClassifier.load_from_checkpoint(feature_model)
            # self.feature_model.freeze()
            self.feature_extractor = ImageClassifier.load_from_checkpoint(feature_model)
        else:
            self.feature_extractor = FeatureExtractor()
        # self.pos_embed = nn.Linear(2, hidden_dim)
        self.pred_head = nn.Linear(hidden_dim, n_genes)
        
        self.learning_rate = learning_rate
        self.n_genes = n_genes

    def forward(self, patch, center):
        feature = self.feature_extractor(patch).flatten(1)
        h = feature
        pred = self.pred_head(F.relu(h))
        return pred

    def training_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('valid_loss', loss)
        
    def test_step(self, batch, batch_idx):
        patch, center, exp, mask, label = batch
        if self.use_mask:
            pred, mask_pred = self(patch, center)
        else:
            pred = self(patch, center)

        loss = F.mse_loss(pred, exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser


class HisToGene(pl.LightningModule):
    def __init__(self, patch_size=112, n_layers=4, n_genes=1000, dim=1024, learning_rate=1e-4, dropout=0.1, n_pos=64):
        super().__init__()
        # self.save_hyperparameters()
        self.learning_rate = learning_rate
        patch_dim = 3*patch_size*patch_size
        self.patch_embedding = nn.Linear(patch_dim, dim)
        self.x_embed = nn.Embedding(n_pos,dim)
        self.y_embed = nn.Embedding(n_pos,dim)
        self.vit = ViT(dim=dim, depth=n_layers, heads=16, mlp_dim=2*dim, dropout = dropout, emb_dropout = dropout)

        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes)
        )

    def forward(self, patches, centers):
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:,:,0])
        centers_y = self.y_embed(centers[:,:,1])
        x = patches + centers_x + centers_y
        h = self.vit(x)
        x = self.gene_head(h)
        return x

    def training_step(self, batch, batch_idx):        
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('valid_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        patch, center, exp = batch
        pred = self(patch, center)
        loss = F.mse_loss(pred, exp)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        return parser

def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)



from scipy import stats

def plot_correlation(df, attr_1, attr_2):
    r = stats.pearsonr(df[attr_1], 
                       df[attr_2])[0] **2

    g = sns.lmplot(data=df,
        x=attr_1, y=attr_2,
        height=5, legend=True
    )
    # g.set(ylim=(0, 360), xlim=(0,360))

    g.set_axis_labels(attr_1, attr_2)
    plt.annotate(r'$R^2:{0:.2f}$'.format(r),
                (max(df[attr_1])*0.9, max(df[attr_2])*0.9))
    return g


def calculate_correlation(attr_1, attr_2):
    r = stats.pearsonr(attr_1, 
                       attr_2)[0]
    return r

def calculate_correlation_2(attr_1, attr_2):
    r = stats.spearmanr(attr_1, 
                       attr_2)[0]
    return r




gene_list_path = "/scratch/imb/Xiao/STimage/development/stimage_compare_histogene_1000hvg/gene_list_OOD.pkl"
with open(gene_list_path, 'rb') as f:
    gene_list = pickle.load(f)
n_genes = len(gene_list)

df = pd.DataFrame()
i = 2
fold = i
test_sample = "FFPE"
print("process {}".format(test_sample))
tag = '-htg_FFPE_1000_32_cv'




dataset = ViT_HER2ST(train=True, fold=fold)



train_loader = DataLoader(dataset, batch_size=1, num_workers=4, shuffle=True)



model = HisToGene(n_layers=8, n_genes=n_genes, learning_rate=1e-6)
trainer = pl.Trainer(gpus=1, max_epochs=100)
trainer.fit(model, train_loader)



device = torch.device('cuda')
dataset = ViT_HER2ST(train=False,sr=False,fold=fold)
test_loader = DataLoader(dataset, batch_size=1, num_workers=4)




adata_pred_list = []
adata_truth_list = []
for data in test_loader:
    
    adata_pred, adata_truth = model_predict(model, [data],'cuda')
    adata_pred_list.append(adata_pred)
    adata_truth_list.append(adata_truth)




from anndata import AnnData
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
    df = df.append(pd.Series([gene, cor_val, "FFPE", "His2gene"], 
                         index=["Gene", "Pearson correlation", "Slide", "Method"]),
              ignore_index=True)



pred_adata.write_h5ad("/clusterdata/uqxtan9/Xiao/STimage/development/stimage_benchmarking_1000hvg_OOD/His2gene_pred_adata.h5ad")
test_dataset.write_h5ad("/clusterdata/uqxtan9/Xiao/STimage/development/stimage_benchmarking_1000hvg_OOD/His2gene_test_adata.h5ad")



df.to_csv("../stimage_benchmarking_1000hvg_OOD/histogene_cor_{}.csv".format("FFPE"))






