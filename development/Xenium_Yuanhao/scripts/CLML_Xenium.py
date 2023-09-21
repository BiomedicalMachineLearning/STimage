#!/usr/bin/env python
# coding: utf-8

# # Read adata

names = ["1302A", "1956A", "4851A"]


# Read the correct adata
import pickle
  
with open('adata_3samples.pkl', 'rb') as file:
    adata_dict = pickle.load(file)


from squidpy.im import ImageContainer 
import numpy as np
import torch

img_paths = ["1302A_0007311_he.ome.tif", "1956A_0007330_he.ome.tif", "4851A_0007473_he.ome.tif"]
img_source_path = "/scratch/imb/uqyjia11/Yuanhao/xenium/image/"

with open('Images_dic.pkl', 'rb') as file:
    img_dic = pickle.load(file)


# # Tiling

import torch
import torchvision.transforms as tf

class Adataloader(torch.utils.data.Dataset):
    def __init__(self, adata, name, img_dic, r=32):
        super(Adataloader, self).__init__()
        """        
        Image shape [N, 3, 64, 64]
        Gene expression shape [N, 280]
        Cell type label [N, 1]
        """
        self.ct_x = adata.obs["Centroid X"].astype(int)
        self.ct_y = adata.obs["Centroid Y"].astype(int)
        self.r = r
        self.adata = adata
        self.name = name
        self.img_dic = img_dic
        self.lbl2id = {'B-cells':0,
                         'CAFs':1,
                         'Cancer Epithelial':2,
                         'Endothelial':3,
                         'Myeloid':4,
                         'PVL':5,
                         'Plasmablasts':6,
                         'T-cells':7}
        self.expr = adata.X
        self.CellTypes = adata.obs['celltype_major'].to_list()
        self.classes = [self.lbl2id[v] for v in self.CellTypes]

    def __getitem__(self, index):
        x, y = self.ct_x[index], self.ct_y[index]
        self.img_dic = img_dic
        img = self.img_dic[self.name]
        patch = img[:,(y-self.r):(y+self.r),(x-self.r):(x+self.r)]
        expr = torch.tensor(self.expr[index])
        classs = torch.tensor(self.classes[index])
        data = [patch, expr, classs]
        return data
        
    def __len__(self):
        return len(self.ct_x)


from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

r=32
print(names)
full_train_dataset = ConcatDataset([Adataloader(adata_dict[names[0]], names[0], img_dic, r), Adataloader(adata_dict[names[1]], names[1], img_dic, r)])
train_size = int(0.7 * len(full_train_dataset))
test_size = len(full_train_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])
test_dataset = Adataloader(adata_dict[names[2]], names[2], img_dic, r)
tr_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)
te_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


import psutil

# Get CPU usage percentage
cpu_usage = psutil.cpu_percent(interval=1)  # Interval is in seconds
print(f"CPU Usage: {cpu_usage}%")

# Get RAM usage
ram = psutil.virtual_memory()
print(f"Total RAM: {ram.total / (1024 ** 3):.2f} GB")
print(f"Available RAM: {ram.available / (1024 ** 3):.2f} GB")
print(f"Used RAM: {ram.used / (1024 ** 3):.2f} GB")
print(f"RAM Usage Percentage: {ram.percent}%")


# # Define the model

# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as tf
import random
import timm
import pytorch_lightning as pl

from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from copy import deepcopy as dcp
from collections import defaultdict as dfd
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# In[8]:


class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name='resnet50', pretrained=True, trainable=True
    ):
        super().__init__()
        if model_name == "resnet50":
            self.model = timm.create_model(model_name, pretrained, num_classes=0, global_pool="avg")

        elif model_name == "resnet18":
            self.model = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT)
            self.model.fc = nn.Identity()

        elif model_name == "swin_v2_t":
            self.model = torchvision.models.swin_v2_t(weights=torchvision.models.Swin_V2_T_Weights.DEFAULT)
            self.model.head=nn.Identity()

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


# In[9]:


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.2
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x


# In[10]:


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


# In[11]:


class CLIPModel_CLIP(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        model_name="resnet18",
        temperature=1.0,
        image_embedding=512,
        spot_embedding=280,
        projection_dim=256,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.image_encoder = ImageEncoder(model_name) # Image encoder
        self.image_projection = ProjectionHead(embedding_dim=image_embedding, projection_dim=projection_dim) # Image projection
        self.spot_projection = ProjectionHead(embedding_dim=spot_embedding, projection_dim=projection_dim) # Gene expression encoder
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and spot Features
        image_features = self.image_encoder(batch["image"])
        spot_features = batch["reduced_expression"]

        # Getting Image and Spot Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features) # Resnet50 extraction for img
        spot_embeddings = self.spot_projection(spot_features) # MLP projection for exp

        # Calculating the Contrastive Loss
        logits = (spot_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        spots_similarity = spot_embeddings @ spot_embeddings.T
        targets = F.softmax((images_similarity + spots_similarity) / 2 * self.temperature, dim=-1)
        
        spots_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + spots_loss) / 2.0

        return loss.mean()

    def training_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        exp = exp.to(torch.float32)
        patch = tf.Resize(224)(patch.to(torch.float32))
        batch = {
        'image': patch,
        'reduced_expression': exp,
        'label': label,
        }

        loss = self(batch)
        self.log('Train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
   
    def validation_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        exp = exp.to(torch.float32)
        patch = tf.Resize(224)(patch.to(torch.float32))
        batch = {
        'image': patch,
        'reduced_expression': exp,
        'label': label,
        }

        loss = self(batch)
        self.log('Validation_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        exp = exp.to(torch.float32)
        patch = tf.Resize(224)(patch.to(torch.float32))
        batch = {
        'image': patch,
        'reduced_expression': exp,
        'label': label,
        }

        loss = self(batch)
        self.log('Test_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optim_dict = {'optimizer': optim}
        return optim_dict


import gc
gc.collect()

model_name = "resnet18"
if model_name== "resnet50":
    image_embedding=2048
elif model_name== "resnet18":
    image_embedding=512
elif model_name== "swin_v2_t":
    image_embedding=768
    
model = CLIPModel_CLIP(model_name=model_name, image_embedding=image_embedding)
tr_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
logger = pl.loggers.CSVLogger("./logs", name=f"CLIP-{model_name}_Xenium")
trainer = pl.Trainer(accelerator='auto', callbacks=[EarlyStopping(monitor='Validation_loss',mode='min')], min_epochs=10, logger=logger)
trainer.fit(model, tr_loader, val_loader)
torch.save(model.state_dict(),f"./model/CLML-{model_name}_Xenium.ckpt")


