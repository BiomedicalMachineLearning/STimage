#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--Data_Name', type=str, default='Pfizer', help='Data source')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate.')
parser.add_argument('--max_epochs', type=int, default=300, help='maximum number of epochs.')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of dataset.')
parser.add_argument('--hidden_dim', type=int, default=512, help='hidden dimension in contrastive embedding.')
parser.add_argument('--seed', type=int, default=42, help='Randomseeds for reproducibility.')

args = parser.parse_args()


# # Read adata

import scanpy as sc

# Read the correct adata
import pickle

Data_Name = args.Data_Name # ["Peter", "Pfizer"]

if Data_Name == "Pfizer":
    names = ["QMDL01", "QMDL02", "QMDL03","QMDL04","QMDL05"]
    
    with open('adata_5samples.pkl', 'rb') as file:
        adata_dict = pickle.load(file)

    with open('Images_5samples.pkl', 'rb') as file:
        img_dic = pickle.load(file)
        
elif Data_Name == "Peter":
    names = ["1302A", "1956A", "4851A"]
    with open('adata_3samples.pkl', 'rb') as file:
        adata_dict = pickle.load(file)
    
    with open('Images_dic_Pfizer.pkl', 'wb') as file:
        pickle.dump(imgs, file)

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


# In[41]:


import numpy as np
import torch
import torchvision.transforms as tf

class Adataloader(torch.utils.data.Dataset):
    def __init__(self, adata, name, img_dic, r=29):
        super(Adataloader, self).__init__()
        """        
        Image shape [N, 3, 64, 64]
        Gene expression shape [N, 280]
        Cell type label [N, 1]
        """
        self.ct_x = adata.obs["x_transformed"].astype(int)
        self.ct_y = adata.obs["y_transformed"].astype(int)
        self.r = r
        self.adata = adata
        self.name = name
        self.img_dic = img_dic
        self.lbl2id = {
                     'CAFs':0, 
                     'Cancer Epithelial':1, 
                     'Myeloid':2, 
                     'Normal Epithelial':3, 
                     'T-cells':4, 
                     'Endothelial':5, 
                     'PVL':6, 
                     'Plasmablasts':7, 
                     'B-cells':8 }
        self.expr = np.array(adata.to_df())
        self.CellTypes = adata.obs['predicted.id'].to_list()
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


# In[42]:


from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

r=29 # Tile size is 58

if Data_Name == "Peter":
    full_train_dataset = ConcatDataset([Adataloader(adata_dict[names[i]], names[i], img_dic, r) for i in range(4)])
    train_size = int(0.7 * len(full_train_dataset))
    test_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])
    test_dataset = Adataloader(adata_dict[names[4]], names[4], img_dic, r)
    
elif Data_Name == "Pfizer":
    full_train_dataset = ConcatDataset([Adataloader(adata_dict[names[i]], names[i], img_dic, r) for i in range(2)])
    train_size = int(0.7 * len(full_train_dataset))
    test_size = len(full_train_dataset) - train_size
    train_dataset, validation_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])
    test_dataset = Adataloader(adata_dict[names[3]], names[3], img_dic, r)
    
import torchvision.transforms as transforms

# Data Augmentation methods
contrast_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)


import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# SimCLR model by 
class SimCLR(L.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = torchvision.models.resnet50(
            pretrained=True,
        )  # num_classes is the output size of the last linear layer
        
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Identity(),
            nn.Linear(2048, hidden_dim),
        )
        self.tf = contrast_transforms

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, patch, mode="train"):
        img1 = self.tf(patch)
        img2 = self.tf(patch)
        imgs = torch.cat([img1, img2], dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        patch, exp, label = batch
        patch = patch.to(torch.float32)
        return self.info_nce_loss(patch, mode="train")

    def validation_step(self, batch, batch_idx):
        patch, exp, label = batch
        patch = patch.to(torch.float32)
        self.info_nce_loss(patch, mode="val")


# Create DataLoader instances for training, validation, and test datasets
tr_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
te_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


import pytorch_lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint


model = SimCLR(hidden_dim=args.hidden_dim, lr=args.lr, temperature=0.07, weight_decay=1e-4, max_epochs=args.max_epochs)

logger = pl.loggers.CSVLogger("./logs", name=f"SimCLR_{Data_Name}_Xenium")
trainer = L.Trainer(
    accelerator="auto",
    max_epochs=args.max_epochs,
    callbacks=[
    ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
    LearningRateMonitor("epoch"),],
    logger = logger,
)

L.seed_everything(args.seed)

trainer.fit(model, tr_loader, val_loader)

torch.save(model.state_dict(),f"./model/SimCLR_{Data_Name}_Xenium.ckpt")



