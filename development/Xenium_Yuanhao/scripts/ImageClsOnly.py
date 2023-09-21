#!/usr/bin/env python
# coding: utf-8
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='vgg16', help='Backbone')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
parser.add_argument('--max_epochs', type=int, default=100, help='minimum number of epochs.')

args = parser.parse_args()



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

# Define a function to calculate weights for balanced classes in a dataset
def make_weights_for_balanced_classes(vec_classes, nclasses):
    # Initialize a count list for each class
    count = [0] * nclasses
    
    # Count the occurrences of each class in the dataset
    for i in range(len(vec_classes)):
        count[vec_classes[i].item()] += 1
    
    # Calculate the weight for each class to balance the dataset
    weight_per_class = [0.] * nclasses 
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N / float(count[i])                                 
    
    # Create a list of weights for each sample in the dataset
    weight_for_sample = [0] * len(vec_classes)   
    for i in range(len(vec_classes)):
        c = vec_classes[i].item()
        weight_for_sample[i] = weight_per_class[c]
    
    return weight_for_sample

# In[6]:

"""Default"""
# from torch.utils.data import DataLoader
# from torch.utils.data import ConcatDataset

# r=32
# print(names)
# full_train_dataset = ConcatDataset([Adataloader(adata_dict[names[0]], names[0], img_dic, r)])
# train_size = int(0.7 * len(full_train_dataset))
# test_size = len(full_train_dataset) - train_size
# train_dataset, validation_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])
# test_dataset = Adataloader(adata_dict[names[1]], names[1], img_dic, r)
# tr_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# val_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
# te_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

"""Weighted Random Sample Dataset"""
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

r=32
print(names)
full_train_dataset = ConcatDataset([Adataloader(adata_dict[names[1]], names[1], img_dic, r), Adataloader(adata_dict[names[2]], names[2], img_dic, r)])
train_size = int(0.7 * len(full_train_dataset))
test_size = len(full_train_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])
test_dataset = Adataloader(adata_dict[names[0]], names[0], img_dic, r)

# Create an empty list to store class labels
list_classes = []

# Iterate through the training dataset and collect class labels
for patch, expr, c in iter(train_dataset):
    list_classes.append(c)                                                                         
vec_classes = torch.stack(list_classes, axis=0)                                                      

# For an unbalanced dataset, create a weighted sampler
# Calculate weights for each class in the dataset
weights = make_weights_for_balanced_classes(vec_classes, 8)                                                             
weights = torch.DoubleTensor(weights)                                       

# Create a WeightedRandomSampler for the DataLoader
sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

# Create DataLoader instances for training, validation, and test datasets
tr_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
val_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)
te_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


# In[7]:


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

# In[8]:


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
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# In[9]:


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

        elif model_name == "vgg16":
            self.model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
            self.model.classifier[6]=nn.Identity()
            
        elif model_name == "densenet121":
            self.model = torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT)
            self.model.classifier=nn.Identity()

        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


# In[10]:


class Classifier(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        num_class=8,
        dropout=0.2,
    ):
        super().__init__()
        self.cls = nn.Linear(embedding_dim, num_class)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(num_class, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cls = self.cls(x)
        x = self.gelu(cls)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + cls
        x = nn.Softmax(dim=1)(x)
        return x


# In[11]:


class ImageClassifier(pl.LightningModule):
#     pl.LightningModule
    def __init__(
        self,
        learning_rate=1e-3,
        model_name="resnet18",
        temperature=1.0,
        image_embedding=512,
        num_class=8,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.image_encoder = ImageEncoder(model_name) # Image encoder
        self.classifier = Classifier(embedding_dim=image_embedding, num_class=num_class)
            
    def forward(self, patch):
        # Getting Image and spot Features
        image_features = self.image_encoder(tf.Resize(224)(patch))
        
        # Classification
        pred = self.classifier(image_features)

        return pred

    def training_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        patch = tf.Resize(224)(patch.to(torch.float32))
        pred = self(patch)
        loss = F.cross_entropy(pred, label.view(-1).long())
        self.log('Train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        patch = tf.Resize(224)(patch.to(torch.float32))
        pred = self(patch)
        loss = F.cross_entropy(pred, label.view(-1).long())
        self.log('Validation_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        patch = tf.Resize(224)(patch.to(torch.float32))
        
        pred_prob = self(patch)
        _, pred = pred_prob.topk(1, dim=1)
        acc = (pred == label.view(-1)).float().mean()
        self.log('Test_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)
        return acc

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optim_dict = {'optimizer': optim}
        return optim_dict


# In[15]:


import gc
gc.collect()


# In[16]:

model_name = args.model_name
fusion_method = "ImageOnly"

if model_name== "resnet50":
    image_embedding=2048
elif model_name== "resnet18":
    image_embedding=512
elif model_name== "densenet121":
    image_embedding=1024
elif model_name== "vgg16":
    image_embedding=4096
    
    
model = ImageClassifier(model_name=args.model_name, image_embedding=image_embedding, learning_rate=args.lr,)

logger = pl.loggers.CSVLogger("./logs", name=f"{model_name}_3samples_Xenium")

trainer = pl.Trainer(accelerator='auto', callbacks=[EarlyStopping(monitor='Validation_loss',mode='min')], max_epochs=args.max_epochs, logger=logger)

trainer.fit(model, tr_loader, val_loader)
torch.save(model.state_dict(),f"./model/{model_name}_3samples_Xenium.ckpt")
trainer.test(model, te_loader)

# Prediction
import tqdm

model.eval()
model.cuda()
prediction = []
gt = []
prob = []
with torch.no_grad():
    for i in tqdm.tqdm(range(len(test_dataset))):
        patch, exp, label = test_dataset[i]
        patch = tf.Resize(224)(patch.to(torch.float32)).unsqueeze(0)
        pred_prob = model(patch.cuda())
        _, pred = pred_prob.topk(1, dim=1)
        prediction.append(pred.cpu().squeeze(0)[0].numpy())
        gt.append(label.numpy())
        prob.append(pred_prob.squeeze(0).cpu().numpy())
        
# Save adata with coordinates correction
import pickle
    
with open(f'./Prediction_results/{model_name}_preds(1302A).pkl', 'rb') as file:
    prediction1 = pickle.load(file)
    
    
# Map the id to cell types
lbl2id = {'B-cells':0,
         'CAFs':1,
         'Cancer Epithelial':2,
         'Endothelial':3,
         'Myeloid':4,
         'PVL':5,
         'Plasmablasts':6,
         'T-cells':7}
id2lbl = {v: k for k, v in lbl2id.items()}

name = "1302A"
adata_dict[name].obs["predicted_celltypes"] = [id2lbl[int(v)] for v in prediction]

import scanpy as sc
import matplotlib.pyplot as plt
gc.collect()
sc.pl.spatial(adata_dict[name], color="predicted_celltypes", save=f"CT-PRED-{model_name}-{name}-3samples.png", title=f"{model_name} Prediction ({name})",spot_size=50)

plt.show()
