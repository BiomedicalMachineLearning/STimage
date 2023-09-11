[TOC]



# Dataset Availability

We have 3 samples from Pfizer. We use 1302A and 1956A as training samples, 4851A as testing sample.



## 1302A

![1302A](/Users/yuanhaojiang/Documents/Xenium Project/Figures/1302A.png)

## 1956A 

![1956A](/Users/yuanhaojiang/Documents/Xenium Project/Figures/1956A.png)

## 4851A

![4851A](/Users/yuanhaojiang/Documents/Xenium Project/Figures/4851A.png)

# Exploratory data analysis

* Different view for different crop size.

* Cell Type Distribution

* Cell Area Distribution

* Gene Expression Distribution

  

## Different view for different crop size.

**Tile size=32**

![crop size 16](/Users/yuanhaojiang/Documents/Xenium Project/Figures/crop_r16.png)

**Tile size=64**

![crop size 32](/Users/yuanhaojiang/Documents/Xenium Project/Figures/crop_r32.png)

**Tile size=96**

![crop size 48](/Users/yuanhaojiang/Documents/Xenium Project/Figures/crop_r48.png)

## Cell Type Distribution

![CellTypeDistribution](/Users/yuanhaojiang/Documents/Xenium Project/Figures/CellTypeDistribution.png)

## Cell Area Distribution

### 1302A

![CellArea4851A](/Users/yuanhaojiang/Documents/Xenium Project/Figures/CellArea-1302A.png)



### 1956A

![CellArea1956A](/Users/yuanhaojiang/Documents/Xenium Project/Figures/CellArea-1956A.png)



### 4851A

![CellArea4851A](/Users/yuanhaojiang/Documents/Xenium Project/Figures/CellArea-4851A.png)



### 



## Gene Expression Distribution





![AGG Number of genes detected](/Users/yuanhaojiang/Downloads/AGG Number of genes detected.png)



# Dataset Preprocessing

## Remove the tiles with zero or low gene expression values.

Threshold=

|                 | Before | After |
| --------------- | ------ | ----- |
| Number of Tiles |        |       |







## Remove the tiles with outlier cell area.

Threshold=

|                 | Before | After |
| --------------- | ------ | ----- |
| Number of Tiles |        |       |





## Down-Scaling the dominant cell types & Up-Scaling the minor cell types.

### Original 

|                   | Number of Tiles |
| ----------------- | --------------- |
| CAFs              |                 |
| Cancer Epithelial |                 |
| Endothelial       |                 |
| Myeloid           |                 |
| PVL               |                 |
| Plasmablasts      |                 |
| T-cells           |                 |

### Down scaling dominant cell types

|                   | Number of Tiles |
| ----------------- | --------------- |
| CAFs              |                 |
| Cancer Epithelial |                 |
| Endothelial       |                 |
| Myeloid           |                 |
| PVL               |                 |
| Plasmablasts      |                 |
| T-cells           |                 |

### Up Scaling minor cell types

|                   | Number of Tiles |
| ----------------- | --------------- |
| CAFs              |                 |
| Cancer Epithelial |                 |
| Endothelial       |                 |
| Myeloid           |                 |
| PVL               |                 |
| Plasmablasts      |                 |
| T-cells           |                 |



# Data Augmentation

* Gemetric Transformation
* Color Transformation

## Gemetric Transformation

### Flipping





### Rotation





## Color Transformation

### Color Augmentation

H&E intensity augmentation





### Color Normalization

Reinhard normalization





# Model

## Baseline

### Resent18



### Resnet50



### Swin-Transformer



## Contrastive Learning (Xenium Dataset)

### CLIP ML Model

![CLIPFC](/Users/yuanhaojiang/Documents/Xenium Project/Figures/CLIP_FC.png)

### CLIP FC Model (FineTune Image Encoder)

![CLIPFC](/Users/yuanhaojiang/Documents/Xenium Project/Figures/CLIP_FC.png)





## Contrastive Learning (Xenium Dataset+SC Dataset)

### CLIP ML Model

Drawbacks:

* Can not finetune image encoder during training machine learning models.
* Sklearn based machine learning model can not use CUDA to speed up training process.



### CLIP FC Model (FineTune Image Encoder)

Benefits:

* Contrastive learning can align the latent space between H&E image and gene expression. Image encoder can learn some features from gene expression.
* Finetune the image encoder during supervised training can make the image encoder more label oriented.

## Multi-modal fusion

Feature Fusion Methods:

* Summation

* Concatenation

* Summation Attention

* Concatenation Attention

![Multi modal fusion](/Users/yuanhaojiang/Documents/Xenium Project/Figures/Multi modal fusion.png)

### Summation



### Concatenation



### Summation Attention



### Concatenation Attention





## Deep Kernel Learning

### SVDKL (Stochastic Variational Deep Kernel Learning)

Benefits:

* 

**Pending**

Reference:
https://pyro.ai/examples/dkl.html

https://docs.gpytorch.ai/en/stable/examples/06_PyTorch_NN_Integration_DKL/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.html



# Interpretability Assessment

## Integrated Gradients to analyze feature importance

https://captum.ai/tutorials/Resnet_TorchVision_Ablation

**Pending**



# Uncertatinty Assessment

**Pending**
$$
Uncertatinty Score = 1 - Prediction Probability
$$
