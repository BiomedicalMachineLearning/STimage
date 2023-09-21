import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='vgg16', help='Backbone')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate.')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum number of epochs.')

args = parser.parse_args()

# # Read adata

# In[1]:


names = ["1302A", "1956A", "4851A"]


# In[2]:


# Read the correct adata
import pickle
  
with open('adata_3samples.pkl', 'rb') as file:
    adata_dict = pickle.load(file)


# In[10]:


from squidpy.im import ImageContainer 
import numpy as np
import torch

img_paths = ["1302A_0007311_he.ome.tif", "1956A_0007330_he.ome.tif", "4851A_0007473_he.ome.tif"]
img_source_path = "/scratch/imb/uqyjia11/Yuanhao/xenium/image/"

with open('Images_dic_Pfizer.pkl', 'rb') as file:
    img_dic = pickle.load(file)


# # Tiling

# In[93]:


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


# In[94]:


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

# In[129]:


"""Weighted Random Sample Dataset"""
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

r=29
full_train_dataset = ConcatDataset([Adataloader(adata_dict[names[0]], names[0], img_dic, r), Adataloader(adata_dict[names[1]], names[1], img_dic, r)])
train_size = int(0.7 * len(full_train_dataset))
test_size = len(full_train_dataset) - train_size
train_dataset, validation_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, test_size])
test_dataset = Adataloader(adata_dict[names[2]], names[2], img_dic, r)

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
te_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# In[131]:


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

# In[132]:


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


# In[133]:


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


# In[134]:


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


# In[135]:


class Regressor(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        num_genes=280,
        dropout=0.2,
    ):
        super().__init__()
        self.reg = nn.Linear(embedding_dim, num_genes)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(num_genes, num_genes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        reg = self.reg(x)
        x = self.gelu(reg)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + reg
        return x


# In[230]:


class Regclass(pl.LightningModule):
#   pl.LightningModule
    def __init__(
        self,
        learning_rate=1e-3,
        model_name="resnet18",
        temperature=1.0,
        image_embedding=512,
        num_class=8,
        num_genes=280,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.image_encoder = ImageEncoder(model_name) # Image encoder
        self.classifier = Classifier(embedding_dim=image_embedding, num_class=num_class)
        self.regressor = Regressor(embedding_dim=image_embedding, num_genes=num_genes)
            
    def forward(self, patch):
        # Getting Image and spot Features
        image_features = self.image_encoder(tf.Resize(224)(patch))
        
        # Classification
        pred_cls = self.classifier(image_features)
        
        # Regression
        pred_exp = self.regressor(image_features)

        return pred_cls, pred_exp
    def pred_step(self, patch):
        image_features = self.image_encoder(tf.Resize(224)(patch.to(torch.float32)))
        pred_cls = self.classifier(image_features)
        _, pred = pred_cls.topk(1, dim=1)
        return pred, pred_cls
        

    def training_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        patch = tf.Resize(224)(patch.to(torch.float32))
        pred_cls, pred_exp = self(patch)
        
        loss_cls = F.cross_entropy(pred_cls, label.view(-1).long()).to(torch.float64)
        self.log('Train_classification_loss', loss_cls, on_epoch=True, prog_bar=True, logger=True)
        loss_reg = F.mse_loss(pred_exp.float(), exp.float())
        self.log('Train_regression_loss', loss_reg, on_epoch=True, prog_bar=True, logger=True)
        
        loss = loss_reg + loss_cls
        self.log('Train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        patch = tf.Resize(224)(patch.to(torch.float32))
        pred_cls, pred_exp = self(patch)
        loss = F.cross_entropy(pred_cls, label.view(-1).long())
        self.log('Validation_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """ Load data """
        patch, exp, label = batch
        patch = tf.Resize(224)(patch.to(torch.float32))
        pred_cls, pred_exp = self(patch)
        _, pred = pred_cls.topk(1, dim=1)
        acc = (pred == label.view(-1)).float().mean()
        self.log('Test_accuracy', acc, on_epoch=True, prog_bar=True, logger=True)
        return acc

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optim=torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optim_dict = {'optimizer': optim}
        return optim_dict

    


# In[231]:


import gc
gc.collect()


# # Model Training 

# In[232]:


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
    
    
model = Regclass(model_name=model_name, image_embedding=image_embedding, learning_rate=1e-5,)
logger = pl.loggers.CSVLogger("./logs", name=f"{model_name}_Regclass_3samples_Xenium")
trainer = pl.Trainer(accelerator='auto', callbacks=[EarlyStopping(monitor='Validation_loss',mode='min')], max_epochs=args.max_epochs, logger=logger)
trainer.fit(model, tr_loader, val_loader)
torch.save(model.state_dict(),f"./model/{model_name}_Regclass_Xenium_3samples.ckpt")
trainer.test(model, te_loader)


# In[253]:


# Prediction
import tqdm

model.eval()
model.cuda()
prediction = []
gt = []
prob = []
with torch.no_grad():
    for patch, exp, label in tqdm.tqdm(te_loader):
        pred, pred_prob = model.pred_step(patch.cuda())
        prediction.append(pred.cpu().squeeze(0).numpy().reshape(-1))
        gt.append(label.numpy())
        prob.append(pred_prob.squeeze(0).cpu().numpy())


prob = np.concatenate((prob), axis=0)
prediction = np.concatenate((prediction), axis=0)
gt = np.concatenate((gt), axis=0)


# Save adata with coordinates correction
import pickle
    
with open(f'./Prediction_results/{model_name}_Regclass_preds({names[2]}).pkl', 'wb') as f:
    pickle.dump(prediction, f)
    
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

name = names[2]
adata_dict[name].obs["predicted_celltypes"] = [id2lbl[int(v)] for v in prediction]


# In[176]:


import scanpy as sc
import matplotlib.pyplot as plt
gc.collect()
sc.pl.spatial(adata_dict[name], color="predicted_celltypes", save=f"CT-PRED-{model_name}-Regclass-{name}-3samples.png", title=f"{model_name} Prediction ({name})",spot_size=50)


# In[259]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(gt, prediction)

normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]

class_labels = ['B-cells', 'CAFs', 'Cancer Epithelial', 'Endothelial', 'Myeloid', 'PVL', 'Plasmablasts', 'T-cells']

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

# Create a heatmap
sns.heatmap(normalized_conf_matrix, annot=True, fmt=".2f", xticklabels=class_labels, yticklabels=class_labels)

# Add labels and title
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Multiclass Confusion Matrix")

plt.savefig(f"confmat-{model_name}-Regclass-{name}-3samples.png")
# Show the plot
plt.show()


# In[260]:


from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_probas, cls, title='ROC Curves',
                   curves=('micro', 'macro', 'each_class'),
                   ax=None, figsize=None, cmap='nipy_spectral',
                   title_fontsize="large", text_fontsize="medium"):
    """Generates the ROC curves from labels and predicted scores/probabilities

    Args:
        y_true (array-like, shape (n_samples)):
            Ground truth (correct) target values.

        y_probas (array-like, shape (n_samples, n_classes)):
            Prediction probabilities for each class returned by a classifier.

        title (string, optional): Title of the generated plot. Defaults to
            "ROC Curves".

        curves (array-like): A listing of which curves should be plotted on the
            resulting plot. Defaults to `("micro", "macro", "each_class")`
            i.e. "micro" for micro-averaged curve, "macro" for macro-averaged
            curve

        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.

        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.

        cmap (string or :class:`matplotlib.colors.Colormap` instance, optional):
            Colormap used for plotting the projection. View Matplotlib Colormap
            documentation for available options.
            https://matplotlib.org/users/colormaps.html

        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".

        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".

    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.

    Example:
        >>> import scikitplot as skplt
        >>> nb = GaussianNB()
        >>> nb = nb.fit(X_train, y_train)
        >>> y_probas = nb.predict_proba(X_test)
        >>> skplt.metrics.plot_roc_curve(y_test, y_probas)
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()

        .. image:: _static/examples/plot_roc_curve.png
           :align: center
           :alt: ROC Curves
    """
    y_true = np.array(y_true)
    y_probas = np.array(y_probas)

    if 'micro' not in curves and 'macro' not in curves and \
            'each_class' not in curves:
        raise ValueError('Invalid argument for curves as it '
                         'only takes "micro", "macro", or "each_class"')

    classes = np.unique(y_true)
    probas = y_probas

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_true, probas[:, i],
                                      pos_label=classes[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    micro_key = 'micro'
    i = 0
    while micro_key in fpr:
        i += 1
        micro_key += str(i)

    y_true = label_binarize(y_true, classes=classes)
    if len(classes) == 2:
        y_true = np.hstack((1 - y_true, y_true))

    fpr[micro_key], tpr[micro_key], _ = roc_curve(y_true.ravel(),
                                                  probas.ravel())
    roc_auc[micro_key] = auc(fpr[micro_key], tpr[micro_key])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[x] for x in range(len(classes))]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= len(classes)

    macro_key = 'macro'
    i = 0
    while macro_key in fpr:
        i += 1
        macro_key += str(i)
    fpr[macro_key] = all_fpr
    tpr[macro_key] = mean_tpr
    roc_auc[macro_key] = auc(fpr[macro_key], tpr[macro_key])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)

    if 'each_class' in curves:
        for i in range(len(classes)):
            color = plt.cm.get_cmap(cmap)(float(i) / len(classes))
            ax.plot(fpr[i], tpr[i], lw=2, color=color,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(cls[i], roc_auc[i]))

    if 'micro' in curves:
        ax.plot(fpr[micro_key], tpr[micro_key],
                label='micro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc[micro_key]),
                color='deeppink', linestyle=':', linewidth=4)

    if 'macro' in curves:
        ax.plot(fpr[macro_key], tpr[macro_key],
                label='macro-average ROC curve '
                      '(area = {0:0.2f})'.format(roc_auc[macro_key]),
                color='navy', linestyle=':', linewidth=4)

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=text_fontsize)
    ax.set_ylabel('True Positive Rate', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)
    ax.legend(loc='lower right', fontsize=text_fontsize)
    return ax

y_true = gt
y_score = prob

fig, ax = plt.subplots(figsize=(10,10))
plot_roc_curve(y_true, y_score, cls=class_labels, ax=ax)
plt.savefigg(f"ROC-{model_name}-Regclass-{name}-3samples.png")
plt.show()

