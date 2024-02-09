#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage.transform import resize
from skimage.color import rgb2hed
from skimage.feature import peak_local_max
from skimage.measure import label, regionprops
from skimage.segmentation import mark_boundaries, watershed
from skimage.segmentation import slic
from skimage.morphology import area_opening
import skimage
import scipy as sp
from scipy import ndimage as ndi

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import anndata
plt.rcParams['savefig.facecolor']='#111111'


# In[2]:


def scale_mask(mat, N):
    mat = mat.reshape(-1, N * 89401)
    mat_std = (mat - mat.min()) / (mat.max() - mat.min())
    mat_scaled = mat_std * (1 - 0) + 0  
    return mat_scaled.reshape(N, 299, 299)

def scale_one_mask(list_mat):
    scaled_mask = []
    for i in range(0,len(list_mat)):
        mat = list_mat[i].copy()
        mat_std = (mat - mat.min()) / (mat.max() - mat.min())
        mat_scaled = mat_std * (1 - 0) + 0
        scaled_mask.append(mat_scaled)
    return scaled_mask

def no_bg_img_save(img):
    no_bg_img = img.convert("RGBA")
    datas = no_bg_img.getdata()
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    no_bg_img.putdata(newData)
    return no_bg_img
num_steps = 256
viridis_colors = plt.cm.viridis(np.linspace(0, 1, num_steps - 1))
colors = np.vstack([[1, 1, 1, 1], viridis_colors])
new_cmap = LinearSegmentedColormap.from_list('viridis_white', colors, N=5)

def watershed_segment(image):
    annotation_hed = rgb2hed(image)
    annotation_h = annotation_hed[:, :, 0]
    annotation_h *= 255.0 / np.percentile(annotation_h, q=0.01)
    thresh = skimage.filters.threshold_otsu(annotation_h) * 0.7
    im_fgnd_mask = sp.ndimage.binary_fill_holes(annotation_h < thresh)
    distance = ndi.distance_transform_edt(im_fgnd_mask)
    coords = peak_local_max(distance, footprint=np.ones((5, 5)), labels=im_fgnd_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(annotation_h, markers, mask=im_fgnd_mask)
    im_nuclei_seg_mask = area_opening(labels, area_threshold=64).astype(np.int32)
    map_dic = dict(zip(np.unique(im_nuclei_seg_mask), np.arange(len(np.unique(im_nuclei_seg_mask)))))
    im_nuclei_seg_mask = np.vectorize(map_dic.get)(im_nuclei_seg_mask)
    return im_nuclei_seg_mask

def only_segment_scores(adata, LIME_score):
    LIME_mask_score = []
    for i in tqdm(range(len(adata.obs["tile_path"])), desc="Processing images"):
        try:
            mask = watershed_segment(Image.open(adata.obs["tile_path"][i]))
        except:
            mask = np.full((299, 299), 0)
        mask = np.where(mask == 0, 0, 1)
        minimum_val = LIME_score[i].min()
        mask_score = LIME_score[i] * mask
        mask_score = np.where(mask_score == 0, LIME_score[i].min(), mask_score)
        LIME_mask_score.append(mask_score)
    return LIME_mask_score

def LIME_compare(gene2_LimeMask_scaled, gene1_LimeMask_scaled):
    """
    Compares LIME explanations for two classes by creating difference images.

    Args:
        gene2_LimeMask_scaled: A list of LIME heatmaps (scaled 0-1) for class gene2.
        gene1_LimeMask_scaled: A list of LIME heatmaps (scaled 0-1) for class gene1.

    Returns:
        A list of difference images, where each pixel represents the difference
        in LIME scores between the corresponding pixels in the gene2 and gene1 heatmaps.
    """

    gene2_list = []
    gene1_list = []
    gene2_gene1 = []

    # Create blank background arrays for RGB conversion
    a = np.array([[255]*299]*299)
    b = np.array([[255]*299]*299)

    # Process gene2 heatmaps
    for i in range(len(gene2_LimeMask_scaled)):
        rgb_uint8 = (np.dstack((gene2_LimeMask_scaled[i], a, b)) * 255).astype(np.uint8)

        # Highlight important regions in green
        black_pixels = np.where((rgb_uint8[:, :, 0] < 15) & (rgb_uint8[:, :, 1] < 15) & (rgb_uint8[:, :, 2] < 15))
        rgb_uint8[black_pixels] = [0, 255, 0]

        # Set background pixels to white
        rgb_uint8_im = Image.fromarray(rgb_uint8)
        rgb_uint8_im_info = max(rgb_uint8_im.getcolors(rgb_uint8_im.size[0] * rgb_uint8_im.size[1]))
        bcg_pixels = np.where((rgb_uint8[:, :, 0] == rgb_uint8_im_info[1][0]) &
                               (rgb_uint8[:, :, 1] == rgb_uint8_im_info[1][1]) &
                               (rgb_uint8[:, :, 2] == rgb_uint8_im_info[1][2]))
        rgb_uint8[bcg_pixels] = [255, 255, 255]

        gene2_list.append(rgb_uint8)

    # Perform the same processing for gene1 heatmaps
    for i in range(0,len(gene1_LimeMask_scaled)):
        rgb_uint8 = (np.dstack((gene1_LimeMask_scaled[i],
                                a,b))*255).astype(np.uint8)
        black_pixels = np.where(
            (rgb_uint8[:, :, 0] < 15) & 
            (rgb_uint8[:, :, 1] < 15) & 
            (rgb_uint8[:, :, 2] < 15)
        )
        rgb_uint8[black_pixels] = [0, 255, 0]

        rgb_uint8_im = Image.fromarray(rgb_uint8)
        rgb_uint8_im_info = max(rgb_uint8_im.getcolors(rgb_uint8_im.size[0]*rgb_uint8_im.size[1]))

        bcg_pixels = np.where(
            (rgb_uint8[:, :, 0] == rgb_uint8_im_info[1][0]) & 
            (rgb_uint8[:, :, 1] == rgb_uint8_im_info[1][1]) & 
            (rgb_uint8[:, :, 2] == rgb_uint8_im_info[1][2])
        )
        rgb_uint8[bcg_pixels] = [255, 255, 255]
        gene1_list.append(rgb_uint8)
    
    for i in range(0,len(gene2_list)):
        gene2_gene1.append(gene2_list[i]-gene1_list[i])
    return gene2_gene1


# In[4]:


DATA_PATH = "/scratch/project/stseq/Onkar/STimage_v1/"
adata_all = anndata.read_h5ad(DATA_PATH+"all_adata.h5ad")

FFPE = adata_all[adata_all.obs["library_id"]=="FFPE"]
FFPE.obs["tile_path"] = DATA_PATH+"tiles/tiles/"+FFPE.obs["tile_path"].str.split('/', expand=True)[6]
# FFPE = FFPE[FFPE.obs.index.isin(FFPE.obs.iloc[:10,:].index)]
coor_FFPE = list(zip(FFPE.obs["imagerow"],FFPE.obs["imagecol"]))


# In[5]:


LIME_PATH = "/scratch/project/stseq/Onkar/STimage_v1/Outputs/pickle/npy/"
FFPE_CD24 = np.load(LIME_PATH+"LIMEMask_FFPE_CD24_all.npy")
FFPE_CD52 = np.load(LIME_PATH+"LIMEMask_FFPE_CD52_all.npy")
FFPE_CD24_reg = np.load(LIME_PATH+"LIMEMaskReg_FFPE_CD24_all.npy")
FFPE_CD52_reg = np.load(LIME_PATH+"LIMEMaskReg_FFPE_CD52_all.npy")

FFPE_CD24_seg_sc = only_segment_scores(FFPE,FFPE_CD24)
FFPE_CD52_seg_sc = only_segment_scores(FFPE,FFPE_CD52)
FFPE_CD24_reg_seg_sc = only_segment_scores(FFPE,FFPE_CD24_reg)
FFPE_CD52_reg_seg_sc = only_segment_scores(FFPE,FFPE_CD52_reg)

FFPE_CD24_seg_sc_pre_scaled=FFPE_CD24_seg_sc.copy()
FFPE_CD52_seg_sc_pre_scaled=FFPE_CD52_seg_sc.copy()
FFPE_CD24_reg_seg_sc_pre_scaled=FFPE_CD24_reg_seg_sc.copy()
FFPE_CD52_reg_seg_sc_pre_scaled=FFPE_CD52_reg_seg_sc.copy()

for i in range(0,len(FFPE.obs["tile_path"])):    
    FFPE_CD24_seg_sc_pre_scaled[i] = np.where(FFPE_CD24_seg_sc_pre_scaled[i] > 0, FFPE_CD24_seg_sc_pre_scaled[i], 0)
    FFPE_CD52_seg_sc_pre_scaled[i] = np.where(FFPE_CD52_seg_sc_pre_scaled[i] > 0, FFPE_CD52_seg_sc_pre_scaled[i], 0)
    FFPE_CD24_reg_seg_sc_pre_scaled[i] = np.where(FFPE_CD24_reg_seg_sc_pre_scaled[i] > 0, 
                                                  FFPE_CD24_reg_seg_sc_pre_scaled[i], 0)
    FFPE_CD52_reg_seg_sc_pre_scaled[i] = np.where(FFPE_CD52_reg_seg_sc_pre_scaled[i] > 0, 
                                                  FFPE_CD52_reg_seg_sc_pre_scaled[i], 0)
    
FFPE_CD24_seg_sc_scaled = scale_one_mask(FFPE_CD24_seg_sc_pre_scaled)
FFPE_CD52_seg_sc_scaled = scale_one_mask(FFPE_CD52_seg_sc_pre_scaled)
FFPE_CD24_reg_seg_sc_scaled = scale_one_mask(FFPE_CD24_reg_seg_sc)
FFPE_CD52_reg_seg_sc_scaled = scale_one_mask(FFPE_CD52_reg_seg_sc)

FFPE_CD24_vs_CD52 = LIME_compare(FFPE_CD24_seg_sc,FFPE_CD52_seg_sc)
FFPE_CD24_vs_CD52_reg = LIME_compare(FFPE_CD24_reg_seg_sc,FFPE_CD52_reg_seg_sc)


# #### Classification FFPE

# In[ ]:


f1 = plt.figure("GOOD1")
ax = f1.add_subplot(111)
ax.axis('off')
ax.set_position([0, 0, 1, 1])
f1.patch.set_alpha(0.)
ax.patch.set_alpha(0.)

for i in range(0,len(FFPE_CD52_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(FFPE_CD52_seg_sc_scaled[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD52/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD52/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD52/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(FFPE_CD52_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD52/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")    

tissue = Image.fromarray(FFPE.uns["spatial"]["FFPE"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(FFPE.obs["imagecol"],FFPE.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/FFPE_LIME_CD52.png'
tissue.save(result_image_path)


# In[ ]:


for i in range(0,len(FFPE_CD24_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(FFPE_CD24_seg_sc_scaled[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD24/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD24/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD24/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(FFPE_CD24_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD24/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")    

tissue = Image.fromarray(FFPE.uns["spatial"]["FFPE"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(FFPE.obs["imagecol"],FFPE.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/FFPE_LIME_CD24.png'
tissue.save(result_image_path)


# In[ ]:


for i in range(0,len(FFPE_CD24_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(255-FFPE_CD24_vs_CD52[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/Combined_FFPE/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/Combined_FFPE/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/Combined_FFPE/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(FFPE_CD24_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/Combined_FFPE/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")    

tissue = Image.fromarray(FFPE.uns["spatial"]["FFPE"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(FFPE.obs["imagecol"],FFPE.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/FFPE_LIME_CD24_CD52.png'
tissue.save(result_image_path)


# #### Regression FFPE 

# In[ ]:


for i in range(0,len(FFPE_CD52_reg_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(FFPE_CD52_reg_seg_sc_scaled[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD52/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD52/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD52/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(FFPE_CD52_reg_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD52/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")    

tissue = Image.fromarray(FFPE.uns["spatial"]["FFPE"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(FFPE.obs["imagecol"],FFPE.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/FFPE_LIME_CD52_reg.png'
tissue.save(result_image_path)


# In[ ]:


for i in range(0,len(FFPE_CD24_reg_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(FFPE_CD24_reg_seg_sc_scaled[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD24/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD24/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD24/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(FFPE_CD24_reg_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_FFPE_CD24/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")    

tissue = Image.fromarray(FFPE.uns["spatial"]["FFPE"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(FFPE.obs["imagecol"],FFPE.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/FFPE_LIME_CD24_reg.png'
tissue.save(result_image_path)


# In[ ]:


for i in range(0,len(FFPE_CD24_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(255-FFPE_CD24_vs_CD52_reg[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/Combined_FFPE/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/Combined_FFPE/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/Combined_FFPE/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(FFPE_CD24_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/Combined_FFPE/"+str(coor_FFPE[i][0])+"_"+str(coor_FFPE[i][1])+".png")    

tissue = Image.fromarray(FFPE.uns["spatial"]["FFPE"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(FFPE.obs["imagecol"],FFPE.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/FFPE_LIME_CD24_CD52_reg.png'
tissue.save(result_image_path)


# #### Classification 1160920F

# In[ ]:


DATA_PATH = "/scratch/project/stseq/Onkar/STimage_v1/"
adata_all = anndata.read_h5ad(DATA_PATH+"all_adata.h5ad")

F1160920 = adata_all[adata_all.obs["library_id"]=="1160920F"]
F1160920.obs["tile_path"] = DATA_PATH+"tiles/tiles/"+F1160920.obs["tile_path"].str.split('/', expand=True)[6]
# F1160920 = F1160920[F1160920.obs.index.isin(F1160920.obs.iloc[:10,:].index)]
coor_F1160920 = list(zip(F1160920.obs["imagerow"],F1160920.obs["imagecol"]))


# In[ ]:


LIME_PATH = "/scratch/project/stseq/Onkar/STimage_v1/Outputs/pickle/npy/"
F1160920_CD24 = np.load(LIME_PATH+"LIMEMask_1160920F_CD24_all.npy")
F1160920_CD52 = np.load(LIME_PATH+"LIMEMask_1160920F_CD52_all.npy")
F1160920_CD24_reg = np.load(LIME_PATH+"LIMEMaskReg_1160920F_CD24_all.npy")
F1160920_CD52_reg = np.load(LIME_PATH+"LIMEMaskReg_1160920F_CD52_all.npy")

F1160920_CD24_seg_sc = only_segment_scores(F1160920,F1160920_CD24)
F1160920_CD52_seg_sc = only_segment_scores(F1160920,F1160920_CD52)
F1160920_CD24_reg_seg_sc = only_segment_scores(F1160920,F1160920_CD24_reg)
F1160920_CD52_reg_seg_sc = only_segment_scores(F1160920,F1160920_CD52_reg)

F1160920_CD24_seg_sc_pre_scaled=F1160920_CD24_seg_sc.copy()
F1160920_CD52_seg_sc_pre_scaled=F1160920_CD52_seg_sc.copy()
F1160920_CD24_reg_seg_sc_pre_scaled=F1160920_CD24_reg_seg_sc.copy()
F1160920_CD52_reg_seg_sc_pre_scaled=F1160920_CD52_reg_seg_sc.copy()

for i in range(0,len(F1160920.obs["tile_path"])):    
    F1160920_CD24_seg_sc_pre_scaled[i] = np.where(F1160920_CD24_seg_sc_pre_scaled[i] > 0, F1160920_CD24_seg_sc_pre_scaled[i], 0)
    F1160920_CD52_seg_sc_pre_scaled[i] = np.where(F1160920_CD52_seg_sc_pre_scaled[i] > 0, F1160920_CD52_seg_sc_pre_scaled[i], 0)
    F1160920_CD24_reg_seg_sc_pre_scaled[i] = np.where(F1160920_CD24_reg_seg_sc_pre_scaled[i] > 0, 
                                                  F1160920_CD24_reg_seg_sc_pre_scaled[i], 0)
    F1160920_CD52_reg_seg_sc_pre_scaled[i] = np.where(F1160920_CD52_reg_seg_sc_pre_scaled[i] > 0, 
                                                  F1160920_CD52_reg_seg_sc_pre_scaled[i], 0)
    
F1160920_CD24_seg_sc_scaled = scale_one_mask(F1160920_CD24_seg_sc_pre_scaled)
F1160920_CD52_seg_sc_scaled = scale_one_mask(F1160920_CD52_seg_sc_pre_scaled)
F1160920_CD24_reg_seg_sc_scaled = scale_one_mask(F1160920_CD24_reg_seg_sc)
F1160920_CD52_reg_seg_sc_scaled = scale_one_mask(F1160920_CD52_reg_seg_sc)

F1160920_CD24_vs_CD52 = LIME_compare(F1160920_CD24_seg_sc,F1160920_CD52_seg_sc)
F1160920_CD24_vs_CD52_reg = LIME_compare(F1160920_CD24_reg_seg_sc,F1160920_CD52_reg_seg_sc)


# In[ ]:


for i in range(0,len(F1160920_CD52_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(F1160920_CD52_seg_sc_scaled[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD52/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD52/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD52/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(F1160920_CD52_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD52/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")    

tissue = Image.fromarray(F1160920.uns["spatial"]["1160920F"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(F1160920.obs["imagecol"],F1160920.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/1160920F_LIME_CD52.png'
tissue.save(result_image_path)


# In[ ]:


for i in range(0,len(F1160920_CD24_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(F1160920_CD24_seg_sc_scaled[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(F1160920_CD24_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")    

tissue = Image.fromarray(F1160920.uns["spatial"]["1160920F"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(F1160920.obs["imagecol"],F1160920.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/1160920F_LIME_CD24.png'
tissue.save(result_image_path)


# In[ ]:


for i in range(0,len(F1160920_CD24_reg_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(255-F1160920_CD24_vs_CD52[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(F1160920_CD24_reg_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")    

tissue = Image.fromarray(F1160920.uns["spatial"]["1160920F"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(F1160920.obs["imagecol"],F1160920.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/1160920F_LIME_CD24_CD52.png'
tissue.save(result_image_path)


# #### Regression 1160920F

# In[ ]:


for i in range(0,len(F1160920_CD52_reg_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(F1160920_CD52_reg_seg_sc_scaled[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD52/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD52/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD52/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(F1160920_CD52_reg_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD52/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")    

tissue = Image.fromarray(F1160920.uns["spatial"]["1160920F"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(F1160920.obs["imagecol"],F1160920.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/1160920F_LIME_CD52_reg.png'
tissue.save(result_image_path)


# In[ ]:


for i in range(0,len(F1160920_CD24_reg_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(F1160920_CD24_reg_seg_sc_scaled[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(F1160920_CD24_reg_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")    

tissue = Image.fromarray(F1160920.uns["spatial"]["1160920F"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(F1160920.obs["imagecol"],F1160920.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/1160920F_LIME_CD24_reg.png'
tissue.save(result_image_path)


# In[ ]:


for i in range(0,len(F1160920_CD24_reg_seg_sc_scaled)):    
    plt.figure(figsize=(3.89,3.89))
    plt.rcParams['savefig.facecolor']='#111111'

    plt.imshow(255-F1160920_CD24_vs_CD52_reg[i],alpha=0.5,cmap=new_cmap)
    plt.axis('off')
    plt.savefig("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", 
                transparent=True)
    img = Image.open("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")
    no_bg_img = no_bg_img_save(img)
    no_bg_img.save("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png", "PNG")

small_image_paths = []
for i in range(0,len(F1160920_CD24_reg_seg_sc_scaled)):
    small_image_paths.append("/scratch/user/s4634945/Self_Data/STimage/tiles/LIME_tiles_1160920F_CD24/"+str(coor_F1160920[i][0])+"_"+str(coor_F1160920[i][1])+".png")    

tissue = Image.fromarray(F1160920.uns["spatial"]["1160920F"]["images"]["fulres"]).convert("RGBA")
transparency = 128 
tissue_image = Image.new('RGBA', tissue.size, (0, 0, 0, 0))
tissue_image.paste(tissue, (0, 0), mask=tissue)
tissue_image.putalpha(transparency)
small_image_coords = list(zip(F1160920.obs["imagecol"],F1160920.obs["imagerow"]))
small_images = [Image.open(path) for path in small_image_paths]
for (center_x, center_y), small_image in zip(small_image_coords, small_images):
    x = center_x - small_image.width // 2
    y = center_y - small_image.height // 2    
    tissue.paste(small_image, (x, y), small_image)
result_image_path = '/scratch/user/s4634945/Self_Data/STimage/1160920F_LIME_CD24_CD52_reg.png'
tissue.save(result_image_path)


# In[ ]:





# In[14]:



