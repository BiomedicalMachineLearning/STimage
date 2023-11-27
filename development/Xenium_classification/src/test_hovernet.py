#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import json
import argparse
import numpy as np
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
import albumentations as A

from pathml.datasets.pannuke import PanNukeDataModule
from custom_dataloader import XDataModule
from pathml.ml.hovernet import HoVerNet, loss_hovernet, post_process_batch_hovernet
from pathml.ml.utils import wrap_transform_multichannel, dice_score
from pathml.utils import plot_segmentation
from eval_utils import save_mat
from convnext import Convnext_HoverNetDecoder
import timm

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def main(args):

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    n_classes = args.n_classes
    
    # data augmentation transform 
    hover_transform = A.Compose(
        [A.VerticalFlip(p=0.5), 
         A.HorizontalFlip(p=0.5),
         A.RandomRotate90(p=0.5),
         A.GaussianBlur(p=0.5),
         A.MedianBlur(p=0.5, blur_limit=5),
         A.ColorJitter(brightness=0.2, contrast=0.25, saturation=0.2, hue=0.05), # https://arxiv.org/pdf/2203.03415.pdf
         # A.Affine(shear=(-5, 5), scale=(0.8, 1.2)), # https://arxiv.org/pdf/2203.01940.pdf
        ], 
        additional_targets = {f"mask{i}" : "mask" for i in range(n_classes)}
    )
    
    transform = wrap_transform_multichannel(hover_transform)
    if "," in args.dataset:
        dataset = args.dataset.split(",")
    else:
        dataset = [args.dataset]
    print(dataset)
    if args.split:
        split = [float(x) for x in args.split.split(",")]
    else:
        split = None
    
    if "pannuke" in args.dataset:
        data = PanNukeDataModule(
            data_dir="./data/pannuke/", 
            download=False,
            nucleus_type_labels=True, 
            batch_size=8, 
            hovernet_preprocess=True,
            split=1,
            transforms=transform
        )
    else:
        data = XDataModule(
            data_dir=dataset, 
            nucleus_type_labels=True, 
            batch_size=64,  ## MADE IT LARGER 
            hovernet_preprocess=True,
            split=split,
            transforms=transform,
            seed=args.seed,
            return_path=True,
            mod = args.mod
        )
    
    # train_dataloader = data.train_dataloader
    # valid_dataloader = data.valid_dataloader
    test_dataloader = data.test_dataloader
    
    print(f"GPUs used:\t{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
        # device = torch.device("cpu")
    print(f"Device:\t\t{device}")


    hovernet = HoVerNet(n_classes=n_classes)
    model = args.model
    if 'convnextv2' in model:
        model_c = timm.create_model(
            model,
            features_only=True,
        )
        model_c.stem_0 =  nn.Conv2d(3, 96, kernel_size=(7, 7), stride=(1, 1), padding=(3,3))
        hovernet.encoder = model_c
        hovernet.np_branch = Convnext_HoverNetDecoder()
        hovernet.nc_branch = Convnext_HoverNetDecoder()
        hovernet.hv_branch = Convnext_HoverNetDecoder()

    hovernet = torch.nn.DataParallel(hovernet)
    
    checkpoint = torch.load(args.path)
    hovernet.load_state_dict(checkpoint)
    hovernet = hovernet.to(device)
    hovernet.to(device)
    
    hovernet.eval()
    
    ims = None
    mask_truth = None
    mask_pred = None
    tissue_types = []
    
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_dataloader)):
            # send the data to the GPU
            images = data[0].float().to(device)
            masks = data[1].to(device)
            hv = data[2].float().to(device)
            tissue_type = data[3]
    
            # pass thru network to get predictions
            outputs = hovernet(images)
            preds_detection, preds_classification = post_process_batch_hovernet(outputs, n_classes=n_classes)
            
            if i == 0:
                ims = data[0].numpy()
                mask_truth = data[1].numpy()
                mask_pred = preds_classification
                tissue_types.extend(tissue_type)
            else:
                ims = np.concatenate([ims, data[0].numpy()], axis=0)
                mask_truth = np.concatenate([mask_truth, data[1].numpy()], axis=0)
                mask_pred = np.concatenate([mask_pred, preds_classification], axis=0)
                tissue_types.extend(tissue_type)

    mask_pred_save = np.moveaxis(mask_pred, 1, 3)
    mask_truth_save = np.moveaxis(mask_truth, 1, 3)
    if not tissue_types:
        tissue_types = np.array(['' for x in range(mask_truth_save.shape[0])])
    np.save(out_dir / 'preds.npy', mask_pred_save)
    np.save(out_dir / 'truths.npy', mask_truth_save)
    np.save(out_dir / 'types.npy', tissue_types)

    (out_dir / 'centroids_pred').mkdir(parents=True, exist_ok=True)
    (out_dir / 'centroids_truth').mkdir(parents=True, exist_ok=True)
    save_mat(out_dir / 'centroids_pred', mask_pred)
    save_mat(out_dir / 'centroids_truth', mask_truth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=None, help="Dataset path"
    )
    parser.add_argument(
        "--path", type=str, default=None, help="Model path"
    )
    parser.add_argument(
        "--model", type=str, default="Default", help="Default, convnextv2_small, convnextv2_tiny"
    )
    parser.add_argument(
        "--split", type=str, default=None, help="train,val,test split"
    )
    parser.add_argument(
        "--n-classes", type=int, default=10, help="number of classes"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./outs/", help="output directory"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed (for data split +other)"
    )
    parser.add_argument(
        "--mod",
        action="store_true",
        default=False,
        help="mod dataloader",
    )

    args = parser.parse_args()
    print(args)
    main(args)
