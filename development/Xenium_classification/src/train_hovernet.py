#!/usr/bin/env python
# coding: utf-8

import random
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

from convnext import Convnext_HoverNetDecoder
import timm
from soft_ce import loss_hovernet_ce, load_df_targets
from scanpy import read_h5ad
import os
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def save_dict_as_json(dictionary, filename):
    with open(filename, 'w') as file:
        json.dump(dictionary, file)

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
    # Has been verified that the aug's (incl. colorjitter) do not modify the direct values in the mask
    
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

    if args.soft_targets:
        if "," in args.soft_targets:
            labels = args.soft_targets.split(",")
        else:
            labels = [args.soft_targets]
        label_dict = {label : load_df_targets(label) for label in labels}

        print(label_dict)
    else:
        labels = None

    weight_paths = args.weighted_ce or args.weighted_ce2
    if weight_paths:
        if "," in weight_paths:
            weight_paths = weight_paths.split(",")
        else:
            weight_paths = [weight_paths]
        
    if "pannuke" in args.dataset:
        data = PanNukeDataModule(
            data_dir="./data/pannuke/", 
            download=False,
            nucleus_type_labels=True, 
            batch_size=args.batch_size, 
            hovernet_preprocess=True,
            split=1,
            transforms=transform
        )
    else:
        data = XDataModule(
            data_dir=dataset, 
            nucleus_type_labels=True, 
            batch_size=args.batch_size, 
            hovernet_preprocess=True,
            split=split,
            transforms=transform,
            seed=args.seed,
            label_paths = labels,
            mod=args.mod
        )
    
    train_dataloader = data.train_dataloader
    valid_dataloader = data.valid_dataloader
    test_dataloader = data.test_dataloader
    
    print(f"GPUs used:\t{torch.cuda.device_count()}")
    device = torch.device("cuda:0")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:\t\t{device}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    
    # if args.pretrained:
    #     n_classes_pretrained = args.pretrained_classes
    #     hovernet = HoVerNet(n_classes=n_classes_pretrained)
    #     hovernet = torch.nn.DataParallel(hovernet)
    #     print(f"Loading pretrained model {args.pretrained}")
    #     # load pretrained and replace head
    #     checkpoint = torch.load(args.pretrained)
    #     hovernet.load_state_dict(checkpoint)
    #     hovernet.module.nc_head = nn.Sequential(
    #                     # one channel in output for each class
    #                     nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
    #                 )
    
    hovernet = torch.nn.DataParallel(hovernet)
    
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        hovernet.load_state_dict(checkpoint)
    
    # send model to GPU
    hovernet.to(device);
    
    # set up optimizer
    opt = torch.optim.AdamW(hovernet.parameters(), lr = args.lr, weight_decay=args.weight_decay) # change from 1e-4 -> 1e-5
    # learning rate scheduler to reduce LR by factor of 1/gamma each 25 epochs
    scheduler = StepLR(opt, step_size=25, gamma=args.step_gamma) # change from 0.1 -> 0.5

    # whether to freeze encoder for first n epochs
    # NOTE: could also investigate freezing other branches
    if args.freeze_epochs != 0:
        hovernet.module.encoder.requires_grad_(False)
        # hovernet.module.np_branch.requires_grad_(False)
        # hovernet.module.hv_branch.requires_grad_(False)
    
    if args.weighted_ce:
        if args.soft_targets:
            for i,v in enumerate(label_dict.values()):
                class_count = v.sum(axis=0)
                if i == 0:
                    class_counts = class_count
                else:
                    class_counts += class_count
        else:    
            for i,path in enumerate(weight_paths):
                adata = read_h5ad(path)
                class_count = adata.obs.groupby('predicted.id').count()['cell_id']
                if i == 0:
                    class_counts = class_count
                else:
                    class_counts += class_count
        weights = 1/(class_counts/((1/args.bg_weight)*class_counts.sum()))
        weights = np.append(weights.to_numpy(), 1/(args.bg_weight))
        weights = torch.from_numpy(weights).float().to(device)
        print(weights)
    elif args.weighted_ce2:
        df = pd.concat([read_h5ad(path).obs for path in weight_paths])
        weights = compute_class_weight('balanced', classes=np.unique(df['predicted.id']),y=df['predicted.id'])
        weights = np.append(weights, 1.0)
        weights = torch.from_numpy(weights).float().to(device)
        print(weights)

    else:
        weights = None
        
    # #### Main training loop
    
    n_epochs = 50
    
    # print performance metrics every n epochs
    print_every_n_epochs = 1
    
    # evaluating performance on a random subset of validation mini-batches
    # this saves time instead of evaluating on the entire validation set
    n_minibatch_valid = 50
    
    epoch_train_losses = {}
    epoch_valid_losses = {}
    epoch_train_dice = {}
    epoch_valid_dice = {}
    
    best_epoch = 0

    counter = 0

    factors = args.factors
    if factors:
        factors = factors.split(',')
        assert len(factors) == 6
        factors = np.array(factors).astype(float)
    # main training loop
    for i in tqdm(range(n_epochs)):
        minibatch_train_losses = []
        minibatch_train_dice = []

        # unfreeze encoder if needed
        if args.freeze_epochs != 0 and i == args.freeze_epochs - 1:
            hovernet.module.encoder.requires_grad_(True)
            # hovernet.module.np_branch.requires_grad_(True)
            # hovernet.module.hv_branch.requires_grad_(True)
       
        # put model in training mode
        hovernet.train()
        
        for data in train_dataloader:
            # send the data to the GPU
            images = data[0].float().to(device)
            masks = data[1].to(device)
            hv = data[2].float().to(device)
            label = data[3]

            #FIXME: bg mask should be a proper binary tensor/array
            # This line is temporary until I rerun the fixed save_masks.py
            # masks[:,-1,:,:] = (masks[:,-1,:,:] != 0)*1.
            
            # zero out gradient
            opt.zero_grad()
            
            # forward pass
            outputs = hovernet(images)
            
            # compute loss
            if args.soft_targets:
                df_targets = [label_dict[lbl] for lbl in label]
                loss = loss_hovernet_ce(outputs = outputs, ground_truth = [masks, hv], df_targets = df_targets, n_classes=n_classes, weight=weights, factors=factors)
            else:
                # loss = loss_hovernet(outputs = outputs, ground_truth = [masks, hv], n_classes=n_classes)
                loss = loss_hovernet_ce(outputs = outputs, ground_truth = [masks, hv], df_targets = None, n_classes=n_classes, weight=weights, factors=factors)
            
            # track loss
            minibatch_train_losses.append(loss.item())
            
            # also track dice score to measure performance
            preds_detection, preds_classification = post_process_batch_hovernet(outputs, n_classes=n_classes)
            truth_binary = masks[:, -1, :, :] == 0
            dice = dice_score(preds_detection, truth_binary.cpu().numpy())
            minibatch_train_dice.append(dice)
            
            # compute gradients
            loss.backward()
            
            # step optimizer and scheduler
            opt.step()

            counter+=1
            print(counter)
        
        #step LR scheduler
        scheduler.step()
        
        # evaluate on random subset of validation data
        hovernet.eval()
        minibatch_valid_losses = []
        minibatch_valid_dice = []
        # randomly choose minibatches for evaluating
        minibatch_ix = np.random.choice(range(len(valid_dataloader)), replace=False, size=n_minibatch_valid)
        with torch.no_grad():
            for j, data in enumerate(valid_dataloader):
                if j in minibatch_ix:
                    # send the data to the GPU
                    images = data[0].float().to(device)
                    masks = data[1].to(device)
                    hv = data[2].float().to(device)
                    label = data[3]
    
                    #FIXME: bg mask should be a proper binary tensor/array
                    # This line is temporary until I rerun the fixed save_masks.py
                    # masks[:,-1,:,:] = (masks[:,-1,:,:] != 0)*1.

                    # forward pass
                    outputs = hovernet(images)
    
                    # compute loss
                    loss = loss_hovernet_ce(outputs = outputs, ground_truth = [masks, hv], df_targets = None, n_classes=n_classes, weight=weights, factors=factors)
                    
    
                    # track loss
                    minibatch_valid_losses.append(loss.item())
    
                    # also track dice score to measure performance
                    preds_detection, preds_classification = post_process_batch_hovernet(outputs, n_classes=n_classes)
                    truth_binary = masks[:, -1, :, :] == 0
                    dice = dice_score(preds_detection, truth_binary.cpu().numpy())
                    minibatch_valid_dice.append(dice)
        
        # average performance metrics over minibatches
        mean_train_loss = np.mean(minibatch_train_losses)
        mean_valid_loss = np.mean(minibatch_valid_losses)
        mean_train_dice = np.mean(minibatch_train_dice)
        mean_valid_dice = np.mean(minibatch_valid_dice)
        
        # save the model with best performance
        if i != 0:
            if mean_valid_loss < min(epoch_valid_losses.values()):
                best_epoch = i
                torch.save(hovernet.state_dict(), out_dir / f"{args.seed}_hovernet_best_perf.pt")

        if args.save_all:
            torch.save(hovernet.state_dict(), out_dir / f"{args.seed}_hovernet_epoch_{i}.pt")
            
        # track performance over training epochs
        epoch_train_losses.update({i : mean_train_loss})
        epoch_valid_losses.update({i : mean_valid_loss})
        epoch_train_dice.update({i : mean_train_dice})
        epoch_valid_dice.update({i : mean_valid_dice})
        
        if print_every_n_epochs is not None:
            if i % print_every_n_epochs == print_every_n_epochs - 1:
                print(f"Epoch {i+1}/{n_epochs}:")
                print(f"\ttraining loss: {np.round(mean_train_loss, 4)}\tvalidation loss: {np.round(mean_valid_loss, 4)}")
                print(f"\ttraining dice: {np.round(mean_train_dice, 4)}\tvalidation dice: {np.round(mean_valid_dice, 4)}")
    
    # save fully trained model
    torch.save(hovernet.state_dict(), out_dir / "hovernet_fully_trained.pt")
    print(f"\nEpoch with best validation performance: {best_epoch}")

    # save logs
    save_dict_as_json(epoch_train_losses, out_dir / 'epoch_train_losses.json')
    save_dict_as_json(epoch_valid_losses, out_dir / 'epoch_valid_losses.json')
    save_dict_as_json(epoch_train_dice, out_dir / 'epoch_train_dice.json')
    save_dict_as_json(epoch_valid_dice, out_dir / 'epoch_valid_dice.json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="./tile_data/rep1/", help="Dataset path"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="batch size"
    )
    parser.add_argument(
        "--split", type=str, default="0.75,0.15,0.1", help="train,val,test split"
    )
    parser.add_argument(
        "--n-classes", type=int, default=10, help="number of classes"
    )
    parser.add_argument(
        "--freeze-epochs", type=int, default=0, help="number of epochs to freeze encoder"
    )
    parser.add_argument(
        "--step-gamma", type=float, default=0.1, help="gamma factor for StepLR"
    )
    parser.add_argument(
        "--out-dir", type=str, default="./outs/", help="output directory"
    )
    parser.add_argument(
        "--lr", type=float, default=0.0003, help="learning rate"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.005, help="weight decay"
    )
    parser.add_argument(
        "--pretrained", type=str, default=None, help="path of pretrained model"
    )
    parser.add_argument(
        "--model", type=str, default="Default", help="Default, convnextv2_small, convnextv2_tiny"
    )
    parser.add_argument(
        "--soft-targets", type=str, default=None, help="Compute the soft CE loss with respect to provided targets (anndata object)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (for data split and other)"
    )
    parser.add_argument(
        "--weighted-ce", type=str, default=None, help="Pass the anndata path to calculate the weights"
    )
    parser.add_argument(
        "--weighted-ce2", type=str, default=None, help="UPDATED ver: Pass the anndata path to calculate the weights"
    )
    parser.add_argument(
        "--bg-weight", type=float, default=0.5, help="Weight of the bg for weighted-ce"
    )
    parser.add_argument(
        "--color-aug", type=str, default=None, help="Pass anything to use color jitter"
    )
    parser.add_argument(
        "--factors", type=str, default=None, help="six element comma separated list"
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        default=False,
        help="Save checkpoints for every epoch",
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
