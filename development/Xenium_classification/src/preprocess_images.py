import numpy as np
import sys
import argparse
import cv2
import pandas as pd
import math
import time
import os
import warnings
from tqdm import tqdm
import re
import shutil
import zipfile
from pathlib import Path
import torch
import torch.utils.data as data
from loguru import logger
from PIL import Image
from pathml.ml.hovernet import remove_small_objs
# from pathml.preprocessing import StainNormalizationHE
# from torchvision import transforms
# import torchstain

def white_filter(img, white_threshold=200, percent_threshold=0.85, return_mono=False):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    # Grayscale
    img = img.convert('L')
    # Threshold
    img = img.point( lambda p: 255 if p > white_threshold else 0)
    # To mono
    img = img.convert('1')
    # Percentage of white values
    if np.all(img):
        percent_white = 1
    elif np.all(~np.array(img)):
        percent_white = 1
    else:
        percent_white = np.unique(img, return_counts=True)[1][1] / (img.size[0]**2)
    return (percent_white > percent_threshold) if (not return_mono) else img 

def main(args):
	data_dir = Path(args.data)

	# dirs for images, masks
	imdir = data_dir / "images"
	maskdir = data_dir / "masks"

	# stop if the images and masks directories don't already exist
	assert imdir.is_dir(), f"Error: 'images' directory not found: {imdir}"
	assert maskdir.is_dir(), f"Error: 'masks' directory not found: {maskdir}"

	paths = list(imdir.glob("*"))
	paths = [p.stem for p in paths]

	remove_dir = data_dir / "removed"
	if not args.dry_run: 
		remove_dir.mkdir(parents=True, exist_ok=True)
		(remove_dir  / "images").mkdir(parents=True, exist_ok=True)
		(remove_dir  / "masks").mkdir(parents=True, exist_ok=True)

	filter_list = []

	for path in tqdm(paths):
		impath = imdir / f"{path}.png"
		maskpath = maskdir / f"{path}.npy"

		img = np.array(Image.open(str(impath)))
		mask = np.load(str(maskpath))

		# Filter 0-filled values from cropping
        # NOTE: I think this is no longer needed
		if img.any(axis=-1).mean() < 0.75:
			filter_list.append(path)

        # White space removal
		if white_filter(img):
			filter_list.append(path)
        
		# Filter small objects
		# This could remove nuclei so run this first
		nucleus_mask = (mask[-1] == 0).astype(np.uint8) # invert the bg mask
		filter_mask = (remove_small_objs(nucleus_mask, args.obj_threshold) != 0)
		filter_mask_bg = (filter_mask == 0)
		mask[:-1] = np.multiply(filter_mask, mask[:-1])
		mask[-1] = np.multiply(filter_mask_bg, mask[-1])

		# Filter images for nuclei
        # 2 is for 0,1 (bg/fg)
		if len(np.unique(mask)) - 2 < args.nuclei_threshold:
			filter_list.append(path)

		# Stain normalisation
		# TBA
		# method = args.stain
		# It doesn't seem like Pathml's function is good

		# Save modified files
		if path not in filter_list:
			if not args.dry_run:
				image = Image.fromarray(img.astype(np.uint8))
				image.save(impath)
				np.save(maskpath, mask)
		else:
			if not args.dry_run:
				impath.rename(remove_dir  / "images" / impath.name)
				maskpath.rename(remove_dir  / "masks" / maskpath.name)
			else:
				print(f"Will move {impath} to {str(remove_dir  / 'images' / impath.name)}")
				print(f"Will move {maskpath} to {str(remove_dir  / 'masks' / maskpath.name)}")

	print("Tiles removed:")
	print(filter_list)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument(
			"--data", type=str, default="./tile_data/rep1/", help="Path to tile data directory"
	)
	parser.add_argument(
			"--nuclei-threshold", type=int, default=2, help="Nuclei threshold (must have at least N)"
	)
	parser.add_argument(
			"--obj-threshold", type=int, default=10, help="Threshold for small objects"
	)
	parser.add_argument(
			"--stain", type=str, default=None, help="Stain normalisation method (TBA)"
	)
	parser.add_argument(
			"--dry-run",
			action="store_true",
			default=False,
			help="Dry run",
	)

	args = parser.parse_args()
	print(args)
	main(args)


