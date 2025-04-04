from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import scanpy as sc
import pandas as pd
import numpy as np
import os
import re


# Path_old = "/QRISdata/Q2051/STimage_project/TCGA_pred/"
Path = "/QRISdata/Q2051/STimage_project/TCGA_pred_selected/"

def get_filenames_in_directory(directory_path):
    h5_filenames = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".h5ad") and os.path.isfile(os.path.join(directory_path, filename)):
            h5_filenames.append(filename)
    return h5_filenames

def TCGA_name(original_array):
    result_array = []
    for item in original_array:
        parts = item.split("_")
        if len(parts) >= 2:
            second_part = parts[1].split(".")[0]
            result_array.append(second_part)
    pattern1 = re.compile(r'-\d+-\w+$')
    pattern2 = re.compile(r'(.+)-\w+-\d+-\w+$')
    name_info = [re.sub(pattern1, '', s) for s in result_array]
    name_surv_info = [re.sub(pattern2, r'\1', s) for s in result_array]   
    return name_info, name_surv_info

#############################################################################################################################

filenames = get_filenames_in_directory(Path)
name_info, name_surv_info = TCGA_name(filenames)
final_gene_counts = pd.DataFrame()
final_gene_mean_counts = pd.DataFrame()
for i in tqdm(range(len(filenames)), desc="Processing files"):
    try:
        adata = sc.read_h5ad(Path + filenames[i])
        gene_counts = adata.to_df()
        gene_counts.index = [name_info[i]] * len(gene_counts)
        gene_counts_mean = pd.DataFrame(gene_counts.mean(axis=0))
        final_gene_counts = pd.concat([final_gene_counts, gene_counts])
        final_gene_mean_counts = pd.concat([final_gene_mean_counts, gene_counts_mean.T])
    except:
        print("File empty:",Path + filenames[i])
        del name_info[i]

final_gene_mean_counts.index = name_info

final_gene_counts.to_csv("/QRISdata/Q2051/Onkar/STimage/project_scratch_STimage/STimage_v1/Survival/Updated3_final_gene_counts.csv")
final_gene_mean_counts.to_csv("/QRISdata/Q2051/Onkar/STimage/project_scratch_STimage/STimage_v1/Survival/Updated3_final_gene_mean_counts.csv")

#############################################################################################################################

survival_info = pd.read_excel("/QRISdata/Q2051/Onkar/STimage/project_scratch_STimage/STimage_v1/Survival/STimage_TCGA_patients.xlsx")
sample_info = pd.read_csv("/QRISdata/Q2051/Onkar/STimage/project_scratch_STimage/STimage_v1/Survival/gdc_sample_sheet.2024-01-24.tsv",sep="\t")

survival_info = survival_info[survival_info["case_submitter_id"].isin(name_surv_info)]
sample_info = sample_info[sample_info["Sample ID"].isin(name_info)]

##############################################################################################################################

all_bulk_ge = pd.DataFrame()
for i in tqdm(range(len(sample_info)), desc="Processing bulk gene expression files"):
    try:
        bulk_ge = pd.read_csv("/scratch/project/stseq/Onkar/STimage_v1/Survival/" + list(sample_info["File Name"])[i],
                              sep="\t", skiprows=1)
        bulk_ge = bulk_ge.iloc[4:, :]
        bulk_ge = bulk_ge.set_index("gene_name")
        bulk_ge = bulk_ge[["tpm_unstranded"]]
        bulk_ge.rename(columns={'tpm_unstranded': name_info[i]}, inplace=True)
        bulk_ge = bulk_ge[bulk_ge.index.isin(list(final_gene_counts.columns))]
        all_bulk_ge = pd.concat([all_bulk_ge, bulk_ge], axis=1)
    except:
        print("File empty:",Path + filenames[i])
all_bulk_ge.to_csv("/QRISdata/Q2051/Onkar/STimage/project_scratch_STimage/STimage_v1/Survival/Updated3_bluk_ge.csv")

##############################################################################################################################

survival_info = survival_info[["case_id","case_submitter_id",
                               "vital_status","days_to_last_follow_up",
                               "days_to_death","ajcc_pathologic_stage"]]
survival_info["time"] = np.where(survival_info["vital_status"]=="Alive",survival_info["days_to_last_follow_up"],survival_info["days_to_death"])
survival_info = survival_info[["case_id","case_submitter_id","time",
                               "vital_status","ajcc_pathologic_stage"]]
survival_info.to_csv("/QRISdata/Q2051/Onkar/STimage/project_scratch_STimage/STimage_v1/Survival/Updated3_survival_info.csv")
