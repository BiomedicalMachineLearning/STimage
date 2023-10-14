import torch
import numpy as np
from scipy.spatial import distance_matrix, minkowski_distance, distance
from scipy.stats import pearsonr

def calcWeights(gene_exp, similarityType='pearsonr',):
  """
  Similarity can be ['pearsonr', 'medianL1', 'rmse', 'cosine'].

  Test the code:
  gene_exp = torch.randn((6, 4))
  [calcWeights(gene_exp, similarityType=i) for i in ['pearsonr', 'medianL1', 'rmse', 'cosine']]

  Reference:
  [1] Feng, X., Fang, F., Long, H., Zeng, R., & Yao, Y. (2022). Single-cell RNA-seq data analysis using graph autoencoders and graph attention networks. Frontiers in Genetics, 13, 1003711.
  [2] Abadi, S.A.R., Laghaee, S.P. & Koohi, S. An optimized graph-based structure for single-cell RNA-seq cell-type classification based on non-linear dimension reduction. BMC Genomics 24, 227 (2023). https://doi.org/10.1186/s12864-023-09344-y
  """
  rows=[]
  for i in range(len(gene_exp)): 
    for j in range(len(gene_exp)): 
      if similarityType=='pearsonr':
        # Calculate the Pearson correlation coefficient
        rows.append(pearsonr(gene_exp[i], gene_exp[j])[0])
      elif similarityType=='medianL1':
        # Calculate the median L1 
        # L1 distance represents the absolute deviation of gene expression between cells. 
        # The lower the value means, the higher the similarity, and the better the obtainable imputation effect.
        rows.append(abs(gene_exp[i].median() - gene_exp[j].median()))
      elif similarityType=='rmse':
        # Calculate the RMSE.
        # RMSE represents the square root of the quadratic mean of gene expression differences between cells
        # the smaller the value, the better the effect.
        rows.append(torch.sqrt(torch.mean((gene_exp[i]-gene_exp[j])**2)))
      elif similarityType=='cosine':
        # Calculate the cosine similarity
        # Cosine similarity refers to the product of gene expression between cells. 
        # The value range of cosine similarity is [0, 1]; the higher the value, the more similar.
        rows.append(F.cosine_similarity(gene_exp[i].unsqueeze(0), gene_exp[j].unsqueeze(0)))

  weight_adj = F.softmax(torch.tensor(rows).reshape(len(gene_exp), len(gene_exp)), dim=-1)
  return weight_adj.numpy()


def calcADJ(coord, k=0, weights=None, distanceType='euclidean', pruneTag='NA'):
    """
    Calculate spatial Matrix directly use X/Y coordinates.
    If weights are not provided, default weights=1.
    If weights are provided, assign the weight to the spatial distance.
    The weight is from gene expression similarity between cells. 
    If the gene expression of cells are similar, the distance between them should be short.
    k is the number of nearest nodes, if k=0 we consider all nodes.

    Test codes:
    ct = torch.randint(0, 40000, (6, 2)).to(torch.float32).long() # 0-40000 is range of whole slide image pixels, 6 is num_cell, 2 is spatial coordinates. 
    exp = torch.randn(6,4) # num_cell x num_gene
    calcADJ(ct, k=0, weights=calcWeights(exp).numpy()) # return an adjancency matrix

    Reference:
    [1] https://github.com/biomed-AI/Hist2ST/blob/main/graph_construction.py
    """
    spatialMatrix=coord#.cpu().numpy()
    nodes=spatialMatrix.shape[0]
    Adj = torch.zeros((nodes,nodes))
    for i in np.arange(spatialMatrix.shape[0]):
        tmp=spatialMatrix[i,:].reshape(1,-1)
        distMat = distance.cdist(tmp,spatialMatrix, distanceType)
        if weights.all() != None:
          distMat = distMat*weights[i]
        else:
          pass
        k = spatialMatrix.shape[0]-1 if k == 0 else k
        res = distMat.argsort()[:k+1]
        tmpdist = distMat[0,res[0][1:k+1]]
        boundary = np.mean(tmpdist)+np.std(tmpdist) #optional
        for j in np.arange(1,k+1):
            # No prune
            if pruneTag == 'NA':
                Adj[i][res[0][j]]=1.0
            elif pruneTag == 'STD':
                if distMat[0,res[0][j]]<=boundary:
                    Adj[i][res[0][j]]=1.0
            # Prune: only use nearest neighbor as exact grid: 6 in cityblock, 8 in euclidean
            elif pruneTag == 'Grid':
                if distMat[0,res[0][j]]<=2.0:
                    Adj[i][res[0][j]]=1.0
    return Adj