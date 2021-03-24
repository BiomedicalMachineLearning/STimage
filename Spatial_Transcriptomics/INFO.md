Spatial Transcriptomics [H&E] images

Image Clustering --> Cluster the images into Cancer vs Non-Cancer

Scanpy --> To get the high expressed genes

Gene_Selection --> 1) Filter those genes which are common in Cancer vs Non-Cancer Image Tiles. 2) Choose only those among these genes which are having different UMI counts from those 2 groups.

GeneExpression --> Continuous/Categorical Prediction of UMI Counts in an image.

CNN_BreastCancer10X --> CNN to predict which genes are present in a single spot. [Multi-Label Classification]



![image](https://user-images.githubusercontent.com/47315377/112247425-4d8e4400-8c7a-11eb-9975-840cdefa128a.png)
