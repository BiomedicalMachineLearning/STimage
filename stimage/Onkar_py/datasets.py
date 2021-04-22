#resnet50
#breast_top200
#link_to_immg
#pivot
#gene_exp
#gene_list

#%%
import pandas as pd; import numpy as np

## Reading the DataFrame

def dataframes_test(Spot_gene, Gene_exp):
    #Aggregate, groupby and join
    Gene_file = Spot_gene.groupby(['x','y']).agg('gene_name').apply(lambda x:" ".join(list(set(x)))).reset_index()
    
    ## Processing the Dataframe - Spots and Img_no.
    Link_img=Gene_file
    Link_img['Sno'] = np.arange(len(Link_img))+1
    Link_img['Sno'] = Link_img['Sno'].astype(str)
    Link_img['Sno']= Link_img['Sno'].str.zfill(4) + 'img'
    Link_img=Link_img[['x','y','Sno']]
    
    ## Gene Exp Mean High
    df=Gene_exp.drop(['X1','X2','bar_name'],axis=1)
    #df2=pd.DataFrame()
    #df2=df.groupby(['x','y','UMI_Count']).agg('gene_name').apply(lambda x:" ".join(list(set(x)))).reset_index()
    df3 = pd.merge(df, Link_img, on=['x','y'], how='inner')
    df3=df3.drop(columns=['Unnamed: 0'])
    df4=pd.pivot_table(df3, values='UMI_Count', index=['Sno'],columns=['gene_name'], aggfunc=np.sum)
    df5=df4.fillna(0)
    #df13.drop([col for col, val in df13.sum().iteritems() if val < 11350], axis=1, inplace=True)
    top=df5.iloc[:,:]
    s = top.sum().sort_values(ascending=False, inplace=False)
    top = top[s.index[:500]]

    #Save Files
    Gene_file.to_csv('genes_file_breast_can2.csv') 
    Link_img.to_csv('link_to_img_breast_can2.csv')
    top.to_csv('Breast_2A_500_top.csv')
    
Spot_gene_test = pd.read_csv('D:/onkar/Projects/Project_Spt.Transcriptomics/Spt_Trans/df_Breast_Can2.csv')
Gene_exp_test = pd.read_csv('C:/Users/Onkar/UntitledFolder/gene_exp_breast_can2.csv')
dataframes_test(Spot_gene_test, Gene_exp_test)
#%%

import pandas as pd; import numpy as np

## Reading the DataFrame

def dataframes_train(Spot_gene, Gene_exp):
    #Aggregate, groupby and join
    Gene_file = Spot_gene.groupby(['x','y']).agg('gene_name').apply(lambda x:" ".join(list(set(x)))).reset_index()
    
    ## Processing the Dataframe - Spots and Img_no.
    Link_img=Gene_file
    Link_img['Sno'] = np.arange(len(Link_img))+1
    Link_img['Sno'] = Link_img['Sno'].astype(str)
    Link_img['Sno']= Link_img['Sno'].str.zfill(4) + 'img'
    Link_img=Link_img[['x','y','Sno']]
    
    ## Gene Exp Mean High
    df=Gene_exp.drop(['X1','X2','bar_name'],axis=1)
    #df2=pd.DataFrame()
    #df2=df.groupby(['x','y','UMI_Count']).agg('gene_name').apply(lambda x:" ".join(list(set(x)))).reset_index()
    df3 = pd.merge(df, Link_img, on=['x','y'], how='inner')
    df3=df3.drop(columns=['Unnamed: 0'])
    df4=pd.pivot_table(df3, values='UMI_Count', index=['Sno'],columns=['gene_name'], aggfunc=np.sum)
    df5=df4.fillna(0)
    #df13.drop([col for col, val in df13.sum().iteritems() if val < 11350], axis=1, inplace=True)
    top=df5.iloc[:,:]
    s = top.sum().sort_values(ascending=False, inplace=False)
    top = top[s.index[:500]]

    #Save Files
    Gene_file.to_csv('genes_file_breast_can2.csv') 
    Link_img.to_csv('link_to_img_breast_can2.csv')
    top.to_csv('Breast_2A_500_top.csv')
    
Spot_gene_train = pd.read_csv('D:/onkar/Projects/Project_Spt.Transcriptomics/Spt_Trans/df.csv')
Gene_exp_train = pd.read_csv('C:/Users/Onkar/UntitledFolder/gene_exp.csv')
dataframes_test(Spot_gene_train, Gene_exp_train)