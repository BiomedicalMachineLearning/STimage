#%%
import pandas as pd; import numpy as np
#%%

## Reading the DataFrame

def dataframes_test(Gene_exp_test):
    #Aggregate, groupby and join
    Spot_gene_test = Gene_exp_test.iloc[:,5:]
    Gene_file = Spot_gene_test.groupby(['x','y']).agg('gene_name').apply(lambda x:" ".join(list(set(x)))).reset_index()
    
    ## Processing the Dataframe - Spots and Img_no.
    Link_img=Gene_file
    Link_img['Sno'] = np.arange(len(Link_img))+1
    Link_img['Sno'] = Link_img['Sno'].astype(str)
    Link_img['Sno']= Link_img['Sno'].str.zfill(4) + 'img'
    Link_img=Link_img[['x','y','Sno']]
    
    ## Gene Exp Mean High
    df=Gene_exp_test.drop(['X1','X2','bar_name'],axis=1)
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
    
Gene_exp_test = pd.read_csv('C:/Users/Onkar/UntitledFolder/gene_exp_breast_can2.csv')
dataframes_test(Gene_exp_test)
#%%
## Reading the DataFrame

def dataframes_train(Gene_exp_train):
    #Aggregate, groupby and join
    Spot_gene_train = Gene_exp_train.iloc[:,5:]
    Gene_file = Spot_gene_train.groupby(['x','y']).agg('gene_name').apply(lambda x:" ".join(list(set(x)))).reset_index()
    
    ## Processing the Dataframe - Spots and Img_no.
    Link_img=Gene_file
    Link_img['Sno'] = np.arange(len(Link_img))+1
    Link_img['Sno'] = Link_img['Sno'].astype(str)
    Link_img['Sno']= Link_img['Sno'].str.zfill(4) + 'img'
    Link_img=Link_img[['x','y','Sno']]
    
    ## Gene Exp Mean High
    df=Gene_exp_train.drop(['X1','X2','bar_name'],axis=1)
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
    Gene_file.to_csv('genes_file_trial.csv') 
    Link_img.to_csv('link_to_img_trial.csv')
    top.to_csv('Breast_2A_500_top.csv')
    
Gene_exp_train = pd.read_csv('C:/Users/Onkar/UntitledFolder/gene_exp.csv')
dataframes_train(Gene_exp_train)

#%%
