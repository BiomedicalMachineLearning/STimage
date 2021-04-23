from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder; from sklearn.model_selection import train_test_split
from sklearn import preprocessing; from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score; from sklearn.neighbors import KNeighborsClassifier

import pandas as pd; import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder; from sklearn.model_selection import train_test_split
from sklearn import preprocessing; from sklearn.linear_model import LogisticRegression
import shap; import numpy as np; shap.initjs()

#%%
def multiclass_roc_auc_score(truth, pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(truth)
        truth = lb.transform(truth)
        pred = lb.transform(pred)
        return roc_auc_score(truth, pred, average=average)

def three_class_auroc(X, test_X, Y, test_Y, number):
    
    X=X.iloc[:,1:]
    Standard_scaler_x = preprocessing.MinMaxScaler(feature_range =(0, 1)) #StandardScaler()#
    X = Standard_scaler_x.fit_transform(X) 
    
    test_X=test_X.iloc[:,1:]
    test_X=Standard_scaler_x.transform(test_X)
    
    Y.drop(['Sno'],axis=1,inplace=True)
    Y=Y.iloc[:,:number]
    Standard_scaler_y = preprocessing.MinMaxScaler(feature_range =(0, 1)) #StandardScaler()#
    Y = Standard_scaler_y.fit_transform(Y) 
    Y = pd.DataFrame(data=Y)
    Y=Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))
    
    test_Y.drop(['Sno'],axis=1,inplace=True)
    test_Y=test_Y.iloc[:,:number]
    test_Y=Standard_scaler_y.transform(test_Y)
    test_Y=pd.DataFrame(data=test_Y)
    test_Y=test_Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))
    
    
    score=[]
    for i in Y:
        Y1=Y[i]
        X_train, X_test, y_train, y_test = train_test_split(X, Y1, test_size = 0.15, random_state = 0)
        clf=lgb.LGBMClassifier().fit(X_train, y_train)
        y_pred_test=clf.predict(test_X)
        score.append(multiclass_roc_auc_score(y_pred_test,test_Y[i]))
        
    Score = pd.DataFrame()
    Score["AUC"] = score
    Score.to_csv('AUROC_scores.csv')


X=pd.read_csv('ResNet50_Trainig_Breast_1A.csv')  
test_X = pd.read_csv('ResNet50_Trainig_Breast_2A.csv')
Y=pd.read_csv('Breast_1A_500_top.csv')
test_Y=pd.read_csv('Breast_2A_500_top.csv')
number=2
AUROC = three_class_auroc(X, test_X, Y, test_Y, number)
#%%


def Biomarker_Identicals(train_X, test_X, train_Y, test_Y):

    
    train_X = train_X.iloc[:,1:]
    test_X = test_X.iloc[:,1:]
    train_Y = train_Y[['Cluster']]
    test_Y['Cluster'] = test_Y['Cluster'].replace({2:0})
    test_Y = test_Y.sort_values(by='Sno')
    test_Y = test_Y.reset_index(drop=True)
    test_Y = test_Y[['Cluster']]
    X_train, X_test, y_train, y_test = train_test_split(train_X, train_Y, test_size = 0.20, random_state = 0)
    clf = lgb.LGBMClassifier()
    clf.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(clf, X_train)
    shap_values = explainer.shap_values(test_X)
    return shap.summary_plot(shap_values, test_X, max_display=10)


train_X = pd.read_csv('Breast_1A_500_top.csv')
test_X = pd.read_csv('Breast_2A_500_top.csv')
train_Y = pd.read_csv('Cluster_Img_trial.csv')
test_Y = pd.read_csv('Cluter_Img2.csv')
Biomarker_Identicals(train_X, test_X, train_Y, test_Y)