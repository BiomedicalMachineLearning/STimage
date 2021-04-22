from sklearn.multioutput import MultiOutputClassifier
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

def multiclass_roc_auc_score(truth, pred, average="macro"):
        lb = LabelBinarizer()
        lb.fit(truth)
        truth = lb.transform(truth)
        pred = lb.transform(pred)
        return roc_auc_score(truth, pred, average=average)

def three_class_auroc():
    
    X=pd.read_csv('ResNet50_Trainig_Breast_1A.csv')
    X=X.iloc[:,1:]
    Standard_scaler_x = preprocessing.MinMaxScaler(feature_range =(0, 1)) #StandardScaler()#
    X = Standard_scaler_x.fit_transform(X) 
    
    test_X = pd.read_csv('ResNet50_Trainig_Breast_2A.csv')
    test_X=test_X.iloc[:,1:]
    test_X=Standard_scaler_x.transform(test_X)
    
    Y=pd.read_csv('Breast_1A_500_top.csv')
    Y.drop(['Sno'],axis=1,inplace=True)
    Y=Y.iloc[:,:2]
    Standard_scaler_y = preprocessing.MinMaxScaler(feature_range =(0, 1)) #StandardScaler()#
    Y = Standard_scaler_y.fit_transform(Y) 
    Y = pd.DataFrame(data=Y)
    Y=Y.apply(lambda x: pd.qcut(x, 3,duplicates='drop',labels=False))
    
    test_Y=pd.read_csv('Breast_2A_500_top.csv')
    test_Y.drop(['Sno'],axis=1,inplace=True)
    test_Y=test_Y.iloc[:,:2]
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
    
    X=pd.read_csv('Breast_2A_500_top.csv')
    X.drop(["Sno"], axis=1, inplace=True)
    res = pd.DataFrame(index=[k for k in X])
    res = res.iloc[:2,:]
    res["AUC"] = score
    return res

#three_class_auroc()

#%%