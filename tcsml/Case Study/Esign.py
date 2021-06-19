# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 03:14:54 2021

@author: rohan
"""
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score, auc, plot_roc_curve
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statscal import calc_vif, cal_corr_pval
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def calc_vif(X):

    # Calculating VIF
    '''
    This function will return a dataframe that will contain the vif values of each of the features
    '''
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)
def cal_corr_pval(df, series):
    corr_df = pd.DataFrame(columns=['r','p'])
    for col in df:
        if pd.api.types.is_numeric_dtype(df[col]):
            r, p = stats.pearsonr(series, df[col])
            corr_df.loc[col] = [r, p]
    return corr_df

def load_df():
    df = pd.read_csv('P39-Financial-Data.csv')
    df.drop(['entry_id'], axis=1, inplace=True)
    return df
def get_encoded_df(df):
    one_hot = pd.get_dummies(df['pay_schedule'])
    df = df.drop('pay_schedule',axis = 1)
    df = df.join(one_hot)
    return df
def reorg_df(df):
    new_cols = ['home_owner','has_debt', 'bi-weekly','monthly', 'semi-monthly',
        'weekly','age',  'income', 'months_employed', 'years_employed',
       'current_address_year', 'personal_account_m', 'personal_account_y',
       'amount_requested', 'risk_score', 'risk_score_2',
       'risk_score_3', 'risk_score_4', 'risk_score_5', 'ext_quality_score',
       'ext_quality_score_2', 'inquiries_last_month', 'e_signed']
    df = df.reindex(columns = new_cols)
    return df
 
def show_heatmap(df):
    cor = df.corr()
    print(cor)
    sns.heatmap(cor)
    value_cols =['age', 'income', 'months_employed', 'years_employed',
       'current_address_year', 'personal_account_m', 'personal_account_y',
       'amount_requested', 'risk_score', 'risk_score_2',
       'risk_score_3', 'risk_score_4', 'risk_score_5', 'ext_quality_score',
       'ext_quality_score_2', 'inquiries_last_month']
    return cor
def preprocessing(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test

def fit_scale(X_train, X_test):
    sc = StandardScaler()  
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test

def LDA_implmnt(X_train, X_test, y_train, y_test):
    lda = LDA(n_components = 1)
    X_train = lda.fit_transform(X_train, y_train)
    X_test = lda.transform(X_test)
    return X_train, X_test, y_train, y_test

def model_logistic(X_train, y_train):
    classifier = LogisticRegression(random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier
def model_rf(X_train, y_train):
    classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    classifier.fit(X_train, y_train)
    return classifier
def model_xgb(X_train, y_train):
    classifier = XGBClassifier()
    classifier.fit(X_train, y_train)
    return classifier    
def get_model_performance(classifier,X, y):
    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    print(cm)
    print(f'Accuracy: {accuracy_score(y, y_pred)}')
    print(f'f1-score: {f1_score(y,y_pred)}')

def roc_plot(classifier, X_test, y_test):
    probs = classifier.predict_proba(X_test)
    #a  = classifier.predict()
    preds = probs[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.plot(fpr, tpr, label = 'AUC = %0.4f' %roc_auc, color = 'blue')
    plt.legend(loc = 'lower right')
    plt.plot([0,1],[0,1], 'r--')
    
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
def optimal_ts(classifier, X_test, y_test):
    probs = classifier.predict_proba(X_test)
    #a  = classifier.predict()
    preds = probs[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print("Threshold value is:", optimal_threshold)
    
    new_preds = list()
    
    for i in preds:
        if i > optimal_threshold:
            new_preds.append(1)
        else:
            new_preds.append(0)
    
    cm = confusion_matrix(y_test, new_preds)
    print(cm)
    print(f'Accuracy: {accuracy_score(y_test, new_preds)}')
    print(f'f1-score: {f1_score(y_test,new_preds)}')
    roc_df= pd.DataFrame(list(zip(fpr, tpr, thresholds)), columns=['FPR','TPR', 'Threshold'])
    return roc_df
            
df = load_df()
df = get_encoded_df(df)
df = reorg_df(df)
cor = show_heatmap(df)
cor.to_csv('corr_esign.csv')
df.drop('risk_score_4', axis =1, inplace=True)
show_heatmap(df)
X_train, X_test, y_train, y_test = preprocessing(df)
X_train, X_test = fit_scale(X_train, X_test)
X_train, X_test, y_train, y_test = LDA_implmnt(X_train, X_test, y_train, y_test)
log_model = model_logistic(X_train, y_train)
rf_model = model_rf(X_train, y_train)
xgb_model = model_xgb(X_train, y_train)
get_model_performance(xgb_model, X_test, y_test)
roc_df = optimal_ts(xgb_model,X_test, y_test)






roc_plot(xgb_model, X_test, y_test)
dataNew = df
F, p = stats.f_oneway(dataNew['risk_score'],dataNew['risk_score_2'],dataNew['risk_score_3'],dataNew['risk_score_4'],dataNew['risk_score_5'])
# Seeing if the overall model is significant
print('F-Statistic=%.3f, p=%.5f' % (F, p))
model = sm.OLS('age ~ (e_signed)', df).fit()
model.summary()

df[df.iloc[:,-1]==1].count
df[df.iloc[:,-1]==0].count

sns.jointplot(x='age', y='income', data=df ,kind='kde')

sns.countplot(x='e_signed', data = df)
print(df.columns)
plot3d=px.scatter_3d(df, x='age',y='income', z='ext_quality_score')
plot3d.show()

df_house = df[df['home_owner']==1]
df_not_house = df[df['home_owner']==0]

plt.scatter(df_house.age,df_house.ext_quality_score, marker='^')
plt.scatter(df_not_house.age,df_not_house.ext_quality_score)

vif = calc_vif(df[['risk_score', 'risk_score_2','risk_score_3', 'risk_score_4', 'risk_score_5']])
vif2 = calc_vif(df[['risk_score', 'risk_score_2', 'risk_score_4', 'risk_score_5']])
vif3 = calc_vif(df[['risk_score', 'risk_score_2', 'risk_score_4']])

