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
    classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
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
    return roc_auc
def optimal_ts(classifier, X_test, y_test):
    probs = classifier.predict_proba(X_test)
    #a  = classifier.predict()
    preds = probs[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    #print("Threshold value is:", optimal_threshold)
    
    new_preds = list()
    
    for i in preds:
        if i > optimal_threshold:
            new_preds.append(1)
        else:
            new_preds.append(0)
    
    cm = confusion_matrix(y_test, new_preds)
    print(cm)
    
    print_str = 'Threshold value is: {}\nAccuracy: {}\nf1-score: {}'.format(optimal_threshold,accuracy_score(y_test, new_preds),f1_score(y_test,new_preds))
    #"Threshold value is:" optimal_threshold,f'Accuracy: {accuracy_score(y_test, new_preds)}',f'f1-score: {f1_score(y_test,new_preds)}'
    #with open('model_value.txt', 'w') as w:
        #w.write(print_str)
    #print(f'Accuracy: {accuracy_score(y_test, new_preds)}')
    #print(f'f1-score: {f1_score(y_test,new_preds)}')
    print(print_str)
    roc_df= pd.DataFrame(list(zip(fpr, tpr, thresholds)), columns=['FPR','TPR', 'Threshold'])
    return roc_df, optimal_threshold,accuracy_score(y_test, new_preds), f1_score(y_test,new_preds)
            
df = load_df()
sns.countplot(x='e_signed',hue='has_debt', data = df)
sns.countplot(x='e_signed',hue='home_owner', data = df)

df['pay_schedule'].unique()



df_0 = df.loc[df['e_signed']==0]
df_1 = df.loc[df['e_signed']==1]
plt.plot(df_0['age'], np.zeros_like(df_0['age']), color='red', marker='^')
plt.plot(df_1['age'], np.zeros_like(df_1['age']), color='blue', marker = 'o')
plt.show()
sns.FacetGrid(df,hue='e_signed',size=8).map(plt.scatter, 'risk_score', 'ext_quality_score').add_legend();plt.show()
sns.pairplot(df[['risk_score', 'risk_score_2','risk_score_3', 'risk_score_4', 'risk_score_5','e_signed']], hue='e_signed', height=2)

sns.pairplot(df[['risk_score', 'ext_quality_score','ext_quality_score_2', 'inquiries_last_month', 'e_signed']], hue='e_signed', height=2)
#df.drop('ext_quality_score_2', axis =1, inplace=True)

df = get_encoded_df(df)
df = reorg_df(df)
dups = df.duplicated()
print(dups.any())
#cor = show_heatmap(df)
#cor.to_csv('corr_esign.csv')
#df.drop('risk_score_4', axis =1, inplace=True)
#show_heatmap(df)
X_train, X_test, y_train, y_test = preprocessing(df)
X_train, X_test = fit_scale(X_train, X_test)
#X_train, X_test, y_train, y_test = LDA_implmnt(X_train, X_test, y_train, y_test)
model = model_xgb(X_train, y_train)
#get_model_performance(model, X_test, y_test)
roc = optimal_ts(model,X_test, y_test)
roc_auc = roc_plot(model, X_test, y_test)
#X_train = np.delete(X_train, 11, axis = 1)
def var_auc(X_train, y_train, X_test, y_test):
    model_dict=dict()
    acc=list()
    f1 = list()
    auc = list()
    thr = list()
    print(X_train.shape[1])
    X_test_d = X_test
    X_train_d = X_train
    for i in range(X_train.shape[1]):
        print(f'in loop: {i}')
        X_train_d = np.delete(X_train_d, i, axis = 1)
        print(X_train.shape)
        X_test_d = np.delete(X_test_d, i, axis = 1)
        model = model_xgb(X_train_d, y_train)
        roc = optimal_ts(model, X_test_d, y_test)
        roc_auc = roc_plot(model,X_test_d, y_test)
        auc.append(roc_auc)
        acc.append(roc[2])
        f1.append(roc[3])
        thr.append(roc[1])
        X_test_d = X_test
        X_train_d = X_train
        
    model_dict = {'AUC':auc, 'Accuracy':acc, 'F1 Score':f1,'Thresold Value': thr}
    model_df = pd.DataFrame.from_dict(model_dict)
    return model_df
model_df1 = var_auc(X_train, y_train, X_test, y_test)

X_train = np.delete(X_train, 18, axis = 1)
X_test = np.delete(X_test, 18, axis=1)
model_df2 = var_auc(X_train, y_train, X_test, y_test)

X_train = np.delete(X_train, 20, axis = 1)
X_test = np.delete(X_test, 20, axis=1)
model_df3 = var_auc(X_train, y_train, X_test, y_test)

X_train = np.delete(X_train, 18, axis = 1)
X_test = np.delete(X_test, 18, axis=1)
model_df4 = var_auc(X_train, y_train, X_test, y_test)

X_train = np.delete(X_train, 4, axis = 1)
X_test = np.delete(X_test, 4, axis=1)
model_df5 = var_auc(X_train, y_train, X_test, y_test)

X_train = np.delete(X_train, 4, axis = 1)
X_test = np.delete(X_test, 4, axis=1)
model_df6 = var_auc(X_train, y_train, X_test, y_test)

X_train = np.delete(X_train, 6, axis = 1)
X_test = np.delete(X_test, 6, axis=1)
model_df7 = var_auc(X_train, y_train, X_test, y_test)


model = model_xgb(X_train, y_train)
roc = optimal_ts(model,X_test, y_test)
roc_auc = roc_plot(model, X_test, y_test)

final_model_specs = var_auc(X_train, y_train, X_test, y_test)



dataNew = df
#risks = stats.f_oneway(dataNew['risk_score'],dataNew['risk_score_2'],dataNew['risk_score_3'],dataNew['risk_score_4'],dataNew['risk_score_5'])
risks_f,risks_p = stats.f_oneway(dataNew['risk_score_2'],dataNew['risk_score_3'],dataNew['risk_score_4'],dataNew['risk_score_5'])
quality_f, quality_p = stats.f_oneway(dataNew['ext_quality_score'], dataNew['ext_quality_score_2'])

sns.histplot(data = df, x='risk_score', hue='home_owner', kde=True)
sns.histplot(data = df, x='risk_score', hue='has_debt', kde=True)
sns.histplot(data = df, x='risk_score', hue='e_signed', kde=True)
sns.histplot(data = df, x='ext_quality_score', hue='e_signed', kde=True)


df_0 = df.loc[df['e_signed']==0]
df_1 = df.loc[df['e_signed']==1]
t_q,p_q =stats.ttest_ind(df_0['ext_quality_score'], df_1['ext_quality_score'])
t_r, p_r= stats.ttest_ind(df_0['risk_score'], df_1['risk_score'])
var= stats.chi2

print('F-Statistic=%.3f, p=%.5f' % (F, p))
model = sm.OLS('age ~ (e_signed)', np.asarray(df)).fit()
model.summary()

df[df.iloc[:,-1]==1].count
df[df.iloc[:,-1]==0].count

sns.jointplot(x='age', y='income', hue='e_signed', data=df)

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

