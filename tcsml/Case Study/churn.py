# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 14:12:47 2021

@author: rohan
"""
'''['user', 'churn', 'age', 'housing', 'credit_score', 'deposits',
       'withdrawal', 'purchases_partners', 'purchases', 'cc_taken',
       'cc_recommended', 'cc_disliked', 'cc_liked', 'cc_application_begin',
       'app_downloaded', 'web_user', 'app_web_user', 'ios_user',
       'android_user', 'registered_phones', 'payment_type', 'waiting_4_loan',
       'cancelled_loan', 'received_loan', 'rejected_loan', 'zodiac_sign',
       'left_for_two_month_plus', 'left_for_one_month', 'rewards_earned',
       'reward_rate', 'is_referred']'''
from scipy import stats

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import NearMiss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc, plot_roc_curve
from xgboost import XGBClassifier

def load_df():
    return pd.read_csv('churn_data.csv')
def impute_missing_reward_points(cols):
    rewards_earned = cols[0]
    rejected_loan = cols[1]
    if pd.isnull(rewards_earned):
        if rejected_loan:
            return 9
        else:
            return 25
    else:
        return rewards_earned
def get_encoded_df(df):
    one_hot = pd.get_dummies(df[['payment_type','housing']])
    df = df.drop(['housing','payment_type'],axis = 1)
    df = df.join(one_hot)
    return df
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
'''
def ttest_cols(df):
    cols = df.columns
    cols = cols.tolist().remove('churn')
    print(cols)
    t_lst = list()
    p_lst = list()
    df_churn = df.loc[df['churn']==1]
    df_not_churn = df.loc[df['churn']==0]
    for col in cols:
        t,p = stats.ttest_ind(df_churn.loc[col], df_not_churn.loc[col])
        t_lst.append(t)
        p_lst.append(p)
    return pd.DataFrame(zip(col,t,p),columns=['Features','t', 'p'])'''
df = load_df()
df.columns
type(df.columns)
corr = df.corr()
sns.heatmap(corr)

sns.heatmap(df.isnull())
print(df.isnull().any())
sns.boxplot(df['rejected_loan'], df['rewards_earned'])
sns.boxplot(df['left_for_two_month_plus'], df['rewards_earned'])
sns.boxplot(df['registered_phones'], df['rewards_earned'])
sns.boxplot(df['churn'], df['rewards_earned'])
sns.countplot(x='churn', hue = 'housing', data = df)
df['rewards_earned'] = df[['rewards_earned', 'rejected_loan']].apply(impute_missing_reward_points,axis = 1)
sns.heatmap(df.isnull())
sns.histplot(data = df, x='deposits', hue='churn', kde=True)
risks_f,risks_p = stats.f_oneway(df['age'],df['deposits'],df['withdrawal'])
df_churn = df.loc[df['churn']==1]
df_not_churn = df.loc[df['churn']==0]
df.shape
df_churn.shape
df_not_churn.shape
t,p = stats.ttest_ind(df_churn['purchases'], df_not_churn['purchases'])
t,p = stats.ttest_ind(df_churn['deposits'], df_not_churn['deposits'])
t,p = stats.ttest_ind(df_churn['cc_liked'], df_not_churn['cc_liked'])
#df_tp = ttest_cols(df)
df.drop(['credit_score', 'user','cc_liked'], axis = 1, inplace = True)
sns.heatmap(df.isnull(), cmap = 'Blues_r')
df = get_encoded_df(df)
df.drop(['zodiac_sign','app_web_user','android_user','ios_user','housing_na','payment_type_na','app_downloaded'], axis = 1, inplace = True)

df.columns

X = df.drop('churn', axis = 1).values
y = df[['churn']].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.25, random_state=0)
classifier =RandomForestClassifier(n_estimators=120)
#classifier= XGBClassifier()
classifier.fit(X_train, y_train)
print(f'F1 Score: {f1_score(y_test, classifier.predict(X_test))}')
print(f'Accuracy: {accuracy_score(y_test, classifier.predict(X_test))}')
roc_plot(classifier, X_test, y_test)
