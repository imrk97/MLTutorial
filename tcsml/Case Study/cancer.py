# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 23:41:56 2021

@author: rohan
"""

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, auc, plot_roc_curve


def calc_vif(X):

    # Calculating VIF
    '''
    This function will return a dataframe that will contain the vif values of each of the features
    '''
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]

    return(vif)


def get_model_performance(classifier, X, y):
    y_pred = classifier.predict(X)
    cm = confusion_matrix(y, y_pred)
    print(cm)
    print(f'Accuracy: {accuracy_score(y, y_pred)}')
    print(f'f1-score: {f1_score(y,y_pred)}')


def roc_plot(classifier, X_test, y_test):
    probs = classifier.predict_proba(X_test)
    #a  = classifier.predict()
    preds = probs[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc}')
    plt.plot(fpr, tpr, label='AUC = %0.4f' % roc_auc, color='blue')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')

    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    return roc_auc



# Import Cancer data drom the Sklearn library
cancer = load_breast_cancer(return_X_y=True, as_frame=True)
df = cancer[0]
df['category'] = cancer[1]
df.columns
cols = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension',
        'category']

sns.heatmap(df.isnull())
sns.scatterplot(df['mean radius'], df['worst radius'])

sns.pairplot(df[['worst radius', 'worst perimeter', 'worst area']])
sns.pairplot(df[['worst smoothness', 'worst compactness', 'worst concavity',
                 'worst concave points', 'worst symmetry', 'worst fractal dimension']])
sns.heatmap(df.isnull(), cmap='Blues_r')
corr1 = df[['mean radius', 'mean perimeter', 'mean area']].corr()
sns.heatmap(corr1)
sns.pairplot(df[['mean radius', 'mean perimeter','mean area', 'category']], hue='category')
sns.pairplot(df[['worst radius', 'worst perimeter','worst area', 'category']], hue='category')
sns.pairplot(hue='category', data=df[['radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error','compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'category']])
sns.pairplot(hue='category', data=df[['mean smoothness', 'mean compactness', 'mean concavity','mean concave points', 'mean symmetry', 'mean fractal dimension', 'category']])
sns.heatmap(df[['mean smoothness', 'mean compactness', 'mean concavity','mean concave points', 'mean symmetry', 'mean fractal dimension', 'category']].corr())
sns.boxplot(x='category', y='mean radius', data=df)
sns.boxplot(x='category', y='mean smoothness', data=df)

df1 = df[['mean radius', 'mean concavity', 'radius error', 'texture error','smoothness error', 'concavity error', 'symmetry error', 'fractal dimension error']]
df2 = df[['worst radius', 'worst concavity', 'radius error', 'texture error','smoothness error', 'concavity error', 'symmetry error', 'fractal dimension error']]
df3 = df[['worst radius','worst concavity','mean radius', 'mean concavity', 'radius error', 'texture error','smoothness error', 'concavity error', 'symmetry error', 'fractal dimension error']]


vif = calc_vif(df)
vif1 = calc_vif(df1)



df1[['category']] = df[['category']]
df2[['category']] = df[['category']]
df3[['category']] = df[['category']]



X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = SVC(kernel='rbf', random_state=0, probability=True)
classifier.fit(X_train, y_train)


cm = confusion_matrix(y_test, classifier.predict(X_test))
print(cm)
get_model_performance(classifier, X_test, y_test)
roc_plot(classifier, X_test, y_test)

# sns.pairplot(df)

sns.heatmap(df.isnull())
print(df.columns)
sns.scatterplot(df['mean radius'], df['mean perimeter'])
sns.pairplot(df[['mean radius', 'mean perimeter', 'mean area']])
df_mod1 = df.drop(['mean perimeter', 'mean area'], axis=1)

sns.scatterplot(df['mean radius'], df['worst radius'])

sns.pairplot(df[['worst radius', 'worst perimeter', 'worst area']])
sns.pairplot(df[['worst smoothness', 'worst compactness', 'worst concavity',
                 'worst concave points', 'worst symmetry', 'worst fractal dimension']])


df_mod2 = df_mod1.drop(['worst perimeter', 'worst area'], axis=1)

vif2 = calc_vif(df_mod2)

df_mod3 = df_mod2.drop('worst radius', axis=1)

vif3 = calc_vif(df_mod3)

df_mod4 = df_mod3.drop('mean fractal dimension', axis=1)

vif4 = calc_vif(df_mod4)

df_mod5 = df_mod4.drop('worst smoothness', axis=1)
vif5 = calc_vif(df_mod5)
