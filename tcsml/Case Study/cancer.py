# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 23:41:56 2021

@author: rohan
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statscal import calc_vif



# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer(return_X_y=True,as_frame=True)


df = cancer[0]

del cancer

df.info()

vif = calc_vif(df)

#sns.pairplot(df)

sns.heatmap(df.isnull())

sns.scatterplot(df['mean radius'], df['mean perimeter'])
sns.pairplot(df[['mean radius','mean perimeter']])
#dropping  mean radius cuz is colinear with mean perimeter
df_mod1 =df.drop(['mean perimeter'], axis=1)
vif_mod1 = calc_vif(df_mod1)

df_mod2 =df.drop(['mean radius'], axis=1)
vif_mod2 = calc_vif(df_mod2)

#del [vif_mod2, df_mod2]

df_mod3 = df_mod1.drop(['worst perimeter'], axis = 1)
vif_mod3 = calc_vif(df_mod3)

test_vif = calc_vif(df_mod3[['mean radius', 'worst radius']])

df_mod4 = df_mod3.drop(['mean radius'], axis = 1)
vif_mod4 = calc_vif(df_mod4)

sns.scatterplot(df['mean fractal dimension'], df['worst radius'])

sns.scatterplot(df['mean fractal dimension'], df['worst fractal dimension'])

test_vif2 = calc_vif(df[['mean fractal dimension','worst fractal dimension']])



sns.pairplot(df[['mean radius','mean perimeter','worst radius','worst perimeter']])
cols = df.loc[:,['mean radius','mean perimeter','worst radius','worst perimeter']]
df['new_col'] = cols.mean(axis=1)


df_mod5 = df.drop(['mean radius','mean perimeter','worst radius','worst perimeter', 'mean area', 'worst area'], axis=1)
vif_mod5 = calc_vif(df_mod5)
sns.scatterplot(df['mean area'], df[''])
'''

