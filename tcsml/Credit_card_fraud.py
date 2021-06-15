# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 03:32:35 2021

@author: rohan
"""

import numpy as np
import pandas as pd
import seaborn as sns
data = pd.read_csv('C:/Users/rohan/Desktop/GIT/creditcard.csv')
print(data.head())
sns.pairplot(data[['Amount','Time' ]])
