# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:21:21 2021

@author: rohan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
#import the dataframe

data = pd.read('filename.csv')

x = data.iloc(:,:-1)

y = data.iloc(:,3)
imputer = Imputer(missing_value='NaN', strategy = 'mean', axis = 0)
iputer = imputer.fit(x.iloc[:,1:3])

x.iloc[:,1:3] = imputer.transform(x.iloc[:,1:3])

import LablerEncoder,one hotencoder

lebal = leblen()
x.iloc[:,0] = lebalenco_x.fit_trans(x.iloc[:,0])
onehotencoder = onehotencoder(categorical_features = [0])
x = onehotencoder.fit_transform(X)