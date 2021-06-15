# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 23:41:56 2021

@author: rohan
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt




# Import Cancer data drom the Sklearn library
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer(return_X_y=True,as_frame=True)
cancer

df = cancer[0]

df.info()

