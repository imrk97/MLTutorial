#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


df = pd.read_csv('data/Salary_Data.csv')
df


# In[7]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)
r = LinearRegression()
r.fit(X_train, y_train)

y_pred = r.predict(X_test)


# In[6]:


plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, r.predict(X_train), color= 'red')


# In[ ]:




