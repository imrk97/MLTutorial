#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import statsmodels.formula.api as sm
import seaborn as sns


# In[8]:


ad = pd.read_csv("data/advertising.csv")
ad.head()


# In[9]:


model = sm.ols('Sales ~ TV + Radio + Newspaper', ad).fit()
print(model.params)


# In[10]:


sns.heatmap(ad.corr(), annot=True)


# In[ ]:





# In[ ]:




