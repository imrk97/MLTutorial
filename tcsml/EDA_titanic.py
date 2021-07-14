#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
# plot a line, implicitly creating a subplot(111)
plt.plot([1,2,3])
# now create a subplot which represents the top plot of a grid
# with 2 rows and 1 column. Since this subplot will overlap the
# first, the plot (and its axes) previously created, will be removed
plt.subplot(211)
plt.plot(range(12))
plt.subplot(212, facecolor='y') # creates 2nd subplot with yellow background


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train = pd.read_csv('data/titanic_train.csv')
train.head()


# In[4]:


train.info()


# In[5]:


sns.pairplot(train, hue = 'Survived').add_legend()


# In[6]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False)


# In[7]:


sns.countplot(x='Survived', data = train)


# In[8]:


sns.countplot(x='Survived', hue = 'Sex', data = train)
plt.title('Titanic male female survival')


# In[8]:


sns.countplot(x='Survived', hue = 'Pclass', data = train)
plt.title('Titanic Class survival')


# In[10]:


sns.displot(train['Age'].dropna(), kde = True, bins = 40)


# In[10]:


plt.figure(figsize=(12,8))
sns.boxplot(train['Pclass'], train['Age'], )


# In[11]:


def impute_missing_ages(cols):
    age = cols[0]
    Pclass = cols[1]
    if pd.isnull(age):
        if Pclass == 1:
            return 38
        if Pclass == 2:
            return 31
        if Pclass == 3:
            return 28
    else:
        return age


# In[12]:


train['Age'] = train[['Age', 'Pclass']].apply(impute_missing_ages,axis = 1)


# In[13]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False)


# In[14]:


train.drop('Cabin', axis =1, inplace=True)


# In[16]:


sns.heatmap(train.isnull(), yticklabels=False, cbar=False)


# In[15]:


train.head()


# In[18]:


train.info()


# In[19]:


pd.get_dummies(train['Embarked'], drop_first=True).head()


# In[20]:


sex = pd.get_dummies(train['Sex'], drop_first=True)
embarked = pd.get_dummies(train['Embarked'], drop_first=True)
train.drop(['Sex','Embarked','Name','Ticket'], axis = 1,inplace=True)


# In[21]:


train.head()


# In[22]:


train = pd.concat([train,sex,embarked], axis = 1)


# In[23]:


train.head()


# In[24]:


train.info()


# In[25]:


from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [VIF(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[28]:


calc_vif(train)


# In[30]:


train.shape[1]


# In[ ]:




