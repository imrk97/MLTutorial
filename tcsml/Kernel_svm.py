#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, f1_score


# In[2]:


#reading data from csv
dataset = pd.read_csv('data/Social_Network_Ads.csv')
dataset.head()


# In[3]:


dataset.info()


# In[4]:


sns.pairplot(dataset)


# In[5]:


dataset.describe()


# In[6]:


#dividing features and labels
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[7]:


#train and test set split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train[:5],'\n\n', y_test[-5:])


# In[8]:


#Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
print(X_train[:5],'\n\n', X_test[:5])


# In[9]:


#fitting the train and test set in classifier
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)


# In[10]:


#predicting a value
print(classifier.predict(sc.transform([[25, 77000]])))


# In[11]:


#visualising the confusion matrix
y_hat = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_hat)
print(cm)
print(sns.heatmap((cm/np.sum(cm)), fmt='.2%', annot=True, cmap='Blues'))


# In[12]:


#f1 score of the model
f1 = f1_score(y_test,y_hat);f1

