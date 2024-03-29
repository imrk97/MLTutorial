#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from io import StringIO
from datetime import datetime, timedelta


# In[2]:


data = StringIO('''Name Department BMI ID Timestamp
John Computers 200 1011000545983 2021-04-28T17:09:32.092
Mark Mechanical 200 1011000565799 2021-04-27T09:49:45.023
Peter Electronics 200 1011000569814 2021-04-27T09:43:14.378
Alex Computers 400 1011000574698 2021-04-27T14:36:31.521
Karry Computers 200 1011000509788 2021-04-28T16:44:07.924
Saul Electronics 200 1011000548188 2021-04-27T12:18:12.099
Thomas Electronics 200 1011000558720 2021-04-27T09:18:14.439
Steve Electronics 300 1011000566286 2021-04-28T12:27:40.503
Wesley Electronics 200 1011000570818 2021-04-28T14:27:40.504
Pet Mechanical 300 1011000571747 2021-04-29T16:27:40.515
Jenny Electronics 200 1011000572064 2021-04-28T12:27:40.506
Matthews Mechanical 200 1011000572066 2021-04-28T12:27:40.507
Rio Mechanical 300 1011000572067 2021-04-29T12:27:40.508
Pat Mechanical 200 1011000572069 2021-04-29T12:27:40.509
Roy Computers 200 1011000572070 2021-04-28T12:27:40.510
Robbie Mechanical 200 1011000572072 2021-04-29T19:27:40.511
Dave Computers 200 1011000572073 2021-04-30T12:27:40.512
Alex Computers 200 1011000574698 2021-04-28T12:27:40.513
Shaun Mechanical 200 1011000577065 2021-04-27T22:47:40.514''')
print(data)


# In[3]:


df = pd.read_csv(data, sep = ' ')
df


# In[4]:


df['Timestamp'] = [datetime.strptime(d, '%Y-%m-%dT%H:%M:%S.%f') for d in df['Timestamp']]
df


# In[5]:


df['Date'] = [d.date() for d in df['Timestamp']]
df['Time'] = [d.time() for d in df['Timestamp']]
df


# In[6]:


df.keys()
temp = df[df['BMI'] != 200]
temp


# In[7]:


Dropout = temp[['ID', 'Name', 'Department', 'Time']].copy()
Dropout


# In[8]:


#just testing with time
time1 = timedelta(hours = 48)
time2 = timedelta(hours = 24, days = 23, seconds = 33)


# In[9]:


print(time1 + time2)


# In[10]:


#temp['Timestamp'] - df['Timestamp'] > timedelta(hours = 48)
duplicate = temp[df.duplicated(subset = 'ID', keep = False) ]
duplicate


# In[11]:


duplicate = duplicate[['Name', 'Department', 'ID']]
duplicate


# In[ ]:




