#!/usr/bin/env python
# coding: utf-8

# # Flight Fare Prediction

# In[1]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 


# ## Importing the dataset
#  Since data is in form of excel file we have to use pandas read_excel to load the data

# In[2]:


data_train=pd.read_excel('Data_Train.xlsx')


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


data_train.head()


# In[5]:


data_train.shape


# In[6]:


data_train.info()


# In[7]:


# Price is my dependent feature or the target variable while rest are independent features


# In[8]:


#checking for null values

data_train.isnull().sum()  


# In[9]:


#removing all the null values

data_train.dropna(inplace=True)  


# In[10]:


data_train.isnull().sum()        


# ## EDA

# In[11]:


# we will take out day and month from Date_of_Journey to make our prediction easy. 
# It can be done by using pandas to_datetime to convert object data type to datetime dtype.
# .dt.day method will extract only day of that date
# .dt.month method will extract only month of that date


# In[ ]:





# In[ ]:




