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


# In[12]:


data_train['Journey_day'] = pd.to_datetime(data_train['Date_of_Journey'], format='%d/%m/%Y').dt.day


# In[13]:


data_train['Journey_month'] = pd.to_datetime(data_train.Date_of_Journey, format='%d/%m/%Y').dt.month


# In[14]:


data_train.head()


# In[15]:


# Since we have converted Date_of_Journey into integers, so now we can drop as it is of no use  

data_train.drop(['Date_of_Journey'], axis=1, inplace=True)


# In[16]:


data_train.head()


# In[17]:


# Departure time is when a plane leaves the gate. 
# Similar to Date_of_Journey we can extract values from Dep_Time.


# In[18]:


# Extracting hours

data_train['Dep_hour'] = pd.to_datetime(data_train['Dep_Time']).dt.hour

# Extracting Minutes

data_train['Dep_min'] = pd.to_datetime(data_train['Dep_Time']).dt.minute

# Dropping Dep_Time column

data_train.drop(['Dep_Time'], axis=1, inplace=True)


# In[19]:


data_train.head()


# In[20]:


# Arrival time is when the plane pulls up to the gate.
# Similar to Date_of_Journey we can extract values from Arrival_Time


# In[21]:


# Extracting Hours
data_train['Arrival_hour'] = pd.to_datetime(data_train['Arrival_Time']).dt.hour

#Extracting minutes
data_train['Arrival_min'] = pd.to_datetime(data_train['Arrival_Time']).dt.minute


# In[22]:


# Dropping Arrival_time column
data_train.drop(["Arrival_Time"], axis=1, inplace=True)


# In[23]:


data_train.head()


# In[24]:



# It is the difference between the Departure Time and the Arrival Time


# In[ ]:




