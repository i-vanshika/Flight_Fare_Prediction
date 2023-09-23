#!/usr/bin/env python
# coding: utf-8

# # Flight Fare Prediction

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## Importing the dataset
# Since data is in form of excel file we have to use pandas read_excel to load the data

# In[2]:


train_data=pd.read_excel('Data_Train.xlsx')


# In[3]:


pd.set_option('display.max_columns', None)


# In[4]:


train_data.head()


# In[5]:


train_data.shape


# In[6]:


train_data.info()


# In[7]:


# Price is my dependent feature or the target variable while rest are independent features


# In[8]:


#checking for null values

train_data.isnull().sum()  


# In[9]:


train_data.dropna(inplace=True)


# In[10]:


train_data.isnull().sum()


# # EDA

# In[11]:


# we will take out day and month from Date_of_Journey to make our prediction easy. 
# It can be done by using pandas to_datetime to convert object data type to datetime dtype.
# .dt.day method will extract only day of that date
# .dt.month method will extract only month of that date


# In[12]:


train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'], format='%d/%m/%Y').dt.day
train_data['Journey_month'] = pd.to_datetime(train_data.Date_of_Journey, format='%d/%m/%Y').dt.month


# In[13]:


train_data.head()


# In[14]:


# Since we have converted Date_of_Journey into integers, so now we can drop as it is of no use  

train_data.drop(['Date_of_Journey'], axis=1, inplace=True)


# In[15]:


train_data.head()


# In[16]:


# Departure time is when a plane leaves the gate. 
# Similar to Date_of_Journey we can extract values from Dep_Time.


# In[17]:


# Extracting hours

train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour

# Extracting Minutes

train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute

# Dropping Dep_Time column

train_data.drop(['Dep_Time'], axis=1, inplace=True)


# In[18]:


train_data.head()


# In[19]:


# Arrival time is when the plane pulls up to the gate.
# Similar to Date_of_Journey we can extract values from Arrival_Time


# In[20]:


# Extracting Hours
train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour

#Extracting minutes
train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute


# In[21]:


# Dropping Arrival_time column
train_data.drop(["Arrival_Time"], axis=1, inplace=True)


# In[22]:


train_data.head()


# In[23]:


# Time taken by the plane to reach its destination is called Duration
# It is the difference between the Departure Time and the Arrival Time


# In[24]:


# Assigning and converting Duration column into list
duration = list(train_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2: # Checking if duration length is 2 or not i.e,whether the time is in hr and mins or not
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour


# In[25]:


duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
    
# int is converting the string into integer 


# In[26]:


# Adding duration_hours and duration_mins list to train_data dataframe

train_data['Duration_hours'] = duration_hours
train_data['Duration_mins'] = duration_mins


# In[27]:


train_data.head()


# In[28]:


train_data.drop(['Duration'], axis=1, inplace=True)


# In[29]:


train_data.head()


# In[30]:


train_data.shape


# ## Handling Categorical Data
# 
# One can find many ways to handle categorical data. Some of them categorical data are,
# 
# - Nominal Data -> data is not in any order --> OneHotEncoder is used in this case. Example-States
# - Ordinal data --> data are in order --> LabelEncoder is used in this case. Example-Ranking

# In[35]:


train_data.Airline.value_counts()


# In[ ]:




