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

# In[31]:


train_data.Airline.value_counts()


# In[32]:


# Airline vs Price

sns.catplot(x='Airline', y='Price', data=train_data.sort_values('Price', ascending= False), kind='boxen', height=6, aspect=3)
plt.show()

# All the values will be sorted according to the price in descending order


# In[33]:


# From above graph we can see that Jet Airways Business have the highest price
# Apart from the first Airline almost all are having similar median


# In[34]:


# Airline is Nominal Categorical data we will perform OneHotEncoding by using get_dummies

Airline = train_data[["Airline"]]
Airline = pd.get_dummies(Airline, drop_first=True)  #drop_first=true --->hunt more
Airline.head()


# In[35]:


train_data['Source'].value_counts()


# In[36]:


# Source vs Price

sns.catplot(x= 'Source', y='Price', data=train_data.sort_values('Price', ascending= False), height=6, aspect=3, kind='boxen')
plt.show()


# In[37]:


# As Source is Nominal Categorical data we will perform OneHotEncoding

Source= train_data[['Source']]
Source= pd.get_dummies(data=Source, drop_first=True)
Source.head()


# In[38]:


train_data['Destination'].value_counts()


# In[39]:


# As Destination is Nominal Categorical data we will perform OneHotEncoding

Destination=train_data[['Destination']]
Destination=pd.get_dummies(data=Destination, drop_first=True)
Destination.head()


# In[40]:


train_data['Route']


# In[41]:


train_data['Additional_Info'].value_counts()


# In[42]:


train_data.head()


# In[43]:


# Since we have already handled Source and Destination so rout is of no use 
# Similarly Additonal_info contains almost 80% no_info


# In[44]:


train_data.drop(['Additional_Info', 'Route'], axis=1, inplace=True)


# In[45]:


train_data.head()


# In[46]:


train_data['Total_Stops'].value_counts()


# In[47]:


# As this is the case of Ordinal Categorical Type we perform LabelEncoder
# Here values are assigned with corresponding keys


# In[48]:


train_data.replace({'non-stop': 0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4 }, inplace= True)


# In[49]:


train_data.head()


# In[50]:


# Concatenate the dataframe --->train_data + Airline + Source + Destination

train_data=pd.concat([train_data, Airline, Source, Destination], axis=1)


# In[51]:


train_data.head()


# In[52]:


train_data.drop(['Airline','Source','Destination'], axis=1, inplace=True)


# In[53]:


train_data.head()


# In[54]:


train_data.shape


# # Test Set

# ## Why should we do preprocessing steps separately for train and test data?
# - To avoid Data Leakage
# - If we will combine together then model would be knowing some of the information about the test data 

# In[55]:


test_data=pd.read_excel('Test_set.xlsx')


# In[56]:


test_data.head()


# In[57]:


test_data.columns


# In[58]:


# We will not be having Price as the column in the Test data because Price is our dependent feature.
# We will apply all the same steps for Test data as done above for Train data.


# In[59]:


# Pre processing

print('Test data info')
print('-'*75)
print(test_data.info())

print()
print()

print("Null values :")
print("-"*75)
test_data.dropna(inplace=True)
print(test_data.isnull().sum())


# In[60]:


# EDA

# Date_of_Journey
test_data["Journey_day"] = pd.to_datetime(test_data.Date_of_Journey, format="%d/%m/%Y").dt.day
test_data["Journey_month"] = pd.to_datetime(test_data["Date_of_Journey"], format = "%d/%m/%Y").dt.month
test_data.drop(["Date_of_Journey"], axis = 1, inplace = True)

# Dep_Time
test_data["Dep_hour"] = pd.to_datetime(test_data["Dep_Time"]).dt.hour
test_data["Dep_min"] = pd.to_datetime(test_data["Dep_Time"]).dt.minute
test_data.drop(["Dep_Time"], axis = 1, inplace = True)

# Arrival_Time
test_data["Arrival_hour"] = pd.to_datetime(test_data.Arrival_Time).dt.hour
test_data["Arrival_min"] = pd.to_datetime(test_data.Arrival_Time).dt.minute
test_data.drop(["Arrival_Time"], axis = 1, inplace = True)

test_data.head()


# In[61]:


# Duration 

duration = list(test_data["Duration"])

for i in range(len(duration)):
    if len(duration[i].split()) != 2: # Checking if duration length is 2 or not i.e,whether the time is in hr and mins or not
        if "h" in duration[i]:
            duration[i] = duration[i].strip() + " 0m"   # Adds 0 minute
        else:
            duration[i] = "0h " + duration[i]           # Adds 0 hour


duration_hours = []
duration_mins = []
for i in range(len(duration)):
    duration_hours.append(int(duration[i].split(sep = "h")[0]))    # Extract hours from duration
    duration_mins.append(int(duration[i].split(sep = "m")[0].split()[-1]))   # Extracts only minutes from duration
    
# Adding duration_hours and duration_mins list to train_data dataframe

test_data['Duration_hours'] = duration_hours
test_data['Duration_mins'] = duration_mins
test_data.drop(["Duration"], axis=1, inplace =True)
test_data.head()


# In[62]:


# Categorical Data

print("Airline")
print("-"*75)
print(test_data["Airline"].value_counts())
Airline = pd.get_dummies(data=Airline, drop_first=True)

print()

print("Source")
print("-"*75)
print(test_data["Source"].value_counts())
Source = pd.get_dummies(data=Source, drop_first=True)

print()

print("Destination")
print("-"*75)
print(test_data["Destination"].value_counts())
Destination = pd.get_dummies(data=Destination, drop_first=True)

test_data.drop(["Airline","Source","Destination"], axis=1, inplace=True)
test_data.head()


# In[63]:


# Replacing Total_Stops
test_data.replace({"non-stop": 0, "1 stop": 1, "2 stops": 2, "3 stops": 3, "4 stops": 4}, inplace = True)

# Concatenate dataframe --> test_data + Airline + Source + Destination
test_data=pd.concat([test_data,Airline,Source,Destination], axis=1)

# Dropping Route and Additional_Info
test_data.drop(["Route", "Additional_Info"], axis = 1, inplace = True)
test_data.head()


# In[64]:


test_data.shape


# # Feature Selection
# 
# Finding out the best feature which will contribute and have good relation with target variable i.e.,Price Following are some of the feature selection methods,
# - Heatmap
# - feature_importance_
# - SelectKBest

# In[65]:


train_data.shape


# In[66]:


train_data.columns


# In[67]:


# Making separate variables for dependent and independent features 
# X will be indicating independent features and Y will be indicating dependent feature i.e, Price 


# In[68]:


# loc is labeled based operator
# X does not contain Price

X = train_data.loc[:,['Total_Stops', 'Journey_day', 'Journey_month', 'Dep_hour',
       'Dep_min', 'Arrival_hour', 'Arrival_min', 'Duration_hours',
       'Duration_mins', 'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Jet Airways Business',
       'Airline_Multiple carriers',
       'Airline_Multiple carriers Premium economy', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Airline_Vistara Premium economy',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Cochin', 'Destination_Delhi', 'Destination_Hyderabad',
       'Destination_Kolkata', 'Destination_New Delhi']]
X.head()


# In[69]:


y = train_data.iloc[:,1]
y.head()


# In[70]:


# Correlation between the independent and the dependent feature attributes

plt.figure(figsize=(18,18))
sns.heatmap(train_data.corr(), annot=True, cmap = "RdYlGn")
plt.show()


# In[71]:


# Important feature can be found out by ExtraTreesRegressor

from sklearn.ensemble import ExtraTreesRegressor
selection= ExtraTreesRegressor()
selection.fit(X,y)


# In[72]:


print(selection.feature_importances_)


# In[73]:


# plot graph of feature importances for better visualisation

plt.figure(figsize=(12,8))
feat_importances = pd.Series(selection.feature_importances_, index=X.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show()


# # Fitting model using Random Forest
# 
# - Split dataset into train and test set in order to prediction w.r.t X_test
# - If needed do scaling of data but Scaling is not done in Random forest
# - Import model
# - Fit the data
# - Predict w.r.t X_test
# - In regression check RSME (Relative Squared Error) Score
# - Plot graph

# In[74]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y, test_size=0.2, random_state=42)


# In[75]:


from sklearn.ensemble import RandomForestRegressor
reg_rf = RandomForestRegressor()
reg_rf.fit(X_train, y_train)


# In[76]:


y_pred = reg_rf.predict(X_test)


# In[77]:


reg_rf.score(X_train, y_train)


# In[78]:


reg_rf.score(X_test, y_test)


# In[79]:


sns.distplot(y_test- y_pred)
plt.show()


# In[80]:


# Since it is forming Gaussian Distribution, which means results are very good.


# In[81]:


plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[82]:


from sklearn import metrics


# In[83]:


print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[84]:


# RMSE/(max(DV)-min(DV))

2090.5509/(max(y)-min(y))


# In[85]:



metrics.r2_score(y_test, y_pred)


# 
# 
# # Hyperparameter Tuning
# 
# - Choose following method for hyperparameter tuning: (i) RandomizedSearchCV --> Fast   (ii) GridSearchCV
# - Assign hyperparameters in form of dictionary
# - Fit the model
# - Check best paramters and best score

# In[86]:


from sklearn.model_selection import RandomizedSearchCV


# In[87]:


# Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to be considered at every split
max_features= ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int (x) for x in np.linspace(5,30, num=6)]
# Minimum number of samples required to split a node
min_samples_split = [2,5,10,15,100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1,2,5,10]


# In[91]:


# Creating Random Grid

random_grid = {'n_estimators':n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf}


# In[92]:


# Random search of parameters, using 5 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = reg_rf, param_distributions = random_grid,scoring='neg_mean_squared_error', 
                               n_iter= 10, cv=5, verbose=2, random_state=42, n_jobs=1)


# In[93]:


rf_random.fit(X_train, y_train)


# In[94]:


# for best parameters
rf_random.best_params_  


# In[96]:


prediction = rf_random.predict(X_test)


# In[97]:


plt.figure(figsize=(8,8))
sns.distplot(y_test-prediction)
plt.show()


# In[98]:


# above graph also looks like gaussian distributed data


# In[100]:


plt.figure(figsize=(6,6))
plt.scatter(y_test, prediction, alpha=0.5)
plt.xlabel("y_test")
plt.ylabel("y_pred")
plt.show()


# In[101]:


print('MAE:', metrics.mean_absolute_error(y_test, prediction))
print('MSE:', metrics.mean_squared_error(y_test, prediction))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

