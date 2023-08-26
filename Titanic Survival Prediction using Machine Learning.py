#!/usr/bin/env python
# coding: utf-8

# # Importing the dependencies

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[2]:


from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB


# # Data Collection and Preprocessing
# 

# In[3]:


# load the  data from csv to Pandas DataFrame
titanic_data=pd.read_csv('tested.csv')


# In[4]:


titanic_data.sample(5)


# In[5]:


titanic_data.head()


# In[6]:


# number of rows and columns
titanic_data.shape


# In[7]:


# Getting some information about data 
titanic_data.info()


# In[8]:


# check the number of missing values in each col
titanic_data.isnull().sum()


# Handling missing values
# 

# In[9]:


#Drop the 'Cabin' column from the DataFrame
titanic_data = titanic_data.drop(columns='Cabin',axis=1)
#axis=1 for columns and axis=0 for rows


# In[10]:


# Replacing the missing values in age column with mean value
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)


# titanic_data.info()

# In[11]:


#Finding the mode of['Embark'] column
print(titanic_data['Embarked'].mode())


# In[12]:


print(titanic_data['Embarked'].mode()[0])


# In[13]:


#Replacing the missing values in ['Embarked'] column
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode(),inplace=True)


# In[14]:


titanic_data.isnull().sum()


# In[15]:


titanic_data['Fare'].fillna(titanic_data['Fare'].mean(),inplace=True)


# In[16]:


titanic_data.isnull().sum()


# # Data Analysis

# In[17]:


3#Getting some statistical measures about data
titanic_data.describe()


# In[18]:


#Finding the number of people survived or not survived
titanic_data['Survived'].value_counts()


# # Data Visualisation

# In[19]:


sns.set()
# for theme


# In[20]:


# making a count plot for "Survived" columns
sns.countplot("Survived",data=titanic_data)


# In[21]:


#Finding the number of people survived or not survived
titanic_data['Sex'].value_counts()


# In[22]:


# making a count plot for "Survived" columns
sns.countplot("Sex",data=titanic_data)


# In[23]:


# number of surviver genderwise
sns.countplot("Sex",hue="Survived",data=titanic_data)


# In[24]:


sns.countplot("Pclass",data=titanic_data)


# In[25]:


sns.countplot("Pclass",hue="Survived",data=titanic_data)


# #Encoding the categorigal columns 
# 

# In[26]:


titanic_data['Sex'].value_counts()


# In[27]:


titanic_data['Embarked'].value_counts()


# In[28]:


# converting the categorical columns
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}},inplace=True)


# In[29]:


titanic_data.head()


# In[30]:


titanic_data=titanic_data.drop(['Name'],axis=1)


# In[31]:


titanic_data.sample(5)


# In[32]:


titanic_data[["Pclass","Survived"]].groupby(['Pclass'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[33]:


titanic_data[["SibSp","Survived"]].groupby(['SibSp'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# In[34]:


titanic_data[["Embarked","Survived"]].groupby(['Embarked'],as_index=False).mean().sort_values(by='Survived',ascending=False)


# # Model training

# In[35]:


Column_train=['Pclass','Age','SibSp','Parch','Fare','Sex','Embarked']
X=titanic_data[Column_train]
Y=titanic_data['Survived']


# In[36]:


X['Age']=X['Age'].fillna(X['Age'].median())
X['Age'].isnull().sum()


# # Model building and training
# 

# In[37]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[38]:


X_train


# In[39]:


X_test


# In[40]:


print(X.shape,X_train.shape,X_test.shape)


# In[41]:


Y_train


# In[42]:


Y_test


# # LogisticRegression
# 

# In[43]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,Y_train)


# In[44]:


X_train_prediction = model.predict(X_train)


# In[45]:


from sklearn.metrics import accuracy_score
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[46]:



X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# # Checking for a Random Person:
# 

# In[47]:


input_data = (3,0,35,0,0,8.05,0)  
input_data_as_numpy_array = np.asarray(input_data)


# In[48]:


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[49]:


prediction = model.predict(input_data_reshaped)
#print(prediction)
if prediction[0]==0:
    print("Dead")
if prediction[0]==1:
    print("Alive")


# In[ ]:




