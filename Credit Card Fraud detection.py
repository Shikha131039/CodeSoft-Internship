#!/usr/bin/env python
# coding: utf-8

# # Import modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# # Loading the dataset

# In[3]:


df= pd.read_csv('creditcard.csv')


# In[4]:


df.head()


# In[5]:


#stat info
df.describe()


# In[6]:


# datatype
df.info()


# In[7]:


df.isnull().sum()


# In[8]:


df.shape


# In[9]:


# distribution of legit transaction and fraudulent transaction
df['Class'].value_counts()


# # Exploratory Data Analysis

# In[10]:


def draw_histograms(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax,facecolor='midnightblue')
        ax.set_title(feature+" Distribution",color='DarkRed')
        ax.set_yscale('log')
    fig.tight_layout()  
    plt.show()

draw_histograms(df,df.columns,8,4)


# In[11]:


sns.countplot(df['Class'])


# In[12]:


df_temp = df.drop(columns=['Time','Amount','Class'],axis=1)

fig, ax =plt.subplots(ncols=4,nrows=7,figsize=(20,50))
index =0
ax = ax.flatten()

    
for col in df_temp.columns:
    sns.distplot(df_temp[col],ax=ax[index])
    index +=1
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad =5)
                           
                    


# In[13]:


sns.distplot(df['Time'])


# In[14]:


sns.distplot(df['Amount'])


# In[15]:


corr = df.corr()
plt.figure(figsize=(20,40))
sns.heatmap(corr,annot=True,cmap='coolwarm' )


# # Train-Test Split

# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X=df.drop(['Class'],axis =1)
Y=df['Class']


# In[18]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.8,test_size=0.2,random_state=100)


# # Feature Scaling

# In[19]:


from sklearn.preprocessing import StandardScaler


# In[20]:


Scaler = StandardScaler()


# In[21]:


X_train['Amount']=Scaler.fit_transform(X_train[['Amount']])


# In[22]:


X_train.head()


#  # Scale the test set

# In[23]:


X_test['Amount']=Scaler.fit_transform(X_test[['Amount']])


# In[24]:


X_test.head()


#   # Model Training

# In[ ]:


# Logistic Regression


# In[26]:


from sklearn.linear_model import LogisticRegression
model =LogisticRegression()


# In[27]:


# trainig the Logistic Regression Model with Training Data
model.fit(X_train,Y_train)


# # Model Evaluation

# In[28]:


# Accuracy Score


#  #Accuracy on Trainig Data

# In[31]:


from sklearn.metrics import accuracy_score
X_train_prediction =model.predict(X_train)
training_data_accuracy =accuracy_score(X_train_prediction,Y_train)


# In[32]:


print('Accuracy on Training Data :' , training_data_accuracy)


# In[33]:


# Accuracy on Test Data
X_test_prediction = model.predict(X_test)
test_data_accuracy =accuracy_score(X_test_prediction,Y_test)


# In[34]:


print('Accuracy on Test Data :' , test_data_accuracy)


# In[ ]:




