#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io


# # Importing the dataset

# In[2]:


import io
import requests
url="https://raw.githubusercontent.com/amankharwal/Website-data/master/IRIS.csv"
read_data=requests.get(url).content
read_data


# In[3]:


df=pd.read_csv(io.StringIO(read_data.decode('utf-8')))
df.sample(6)


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


df["species"].value_counts()


# # Preprocessing the data 

# In[7]:


df.isnull().sum()


# # Exploratory Data Aanalysis

# In[8]:


df["sepal_length"].hist()


# In[9]:


df["sepal_width"].hist()


# In[10]:


df["petal_length"].hist()


# In[11]:


df["petal_width"].hist()


# In[12]:


colors=['red','orange','blue']
species =['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[13]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['sepal_width'],c=colors[i],label=species[i])
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.legend()


# In[14]:


colors=['red','orange','blue']
species =['Iris-setosa','Iris-versicolor','Iris-virginica']


# In[15]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['petal_length'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.legend()


# In[16]:


for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['petal_length'],x['sepal_width'],c=colors[i],label=species[i])
plt.xlabel('petal_length')
plt.ylabel('sepal_width')
plt.legend()

for i in range(3):
    x=df[df['species']==species[i]]
    plt.scatter(x['sepal_length'],x['petal_width'],c=colors[i],label=species[i])
plt.xlabel('sepal_length')
plt.ylabel('petal_width')
plt.legend()
# In[17]:


df.corr()


# In[18]:


corr=df.corr()
fig,ax=plt.subplots(figsize=(5,4))
sns.heatmap(corr,annot=True,ax=ax)


# In[19]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[20]:


df['species']=le.fit_transform(df['species'])
df.head()


# In[21]:


from sklearn.model_selection import train_test_split
X=df.drop(columns=['species'])
Y=df['species']
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.30)


# In[22]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[23]:


model.fit(x_train,y_train)


# In[24]:


print("Accuracy:",model.score(x_test,y_test)*100)


# In[30]:


# let’s input a set of measurements of the iris flower and use the model to predict the iris species


# In[26]:


X_new = np.array([[3, 2, 1, 0.2], [  4.9, 2.2, 3.8, 1.1 ], [  5.3, 2.5, 4.6, 1.9 ]])
#Prediction of the species from the input vector
prediction = model.predict(X_new)
print("Prediction of Species: {}".format(prediction))


# In[ ]:




