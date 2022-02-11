#!/usr/bin/env python
# coding: utf-8

# # Task 02 - Predicting Using Unsupervised ML. Perdict the optimum no of clusters and represent it visually

# Importing Necessary Libraries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier 

import warnings
warnings.filterwarnings("ignore")


# Loading Data

# In[3]:


# Load the iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)
df.head()


# Data Preprocessing

# In[8]:


df.dtypes


# In[9]:


df.info()


# In[10]:


df.describe()


# In[11]:


df.isnull().sum()


# In[12]:


plt.figure(figsize=(15,8))
sns.heatmap(df.corr(),annot=True, cmap="YlGnBu" )


# # Finding the optimum number of clusters for K-Means and determining the value of K.

# In[13]:


# Finding the optimum number of clusters for k-means classification

x = df.iloc[:, [0, 1, 2, 3]].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# `allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()


# # Best K value is 3

# In[14]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[15]:


# Visualising the clusters - On the first two columns
plt.figure(figsize=(15,8))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = '#7FFF00', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = '#000000', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = '#0000CD', label = 'Iris-virginica')


# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = '#CD00CD', label = 'Centroids')


plt.legend()


# In[ ]:




