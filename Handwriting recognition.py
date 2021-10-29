#!/usr/bin/env python
# coding: utf-8

# ## Fetching Dataset

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


mnist=fetch_openml('mnist_784')


# In[3]:


x,y=mnist["data"],mnist["target"]


# In[4]:


x.shape


# In[5]:


y.shape


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import matplotlib
import matplotlib.pyplot as plt


# In[8]:


some_digit=x[37000]
some_digit_image=some_digit.reshape(28,28)


# In[9]:


plt.imshow(some_digit_image,cmap=matplotlib.cm.binary, interpolation="nearest")
plt.axis("off")


# In[10]:


y[37000]


# In[11]:


x_train=x[:60000]
x_test=x[60000:]


# In[12]:


y_train=y[:60000]
y_test=y[60000:]


# In[13]:


import numpy as np
shuffle_index=np.random.permutation(60000)
x_train, y_train=x_train[shuffle_index],y_train[shuffle_index]


# ## Creating a 2 detector

# In[ ]:


y_train=y_train.astype(np.int8)
y_test=y_test.astype(np.int8)
y_train_4=(y_train==2)
y_test_4=(y_test==2)


# In[ ]:


y_train


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[26]:


clf=LogisticRegression(tol=0.1,solver='lbfgs')


# In[27]:


clf.fit(x_train,y_train_4)


# In[28]:


clf.predict([some_digit])


# In[29]:


from sklearn.model_selection import cross_val_score
a=cross_val_score(clf,x_train,y_train_4,cv=3,scoring="accuracy")


# In[30]:


a.mean()


# In[ ]:




