#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('MLF_GP1_CreditScore.csv',encoding = "ISO-8859-1")
df


# In[ ]:


df.dtypes


# In[3]:


# Checking for missing values
df.isnull().sum()


# In[4]:


# Split the data into training and testing sets

X = df.iloc[:, :-2]  # All columns except the last two
y_InvGrd = df.iloc[:, -2]  # second-to-last column
y_Rating = df.iloc[:, -1]  # Last column

X_train, X_test, y_InvGrd_train, y_InvGrd_test, y_Rating_train, y_Rating_test = train_test_split(
    X, y_InvGrd, y_Rating, test_size=0.2, random_state=42) # 80% training and 20% test


# ### Linear Regression Approach

# In[6]:


##Ridge Regularisation 

# Train the model
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_InvGrd_train)

# Test the model
y_InvGrd_pred = ridge.predict(X_test)
y_InvGrd_pred[y_InvGrd_pred <= 0.5] = 0  # Set threshold to 0.5
y_InvGrd_pred[y_InvGrd_pred > 0.5] = 1
accuracy = accuracy_score(y_InvGrd_test, y_InvGrd_pred)

print("Accuracy of Linear regression with Ridge regularization : ", accuracy)


# In[7]:


##Lasso Regularisation

# Train the model
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_InvGrd_train)

# Test the model
y_InvGrd_pred = lasso.predict(X_test)
y_InvGrd_pred[y_InvGrd_pred <= 0.5] = 0  # Set threshold to 0.5
y_InvGrd_pred[y_InvGrd_pred > 0.5] = 1
accuracy = accuracy_score(y_InvGrd_test, y_InvGrd_pred)

print("Accuracy of Linear regression with Lasso regularization:", accuracy)


# ### Logistic Regression Approach

# In[9]:


##Ridge Regularisation 

lr_ridge = LogisticRegression(penalty='l2', solver='liblinear', C=0.1)
lr_ridge.fit(X_train, y_InvGrd_train)
y_InvGrd_pred_ridge = lr_ridge.predict(X_test)
accuracy_ridge = accuracy_score(y_InvGrd_test, y_InvGrd_pred_ridge)
print("Accuracy of Logistic regression with Ridge regularization :", accuracy_ridge)


## Lasso regularization
lr_lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.1)
lr_lasso.fit(X_train, y_InvGrd_train)
y_InvGrd_pred_lasso = lr_lasso.predict(X_test)
accuracy_lasso = accuracy_score(y_InvGrd_test, y_InvGrd_pred_lasso)
print("Accuracy of Logistic regression with Lasso regularization :", accuracy_lasso)


# ### Neural Networks Approach
# 
# For the neural network approach, i will be loading the dataset again

# In[10]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder



# In[11]:


# Load the dataset
df = pd.read_csv('MLF_GP1_CreditScore.csv',encoding = "ISO-8859-1")
df


# In[12]:


# Defining the features and its target variables
X = df.iloc[:, :-2].values
y_rating = df.iloc[:, -1].values
y_invgrd = df.iloc[:, -2].values


# In[13]:


#splitting the data into train  and test set of 80% and 20% respectively
X_train, X_test, y_rating_train, y_rating_test, y_invgrd_train, y_invgrd_test = train_test_split(X, y_rating, y_invgrd, test_size=0.2, random_state=42)


# In[14]:


# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Convert string labels to integer labels
label_encoder = LabelEncoder()
y_rating_train = label_encoder.fit_transform(y_rating_train)
y_rating_test = label_encoder.transform(y_rating_test)

# One-hot encode rating target variable
y_rating_train = to_categorical(y_rating_train)
y_rating_test = to_categorical(y_rating_test)


# In[15]:


# Defining the model architecture
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='softmax'))


# In[16]:


# Compiling
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training
model.fit(X_train, y_rating_train, epochs=50, batch_size=32, validation_data=(X_test, y_rating_test))

# Evaluation
_, accuracy = model.evaluate(X_test, y_rating_test)
print('Neural network Accuracy: %.2f%%' % (accuracy*100))


# Based on the results gotten,
# 
# Accuracy of Linear regression with Ridge regularization :  0.7676470588235295 approximately 0.7676
# 
# Accuracy of Linear regression with Lasso regularization: 0.7529411764705882 approximately 0.7529
# 
# Accuracy of Logistic regression with Ridge regularization : 0.7647058823529411 approximately 0.7647
# 
# Accuracy of Logistic regression with Lasso regularization : 0.7617647058823529 approximately 0.7618
# 
# Neural network Accuracy: 0.2235 
# 
# The following observations regarding the effectiveness and suitability of each approach for the given problem could be deduced :
# 
# Linear regression with Ridge regularization and Logistic regression with Ridge regularization have similar accuracy scores of 0.7676 and 0.7647 respectively. These results suggest that both these approaches are equally effective in predicting whether the firm is in an investment grade or not.
# 
# Linear regression with Lasso regularization and Logistic regression with Lasso regularization also have similar accuracy scores of 0.7529 and 0.7618 respectively. However, these scores are slightly lower than the accuracy scores of the Ridge regularization models. This could be because Lasso regularization tends to produce sparse models, which might not be well suited for this problem.
# 
# The neural network approach has a significantly lower accuracy score of 0.2235(22.35%). This indicates that the neural network model might not be suitable for this problem. However, it is possible that the model could be enhanced with further fine-tuning and optimization.
# 
# In summary, based on the given results, it can be concluded that the Ridge regularization models are the most effective and suitable approaches for predicting whether the firm is in an investment grade or not. However, further analysis and experimentation might be necessary to confirm these findings.

# 
