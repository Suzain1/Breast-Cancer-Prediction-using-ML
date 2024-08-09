#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


# Load the dataset
data = pd.read_csv('D:\\OHSL\\Breast_cancer_data.csv')


# In[3]:


# Separate features and target
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']


# In[4]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# In[5]:


# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
log_reg_pred = log_reg.predict(X_test_scaled)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
log_reg_report = classification_report(y_test, log_reg_pred)
log_reg_cm = confusion_matrix(y_test, log_reg_pred)


# In[7]:


# Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
gnb_pred = gnb.predict(X_test_scaled)
gnb_accuracy = accuracy_score(y_test, gnb_pred)
gnb_report = classification_report(y_test, gnb_pred)
gnb_cm = confusion_matrix(y_test, gnb_pred)


# In[8]:


# Support Vector Machine
svm = SVC()
svm.fit(X_train_scaled, y_train)
svm_pred = svm.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_report = classification_report(y_test, svm_pred)
svm_cm = confusion_matrix(y_test, svm_pred)


# In[9]:


# Print accuracy and classification report
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Logistic Regression Report:\n", log_reg_report)
print("Gaussian Naive Bayes Accuracy:", gnb_accuracy)
print("Gaussian Naive Bayes Report:\n", gnb_report)
print("SVM Accuracy:", svm_accuracy)
print("SVM Report:\n", svm_report)


# In[10]:


# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Logistic Regression Confusion Matrix
sns.heatmap(log_reg_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Logistic Regression Confusion Matrix')
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')

# Gaussian Naive Bayes Confusion Matrix
sns.heatmap(gnb_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Gaussian Naive Bayes Confusion Matrix')
axes[1].set_xlabel('Predicted Labels')
axes[1].set_ylabel('True Labels')

# SVM Confusion Matrix
sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
axes[2].set_title('SVM Confusion Matrix')
axes[2].set_xlabel('Predicted Labels')
axes[2].set_ylabel('True Labels')

plt.tight_layout()
plt.show()

