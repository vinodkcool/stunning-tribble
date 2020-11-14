import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,accuracy_score
%matplotlib inline
sns.set()



iris = sns.load_dataset("iris")
dataset=iris

#Spliting the dataset in independent and dependent variables
X = dataset.iloc[:,:4].values
y = dataset['species'].values

# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

# Feature Scaling to bring the variable in a single scale

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Fitting Naive Bayes Classification to the Training set with linear kernel

nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = nvclassifier.predict(X_test)
print(y_pred)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

#accuracy score
accuracy_score(y_test,y_pred)