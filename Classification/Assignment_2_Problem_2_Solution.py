##Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.grid_search import GridSearchCV
%matplotlib inline


#Read data (I have created an excel out of the link given)
data=pd.read_excel('car_data1.xlsx')

#EDA starts here

data.head()

#Looks at distributions across columns
for i in data.columns:
    print(data[i].value_counts())
    print()
    
    
#Class distrbution
sns.countplot(data['class'])

##Feature distribution
for i in data.columns[:-1]:
    plt.figure(figsize=(12,6))
    plt.title("For feature '%s'"%i)
    sns.countplot(data[i],hue=data['class'])
    
#Modelling starts here with label encoding
le=LabelEncoder()

for i in data.columns:
    data[i]=le.fit_transform(data[i])
    
    
##X and Y variables
X=data[data.columns[:-1]]
y=data['class']

##Train test split and building the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

logreg=LogisticRegression(solver='newton-cg',multi_class='multinomial')


logreg.fit(X_train,y_train)

pred=logreg.predict(X_test)

logreg.score(X_test,y_test)

##Grid Search with 10 fold cv
param_grid={'C':[0.01,0.1,1,10],
           'solver':['newton-cg', 'lbfgs', 'sag'],
           'multi_class':['multinomial']}
grid=GridSearchCV(estimator=LogisticRegression(n_jobs=-1),param_grid=param_grid,cv=10,n_jobs=-1)

#Fit the model
grid.fit(X_train,y_train)

print(grid.best_params_)
print(grid.best_score_)

##Prediction and Accuracy
pred=grid.best_estimator_.predict(X_test)

print(classification_report(y_test,pred))