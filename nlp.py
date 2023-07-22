#importing useful libraries
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
# import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics, svm
import urllib.parse
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
# import pickle
# import collections
# %matplotlib inline
import pickle
import pandas as pd
# import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

#loading our XSS attack data
data = pd.read_csv('XSS.csv')

#visualizing the data
data.head()

#Vectorizing the data using TF-IDF to make the data trainable
corpus = [d for d in data['Sentence']]
y = [[1,0][d!= 1] for d in data['Label']]
vectorizer2 = TfidfVectorizer()
X = vectorizer2.fit_transform(corpus)

idf2 = vectorizer2.idf_
#print dict(zip(vectorizer2.get_feature_names(), idf2))

#Splitting the data into train and test to train the Logistic Regression Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Implementing Logistic Regression classifier on our data

lgs = LogisticRegression()
param = {'C': np.logspace(-1, 2, 4),'penalty':['l1','l2']}
clf = GridSearchCV(lgs, param, n_jobs = -1)

#Training our Logistic Regression model on the training data


clf.fit(X_train, y_train)


# Computing the metrics for test data
print(" Metrics for test data: ")
predictedDec = clf.predict(X_test)
print("Accuracy: %f" % clf.score(X_test, y_test))
print("Precision: %f" % metrics.precision_score(y_test, predictedDec))
print("Recall: %f" % metrics.recall_score(y_test, predictedDec))
print("F1-Score: %f" % metrics.f1_score(y_test, predictedDec))
print("The confusion matrix is: ")
print(confusion_matrix(y_test, predictedDec))
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf, X_test, y_test) 


# Computing the metrics for train data
print(" Metrics for train data: ")
predictedDec = dec.predict(X_test)
print("Accuracy: %f" % dec.score(X_test, y_test))
print("Precision: %f" % metrics.precision_score(y_test, predictedDec))
print("Recall: %f" % metrics.recall_score(y_test, predictedDec))
print("F1-Score: %f" % metrics.f1_score(y_test, predictedDec))
confusion_matrix(y_test, predictedDec)
plot_confusion_matrix(clf, X_test, y_test) 



# Checking for any code whether it is an attack or not
X_predict = ['<tt onmouseover="alert(1)">test</tt>']
X_predict = vectorizer2.transform(X_predict)
y_Predict = clf.predict(X_predict)
print(y_Predict) #printing predicted values
if(y_Predict == 0):
  print("It is not an attack")
else:
  print("It is a XSS attack!")


#Implementing decision tree instead of Logistic Regression -----------------------------------------------------------------------------------------##
from sklearn.tree import DecisionTreeClassifier
dec = DecisionTreeClassifier(random_state=0)
dec.fit(X_train, y_train)

# Computing the metrics for test data
print(" Metrics for test data: ")
predictedDec = dec.predict(X_test)
print("Accuracy: %f" % dec.score(X_test, y_test))
print("Precision: %f" % metrics.precision_score(y_test, predictedDec))
print("Recall: %f" % metrics.recall_score(y_test, predictedDec))
print("F1-Score: %f" % metrics.f1_score(y_test, predictedDec))
confusion_matrix(y_test, predictedDec)
plot_confusion_matrix(dec, X_test, y_test) 


# Computing the metrics for train data
print(" Metrics for train data: ")
predictedDec = dec.predict(X_train)
print("Accuracy: %f" % dec.score(X_train, y_train))
print("Precision: %f" % metrics.precision_score(y_train, predictedDec))
print("Recall: %f" % metrics.recall_score(y_train, predictedDec))
print("F1-Score: %f" % metrics.f1_score(y_train, predictedDec))
confusion_matrix(y_train, predictedDec)
plot_confusion_matrix(dec, X_train, y_train) 


# Checking for any code whether it is an attack or not using Decision Tree
X_predict = ['<tt onmouseover="alert(1)">test</tt>']
X_predict = vectorizer2.transform(X_predict)
y_Predict = dec.predict(X_predict)
print(y_Predict) #printing predicted values
if(y_Predict == 0):
  print("It is not an attack")
else:
  print("It is a XSS attack!")

pickle.dump(dec, open('nlp_model.pkl', 'wb'))