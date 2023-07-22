import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC  
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

data = pd.read_csv('shuffled_data.csv')
print(data.head())

# input
x = data.iloc[:, 0:46].values

# output
y = data.iloc[:, 46].values

xtrain, xtest, ytrain, ytest = train_test_split(
		x, y, test_size = 0.25, random_state = 0)

sc_x = StandardScaler()
xtrain = sc_x.fit_transform(xtrain)
xtest = sc_x.transform(xtest)

classifier = SVC(kernel='linear') 
  
# fitting x samples and y classes 
classifier.fit(xtrain, ytrain) 
y_pred = classifier.predict(xtest)

cm = confusion_matrix(ytest, y_pred)
print ("Confusion Matrix : \n", cm)

print ("Accuracy : ", accuracy_score(ytest, y_pred))

ypred2 = classifier.predict(xtrain)
cm2 = confusion_matrix(ytrain, ypred2)

print ("Confusion Matrix : \n", cm2)

print ("Accuracy : ", accuracy_score(ytrain, ypred2))

pickle.dump(classifier, open('SVM_model.pkl', 'wb'))
