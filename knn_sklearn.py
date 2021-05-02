# My sys variable
# py --version => Python 3.9.4
# python --version => Python 2.7.17

import numpy as np
from scipy.spatial import distance
def euc(a,b):
    return distance.euclidean(a,b)

import random
class myKNN():
    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
    def predict(self,X_test):
        predictions=[]
        for row in X_test:
            label=self.closest(row)
            predictions.append(label)
        return predictions
    def closest(self,row):
        best_dist=euc(row,self.X_train[0])
        best_index=0
        for i in range(1,len(X_train)):
            dist=euc(row,X_test[i])
            if dist < best_dist:
                best_dist=dist
                best_index=i
        return y_train[best_index]

from sklearn import datasets

#Loading data
iris= datasets.load_iris()

# f(X)=y # f is the classifier function [features,labels]
X=iris.data
y=iris.target

print(X)
# print(y)

# from sklearn.cross_validation import train_test_split
# model_selection
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y ,test_size= .5 )

# classifier
# from sklearn import tree
# my_classifier= tree.DecisionTreeClassifier()

# from sklearn.neighbors import KNeighborsClassifier
# my_classifier=KNeighborsClassifier()
my_classifier=myKNN()

#train it using the training data
# my_classifier=my_classifier.fit(X_train,y_train)
my_classifier.fit(X_train,y_train)

#predict it on the test data
# predictions=my_classifier.predict(X_test,y_test)
predictions=my_classifier.predict(X_test)

# print(predictions)
# print(type(predictions))
# print(y_test)
# print(type(y_test))
# eq=0
# notEq=0
# for i in range(len(y_test)):
#     if(y_test[i]==predictions[i]):
#         eq+=1
#     else:
#         notEq+=1

# print(eq/len(predictions)*100)
# print(notEq/len(predictions)*100)


#Measuring the model

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
