# Importing the libraries
import numpy as np
import pandas as pd
import pickle

dataset = pd.read_csv('titanic.csv')

dummies = pd.get_dummies(dataset['Sex'])
dataset = pd.concat([dataset, dummies], axis=1)
print(dataset.head())

dataset['Pclass'].fillna(3, inplace=True)
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset['male'].fillna(1, inplace=True)

X = dataset[["Pclass", "male", "Age"]]

y = dataset[["Survived"]]

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()

#Fitting model with trainig data
classifier.fit(X, y)

# Saving model to disk
pickle.dump(classifier, open('model2.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
print(model.predict([[2, 1, 15]]))