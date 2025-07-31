import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import pickle
import sklearn.preprocessing

data = pd.read_csv('Churn_Modelling.csv')
# Preprocessing the data
data = data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

data = pd.get_dummies(data, columns=['Geography', 'Gender'])

X = data.drop('Exited', axis=1)
y = data['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
