import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import data
dataset = pd.read_csv('data.csv', header = 0)

#Replace 'M' and 'B' values in diagnosis column with 1 and 0 respectively
dataset['diagnosis'] = dataset['diagnosis'].map({'M':1,'B':0})

#Assign input/output to X and y Matrices
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

#Split into Test/Train sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
my_scalar = StandardScaler()
X_train = my_scalar.fit_transform(X_train)
X_test = my_scalar.transform(X_test)

#import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialize a sequential instance of the NN
model = Sequential()

#Add Input Layer and first hidden layer
model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 30))

#Add Second Hidden Layer
model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

#Add Output Layer
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)