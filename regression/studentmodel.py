"""imports modules"""
import pickle
import matplotlib.pyplot as pyplot
import matplotlib as style
from pandas import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection

# Load dataset from csv file
data = pd.read_csv("student-mat.csv", sep=";")

# create data frames (table-like python object with
# row/columns, and methods to interact with Data frames
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# the target prediction (what you are trying to get)
predict = "G3"

# Pick all labels/attributes
X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.1)

'''
best = 0
for _ in range(30):
    # split into 4 array
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.1)

    # Create a training model
    linear = linear_model.LinearRegression()
    linear.fit(X_train, Y_train)

    # Test the prediction values
    acc = linear.score(X_test, Y_test)
    print(acc)

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            # save model for a file
            pickle.dump(linear, f)
'''
# load model from file
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# outputs Coefficient and Y intercept
print('Coefficient: \n', linear.coef_)
print('Coefficient: \n', linear.intercept_)

# use the model created to predict
predictions = linear.predict(X_test)
for X in range(len(predictions)):
    print(predictions[X], X_test[X], Y_test[X])

# plot a grid
p = "absences"
style.use('MacOSX')
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
