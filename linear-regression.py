import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences", "age", "Medu", "Fedu"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)

# Get model a number of times and save best one
number_of_iterations = 30
def get_best_model(iterations):
    best_score = 0
    for _ in range(iterations):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.1)
        linear = linear_model.LinearRegression()
        linear.fit(x_train, y_train)
        accuracy = int(linear.score(x_test, y_test)*100)

        if accuracy > best_score:
            with open("studentmodel.pickle", "wb") as f:
                pickle.dump(linear, f)
            best_score = accuracy


# Loading linear model
def open_model(file_name):
    pickle_in = open(file_name, "rb")
    return pickle.load(pickle_in)

linear = open_model("studentmodel.pickle")
accuracy = int(linear.score(x_test, y_test)*100)

print("Predictions:")
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
print("=" * 10)
print("Coefficients:", linear.coef_)
print("Accuracy:", accuracy, "%")




def plotting(comparing_attribute, prediction_attribute):
    style.use("ggplot")
    pyplot.scatter(data[comparing_attribute], data[prediction_attribute])
    pyplot.xlabel(comparing_attribute)
    pyplot.ylabel(prediction_attribute)
    pyplot.show()

plotting("G1", predict)