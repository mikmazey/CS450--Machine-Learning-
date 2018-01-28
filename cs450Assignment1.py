#Worked on this with Brady
'''
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
import numpy as np
import warnings
from math import sqrt
from collections import Counter

iris = datasets.load_iris()
data = iris
#This is the original data
#print(iris.data)
#This is the target for how the data should look
#print(iris.target)

#print(iris.target_names)

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#x=iris data, y = iris target
#This train_test_split function randomizes and splits the training and testing sections
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)
#print(x_train, y_train)
#print(x_test, y_test)





#training a model...
classifier = GaussianNB()
model = classifier.fit(x_train, y_train)

#predicts targets for test data
targets_predicted = model.predict(x_test)
#print(targets_predicted)

#compare predicted targets to actual targets and get accuracy
#print(100*(accuracy_score(y_test, targets_predicted)))

class HardCodedClassifier:
    def fit(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        return self

    def predict(self, x_test):
        return [0 for x in range(len(x_test))]

classifier = HardCodedClassifier()
model = classifier.fit(x_train, y_train)
targets_predicted = model.predict(x_test)


targets_predicted = model.predict(x_test)
print(targets_predicted)

#compare predicted targets to actual targets and get accuracy
print(100 *(accuracy_score(y_test, targets_predicted)))

#extra work
import pandas

#this reads a textfile
#data = pandas.read_csv("/Users/darrenfox/PycharmProjects/arcade/data/data_for_CS450/iris_data")
#print(data)

#This code was an attempt to only have int in the data. It doesn't work but we tried.
#for line in data:
   # flowers = line.split(",")
    #for flower in flowers:
       # if flower >= 0:
            #print(flower)

'''

#WEEK 2 ASSIGNMENT
'''
import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train(x_train, y_train):
    return

def predict(x_train, y_train, x_test, k):
    #creating list for distances and targets

    distances = []
    targets = []

    for i in range(len(x_train)):
        #computing the euclidean distance using numpy. comparing each x_test data point to each x_train data point
        e_distance = np.sqrt(np.sum(np.square(x_test - x_train[i, :])))
        #add to list of distances now
        distances.append([e_distance, i])

    #sort the list to make it go from smallest distance to greatest distance
    distances_sorted = sorted(distances)
    print(distances_sorted)
    #make list of the k neighbours' targets. Takes top five (five smallest values) values and puts it into a list called 'targets'
    for i in range(k):
        index = distances_sorted[i][1]
        targets.append(y_train[index])

    #return most common target
    return Counter(targets).most_common(1)[0][0]


def K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k):

    #train on the input data
    train(x_train, y_train)

    for i in range(len(x_test)):
        predictions.append(predict(x_train, y_train, x_test[i, :], k=5))

#making predictions
predictions = []

K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k=5)

#evaluating accuracy
accuracy = 100*accuracy_score(y_test, predictions)
print(accuracy)

#This is the known algorithm
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=5)
model = classifier.fit(x_train, y_train)
predictions2 = model.predict(x_test)

checking_accuracy = 100*accuracy_score(y_test, predictions2)
print(checking_accuracy)

# Our prediction accuracy is 93.33% when the known algorithm is 100$ accurate.


'''
# Week 3 assignment

iris = datasets.load_iris()

import numpy as np
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold



#car data

import pandas
import numpy as np

car_data = pandas.read_csv("/Users/darrenfox/PycharmProjects/arcade/data/data_for_CS450/cardata.csv", header=None)

car_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']


#print(car_data)


cleanup_nums = {'buying': {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1},
                 'maint': {'vhigh': 4, 'high': 3, 'med': 2, 'low': 1},
                  'doors': {'5more': 5, '2':2, '3':3, '4': 4},
                   'persons': {'more': 6, '2':2, '4': 4},
                    'lug_boot': {'small': 1, 'med': 2, 'big': 3},
                      'safety': {'low': 1, 'med': 2, 'high': 3},
                       'class': {'unacc': 1, 'acc': 2, 'good': 3, 'vgood': 4}}
car_data.replace(cleanup_nums, inplace=True)
#print(car_data.dtypes)


def train(x_train, y_train):
    return

def predict(x_train, y_train, x_test, k):
    #creating list for distances and targets

    distances = []
    targets = []

    for i in range(len(x_train)):
        #computing the euclidean distance using numpy. comparing each x_test data point to each x_train data point
        e_distance = np.sqrt(np.sum(np.square(x_test - x_train[i, :])))
        #add to list of distances now
        distances.append([e_distance, i])

    #sort the list to make it go from smallest distance to greatest distance
    distances_sorted = sorted(distances)
    #print(distances_sorted)
    #make list of the k neighbours' targets. Takes top five (five smallest values) values and puts it into a list called 'targets'
    for i in range(k):
        index = distances_sorted[i][1]
        targets.append(y_train[index])

    #return most common target
    return Counter(targets).most_common(1)[0][0]


def K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k):

    #train on the input data
    train(x_train, y_train)

    for i in range(len(x_test)):
        predictions.append(predict(x_train, y_train, x_test[i, :], k=5))

#making predictions
    predictions = []

    K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k=5)



    accuracy = 100*accuracy_score(y_test, predictions)


def cross_validation():
    predictions = []
    kf = KFold(n_splits = 3)
    sum = 0
    for train, test in kf.split(car_data):

        train_data = np.array(car_data)[train]
        x_train = np.array(train_data[:, 0:6])  # data
        y_train = np.array(train_data[:,6])  # target

        test_data = np.array(car_data)[train]
        x_test = np.array(test_data[:, 0:6])
        y_test = np.array(test_data[:,6])

        classifier = K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k=5)
        sum += 100*accuracy_score(classifier.predictions, y_test)
    average = sum/3
    print(average)

cross_validation()





#This is the known algorithm
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=5)
model = classifier.fit(x_train, y_train)
predictions2 = model.predict(x_test)

checking_accuracy = 100*accuracy_score(y_test, predictions2)
print(checking_accuracy)



#diabetes data

diabetes = pandas.read_csv("/Users/darrenfox/PycharmProjects/arcade/data/data_for_CS450/pima-indians-diabetes.data.csv")
diabetes.columns = ['Number of Times Pregnant', 'Plasma Glucose Conc', 'Diastolic Blood Pressure', 'Triceps Skin Thickness', '2-Hour Insulin Thickness', 'BMI', 'Diabetes Pedigree Function', 'Age(years)', 'Class variable (0 or 1)']



#replace all zeros in columns for blood pressure, BMI, and conc for their averages up to that point(meaning, it will only use the data above that placement to get the average) since the zero right now represents and unknown value.

cleanup_zeros = {'Diastolic Blood Pressure': {0: (diabetes['Diastolic Blood Pressure'].mean())},
                 'BMI': {0: (diabetes['BMI'].mean())},
                 'Plasma Glucose Conc': {0: (diabetes['Plasma Glucose Conc'].mean())}}
diabetes.replace(cleanup_zeros, inplace = True)


def train(x_train, y_train):
    return

def predict(x_train, y_train, x_test, k):
    #creating list for distances and targets

    distances = []
    targets = []

    for i in range(len(x_train)):
        #computing the euclidean distance using numpy. comparing each x_test data point to each x_train data point
        e_distance = np.sqrt(np.sum(np.square(x_test - x_train[i, :])))
        #add to list of distances now
        distances.append([e_distance, i])

    #sort the list to make it go from smallest distance to greatest distance
    distances_sorted = sorted(distances)
    #print(distances_sorted)
    #make list of the k neighbours' targets. Takes top five (five smallest values) values and puts it into a list called 'targets'
    for i in range(k):
        index = distances_sorted[i][1]
        targets.append(y_train[index])

    #return most common target
    return Counter(targets).most_common(1)[0][0]


def K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k):

    #train on the input data
    train(x_train, y_train)

    for i in range(len(x_test)):
        predictions.append(predict(x_train, y_train, x_test[i, :], k=5))

#making predictions
    predictions = []

    K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k=5)



    accuracy = 100*accuracy_score(y_test, predictions)


def cross_validation():
    predictions = []
    kf = KFold(n_splits = 3)
    sum = 0
    for train, test in kf.split(car_data):

        train_data = np.array(diabetes)[train]
        x_train = np.array(train_data[:, 0:6])  # data
        y_train = np.array(train_data[:,6])  # target

        test_data = np.array(diabetes)[train]
        x_test = np.array(test_data[:, 0:6])
        y_test = np.array(test_data[:,6])

        classifier = K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k=5)
        sum += 100*accuracy_score(classifier.predictions, y_test)
    average = sum/3
    print(average)

cross_validation()





#This is the known algorithm
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=5)
model = classifier.fit(x_train, y_train)
predictions2 = model.predict(x_test)

checking_accuracy = 100*accuracy_score(y_test, predictions2)
print(checking_accuracy)






#data for automobiles


automobiles = pandas.read_csv("/Users/darrenfox/PycharmProjects/arcade/data/data_for_CS450/automobile.csv")

automobiles = automobiles.replace('?', np.nan)
automobiles = automobiles.dropna()

automobiles.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'car name']

def train(x_train, y_train):
    return

def predict(x_train, y_train, x_test, k):
    #creating list for distances and targets

    distances = []
    targets = []

    for i in range(len(x_train)):
        #computing the euclidean distance using numpy. comparing each x_test data point to each x_train data point
        e_distance = np.sqrt(np.sum(np.square(x_test - x_train[i, :])))
        #add to list of distances now
        distances.append([e_distance, i])

    #sort the list to make it go from smallest distance to greatest distance
    distances_sorted = sorted(distances)
    #print(distances_sorted)
    #make list of the k neighbours' targets. Takes top five (five smallest values) values and puts it into a list called 'targets'
    for i in range(k):
        index = distances_sorted[i][1]
        targets.append(y_train[index])

    #return most common target
    return Counter(targets).most_common(1)[0][0]


def K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k):

    #train on the input data
    train(x_train, y_train)

    for i in range(len(x_test)):
        predictions.append(predict(x_train, y_train, x_test[i, :], k=5))

#making predictions
    predictions = []

    K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k=5)



    accuracy = 100*accuracy_score(y_test, predictions)


def cross_validation():
    predictions = []
    kf = KFold(n_splits = 3)
    sum = 0
    for train, test in kf.split(automobiles):

        train_data = np.array(automobiles)[train]
        x_train = np.array(train_data[:, 0:6])  # data
        y_train = np.array(train_data[:,6])  # target

        test_data = np.array(automobiles)[train]
        x_test = np.array(test_data[:, 0:6])
        y_test = np.array(test_data[:,6])

        classifier = K_Nearest_Neighbour(x_train, y_train, x_test, predictions, k=5)
        sum += 100*accuracy_score(classifier.predictions, y_test)
    average = sum/3
    print(average)

cross_validation()





#This is the known algorithm
from sklearn.neighbors import KNeighborsClassifier

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3)

classifier = KNeighborsClassifier(n_neighbors=5)
model = classifier.fit(x_train, y_train)
predictions2 = model.predict(x_test)

checking_accuracy = 100*accuracy_score(y_test, predictions2)
print(checking_accuracy)










