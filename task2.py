# Implementing linear SVM method using scikit library
#import packages for support vector , accuracy classifier, split train data for the given ratio and datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets
#load the iris datasets
iris_datasets = datasets.load_iris()

#loading  the iris-datasets samples
x = iris_datasets.data
#getting the samples and feactures of the dataset.
y = iris_datasets.target

#split training and testing data set for both x and y for linear kernel for the given ratio
training_set_x, testing_set_x, training_set_y, testing_set_y=train_test_split(x, y, test_size=0.2, random_state=21)

#define the model with linear kernel
lmodel=SVC(kernel='linear')

#fit training data set
lmodel.fit(training_set_x, training_set_y)

#predicting the test data set
prediction=lmodel.predict(testing_set_x)

#calculating the accuracy score for linear kernel
print("Percentage of linear kernel Accuracy score is", accuracy_score(prediction, testing_set_y)*100)