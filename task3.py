#using the SVM with RBF kernel

# import packages for support vector , accuracy, split train data and datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

# load the iris datasets
iris_datasets = datasets.load_iris()

# loading  the iris-datasets samples
x = iris_datasets.data
# getting the samples and feactures of the dataset.
y = iris_datasets.target

# split training and test data for both x and y for linear kernel
training_set_x, testing_set_x, training_set_y, testing_set_y = train_test_split(x, y, test_size=0.2, random_state=21)

# split training and test data for both x and y for rbf kernel
training_set_x1, testing_set_x1, training_set_y1, testing_set_y1 = train_test_split(x, y, test_size=0.2, random_state=21)

# defining the model for linear kernel
lmodel = SVC(kernel='linear')

# defining the model for rbf kernel
rmodel = SVC(kernel='rbf')

# fit training data into linear kernel
lmodel.fit(training_set_x, training_set_y)

# predict the test data using linear kernel
prediction = lmodel.predict(testing_set_x)

# calc accuracy score for linear kernel
print("Percentage of linear kernel Accuracy score is", accuracy_score(prediction, testing_set_y)*100)
# print(prediction)

# fit training data into rbc kernel
rmodel.fit(training_set_x1, training_set_y1)

# predict the test data for rbc kernel
pred = rmodel.predict(testing_set_x1)

# calc accuracy for rbc kernel
print("Percentage of RBF kernel accuracy score is", accuracy_score(pred, testing_set_y1)*100)
# print(pred)
