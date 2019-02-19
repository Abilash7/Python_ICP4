#Implementing Naïve Bayes method using scikit-learn libraryUse iris dataset available

from sklearn import datasets,metrics
#importing the GNB from sklearn library
from sklearn.naive_bayes import GaussianNB
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

#loading the iris dataset
iris_datasets=datasets.load_iris()


#loading  the iris-datasets samples
x = iris_datasets.data
#getting the samples and feactures of the dataset.
y = iris_datasets.target



#using the trian_test_split function which splits a given dataset into a given split ratio
x_training_set,x_testing_set,y_training_set,y_testing_set=train_test_split(x,y,test_size=0.2,random_state=21)

#define the gaussian naive bayes method
gnb=GaussianNB()

#fit the training data into gaussian naive bayes using the fit method
gnb.fit(x_training_set,y_training_set)

#prints the probability of training data
print(f"The Probability of the training_set data is {gnb.score(x_training_set,y_training_set)*100}")


#using the predict function on the test data set
y_pred = gnb.predict(x_testing_set)

#calculating the accuracy classification score using metrics.accuracy_score classifier.
print("The Accuracy score of testing_set data using Naive Bayes is : ",metrics.accuracy_score(y_testing_set, y_pred)*100)