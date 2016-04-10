import pandas
import glob
import ntpath
import numpy as np
import cPickle as pickle
import os.path
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
import math

def main():
	decisonTreeClassify()

def decisonTreeClassify():
	asylum_clean = pandas.read_csv("data/raw/asylum_clean.csv")
	asylum_clean_data = asylum_clean.as_matrix()

	# select specific X columns as features
	X_raw = asylum_clean_data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58]]

	# use column 26 raw_grant as response
	y_raw = asylum_clean_data[:,26]

	# merge X and y
	arrayToClean = np.column_stack((X_raw,y_raw))

	# remove nan rows
	newArray = []
	for row in arrayToClean:
		containsNan = False
		for val in row:
			if math.isnan(val):
				containsNan = True
				break
		if not containsNan:
			newArray.append(row)
	cleanedData = np.array(newArray)


	# extract X and y from cleaned rows
	X = cleanedData[:,:24]
	y = cleanedData[:,-1]
	
	# convert y floats to y int
	y = y.astype(int)


	# classfy using decision tree classifier
	X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)
	for i in np.arange(5,15,2):
		clf = DecisionTreeClassifier(max_depth=i).fit(X_train, y_train)
		print "depth: ", i, " score: ", clf.score(X_test, y_test)

	

if __name__ == '__main__':
  main()