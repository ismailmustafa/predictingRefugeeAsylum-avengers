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
	constrainMaster()

def constrainMaster():
	asylum_clean = pandas.read_csv("data/raw/asylum_clean.csv")
	asylum_clean_data = asylum_clean.as_matrix()

	# delete idncase
	X = np.delete(asylum_clean_data, [0,2,3], axis=1)

	# remove nan rows
	newArray = []
	for row in X:
		containsNan = False
		for val in row:
			if math.isnan(val):
				containsNan = True
				break
		if not containsNan:
			newArray.append(row)
	newX = np.array(newArray)


	# get response (grant column)
	y = newX[:,14]

	y = y.astype(int)

	# remove response
	newX = np.delete(newX, 14, axis=1)


	

	# # delete grant decision
	# X = np.delete(asylum_clean_data, 17, axis=1)

	# # remove string columns
	# X = np.delete(asylum_clean_data, 2, axis=1)
	# X = np.delete(asylum_clean_data, 3, axis=1)

	X_train, X_test, y_train, y_test = cross_validation.train_test_split(newX, y, test_size=0.4, random_state=0)
	clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
	# clf = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
	print "score: ", clf.score(X_test, y_test)


	exit()

	# remove first column (idncase)




	# print asylum_clean.shape
	# print asylum_clean['idncase'].unique().shape
	# print asylumClean
	# exit()
	# mergedDataArray = []
	# if not os.path.isfile("data/mergedData"):
	# 	print "merging data"
	# 	fFrameApplnData = pandas.read_csv("data/raw/court_appln.csv")
	# 	fFrameApplnData = fFrameApplnData[fFrameApplnData['Appl_Code'] == 'ASYL']
	# 	fFrameMasterData = pandas.read_csv("data/raw/master.csv")
	# 	mergedData = fFrameApplnData.join(fFrameMasterData, how="left", on='idnCase')

	# 	mergedDataArray = mergedData.as_matrix()
	# 	pickle.dump(mergedDataArray, open("data/mergedData", "wb"))
	# else:
	# 	print "loading merged data"
	# 	mergedDataArray = pickle.load(open("data/mergedData", "rb"))


	print "iterating"
	dict = {}
	for row in asylum_clean_array:
		if row[0] in dict:
			dict[row[0]] += 1
		else:
			dict[row[0]] = 1

	maxKey = 0
	maxVal = 0
	for key in dict:
		if dict[key] > 1:
			if dict[key] > maxVal:
				maxVal = dict[key]
				maxKey = key
			print key, ": ", dict[key]

	print "max: ", maxKey, ": ", maxVal 
	print "count: ", len(dict)

	

if __name__ == '__main__':
  main()