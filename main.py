import math
import pandas
import numpy as np
from sklearn import svm
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

# specify classifiers used
class Classifier:
	Adaboost, DecisionTree, RandomForest = range(3)

def main():
	# specify classifier to use
	selected_classifier = Classifier.Adaboost

	# run classifier
	if selected_classifier == Classifier.Adaboost:
		X_train, X_test, y_train, y_test = load_and_process_data()
		adaboost_train(X_train, X_test, y_train, y_test, max_rounds=10)
	elif selected_classifier == Classifier.DecisionTree:
		X_train, X_test, y_train, y_test = load_and_process_data()
		decision_tree_train(X_train, X_test, y_train, y_test)
	elif selected_classifier == Classifier.RandomForest:
		X, y = load_all_clean_data()
		random_forest_train(X, y)
	else:
		X_train, X_test, y_train, y_test = load_and_process_data()
		decision_tree_train(X_train, X_test, y_train, y_test)
		
# -------------------------------- DATA / FEATURES ------------------------------

# load data, clean values, return train and test set split
def load_and_process_data():
	print "loading data and processing..."
	asylum_clean = pandas.read_csv("data/raw/asylum_clean_full_mood.csv")
	asylum_clean_data = asylum_clean.as_matrix()

	# select specific X columns as features
	# original (----- use this as baseline -----)
	X_raw = asylum_clean_data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58]]
	# with mood (mood_k2 at index 91, mood_k8 at index 97)
	#X_raw = asylum_clean_data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58, 89, 90]]

	# convert cities to integers
	cities = asylum_clean_data[:,2]
	for i in xrange(len(cities)):
		cities[i] = int(''.join(str(ord(c)) for c in cities[i]))
	X_raw = np.column_stack((X_raw,cities))

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

	# split data into train and test
	return cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

# load all data, including missing values
def load_all_data():
	print "loading data and processing..."
	asylum_clean = pandas.read_csv("data/raw/asylum_clean_full_mood.csv")
	asylum_clean_data = asylum_clean.as_matrix()

	# select specific X columns as features
	# original (----- use this as baseline -----)
	X_raw = asylum_clean_data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58]]
	# with mood (mood_k2 at index 91, mood_k8 at index 97)
	#X_raw = asylum_clean_data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58, 91, 92, 93, 94, 95, 96, 97]]

	# convert cities to integers
	cities = asylum_clean_data[:,2]
	for i in xrange(len(cities)):
		cities[i] = int(''.join(str(ord(c)) for c in cities[i]))
	X_raw = np.column_stack((X_raw,cities))

	# use column 26 raw_grant as response
	y_raw = asylum_clean_data[:,26]

	# return all data
	return X_raw, y_raw

def load_all_clean_data():
	print "loading data and processing..."
	asylum_clean = pandas.read_csv("data/raw/asylum_clean_full_mood.csv")
	asylum_clean_data = asylum_clean.as_matrix()

	# select specific X columns as features
	# original (----- use this as baseline -----)
	X_raw = asylum_clean_data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58]]
	# with mood (mood_k2 at index 91, mood_k8 at index 97)
	#X_raw = asylum_clean_data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58, 91, 92, 93, 94, 95, 96, 97]]

	# convert cities to integers
	cities = asylum_clean_data[:,2]
	for i in xrange(len(cities)):
		cities[i] = int(''.join(str(ord(c)) for c in cities[i]))
	X_raw = np.column_stack((X_raw,cities))

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

	# return all data
	return X, y

# -------------------------------- DATA / FEATURES ------------------------------

# -------------------------------- DECISION TREE --------------------------------

# run decision tree classifier and plot for different depths
def decision_tree_train(X_train, X_test, y_train, y_test, depth=10):
	print "Running Decision Tree Classifier to max depth of", depth
	depths = np.arange(1,11)
	train_errors = []
	test_errors = []
	for d in depths:
		clf = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)
		train_error = 1.0 - clf.score(X_train, y_train)
		test_error = 1.0 - clf.score(X_test, y_test)
		print "depth:", d, "train error:", train_error, "test_error:", test_error
		train_errors.append(train_error)
		test_errors.append(test_error)

	# plot
	train_test_error_plot(depths, train_errors, test_errors, "Decision Tree Classifier", "error")

# -------------------------------- DECISION TREE --------------------------------

# -------------------------------- ADABOOST -------------------------------------

# train adaboost on data and plot
def adaboost_train(X_train, X_test, y_train, y_test, max_rounds=10):
	print "Running Adaboost Classifier to max rounds of", max_rounds
	# change 0 to -1 for adaboost
	y_train[y_train == 0] = -1
	y_test[y_test == 0] = -1

	rounds = np.arange(1,max_rounds+1)
	train_errors = []
	test_errors = []
	for i in rounds:
		train_error, test_error = adaboost(X_train, X_test, y_train, y_test, i)
		print "num rounds:", i, "train error:", train_error, "test_error:", test_error
		train_errors.append(train_error)
		test_errors.append(test_error)

	train_test_error_plot(rounds, train_errors, test_errors, "Adaboost Classifier", "num rounds")

# train adaboost using decision tree weak classfier of depth 3
def adaboost(X_train, X_test, y_train, y_test, M=10):

	weights = np.ones(y_train.shape[0])/y_train.shape[0]
	alphas = []
	weak_classifiers = []

	for i in range(M):
		# weak classifier
		clf = DecisionTreeClassifier(max_depth=3).fit(X_train, y_train, sample_weight=weights)
		weak_classifiers.append(clf)

		# calculate error
		error = 1 - clf.score(X_train, y_train, sample_weight=weights)

		# calculate alpha
		alpha = np.log((1 - error)/error)
		alphas.append(alpha)

		# adjust weights
		p = clf.predict(X_train)
		for i in range(len(weights)):
			if p[i] != y_train[i]:
				weights[i] = (weights[i] * np.exp(alpha))

	train_error = adaboost_error(X_train, y_train, alphas, weak_classifiers)
	test_error  = adaboost_error(X_test, y_test, alphas, weak_classifiers)

	return (train_error, test_error)

# calculate error from adaboost training
def adaboost_error(X,y,alphas,weak_classifiers):

	M = len(alphas)

	pred = np.zeros(X.shape[0])
	for i in range(M):
		pred += alphas[i]*weak_classifiers[i].predict(X)
	for i in range(len(pred)):
		if pred[i] > 0:
			pred[i] = 1
		else:
			pred[i] = -1

	err_count = 0
	for i in range(len(pred)):
		if pred[i] != y[i]:
			err_count += 1

	return float(err_count) / len(pred)

# -------------------------------- ADABOOST -------------------------------------

# -------------------------------- RANDOM FOREST --------------------------------

def random_forest_train(X, y):
	# print "imputing data"
	# rng = np.random.RandomState(0)
	# n_samples = X.shape[0]
	# n_features = X.shape[1]
	# missing_rate = 1.0
	# n_missing_samples = np.floor(n_samples * missing_rate)
	# missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
 #                                      dtype=np.bool),
 #                             np.ones(n_missing_samples,
 #                                     dtype=np.bool)))
	# rng.shuffle(missing_samples)
	# missing_features = rng.randint(0, n_features, n_missing_samples)

	# # Estimate the score after imputation of the missing values
	# print "training random forest"
	# X_missing = X.copy()
	# X_missing[np.where(missing_samples)[0], missing_features] = 0
	# y_missing = y.copy()
	# estimator = Pipeline([("imputer", Imputer(missing_values=0,
	# 																					strategy="mean",
	# 																					axis=0)),
	# 											("forest", RandomForestRegressor(random_state=0,
	# 																											 n_estimators=100))])
	# score = cross_val_score(estimator, X_missing, y_missing).mean()
	# print("Score after imputation of the missing values = %.2f" % score)

	print "training random forest"
	estimator = RandomForestRegressor(random_state=0, n_estimators=100)
	score = cross_val_score(estimator, X, y).mean()
	print("Score = %.2f" % score)

# -------------------------------- RANDOM FOREST --------------------------------

# -------------------------------- PLOTTING -------------------------------------
# take train/test errors and x axis to plot
def train_test_error_plot(depths, train_errors, test_errors, plot_title, xlabel):
	plt.plot(depths, train_errors, label="train error")
	plt.plot(depths, test_errors, label="test error")
	plt.title(plot_title)
	plt.xlabel(xlabel)
	plt.ylabel("error")
	plt.legend()
	plt.show()
# -------------------------------- PLOTTING -------------------------------------

if __name__ == '__main__':
  main()