import math
import pandas
import numpy as np
import datetime as dt
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
from sklearn import preprocessing
from collections import defaultdict

# specify classifiers used
class Classifier:
	Adaboost, DecisionTree, RandomForest = range(3)

def main():

	# specify classifier to use
	selected_classifier = Classifier.Adaboost

	# run classifier
	if selected_classifier == Classifier.Adaboost:
		X_train, X_test, y_train, y_test = load_and_process_data()
		adaboost_train(X_train, X_test, y_train, y_test, max_rounds=10000)
	elif selected_classifier == Classifier.DecisionTree:
		X_train, X_test, y_train, y_test = base_line_data()
		decision_tree_train(X_train, X_test, y_train, y_test,depth=10)
	elif selected_classifier == Classifier.RandomForest:
		X, y = load_all_data()
		random_forest_train(X, y)
	else:
		X_train, X_test, y_train, y_test = load_and_process_data()
		decision_tree_train(X_train, X_test, y_train, y_test)
		
# -------------------------------- DATA / FEATURES ------------------------------

def convert_from_string_column(column, X_raw):
	le = preprocessing.LabelEncoder()
	le.fit(column)
	X_raw = np.column_stack((X_raw, le.transform(column))) 
	return X_raw

def convert_string_data(X_raw, data, columns):
	for i in columns:
		X_raw = convert_from_string_column(data[:,i], X_raw)
	return X_raw

def generate_custom_features(X_raw, data):

	# decision made after 9/11/2001
	dates = [getCompletionDate(x) for x in data[:,4]]
	after_911 = [greater_than_911(x) for x in dates]
	X_raw = np.column_stack((X_raw, after_911))

	# multiply features 7 (nationality) and 8 (written)
	nationality = data[:,7]
	written = data[:,8]
	result = []
	for n,w in zip(nationality, written):
		if not math.isnan(n) and not math.isnan(w):
			result.append(n*w)
		else:
			result.append(np.nan)
	X_raw = np.column_stack((X_raw, result))

	return X_raw
	

# Convert stata date to date object
def getCompletionDate(stata_date):
	date1960Jan1 = dt.datetime(1960,01,01)
	return date1960Jan1 + dt.timedelta(days=stata_date)

def greater_than_911(date):
	if dt.datetime(2001, 9, 11) < date:
		return 1.0
	else:
		return 0.0

def base_line_data():

	print "loading data and processing..."
	asylum_data = pandas.read_csv("data/raw/complete_data.csv")
	data = asylum_data.as_matrix()

	# original (----- use this as baseline -----)
	X_raw = data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58]]

	# use column 26 raw_grant as response
	y_raw = data[:,26]

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

	print cleanedData.shape

	# extract X and y from cleaned rows
	X = cleanedData[:,:24]
	y = cleanedData[:,-1]
	
	# convert y floats to y int
	y = y.astype(int)

	# split data into train and test
	return cross_validation.train_test_split(X, y, test_size=0.4, random_state=0)

# load data, clean values, return train and test set split
def load_and_process_data():
	print "loading data and processing..."
	asylum_data = pandas.read_csv("data/raw/complete_data.csv")
	data = asylum_data.as_matrix()

	# original (----- use this as baseline -----)
	#X_raw = data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58]]

	# numeric data features
	X_raw = data[:, [4		# hearing date
								  ,5		# laywer
									,6		# defensive 
									,7		# nationality id
									,8		# decision is written
									,10		# adj_time_start
									,11		# early start time
									,16		# court code (same as column 15 courtid)
									,19		# number of family members
									,21		# number of families in current time slot
									,22		# number of families per day
									,23 	# order within day
									,27		# lag
									,28		# lag on same day
									,29		# l2 grant
									,30		# how many of the previous 5 decision were grants
									,33		# number of grants within same court excluding current judge
									,34		#	number of previous 5 decided by current judge
									,35		# numcourtgrantother_prev5
									,36		# courtprevother5_dayslapse
									,37		# year
									,38		# mean grant rate per judge x nationality x defensive
									,41		# nationality defensive court id
									#,42		# judgemeannatdefyear
									#,43 	# judgenumdecnatdefyear
									#,44		# control variable - leave-1-out-mean of how judge decides on these refugees in nationality X defensive X year
									,46		# whether previous case was of same nationality
									,47		# grant grant
									,48		# grant deny
									,49		# deny grant
									,50		# deny deny
									,51		# hour start
									,52		# morning
									,53		# lunchtime
									,54		# numcases_judgeday
									,55		# numcases_judge
									,56		# minimum number of hearings in a courthouse
									,57		# year appointed
									,58		# year of experience
									,59 	# whether judge has 8 or more years of experience
									,71		# male judge
									,75		# year of first undergrad graduation
									,76   # year college slr
									,77		# year law school
									,79		# years in government
									,80		# years in government and not INS
									,81		# years in INS
									,82		# INS five year count
									,83		# years in the military
									,84		# NGO years
									,85		# years in private practice
									,86		# years in academia
									# ,124
									# ,125
									# ,126
									# ,127
									# ,128
									# ,129
									# ,130
									# ,131
									# ,132
									# ,133
									# ,134
									# ,135
									#,124
									# ,92
									# ,93
									# ,94
									# ,95
									# ,96
									# #,97
									# ,98
									# ,99
									# ,100
									# ,101
									# ,102
									# #,103
									# ,104
									# ,105
									# ,106
									# ,107
									# ,108
									# #,109
									# ,110
									# ,111
									# ,112
									# ,113
									# ,114
									# #,115
									# ,116
									# ,117
									# ,118
									# ,119
									# ,120
									# #,121
									#,122
									#,123


									# weather data
									# ,92		# precipitation
									# ,93		# snow fall
									# ,94		# snow depth
									# ,95		# temperature max
									# ,96		# temperature min
									# ,97		# minutes of sunshine
									]]

	# string data
	X_raw = convert_string_data(X_raw, data, [2		#hearing location code
																					 ,3		# refugee code
																					 ,67	# location of judge's bar
																					 ,68	# other locations mentioned in bio
																					 ,72	# court city
																					 ,78	# president
																					 ])

	# custom features
	X_raw = generate_custom_features(X_raw, data)

	# use column 26 raw_grant as response
	y_raw = data[:,26]

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

	print cleanedData.shape


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
	asylum_data = pandas.read_csv("data/raw/complete_data.csv")
	data = asylum_data.as_matrix()

	# select specific X columns as features
	# original (----- use this as baseline -----)
	#X_raw = data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58]]
	# with mood (mood_k2 at index 91, mood_k8 at index 97)
	#X_raw = data[:, [5, 6, 8, 11, 12, 15, 19, 21, 22, 23, 30, 31, 33, 34, 35, 36, 37, 39, 52, 53, 54, 55, 56, 58, 89, 90]]

	# numeric data features
	X_raw = data[:, [4		# hearing date
								  ,5		# laywer
									,6		# defensive 
									,7		# nationality id
									,8		# decision is written
									,10		# adj_time_start
									,11		# early start time
									,16		# court code (same as column 15 courtid)
									,19		# number of family members
									,21		# number of families in current time slot
									,22		# number of families per day
									,23 	# order within day
									,27		# lag
									,28		# lag on same day
									,29		# l2 grant
									,30		# how many of the previous 5 decision were grants
									,33		# number of grants within same court excluding current judge
									,34		#	number of previous 5 decided by current judge
									,35		# numcourtgrantother_prev5
									,36		# courtprevother5_dayslapse
									,37		# year
									,38		# mean grant rate per judge x nationality x defensive
									,41		# nationality defensive court id
									,46		# whether previous case was of same nationality
									,47		# grant grant
									,48		# grant deny
									,49		# deny grant
									,50		# deny deny
									,51		# hour start
									,52		# morning
									,53		# lunchtime
									,54		# numcases_judgeday
									,55		# numcases_judge
									,56		# minimum number of hearings in a courthouse
									,57		# year appointed
									,58		# year of experience
									,59 	# whether judge has 8 or more years of experience
									,71		# male judge
									,75		# year of first undergrad graduation
									,76   # year college slr
									,77		# year law school
									,79		# years in government
									,80		# years in government and not INS
									,81		# years in INS
									,82		# INS five year count
									,83		# years in the military
									,84		# NGO years
									,85		# years in private practice
									,86		# years in academia
									]]

	# string data
	X_raw = convert_string_data(X_raw, data, [2		#hearing location code
																					 ,3		# refugee code
																					 ,67	# location of judge's bar
																					 ,68	# other locations mentioned in bio
																					 ,72	# court city
																					 ,78	# president
																					 ])

	# custom features
	X_raw = generate_custom_features(X_raw, data)

	# use column 26 raw_grant as response
	y_raw = data[:,26]

	# split data into train and test
	return X_raw, y_raw

def load_all_clean_data():
	print "loading data and processing..."
	asylum_clean = pandas.read_csv("data/raw/complete_data.csv")
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
		print "depth:", d, "train score:", 1.0 - train_error, "test score:", 1.0 - test_error
		train_errors.append(train_error)
		test_errors.append(test_error)

	# plot
	bar_plot(depths, train_errors, test_errors, "Decision Tree Classifier", "depths")

# -------------------------------- DECISION TREE --------------------------------

# -------------------------------- ADABOOST -------------------------------------

# train adaboost on data and plot
def adaboost_train(X_train, X_test, y_train, y_test, max_rounds=10):
	print "Running Adaboost Classifier to max rounds of", max_rounds
	# change 0 to -1 for adaboost
	y_train[y_train == 0] = -1
	y_test[y_test == 0] = -1

	# run adaboost
	adaboost(X_train, X_test, y_train, y_test, max_rounds)

# train adaboost using decision tree weak classfier of depth 3
def adaboost(X_train, X_test, y_train, y_test, M=10):

	weights = np.ones(y_train.shape[0])/y_train.shape[0]
	alphas = []
	weak_classifiers = []

	train_errors = []
	test_errors = []

	rounds = np.arange(1,M+1)
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
		for j in range(len(weights)):
			if p[j] != y_train[j]:
				weights[j] = (weights[j] * np.exp(alpha))

		train_error = adaboost_error(X_train, y_train, alphas, weak_classifiers)
		test_error  = adaboost_error(X_test, y_test, alphas, weak_classifiers)
		train_errors.append(train_error)
		test_errors.append(test_error)
		print "num rounds:", i+1, "train score:", 1.0 - train_error, "test score:", 1.0 - test_error

	train_test_score_plot(rounds, train_errors, test_errors, "Adaboost Classifier", "num rounds")

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
	print "imputing data"
	rng = np.random.RandomState(0)
	n_samples = X.shape[0]
	n_features = X.shape[1]
	missing_rate = 1.0
	n_missing_samples = np.floor(n_samples * missing_rate)
	missing_samples = np.hstack((np.zeros(n_samples - n_missing_samples,
                                      dtype=np.bool),
                             np.ones(n_missing_samples,
                                     dtype=np.bool)))
	rng.shuffle(missing_samples)
	missing_features = rng.randint(0, n_features, n_missing_samples)

	# Estimate the score after imputation of the missing values
	print "training random forest"
	X_missing = X.copy()
	X_missing[np.where(missing_samples)[0], missing_features] = 0
	y_missing = y.copy()

	trees = [int(x) for x in np.arange(1,140)]
	scores = []
	for t in trees:
		estimator = Pipeline([("imputer", Imputer(missing_values=0,
																							strategy="most_frequent",
																							axis=0)),
													("forest", RandomForestRegressor(random_state=0,
																													 n_estimators=t, n_jobs=-1))])
		score = cross_val_score(estimator, X_missing, y_missing).mean()
		print "trees:", t, "score:", score
		scores.append(score)

	plt.rcParams.update({'font.size': 18})
	plt.plot(trees, scores)
	plt.xlabel("Number of Trees")
	plt.ylabel("Score")
	plt.title("Random Forest Regressor")
	plt.show()

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

def train_test_score_plot(depths, train_errors, test_errors, plot_title, xlabel):
	train_scores = [1.0-x for x in train_errors]
	test_scores = [1.0-x for x in test_errors]
	plt.plot(depths, train_scores, label="train score")
	plt.plot(depths, test_scores, label="test score")
	plt.title(plot_title)
	plt.xlabel(xlabel)
	plt.ylabel("score")
	plt.legend(loc=4)
	plt.show()

def bar_plot(depths, train_errors, test_errors, plot_title, xlabel):
	train_scores = [1.0-x for x in train_errors]
	test_scores = [1.0-x for x in test_errors]

	plt.rcParams.update({'font.size': 18})

	width = 0.40
	fig, ax = plt.subplots()
	rects1 = ax.bar(depths, train_scores, width, color='r')
	rects2 = ax.bar(depths + width, test_scores, width, color='b')

	ax.set_ylabel('Scores')
	ax.set_title(plot_title)
	ax.set_xlabel(xlabel)

	ax.set_xticks(depths)
	ax.legend((rects1[0], rects2[0]), ('train', 'test'), loc=4)

	plt.show()
# -------------------------------- PLOTTING -------------------------------------

if __name__ == '__main__':
  main()

def plot_random_forest():
	x_y = [(1 ,-0.371003354709),
	(2 ,-0.00701635196145),
	(3 ,0.112720357395),
	(4 ,0.172282300268),
	(5 ,0.209648590076),
	(6 ,0.23521886899),
	(7 ,0.252525159451),
	(8 ,0.265074523417),
	(9 ,0.274840153833),
	(10 ,0.281511652427),
	(11 ,0.288662235109),
	(12 ,0.292363234128),
	(13 ,0.298108873568),
	(14 ,0.302763071533),
	(15 ,0.307304420584),
	(16 ,0.310044049182),
	(17 ,0.312804911985),
	(18 ,0.315270263505),
	(19 ,0.317178211003),
	(20 ,0.319853239463),
	(21 ,0.321776840602),
	(22 ,0.322783556182),
	(23 ,0.324006600206),
	(24 ,0.325049813113),
	(25 ,0.326170702054),
	(26 ,0.326938160419),
	(27 ,0.328887749904),
	(28 ,0.33001058743),
	(29 ,0.331083203756),
	(30 ,0.332057776752),
	(31 ,0.332586715467),
	(32 ,0.333209795529),
	(33 ,0.334091025764),
	(34 ,0.334781724845),
	(35 ,0.335542360378),
	(36 ,0.336311262813),
	(37 ,0.336785200747),
	(38 ,0.337244930104),
	(39 ,0.337561498376),
	(40 ,0.338184930985),
	(41 ,0.338904645998),
	(42 ,0.339603604144),
	(43 ,0.34023077572),
	(44 ,0.340543623832),
	(45 ,0.340546215228),
	(46 ,0.341160673234),
	(47 ,0.341601366958),
	(48 ,0.341807764046),
	(49 ,0.342294718733),
	(50 ,0.342314307811),
	(51 ,0.342321184527),
	(52 ,0.342423462701),
	(53 ,0.342488836763),
	(54 ,0.342905737705),
	(55 ,0.343209785687),
	(56 ,0.343378896655),
	(57 ,0.343759329832),
	(58 ,0.344047916341),
	(59 ,0.344158432407),
	(60 ,0.344387023309),
	(61 ,0.344246425057),
	(62 ,0.344384350668),
	(63 ,0.344465165605),
	(64 ,0.344603823124),
	(65 ,0.344659241775),
	(66 ,0.34482758456),
	(67 ,0.345197997626),
	(68 ,0.345478271292),
	(69 ,0.345590477941),
	(70 ,0.34579955671),
	(71 ,0.345927218147),
	(72 ,0.34597392378),
	(73 ,0.346049972025),
	(74 ,0.346171689376),
	(75 ,0.346271961006),
	(76 ,0.346400862357),
	(77 ,0.346727045319),
	(78 ,0.346950748767),
	(79 ,0.346919829288),
	(80 ,0.347083798665),
	(81 ,0.347000870612),
	(82 ,0.347228247389),
	(83 ,0.347321033641),
	(84 ,0.347345973808),
	(85 ,0.347531155627),
	(86 ,0.347495209002),
	(87 ,0.347639669182),
	(88 ,0.347634544646),
	(89 ,0.347619835346),
	(90 ,0.347715359261),
	(91 ,0.347795103602),
	(92 ,0.347949885208),
	(93 ,0.348030177396),
	(94 ,0.348341368808),
	(95 ,0.348240099823),
	(96 ,0.348337795066),
	(97 ,0.348371122257),
	(98 ,0.348435935731),
	(99 ,0.34852554761),
	(100 ,0.34853855018),
	(101 ,0.348562738164),
	(102 ,0.348675969867),
	(103 ,0.34877932027),
	(104 ,0.348804290105),
	(105 ,0.348910454579),
	(106 ,0.349000878442),
	(107 ,0.34903216006),
	(108 ,0.349149453923),
	(109 ,0.349202812034),
	(110 ,0.34930116503),
	(111 ,0.349321424487),
	(112 ,0.349298724708),
	(113 ,0.349470340049),
	(114 ,0.349540599414),
	(115 ,0.349649567516),
	(116 ,0.349638198327),
	(117 ,0.349675520494),
	(118 ,0.349776589037),
	(119 ,0.349968376958),
	(120 ,0.350134450019),
	(121 ,0.350282182894),
	(122 ,0.35024287711),
	(123 ,0.350348136587),
	(124 ,0.350367938462),
	(125 ,0.350517964002),
	(126 ,0.350593828687),
	(127 ,0.350617717739),
	(128 ,0.3506792667),
	(129 ,0.350806942903),
	(130 ,0.350822949144),
	(131 ,0.350831682128),
	(132 ,0.350776829392),
	(133 ,0.350810753966),
	(134 ,0.350857520648),
	(135 ,0.350889603073),
	(136 ,0.351044762536)]

	plt.rcParams.update({'font.size': 18})
	plt.plot([x[0] for x in x_y], [x[1] for x in x_y])
	plt.xlabel("Number of Trees")
	plt.ylabel("Score")
	plt.title("Random Forest Regressor")
	plt.show()