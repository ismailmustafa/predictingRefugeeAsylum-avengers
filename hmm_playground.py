import pandas
import glob
import ntpath
import numpy as np
import datetime as dt
import cPickle as pickle
import os.path
import math
import sys
from collections import Counter
import matplotlib.pyplot as plt
from hmmlearn import hmm
import datetime

# Trains HMM and prints out list of transition probabilities where
# score is greater than 0.9
def main():
	train, test = hmm_train_test_test_split()
	probabilities = hmm_matrix_calculation_evaluation(train)

	increment = 0.1
	happy_happy = np.arange(0.0,1.0 + increment, increment)
	happy_sad = np.arange(0.0,1.0 + increment, increment)
	sad_happy = np.arange(0.0,1.0 + increment, increment)
	sad_sad = np.arange(0.0,1.0 + increment, increment)

	maximum = calculate_maximum(test)

	for hh in happy_happy:
		for hs in happy_sad:
			for sh in sad_happy:
				for ss in sad_sad:
					if (hh + hs) < 1.001 and (hh + hs) > 0.999 and (sh + ss) < 1.001 and (sh + ss) > 0.999:
						transition_probability = {
							'happy' : {'happy': hh, 'sad': hs},
							'sad' 	: {'happy': sh, 'sad': ss}
						}
						score = hmm_score(probabilities, test, transition_probability, maximum)
						if score > 0.9:
							print hh, hs, sh, ss, "=", score

def calculate_maximum(hmm_values):
	# calculate maximums
	maximum = {}
	maximum['prcp'] = max([x[2] for x in hmm_values if not math.isnan(x[2])])
	maximum['tavg'] = max([x[3] for x in hmm_values if not math.isnan(x[3])])

	maximum['prcp_minus_1'] = max([x[4] for x in hmm_values if not math.isnan(x[4])])
	maximum['tavg_minus_1'] = max([x[5] for x in hmm_values if not math.isnan(x[5])])

	maximum['prcp_minus_2'] = max([x[6] for x in hmm_values if not math.isnan(x[6])])
	maximum['tavg_minus_2'] = max([x[7] for x in hmm_values if not math.isnan(x[7])])

	maximum['prcp_minus_3'] = max([x[8] for x in hmm_values if not math.isnan(x[8])])
	maximum['tavg_minus_3'] = max([x[9] for x in hmm_values if not math.isnan(x[9])])

	maximum['prcp_minus_4'] = max([x[10] for x in hmm_values if not math.isnan(x[10])])
	maximum['tavg_minus_4'] = max([x[11] for x in hmm_values if not math.isnan(x[11])])
	return maximum

def hmm_score(probabilities, hmm_values, transition_probability, maximum):
	emission_probability = probabilities['emission_probability']
	# transition_probability = probabilities['transition_probability']
	states = probabilities['states']
	start_probability = {'sad': 0.5, 'happy': 0.5}

	correct_count = 0
	total_count = 0
	for date, mood, prcp, tavg, prcp_minus_1, tavg_minus_1, prcp_minus_2, tavg_minus_2, prcp_minus_3, tavg_minus_3, prcp_minus_4, tavg_minus_4 in hmm_values:
		total_count += 1
		row_observations = [
											 		weather_score(maximum['prcp_minus_4'], maximum['tavg_minus_4'], prcp_minus_4, tavg_minus_4),
											 		weather_score(maximum['prcp_minus_3'], maximum['tavg_minus_3'], prcp_minus_3, tavg_minus_3),
											 		weather_score(maximum['prcp_minus_2'], maximum['tavg_minus_2'], prcp_minus_2, tavg_minus_2),
											 		weather_score(maximum['prcp_minus_1'], maximum['tavg_minus_1'], prcp_minus_1, tavg_minus_1),
											 		weather_score(maximum['prcp'], maximum['tavg'], prcp, tavg)
											 ]

		# reverse to ascending order and covert to observations
		obs = []
		for x in row_observations:
			if x > 0.5:
				obs.append("good_weather")
			else:
				obs.append("bad_weather")

		# use viterbi to calculate latent variables
		latent_results = viterbi(obs, states, start_probability, transition_probability, emission_probability)

		mood_str = latent_results[-1]
		if mood_str == "happy" and mood > 6.01:
			correct_count += 1
		elif mood_str == "sad" and mood <= 6.01:
			correct_count += 1

	return float(correct_count) / float(total_count)

def hmm_train_test_test_split(split=0.5):
	data = pandas.read_csv("data/raw/complete_data.csv")
	hmm_values = []
	for date, mood, prcp, tmax, tmin, prcp_minus_1, tmax_minus_1, tmin_minus_1, prcp_minus_2, tmax_minus_2, tmin_minus_2, prcp_minus_3, tmax_minus_3, tmin_minus_3, prcp_minus_4, tmax_minus_4, tmin_minus_4 in zip(data['comp_date'].values, data['twitter_score'].values, data['prcp'].values, data['tmax'].values, data['tmin'].values, data['prcp_minus_1'].values, data['tmax_minus_1'].values, data['tmin_minus_1'].values, data['prcp_minus_2'].values, data['tmax_minus_2'].values, data['tmin_minus_2'].values, data['prcp_minus_3'].values, data['tmax_minus_3'].values, data['tmin_minus_3'].values, data['prcp_minus_4'].values, data['tmax_minus_4'].values, data['tmin_minus_4'].values):
		if not math.isnan(date) and not math.isnan(mood) and not math.isnan(tmax) and not math.isnan(tmin) and not math.isnan(prcp):
			hmm_values.append( 
											   ( 
											   	 getCompletionDate(int(date)),
													 mood,
													 prcp,
													 ((float(tmax) + float(tmin)) / 2.0),
													 prcp_minus_1,
													 ((float(tmax_minus_1) + float(tmin_minus_1)) / 2.0),
													 prcp_minus_2,
													 ((float(tmax_minus_2) + float(tmin_minus_2)) / 2.0),
													 prcp_minus_3,
													 ((float(tmax_minus_3) + float(tmin_minus_3)) / 2.0),
													 prcp_minus_4,
													 ((float(tmax_minus_4) + float(tmin_minus_4)) / 2.0)
												 )
											 )

	split_index = int(math.floor(split*len(hmm_values)))
	train = hmm_values[:split_index]
	test = hmm_values[split_index:]
	return train, test
	hmm_matrix_calculation_evaluation(train)

def hmm_matrix_calculation_evaluation(hmm_values):

	probabilities = {}
	emission_probability = {
		'happy' : {'good_weather': 0.0, 'bad_weather': 0.0},
		'sad' 	: {'good_weather': 0.0, 'bad_weather': 0.0}
	}
	transition_probability = {
		'happy' : {'happy': 0.8, 'sad': 0.2},
		'sad' 	: {'happy': 0.2, 'sad': 0.8}
	}

	data = pandas.read_csv("data/raw/complete_data.csv")

	# calculate averages
	average = {}
	average['mood'] = 6.01 # threshold in research paper
	average['prcp'] = sum([x[2] for x in hmm_values]) / float(len([x[2] for x in hmm_values]))
	average['tavg'] = sum([x[3] for x in hmm_values]) / float(len([x[3] for x in hmm_values]))

	maximum = {}
	maximum['prcp'] = max([x[2] for x in hmm_values])
	maximum['tavg'] = max([x[3] for x in hmm_values])

	# weather score calculated here
	hmm_values_score = []
	for date, mood, prcp, tavg, _, _, _, _, _, _, _, _ in hmm_values:
		hmm_values_score.append(	
													   (
													   	 date
													   , mood
													   , calculate_weather_score(maximum, tavg, prcp)
													   )
													 )

	# add average weather
	average['weather'] = sum([x[2] for x in hmm_values_score]) / float(len([x[2] for x in hmm_values_score]))

	# emission counts
	emission_counts = Counter()
	for date, mood, weather_score in hmm_values_score:

		# mood count
		if mood > average['mood']:
			emission_counts['happy'] += 1
		else:
			emission_counts['sad'] += 1

		# happy combined counts
		if mood > average['mood'] and weather_score > average['weather']:
			emission_counts['happy_good_weather'] += 1
		elif mood > average['mood'] and weather_score <= average['weather']:
			emission_counts['happy_bad_weather'] += 1

		# sad combined counts
		if mood <= average['mood'] and weather_score > average['weather']:
			emission_counts['sad_good_weather'] += 1
		elif mood <= average['mood'] and weather_score <= average['weather']:
			emission_counts['sad_bad_weather'] += 1

	# calculate emission probabilities
	emission_probability['happy']['good_weather'] = float(emission_counts['happy_good_weather']) / float(emission_counts['happy'])
	emission_probability['happy']['bad_weather'] = float(emission_counts['happy_bad_weather']) / float(emission_counts['happy'])

	emission_probability['sad']['good_weather'] = float(emission_counts['sad_good_weather']) / float(emission_counts['sad'])
	emission_probability['sad']['bad_weather'] = float(emission_counts['sad_bad_weather']) / float(emission_counts['sad'])

	# emission probabiliiies
	for key in emission_probability:
		print "--", key, "--"
		for sub_key in emission_probability[key]:
			print "   ", sub_key, emission_probability[key][sub_key]

	probabilities = {
		'transition_probability' : transition_probability,
		'emission_probability' 	: emission_probability
	}

	probabilities['states'] = ('happy', 'sad')

	return probabilities

def getCompletionDate(stata_date):
	date1960Jan1 = dt.datetime(1960,01,01)
	return date1960Jan1 + dt.timedelta(days=stata_date)

# normalize tavg and prcp, then subtract four times prcp from tavg
def calculate_weather_score(maximum, tavg, prcp):
	t = tavg / maximum['tavg']
	p = prcp / maximum['prcp']
	return t - 4*p

# split up prcp_max and tavg_max
def weather_score(prcp_max, tavg_max, prcp, tavg):
	t = tavg / tavg_max
	p = prcp / prcp_max
	return t - 4*p

# viterbi algorithm taken from wikipedia
def viterbi(obs, states, start_p, trans_p, emit_p):
			V = [{}]
			for i in states:
				V[0][i] = start_p[i]*emit_p[i][obs[0]]
			# Run Viterbi when t > 0
			for t in range(1, len(obs)):
				V.append({})
				for y in states:
					prob = max(V[t - 1][y0]*trans_p[y0][y]*emit_p[y][obs[t]] for y0 in states)
					V[t][y] = prob
			opt = []
			for j in V:
				for x, y in j.items():
					if j[x] == max(j.values()):
						opt.append(x)
			# The highest probability
			h = max(V[-1].values())
			return opt

if __name__ == '__main__':
  main()