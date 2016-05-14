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

def main():
	loadNationalityLookup()
	loadNationalityIdncaseLookup()
	addNationality()

def addNationality():
	print "loading"
	data = pandas.read_csv("data/raw/testing.csv")
	nationality_idncase_lookup = loadNationalityIdncaseLookup()
	print "loaded stuff"
	all_nat = []
	all_nationality = []
	for row in data.iterrows():
		idncase = row[1][0]
		lookup = nationality_idncase_lookup[idncase]
		nat = lookup[0]
		nationality = lookup[1]
		all_nat.append(nat)
		all_nationality.append(nationality)
	data["nat_code"] = all_nat
	data["nationality"] = all_nationality
	data.to_csv('data/raw/testing_nationality.csv', index=False, index_label=False)

def loadNationalityLookup():
	if os.path.isfile("data/raw/nationality_fast_lookup"):
		nationality_fast_lookup = pickle.load(open("data/raw/nationality_fast_lookup", "rb"))
		return nationality_fast_lookup
	else:
		print "creating nationality lookup"
		nationality_fast_lookup = {}
		nationality = pandas.read_csv("data/raw/tblLookupNationality.csv", header=None).as_matrix()
		for row in nationality:
			code = row[1]
			city = row[2]
			nationality_fast_lookup[code] = city
		pickle.dump(nationality_fast_lookup, open("data/raw/nationality_fast_lookup", "wb"))
		print "nationality lookup already created"
		return nationality_fast_lookup

def loadNationalityIdncaseLookup():
	nationality_idncase_lookup = {}
	if os.path.isfile("data/raw/nationality_idncase_lookup"):
		nationality_fast_lookup = pickle.load(open("data/raw/nationality_idncase_lookup", "rb"))
		return nationality_fast_lookup
	else:
		print "creating nationality idncase lookup"
		master = pandas.read_csv("data/raw/master.csv").as_matrix()
		nationality_fast_lookup = loadNationalityLookup()
		for row in master:
			idncase = 999999999
			if not math.isnan(row[0]):
				idncase = int(row[0])
			nat = row[1]
			natLookup = "??"
			if nat in nationality_fast_lookup:
				natLookup = nationality_fast_lookup[nat]
			nationality_idncase_lookup[idncase] = (nat, natLookup)
		pickle.dump(nationality_idncase_lookup, open("data/raw/nationality_idncase_lookup", "wb"))
		return nationality_idncase_lookup

def mergeBiosAndCityWthAsylumClean():
	asylum_clean = pandas.read_csv("data/raw/asylum_clean.csv")
	bios_clean = pandas.read_stata("data/raw/bios_clean.dta")
	city_lookup = pandas.read_csv("data/raw/cityLookup.csv")
	mergedDataFrameWithBios = asylum_clean.merge(bios_clean, on='ij_code', how='left')
	mergedDataFrameWithCityAndBios = mergedDataFrameWithBios.merge(city_lookup, on='hearing_loc_code', how='left')
	mergedDataFrameWithCityAndBios.to_csv('data/raw/testing.csv', index=False, index_label=False)

# This takes in te tblLookupBaseCity.csv and cleans it up. Saves to cityLookup.csv
def saveCityLookup():
	input_data = pandas.read_csv("data/raw/tblLookupBaseCity.csv")
	city_lookup_matrix = input_data.as_matrix()
	city_lookup_short = []
	for row in city_lookup_matrix:
		city_lookup_short.append([row[1].strip(), row[5].strip()])
	data = np.asarray(city_lookup_short)
	df = pandas.DataFrame(data=data, columns=["hearing_loc_code", "city"])
	df.to_csv('data/raw/cityLookup.csv', index=False, index_label=False)

def hmm_matrix_calculation():

	probabilities = {}
	if os.path.isfile("data/raw/hmm_probabilities"):
		probabilities = pickle.load(open("data/raw/hmm_probabilities", "rb"))
	else:
		print "--- probabilities not on file. Loading..."
		emission_probability = {
			'happy' : {'good_weather': 0.0, 'bad_weather': 0.0},
			'sad' 	: {'good_weather': 0.0, 'bad_weather': 0.0}
		}
		transition_probability = {
			'happy' : {'happy': 0.8, 'sad': 0.2},
			'sad' 	: {'happy': 0.2, 'sad': 0.8}
		}

		data = pandas.read_csv("data/raw/asylum_data_full_weather_five.csv")
		hmm_values = []
		for date, mood, prcp, snow, snwd, tmax, tmin in zip(data['comp_date'].values, data['twitter_score'].values, data['prcp'].values, data['snow'].values, data['snwd'].values, data['tmax'].values, data['tmin'].values):
			if not math.isnan(date) and not math.isnan(mood) and not math.isnan(tmax) and not math.isnan(tmin) and not math.isnan(prcp):
				hmm_values.append( 
												   ( 
												   	 getCompletionDate(int(date)),
														 mood,
														 ((float(tmax) + float(tmin)) / 2.0),
														 prcp
													 )
												 )

		# calculate averages
		average = {}
		average_mood = sum([x[1] for x in hmm_values]) / float(len([x[1] for x in hmm_values])) - 0.3
		average['mood'] = 6.01 # threshold in research paper
		average['tavg'] = sum([x[2] for x in hmm_values]) / float(len([x[2] for x in hmm_values]))
		average['prcp'] = sum([x[3] for x in hmm_values]) / float(len([x[3] for x in hmm_values]))

		maximum = {}
		maximum['tavg'] = max([x[2] for x in hmm_values])
		maximum['prcp'] = max([x[3] for x in hmm_values])

		# weather score calculated here
		hmm_values_score = []
		for date, mood, tavg, prcp in hmm_values:
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

		pickle.dump(probabilities, open("data/raw/hmm_probabilities", "wb"))
		print "saved probabilities"

	probabilities['states'] = ('happy', 'sad')
	return probabilities

# extra feature. Attach mood of state to city
def append_state_mood():
	data = pandas.read_csv("data/raw/asylum_data_complete.csv")
	state_mood = pandas.read_csv("data/raw/all_city_moods.csv")
	city_mood_dict = {}
	for row in state_mood.iterrows():
		city = row[1][2]
		state_mood = row[1][3]
		city_mood_dict[city.upper()] = state_mood

	# append to new dataset
	final_moods = []
	for row in data.iterrows():
		city = row[1][88]
		if city in city_mood_dict:
			final_moods.append(city_mood_dict[city])
		else:
			final_moods.append("")

	data['state_mood'] = final_moods
	data.to_csv('data/raw/asylum_data_complete_state_mood.csv', index=False, index_label=False)

def save_city_moods():
	cities = pandas.read_csv("data/raw/cities.csv")
	state_mood = pandas.read_csv("data/raw/state_mood.csv")

	state_mood_dict = {}
	for row in state_mood.iterrows():
		state = row[1][0]
		mood = row[1][1]
		state_mood_dict[state] = mood

	all_moods = []
	for row in cities.iterrows():
		state = row[1][0]
		city = row[1][2]
		if state in state_mood_dict:
			all_moods.append(state_mood_dict[state])
		else:
			all_moods.append(np.nan)

	cities['state_mood'] = all_moods
	cities.to_csv('data/raw/all_city_moods.csv', index=False, index_label=False)

def date_range():
	data = pandas.read_csv("data/raw/asylum_data_complete.csv")
	all_dates = [getCompletionDate(int(x)) for x in data['comp_date'].values if not math.isnan(x)]
	print len(np.unique(data['ij_code'].values))
	return min(all_dates), max(all_dates)

def calculate_full_mood_data():
	probabilities = hmm_matrix_calculation()
	emission_probability = probabilities['emission_probability']
	transition_probability = probabilities['transition_probability']
	states = probabilities['states']
	start_probability = {'sad': 0.5, 'happy': 0.5}

	data = pandas.read_csv("data/raw/asylum_data_full_weather_five.csv")

	# determine maximum prcp and tavg
	vals = []
	for prcp, tmax, tmin, prcp_minus_1, tmax_minus_1, tmin_minus_1, prcp_minus_2, tmax_minus_2, tmin_minus_2, prcp_minus_3, tmax_minus_3, tmin_minus_3, prcp_minus_4, tmax_minus_4, tmin_minus_4 in zip(data['prcp'].values, data['tmax'].values, data['tmin'].values, data['prcp_minus_1'].values, data['tmax_minus_1'].values, data['tmin_minus_1'].values, data['prcp_minus_2'].values, data['tmax_minus_2'].values, data['tmin_minus_2'].values, data['prcp_minus_3'].values, data['tmax_minus_3'].values, data['tmin_minus_3'].values, data['prcp_minus_4'].values, data['tmax_minus_4'].values, data['tmin_minus_4'].values):
		vals.append((prcp, (tmax + tmin) / 2.0, prcp_minus_1, (tmax_minus_1 + tmin_minus_1) / 2.0, prcp_minus_2, (tmax_minus_2 + tmin_minus_2) / 2.0, prcp_minus_3, (tmax_minus_3 + tmin_minus_3) / 2.0, prcp_minus_4, (tmax_minus_4 + tmin_minus_4) / 2.0))

	# calculate maximums
	maximum = {}
	maximum['prcp'] = max([x[0] for x in vals if not math.isnan(x[0])])
	maximum['tavg'] = max([x[1] for x in vals if not math.isnan(x[1])])

	maximum['prcp_minus_1'] = max([x[2] for x in vals if not math.isnan(x[2])])
	maximum['tavg_minus_1'] = max([x[3] for x in vals if not math.isnan(x[3])])

	maximum['prcp_minus_2'] = max([x[4] for x in vals if not math.isnan(x[4])])
	maximum['tavg_minus_2'] = max([x[5] for x in vals if not math.isnan(x[5])])

	maximum['prcp_minus_3'] = max([x[6] for x in vals if not math.isnan(x[6])])
	maximum['tavg_minus_3'] = max([x[7] for x in vals if not math.isnan(x[7])])

	maximum['prcp_minus_4'] = max([x[8] for x in vals if not math.isnan(x[8])])
	maximum['tavg_minus_4'] = max([x[9] for x in vals if not math.isnan(x[9])])

	all_moods = []
	# calculate observations for each row
	count = 0
	for prcp, tavg, prcp_minus_1, tavg_minus_1, prcp_minus_2, tavg_minus_2, prcp_minus_3, tavg_minus_3, prcp_minus_4, tavg_minus_4 in vals:
		row_observations = []
		count += 1
		# observation one
		if not math.isnan(prcp) and not math.isnan(tavg):
			row_observations.append(weather_score(maximum['prcp'], maximum['tavg'], prcp, tavg))
		else:
			row_observations.append(np.nan)

		# observation two
		if not math.isnan(prcp_minus_1) and not math.isnan(tavg_minus_1):
			row_observations.append(weather_score(maximum['prcp_minus_1'], maximum['tavg_minus_1'], prcp_minus_1, tavg_minus_1))
		else:
			row_observations.append(np.nan)

		# observation three
		if not math.isnan(prcp_minus_2) and not math.isnan(tavg_minus_2):
			row_observations.append(weather_score(maximum['prcp_minus_2'], maximum['tavg_minus_2'], prcp_minus_2, tavg_minus_2))
		else:
			row_observations.append(np.nan)

		# observation four
		if not math.isnan(prcp_minus_3) and not math.isnan(tavg_minus_3):
			row_observations.append(weather_score(maximum['prcp_minus_3'], maximum['tavg_minus_3'], prcp_minus_3, tavg_minus_3))
		else:
			row_observations.append(np.nan)

		# observation five
		if not math.isnan(prcp_minus_4) and not math.isnan(tavg_minus_4):
			row_observations.append(weather_score(maximum['prcp_minus_4'], maximum['tavg_minus_4'], prcp_minus_4, tavg_minus_4))
		else:
			row_observations.append(np.nan)

		# reverse to ascending order and covert to observations
		row_observations.reverse()
		obs = []
		for x in row_observations:
			if math.isnan(x):
				obs.append("")
			elif x > 0.5:
				obs.append("good_weather")
			else:
				obs.append("bad_weather")


		# use viterbi to calculate latent variables
		latent_results = ["","","","",""]
		if not "" in obs:
			latent_results = viterbi(obs, states, start_probability, transition_probability, emission_probability)

		if latent_results[-1] == "happy":
			all_moods.append(1)
		elif latent_results[-1] == "sad":
			all_moods.append(0)
		else:
			all_moods.append(np.nan)

		if count % 10000 == 0:
			print count

	data['mood'] = all_moods
	data.to_csv('data/raw/asylum_data_full_hmm_mood.csv', index=False, index_label=False)


# normalize tavg and prcp, then subtract four times prcp from tavg
def weather_score(prcp_max, tavg_max, prcp, tavg):
	t = tavg / tavg_max
	p = prcp / prcp_max
	return t - 4*p

	# observations = ('good_weather', 'bad_weather', 'bad_weather', 'bad_weather')

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

def calculate_mood(mood, avg):
	if mood > avg:
		return 'happy'
	else:
		return 'sad'

def all_but_index(values, index):
	new_values = []
	for i,v in enumerate(values):
		if i != index:
			new_values.append(v)
	return new_values


# return next day given current day
def get_next_day(date):
	return date + datetime.timedelta(days=1)

def get_prev_day(date):
	return date + datetime.timedelta(days=-1)

# normalize tavg and prcp, then subtract four times prcp from tavg
def calculate_weather_score(maximum, tavg, prcp):
	t = tavg / maximum['tavg']
	p = prcp / maximum['prcp']
	return t - 4*p

def gaussian(u1,u2):
  z1 = np.sqrt(-2*np.log(u1))*np.cos(2*np.pi*u2)
  z2 = np.sqrt(-2*np.log(u1))*np.sin(2*np.pi*u2)
  return z1,z2

def histogram(vals, title):
	n, bins, patches = plt.hist(vals, 50, normed=1, facecolor='green', alpha=0.75)
	plt.title(title)
	plt.grid(True)
	plt.show()

def weather_distribution_average(data):
	weather_data = data['weather_score'].values
	average = sum(weather_data) / len(weather_data)
	return average

	# the histogram of the data
	n, bins, patches = plt.hist(weather_data, 50, normed=1, facecolor='green', alpha=0.75)

	plt.xlabel('bins')
	plt.ylabel('weather')
	plt.title(r'weather distribution')
	plt.grid(True)

	plt.show()

def average(column_values):
	remove_nans = []
	for val in column_values:
		if not math.isnan(val):
			remove_nans.append(val)
	return sum(remove_nans) / float(len(remove_nans))
		

def getCompletionDate(stata_date):
	date1960Jan1 = dt.datetime(1960,01,01)
	return date1960Jan1 + dt.timedelta(days=stata_date)

def load_bios():
	all_bios = []
	bios = glob.glob("data/raw/bio/*.txt")
	for b in bios:
		bio = open(b,'r')
		all_bios.append(bio.read())
		bio.close()
	return all_bios

def clean_format():
	data = pandas.read_csv("data/raw/bios_clean.csv")
	all_output = "ij_code,first_name,last_name,decisions,percent_denial\n"
	for row in data.iterrows():
		output = str(row[1]['ij_code']) + "," + str(row[1]['FirstName']) + "," + str(row[1]['LastName']) + "," ",\n"
		all_output += output
	csv_file = open("data/raw/bio_decisions.csv", "w")
	csv_file.write(all_output)
	csv_file.close()


def extract_degree():
	all_bios = load_bios()
	data = pandas.read_csv("data/raw/bios_clean.csv")
	i = 0
	all_output = "ij_code,FirstName,LastName,degree\n"
	for row in data.iterrows():
		last_name = str(row[1]['LastName'])
		year_appointed = ""
		if not math.isnan(row[1]['Year_Appointed_SLR']):
			year_appointed = str(int(row[1]['Year_Appointed_SLR']))
		elif not math.isnan(row[1]['YearofFirstUndergradGraduatio']):
			year_appointed = str(int(row[1]['YearofFirstUndergradGraduatio']))

		degree = get_degree(all_bios, last_name, year_appointed)
		output = str(row[1]['ij_code']) + "," + str(row[1]['FirstName']) + "," + str(row[1]['LastName']) + "," + degree + "\n"
		all_output += output
	csv_file = open("data/raw/bio_degree_new.csv", "w")
	csv_file.write(all_output)
	csv_file.close()

def get_degree(all_bios, last_name, year_appointed):
	num_found = 0
	bio_found = ""
	for b in all_bios:
		res1 = b.find(last_name)
		res2 = b.find(year_appointed)
		if res1 != -1 and res2 != -1:
			num_found += 1
			bio_found = b
	if num_found == 1:
		degree_index = bio_found.find("bachelor")
		if degree_index != -1:
			degree_list = bio_found[degree_index:degree_index+20].split()
			degree = degree_list[0] + " " + degree_list[1] + " " + degree_list[2]
			return degree
	return ""

def convertWeatherFiles():
	weatherFiles = glob.glob("data/clean/*.dta")
	for f in weatherFiles:
		fileName = ntpath.basename(f).split(".")[0] + ".csv"
		fFrame = pandas.read_stata(f)
		fFrame.to_csv("data/weather/" + fileName)

def nanCheck(val):
	if (type(val) is float and val < -9970.0) or (type(val) is int and val < -9970):
			return float('nan')
	return val

def weatherScore():
	all_weather_dict = {}
	# Load fast lookup for weather data
	if os.path.isfile("data/raw/weather_fast_lookup"):
		print "weather data in memory"
		all_weather_dict = pickle.load(open("data/raw/weather_fast_lookup", "rb"))
		print "loaded weather data from memory"
	# otherwise load non-condensed version
	else:
		print "weather data not in memory"
		all_weather = pandas.read_stata("data/clean/weather_all.dta")
		all_weather_matrix = all_weather.as_matrix()

		# initialize max values
		max_prcp = max_snow = max_snwd = max_tsun = max_tavg = sys.float_info.min

		for weather_row in all_weather_matrix:
			date = weather_row[4]
			city = weather_row[64]
			prcp = nanCheck(weather_row[14]) # precipitation in 10ths of mm
			snow = nanCheck(weather_row[16]) # snowfall in mm
			snwd = nanCheck(weather_row[15]) # depth of snow in mm
			tmax = nanCheck(weather_row[21]) # highest day temperature in 10th of celcius
			tmin = nanCheck(weather_row[22]) # lowest day temperature in 10th of celcius
			tsun = nanCheck(weather_row[20]) # daily total sunshine in minutes
			all_weather_dict[str(date) + city] = (prcp, snow, snwd, 
																					  tmax, tmin, tsun)

			# calculate max values
			if not math.isnan(prcp) and prcp > max_prcp:
				max_prcp = prcp
			if not math.isnan(snow) and snow > max_snow:
				max_snow = snow
			if not math.isnan(snwd) and snwd > max_snwd:
				max_snwd = snwd
			if not math.isnan(tsun) and tsun > max_tsun:
				max_tsun = tsun
			if not math.isnan(tmax) and not math.isnan(tmin) and (abs(float(tmax) + float(tmin))/2.0) > max_tavg:
				max_tavg = (abs(float(tmax) + float(tmin))/2.0)

		# set max values
		all_weather_dict["max_prcp"] = max_prcp
		all_weather_dict["max_snow"] = max_snow
		all_weather_dict["max_snwd"] = max_snwd
		all_weather_dict["max_tsun"] = max_tsun
		all_weather_dict["max_tavg"] = max_tavg

		pickle.dump(all_weather_dict, open("data/raw/weather_fast_lookup", "wb"))
		print "saved weather data"

	print "loading asylum data"
	asylum_clean_full = pandas.read_csv("data/raw/asylum_clean_full_sports.csv")
	print "loaded asylum data"

	lowest_score = sys.float_info.max
	highest_score = sys.float_info.min

	# Iterate through asylum clean data and calculate weather score
	all_scores = []
	for i,row in enumerate(asylum_clean_full.iterrows()):
		# Get date of completion from row
		currentDate = getCompletionDate(row[1][4])
		allDates = [str(currentDate.strftime('%Y%m%d'))]
		for numDays in np.arange(1,5):
			newDate = currentDate - dt.timedelta(days=numDays)
			allDates.append(str(newDate.strftime('%Y%m%d')))
		# Get dates of current and past four days
		city = row[1][-2]

		# Default to 0.5
		score_calculation = 0.5
		dateCount = 0
		# iterate through all 5 days of weather
		weather_scores = []
		for date in allDates:
			# Check if date and city values exist
			if type(date) is str and type(city) is str:
				dateCount += 1
				key = date + city
				if key in all_weather_dict:
					weather_data = all_weather_dict[key]
					prcp = 1.0 - safe_divide(weather_data[0], all_weather_dict["max_prcp"]) # precipitation in 10ths of mm
					snow = 1.0 - safe_divide(weather_data[1], all_weather_dict["max_snow"]) # snowfall in mm
					snwd = 1.0 - safe_divide(weather_data[2], all_weather_dict["max_snwd"]) # depth of snow in mm
					tmax = weather_data[3] 																						# highest day temperature in 10th of celcius
					tmin = weather_data[4] 																						# lowest day temperature in 10th of celcius
					tavg = (float(tmax) + float(tmin)) / 2.0    											# average day temperature in 10th of celcius
					tscore = temp_score(tavg)
					tsun = safe_divide(weather_data[5], all_weather_dict["max_tsun"]) # daily total sunshine in minutes

					calculations = prcp, snow, snwd, tscore, tsun
					weather_scores.append(applyWeights(weights(calculations), calculations))
		if dateCount != 0:
			score_calculation = compute_score(weather_scores)
			if score_calculation > highest_score:
				highest_score = score_calculation
			if score_calculation < lowest_score and score_calculation > 0.01 :
				lowest_score = score_calculation

		all_scores.append(score_calculation)
	asylum_clean_full['weather_score'] = all_scores
	asylum_clean_full.to_csv('data/raw/asylum_clean_full_test.csv', index=False, index_label=False)

	print "highest score:", highest_score
	print "lowest_score:", lowest_score

def temp_score(tavg):
	# return 1 - (abs(tavg - 136.0) / max_tavg)
	if tavg <= 0.0 or tavg >= 272:
		return 0.0
	else:
		return 1.0 - (abs(tavg - 136.0) / 136.0)

def safe_divide(val, max_val):
	if not math.isnan(val):
		return float(val) / float(max_val)
	else:
		return val

# determine weights for weather score
def weights(weather_data):
	weights = []
	weight_count = 0
	for val in weather_data:
		if not math.isnan(val):
			weight_count += 1
	for i in range(weight_count):
		weights.append(1.0/float(weight_count))
	return weights

def applyWeights(weights, weather_data):
	weight_index = 0
	score = 0.0
	for val in weather_data:
		if not math.isnan(val):
			score += weights[weight_index]*float(val)
			weight_index += 1
	return score

def compute_score(weather_scores):
	if len(weather_scores) == 0:
		return 0.0
	return sum(weather_scores) / float(len(weather_scores))

def getCompletionDate(stata_date):
	date1960Jan1 = dt.datetime(1960,01,01)
	return date1960Jan1 + dt.timedelta(days=stata_date)


if __name__ == '__main__':
  main()
