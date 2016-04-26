import pandas
import glob
import ntpath
import numpy as np
import datetime as dt
import cPickle as pickle
import os.path
import math

def main():
	weatherScore()

def convertWeatherFiles():
	weatherFiles = glob.glob("data/clean/*.dta")
	for f in weatherFiles:
		fileName = ntpath.basename(f).split(".")[0] + ".csv"
		fFrame = pandas.read_stata(f)
		fFrame.to_csv("data/weather/" + fileName)

def mergeBiosAndCityWthAsylumClean():
	asylum_clean = pandas.read_csv("data/raw/asylum_clean.csv")
	bios_clean = pandas.read_stata("data/raw/bios_clean.dta")
	city_lookup = pandas.read_csv("data/raw/cityLookup.csv")
	mergedDataFrameWithBios = asylum_clean.merge(bios_clean, on='ij_code', how='left')
	mergedDataFrameWithCityAndBios = mergedDataFrameWithBios.merge(city_lookup, on='hearing_loc_code', how='left')
	mergedDataFrameWithCityAndBios.to_csv('data/raw/asylum_clean_full.csv', index=False, index_label=False)

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
		for weather_row in all_weather_matrix:
			date = weather_row[4]
			city = weather_row[64]
			prcp = weather_row[14] # precipitation in 10ths of mm
			snow = weather_row[16] # snowfall in mm
			snwd = weather_row[15] # depth of snow in mm
			tmax = weather_row[21] # highest day temperature in 10th of celcius
			tmin = weather_row[22] # lowest day temperature in 10th of celcius
			tsun = weather_row[20] # daily total sunshine in minutes
			acsc = weather_row[70] # average cloudiness sunrise to sunset (percent)
			all_weather_dict[str(date) + city] = (nanCheck(prcp), nanCheck(snow), nanCheck(snwd), 
																					  nanCheck(tmax), nanCheck(tmin), nanCheck(tsun), 
																					  nanCheck(acsc))
		pickle.dump(all_weather_dict, open("data/raw/weather_fast_lookup", "wb"))
		print "saved weather data"

	print "loading asylum data"
	asylum_clean_full = pandas.read_csv("data/raw/asylum_clean_full_sample.csv")
	print "loaded asylum data"

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
		city = row[1][-1]

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
					prcp = weather_data[0] # precipitation in 10ths of mm
					snow = weather_data[1] # snowfall in mm
					snwd = weather_data[2] # depth of snow in mm
					tmax = weather_data[3] # highest day temperature in 10th of celcius
					tmin = weather_data[4] # lowest day temperature in 10th of celcius
					tsun = weather_data[5] # daily total sunshine in minutes
					acsc = weather_data[6] # average cloudiness sunrise to sunset (percent)
					weather_scores.append(applyWeights(weights(weather_data), weather_data))
		if dateCount != 0:
			score_calculation = compute_score(weather_scores)
		all_scores.append(score_calculation)
	asylum_clean_full['weather_mood'] = all_scores
	asylum_clean_full.to_csv('data/raw/asylum_clean_full_sample_weathermood.csv', index=False, index_label=False)

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
	maximum = max(weather_scores)
	normalized = [float(x)/maximum for x in weather_scores]
	return sum(normalized) / float(len(normalized))


def getCompletionDate(stata_date):
	date1960Jan1 = dt.datetime(1960,01,01)
	return date1960Jan1 + dt.timedelta(days=stata_date)

# This takes in te tblLookupBaseCity.csv and cleans it up. Saves to cityLookup.csv
def createCityLookup(city_lookup):
	city_lookup_matrix = city_lookup.as_matrix()
	city_lookup_short = []
	for row in city_lookup_matrix:
		city_lookup_short.append([row[1].strip(), row[5].strip()])
	data = np.asarray(city_lookup_short)
	df = pandas.DataFrame(data=data, columns=["hearing_loc_code", "city"])
	df.to_csv('data/raw/cityLookup.csv', index=False, index_label=False)



if __name__ == '__main__':
  main()
