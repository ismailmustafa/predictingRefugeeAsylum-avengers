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

def main():
	clean_format()

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







def addNationality():
	print "loading"
	data = pandas.read_csv("data/raw/asylum_clean_full.csv")
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
	data.to_csv('data/raw/asylum_data_full.csv', index=False, index_label=False)

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
