import pandas as pd
import datetime as dt
from datetime import date, timedelta
import numpy as np
from pandas import Series
import sys
import re
import cPickle as pickle
import os.path
import math
import os
from collections import defaultdict

def nanCheck(val):
    if (type(val) is float and val < -9970.0) or (type(val) is int and val < -9970):
            return float('nan')
    return val

def temp_score(tavg, max_tavg, min_tavg):
    return (tavg - min_tavg) / (max_tavg - min_tavg)

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

# convert stata date to string
def getCompletionDate(stata_date):
    date1960Jan1 = dt.datetime(1960,01,01)
    return date1960Jan1.date() + dt.timedelta(days=stata_date)

class LoadData:

    #------------------- TWITTER DATA -----------------------
    all_twitter_data = {}
    print "---loading twitter data"
    if os.path.isfile("data/raw/twitter_fast_lookup"):
        all_twitter_data = pickle.load(open("data/raw/twitter_fast_lookup", "rb"))
        print "---loaded twitter data"
    else:
        print "---twitter data not on file, loading"
        for row in open('data/raw/mood-city.txt'):
            temp = row.split('\t')
            all_twitter_data[temp[0]+temp[1].lower()] = temp[3]
        pickle.dump(all_twitter_data, open("data/raw/twitter_fast_lookup", "wb"))
        print "---twitter data saved"
    # ------------------- SPORTS DATA ------------------------
    def format_date(row):
        date_list = row.split()
        play_date = date_list[len(date_list)-1]
        return dt.datetime.strptime(play_date, '%m/%d/%y').date()

    all_sports_data = {}
    print "---loading sports data"
    if os.path.isfile("data/raw/sports_fast_lookup"):
        all_sports_data = pickle.load(open("data/raw/sports_fast_lookup", "rb"))
        print "---loaded sports data"
    else:
        print "---sports data not on file, loading"
        nba_data_matrix = pd.read_csv('data/sportsData/NBA.csv', index_col=False).as_matrix()
        nfl_data_matrix = pd.read_csv('data/sportsData/NFL.csv', index_col=False).as_matrix()
        mlb_data_matrix = pd.read_csv('data/sportsData/MLB.csv', index_col=False).as_matrix()
        nhl_data_matrix = pd.read_csv('data/sportsData/NHL.csv', sep=',', index_col=False).as_matrix()

        for row in nba_data_matrix:
            date = format_date(row[2])
            team_name = row[0]
            result = row[9]
            all_sports_data[str(date) + team_name.lower() + "nba"] = result
        for row in nfl_data_matrix:
            date = format_date(row[2])
            team_name = row[0]
            result = row[9]
            all_sports_data[str(date) + team_name.lower() + "nfl"] = result
        for row in mlb_data_matrix:
            date = format_date(row[2])
            team_name = row[0]
            result = row[10]
            all_sports_data[str(date) + team_name.lower() + "mlb"] = result
        for row in nhl_data_matrix:
            date = format_date(row[2])
            team_name = row[0]
            result = row[11]
            all_sports_data[str(date) + team_name.lower() + "nhl"] = result
        pickle.dump(all_sports_data, open("data/raw/sports_fast_lookup", "wb"))
        print "---sports data saved"

    nfl_team_by_state = {'Arizona':['Arizona Cardinals'],
                             'Georgia':['Atlanta Falcons'],
                             'Maryland':['Baltimore Ravens'],
                             'New York':['Buffalo Bills','New York Giants','New York Jets'],
                             'North Carolina':['Carolina Panthers'],
                             'Illinois':['Chicago Bears'],
                             'Ohio':['Cincinnati Bengals','Cleveland Browns'],
                             'Texas':['Dallas Cowboys','Houston Texans''Indianapolis Colts'],
                             'Colorado':['Denver Broncos'],
                             'Michigan':['Detroit Lions'],
                             'Wisconsin':['Green Bay Packers'],
                             'Florida':['Jacksonville Jaguars','Miami Dolphins','Tampa Bay Buccaneers'],
                             'Kansas':['Kansas City Chiefs'],
                             'Minnesota':['Minnesota Vikings'],
                             'Massachusetts':['New England Patriots'],
                             'New Orleans':['New Orleans Saints'],
                             'California':['Oakland Raiders','San Diego Chargers','San Francisco 49ers'],
                             'Pennsylvania':['Philadelphia Eagles','Pittsburgh Steelers'],
                             'Washington':['Seattle Seahawks'],
                             'Missouri':['St. Louis Rams'],
                             'Tennessee':['Tennessee Titans'],
                             'DC':['Washington Redskins']}
    nba_team_by_state = {'Arizona':['Phoenix Suns'],
                            'California':['Golden State Warriors','Los Angeles Clippers','Los Angeles Lakers','Sacramento Kings'],
                            'Colorado':['Denver Nuggets'],
                            'Florida':['Miami Heat','Orlando Magic'],
                            'Georgia':['Atlanta Hawks'],
                            'Illinois':['Chicago Bulls'],
                            'Indiana':['Indiana Pacers'],
                            'Louisiana':['New Orleans Hornets', 'New Orleans Pelicans'],
                            'Massachusetts':['Boston Celtics'],
                            'Michigan':['Detroit Pistons'],
                            'Minnesota':['Minnesota Timberwolves'],
                            'New York':['Brooklyn Nets','New York Knicks'],
							'New Jersey':['New Jersey Nets'],
                            'North Carolina':['Charlotte Bobcats'],
                            'Ohio':['Cleveland Cavaliers'],
                            'Oklahoma':['Oklahoma City Thunder'],
                            'Oregon':['Portland Trail Blazers'],
                            'Pennsylvania':['Philadelphia 76ers'],
                            'Tennessee':['Memphis Grizzlies'],
                            'Texas':['Dallas Mavericks','Houston Rockets','San Antonio Spurs'],
                            'Utah':['Utah Jazz'],
                            'Wisconsin':['Milwaukee Bucks'],
                            'DC':['Washington Wizards']}

    mlb_team_by_state = {
                            'Arizona':['Arizona Diamondbacks'],
                            'California':['Los Angeles Angels','Los Angeles Dodgers','Oakland Athletics','San Diego Padres','San Francisco Giants'],
                            'Colorado':['Colorado Rockies'],
                            'Florida':['Florida Marlins', 'Miami Marlins','Tampa Bay Rays'],
                            'Georgia':['Atlanta Braves'],
                            'Illinois':['Chicago Cubs','Chicago White Sox'],
                            'Maryland':['Baltimore Orioles'],
                            'Massachusetts':['Boston Red Sox'],
                            'Michigan':['Detroit Tigers'],
                            'Minnesota':['Minnesota Twins'],
                            'Missouri':['Kansas City Royals','St. Louis Cardinals'],
                            'New York':['New York Mets','New York Yankees'],
                            'Ohio':['Cleveland Indians','Cincinnati Reds'],
                            'Pennsylvania':['Philadelphia Phillies','Pittsburgh Pirates'],
                            'Texas':['Houston Astros','Texas Rangers'],
                            'Washington':['Seattle Mariners'],
                            'Wisconsin':['Milwaukee Brewers'],
                            'DC':['Washington Nationals']
                                }
    nhl_team_by_state = {
                            'Arizona':['Phoenix Coyotes'],
                            'California':['Anaheim Ducks','Los Angeles Kings','San Jose Sharks'],
                            'Colorado':['Colorado Avalanche'],
                            'Florida':['Florida Panthers','Tampa Bay Lightning'],
                            'Georgia':['Atlanta Thrashers'],
                            'Illinois':['Chicago Blackhawks'],
                            'Massachusetts':['Boston Bruins'],
                            'Michigan':['Detroit Red Wings'],
                            'Minnesota':['Minnesota Wild'],
                            'Missouri':['St. Louis Blues'],
                            'New Jersey':['New Jersey Devils'],
                            'New York':['Buffalo Sabres','New York Islanders','New York Rangers'],
                            'North Carolina':['Carolina Hurricanes'],
                            'Ohio':['Columbus Blue Jackets'],
                            'Pennsylvania':['Philadelphia Flyers','Pittsburgh Penguins'],
                            'Tennessee':['Nashville Predators'],
                            'Texas':['Dallas Stars'],
                            'DC':['Washington Capitals']
                        }

    # ------------------- WEATHER DATA ------------------------
    all_weather_data = {}
    # Load fast lookup for weather data
    if os.path.isfile("data/raw/weather_fast_lookup"):
        print "---weather data in memory"
        all_weather_data = pickle.load(open("data/raw/weather_fast_lookup", "rb"))
        print "---loaded weather data from memory"
    # otherwise load non-condensed version
    else:
        print "---weather data not in memory"
        all_weather = pd.read_stata("data/clean/weather_all.dta")
        all_weather_matrix = all_weather.as_matrix()

        # initialize max values
        max_prcp = max_snow = max_snwd = max_tsun = max_tavg = sys.float_info.min
        min_tavg = sys.float_info.max

        for weather_row in all_weather_matrix:
            date = weather_row[4]
            city = weather_row[64]
            prcp = nanCheck(weather_row[14]) # precipitation in 10ths of mm
            snow = nanCheck(weather_row[16]) # snowfall in mm
            snwd = nanCheck(weather_row[15]) # depth of snow in mm
            tmax = nanCheck(weather_row[21]) # highest day temperature in 10th of celcius
            tmin = nanCheck(weather_row[22]) # lowest day temperature in 10th of celcius
            tsun = nanCheck(weather_row[20]) # daily total sunshine in minutes
            all_weather_data[str(date) + city] = (prcp, snow, snwd, tmax, tmin, tsun)

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
            if not math.isnan(tmax) and not math.isnan(tmin) and (abs(float(tmax) + float(tmin))/2.0) < min_tavg:
                min_tavg = (abs(float(tmax) + float(tmin))/2.0)


        # set max values
        all_weather_data["max_prcp"] = max_prcp
        all_weather_data["max_snow"] = max_snow
        all_weather_data["max_snwd"] = max_snwd
        all_weather_data["max_tsun"] = max_tsun
        all_weather_data["max_tavg"] = max_tavg
        all_weather_data["min_tavg"] = min_tavg

        pickle.dump(all_weather_data, open("data/raw/weather_fast_lookup", "wb"))
        print "---saved weather data"

    # ------------------- NATIONALITY DATA -------------------

    naionality_fast_lookup = {}
    if os.path.isfile("data/raw/nationality_fast_lookup"):
        nationality_fast_lookup = pickle.load(open("data/raw/nationality_fast_lookup", "rb"))
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

    nationality_idncase_lookup = {}
    if os.path.isfile("data/raw/nationality_idncase_lookup"):
        nationality_idncase_lookup = pickle.load(open("data/raw/nationality_idncase_lookup", "rb"))
    else:
        print "creating nationality idncase lookup"
        master = pandas.read_csv("data/raw/master.csv").as_matrix()
        nationality_idncase_lookup = loadNationalityLookup()
        for row in master:
            idncase = 999999999
            if not math.isnan(row[0]):
                idncase = int(row[0])
            nat = row[1]
            natLookup = "??"
            if nat in nationality_idncase_lookup:
                natLookup = nationality_idncase_lookup[nat]
            nationality_idncase_lookup[idncase] = (nat, natLookup)
        pickle.dump(nationality_idncase_lookup, open("data/raw/nationality_idncase_lookup", "wb"))

    # ------------------- BIOS -------------------------------

    input_data = pd.read_csv("data/raw/tblLookupBaseCity.csv")
    city_lookup_matrix = input_data.as_matrix()
    city_lookup = []
    for row in city_lookup_matrix:
        city_lookup.append([row[1].strip(), row[5].strip()])
    data = np.asarray(city_lookup)
    df = pd.DataFrame(data=data, columns=["hearing_loc_code", "city"])
    df.to_csv('data/raw/cityLookup.csv', index=False, index_label=False)

    # ------------------- ASYLUM DATA ------------------------
    print "---loading asylum data"
    asylum_data = pd.read_csv("data/raw/asylum_clean.csv")
    print "---loaded asylum data"


# Need to get these values play_team_nba, play_team_nfl, play_team_mlb, play_team_nhl from map object
def calculate_sports_score(data, play_date, judge_states):
    locations = ["undergrad", "lawschool", "bar"]
    play_team_mlb = defaultdict(list)
    play_team_nba = defaultdict(list)
    play_team_nfl = defaultdict(list)
    play_team_nhl = defaultdict(list)

    # undergrad, lawschool, bar

    for location in locations:
        judge_state = judge_states[location]
        if judge_state in data.mlb_team_by_state.keys():
            for x in data.mlb_team_by_state[judge_state]:
                play_team_mlb[location].append(x)
        if judge_state in data.nba_team_by_state.keys():
            for x in data.nba_team_by_state[judge_state]:
                play_team_nba[location].append(x)
        if judge_state in data.nfl_team_by_state.keys():
            for x in data.nfl_team_by_state[judge_state]:
                play_team_nfl[location].append(x)
        if judge_state in data.nhl_team_by_state.keys():
            for x in data.nhl_team_by_state[judge_state]:
                play_team_nhl[location].append(x)

    days_range = []
    for i in range(0,5):
        days_range.append(dt.datetime.strptime(play_date, '%m/%d/%y').date() - timedelta(days= i))

    column = {"nba_undergrad" : 0, "nba_lawschool" : 0, "nba_bar" : 0,
              "mlb_undergrad" : 0, "mlb_lawschool" : 0, "mlb_bar" : 0,
              "nfl_undergrad" : 0, "nfl_lawschool" : 0, "nfl_bar" : 0,
              "nhl_undergrad" : 0, "nhl_lawschool" : 0, "nhl_bar" : 0}

    for location in locations:
        nba_teams = play_team_nba[location]
        for day in days_range:
            for nba_team in nba_teams:
                key = str(day) + nba_team + "nba"
                if key in data.all_sports_data:
                    outcome = data.all_sports_data[key]
                    if outcome == 'W':
                        column["nba_" + location] += 1

    for location in locations:
        nfl_teams = play_team_nfl[location]
        for day in days_range:
            for nfl_team in nfl_teams:
                key = str(day) + nfl_team + "nfl"
                if key in data.all_sports_data:
                    outcome = data.all_sports_data[key]
                    if outcome == 'W':
                        column["nfl_" + location] += 1

    for location in locations:
        mlb_teams = play_team_mlb[location]
        for day in days_range:
            for mlb_team in mlb_teams:
                key = str(day) + mlb_team + "mlb"
                if key in data.all_sports_data:
                    outcome = data.all_sports_data[key]
                    if outcome == 'W':
                        column["mlb_" + location] += 1

    for location in locations:
        nhl_teams = play_team_nhl[location]
        for day in days_range:
            for nhl_team in nhl_teams:
                key = str(day) + nhl_team + "nhl"
                if key in data.all_sports_data:
                    outcome = data.all_sports_data[key]
                    if outcome == 'W':
                        column["nhl_" + location] += 1

    return column

def sports_weather_handler(data):
    all_sports_scores = []
    all_weather_values = []
    all_twitter_scores = []
    for i in range(len(data.asylum_data)):
        sports_column = sports_handler(data, i)
        all_sports_scores.append(sports_column)
        all_weather_values.append(weather_handler(data, i))
        all_twitter_scores.append(twitter_handler(data, i))
        if i % 10000 == 0:
            print "index at:", i
    return all_sports_scores, all_weather_values, all_twitter_scores

def sports_handler(data, i):
    try:
        date_of_interest = getCompletionDate(data.asylum_data['comp_date'][i].astype(int)).strftime('%m/%d/%y')
        locations_of_interest = {}
        # locations_of_interest = set()
        if isinstance(data.asylum_data['JudgeUndergradLocation'][i], basestring):
            locations_of_interest['undergrad'] = data.asylum_data['JudgeUndergradLocation'][i].split(',')[1].strip()
        if isinstance(data.asylum_data['JudgeLawSchoolLocation'][i], basestring):
            locations_of_interest['lawschool'] = data.asylum_data['JudgeLawSchoolLocation'][i].split(',')[1].strip()
        if isinstance(data.asylum_data['Bar'][i], basestring):
            locations_of_interest['bar'] = map(str.strip, re.split(';|,', data.asylum_data['Bar'][i]))[0]
        if locations_of_interest is not None and len(locations_of_interest) != 0 and date_of_interest is not None:
            return calculate_sports_score(data, date_of_interest, locations_of_interest)
    except:
        return {"nba_undergrad" : 0, "nba_lawschool" : 0, "nba_bar" : 0,
                "mlb_undergrad" : 0, "mlb_lawschool" : 0, "mlb_bar" : 0,
                "nfl_undergrad" : 0, "nfl_lawschool" : 0, "nfl_bar" : 0,
                "nhl_undergrad" : 0, "nhl_lawschool" : 0, "nhl_bar" : 0}
    return {"nba_undergrad" : 0, "nba_lawschool" : 0, "nba_bar" : 0,
            "mlb_undergrad" : 0, "mlb_lawschool" : 0, "mlb_bar" : 0,
            "nfl_undergrad" : 0, "nfl_lawschool" : 0, "nfl_bar" : 0,
            "nhl_undergrad" : 0, "nhl_lawschool" : 0, "nhl_bar" : 0}

def twitter_handler(data, i):
    try:
        date_of_interest = getCompletionDate(data.asylum_data['comp_date'][i].astype(int)).strftime('%Y-%m-%d')
        city = data.asylum_data['city'][i]
        key = str(date_of_interest)+city.lower()
        if key in data.all_twitter_data.keys():
            return data.all_twitter_data[key]
        else:
            return None
    except:
        return None

def weather_handler(data, i):
    # Get date of completion from row
    currentDate = getCompletionDate(data.asylum_data['comp_date'][i].astype(int))
    allDates = [str(currentDate.strftime('%Y%m%d'))]
    for numDays in np.arange(1,5):
        newDate = currentDate - dt.timedelta(days=numDays)
        allDates.append(str(newDate.strftime('%Y%m%d')))
    # Get dates of current and past four days
    city = str(data.asylum_data['city'][i])

    dateCount = 0
    # iterate through all 5 days of weather
    weather_scores = []

    default_weather = [(np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)]

    for date in allDates:
        # Check if date and city values exist
        if type(date) is str and type(city) is str:
            key = date + city
            if key in data.all_weather_data:
                dateCount += 1
                weather_scores.append(data.all_weather_data[key])
            else:
                weather_scores.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
        else:
            weather_scores.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
    if dateCount == 0:
        return default_weather
    return weather_scores

def main():
    data = LoadData()

    # merge bios
    bios_clean = pd.read_stata("data/raw/bios_clean.dta")
    city_lookup = pd.read_csv("data/raw/cityLookup.csv")
    data.asylum_data = data.asylum_data.merge(bios_clean, on='ij_code', how='left')
    data.asylum_data = data.asylum_data.merge(city_lookup, on='hearing_loc_code', how='left')

    # add nationaliy
    all_nat = []
    all_nationality = []
    for row in data.asylum_data.iterrows():
        idncase = row[1][0]
        lookup = data.nationality_idncase_lookup[idncase]
        nat = lookup[0]
        nationality = lookup[1]
        all_nat.append(nat)
        all_nationality.append(nationality)
    data.asylum_data["nat_code"] = all_nat
    data.asylum_data["nationality"] = all_nationality


    # add sports weather and twitter score
    sports_data, weather_data, twitter_scores = sports_weather_handler(data)

    # ---------------------------- TWITTER ----------------------------------------- #

    data.asylum_data['twitter_score'] = twitter_scores

    # ---------------------------- TWITTER ----------------------------------------- #

    # ---------------------------- WEATHER ----------------------------------------- #

    # store weather values
    prcp = []
    snow = []
    snwd = []
    tmax = []
    tmin = []
    tsun = []

    prcp_minus_1 = []
    snow_minus_1 = []
    snwd_minus_1 = []
    tmax_minus_1 = []
    tmin_minus_1 = []
    tsun_minus_1 = []

    prcp_minus_2 = []
    snow_minus_2 = []
    snwd_minus_2 = []
    tmax_minus_2 = []
    tmin_minus_2 = []
    tsun_minus_2 = []

    prcp_minus_3 = []
    snow_minus_3 = []
    snwd_minus_3 = []
    tmax_minus_3 = []
    tmin_minus_3 = []
    tsun_minus_3 = []

    prcp_minus_4 = []
    snow_minus_4 = []
    snwd_minus_4 = []
    tmax_minus_4 = []
    tmin_minus_4 = []
    tsun_minus_4 = []

    for row in weather_data:
        prcp.append(row[0][0])
        snow.append(row[0][1])
        snwd.append(row[0][2])
        tmax.append(row[0][3])
        tmin.append(row[0][4])
        tsun.append(row[0][5])

        prcp_minus_1.append(row[1][0])
        snow_minus_1.append(row[1][1])
        snwd_minus_1.append(row[1][2])
        tmax_minus_1.append(row[1][3])
        tmin_minus_1.append(row[1][4])
        tsun_minus_1.append(row[1][5])

        prcp_minus_2.append(row[2][0])
        snow_minus_2.append(row[2][1])
        snwd_minus_2.append(row[2][2])
        tmax_minus_2.append(row[2][3])
        tmin_minus_2.append(row[2][4])
        tsun_minus_2.append(row[2][5])

        prcp_minus_3.append(row[3][0])
        snow_minus_3.append(row[3][1])
        snwd_minus_3.append(row[3][2])
        tmax_minus_3.append(row[3][3])
        tmin_minus_3.append(row[3][4])
        tsun_minus_3.append(row[3][5])

        prcp_minus_4.append(row[4][0])
        snow_minus_4.append(row[4][1])
        snwd_minus_4.append(row[4][2])
        tmax_minus_4.append(row[4][3])
        tmin_minus_4.append(row[4][4])
        tsun_minus_4.append(row[4][5])

    data.asylum_data["prcp"] = prcp
    data.asylum_data["snow"] = snow
    data.asylum_data["snwd"] = snwd
    data.asylum_data["tmax"] = tmax
    data.asylum_data["tmin"] = tmin
    data.asylum_data["tsun"] = tsun

    data.asylum_data["prcp_minus_1"] = prcp_minus_1
    data.asylum_data["snow_minus_1"] = snow_minus_1
    data.asylum_data["snwd_minus_1"] = snwd_minus_1
    data.asylum_data["tmax_minus_1"] = tmax_minus_1
    data.asylum_data["tmin_minus_1"] = tmin_minus_1
    data.asylum_data["tsun_minus_1"] = tsun_minus_1

    data.asylum_data["prcp_minus_2"] = prcp_minus_2
    data.asylum_data["snow_minus_2"] = snow_minus_2
    data.asylum_data["snwd_minus_2"] = snwd_minus_2
    data.asylum_data["tmax_minus_2"] = tmax_minus_2
    data.asylum_data["tmin_minus_2"] = tmin_minus_2
    data.asylum_data["tsun_minus_2"] = tsun_minus_2

    data.asylum_data["prcp_minus_3"] = prcp_minus_3
    data.asylum_data["snow_minus_3"] = snow_minus_3
    data.asylum_data["snwd_minus_3"] = snwd_minus_3
    data.asylum_data["tmax_minus_3"] = tmax_minus_3
    data.asylum_data["tmin_minus_3"] = tmin_minus_3
    data.asylum_data["tsun_minus_3"] = tsun_minus_3

    data.asylum_data["prcp_minus_4"] = prcp_minus_4
    data.asylum_data["snow_minus_4"] = snow_minus_4
    data.asylum_data["snwd_minus_4"] = snwd_minus_4
    data.asylum_data["tmax_minus_4"] = tmax_minus_4
    data.asylum_data["tmin_minus_4"] = tmin_minus_4
    data.asylum_data["tsun_minus_4"] = tsun_minus_4

    # ---------------------------- WEATHER ----------------------------------------- #

    # ---------------------------- SPORTS ----------------------------------------- #
    nba_undergrad = []
    nba_lawschool = []
    nba_bar = []

    nfl_undergrad = []
    nfl_lawschool = []
    nfl_bar = []

    mlb_undergrad = []
    mlb_lawschool = []
    mlb_bar = []

    nhl_undergrad = []
    nhl_lawschool = []
    nhl_bar = []

    for column in sports_data:
        nba_undergrad.append(column['nba_undergrad'])
        nba_lawschool.append(column['nba_lawschool'])
        nba_bar.append(column['nba_bar'])

        nfl_undergrad.append(column['nfl_undergrad'])
        nfl_lawschool.append(column['nfl_lawschool'])
        nfl_bar.append(column['nfl_bar'])

        mlb_undergrad.append(column['mlb_undergrad'])
        mlb_lawschool.append(column['mlb_lawschool'])
        mlb_bar.append(column['mlb_bar'])

        nhl_undergrad.append(column['nhl_undergrad'])
        nhl_lawschool.append(column['nhl_lawschool'])
        nhl_bar.append(column['nhl_bar'])

    data.asylum_data['nba_undergrad'] = nba_undergrad
    data.asylum_data['nba_lawschool'] = nba_lawschool
    data.asylum_data['nba_bar'] = nba_bar

    data.asylum_data['nfl_undergrad'] = nfl_undergrad
    data.asylum_data['nfl_lawschool'] = nfl_lawschool
    data.asylum_data['nfl_bar'] = nfl_bar

    data.asylum_data['mlb_undergrad'] = mlb_undergrad
    data.asylum_data['mlb_lawschool'] = mlb_lawschool
    data.asylum_data['mlb_bar'] = mlb_bar

    data.asylum_data['nhl_undergrad'] = nhl_undergrad
    data.asylum_data['nhl_lawschool'] = nhl_lawschool
    data.asylum_data['nhl_bar'] = nhl_bar

    # ---------------------------- SPORTS ----------------------------------------- #

    data.asylum_data.to_csv('data/raw/complete_data.csv', index=False, index_label=False)

if __name__ == '__main__':
  main()












