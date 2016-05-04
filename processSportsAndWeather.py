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
def nanCheck(val):
    if (type(val) is float and val < -9970.0) or (type(val) is int and val < -9970):
            return float('nan')
    return val

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

        # set max values
        all_weather_data["max_prcp"] = max_prcp
        all_weather_data["max_snow"] = max_snow
        all_weather_data["max_snwd"] = max_snwd
        all_weather_data["max_tsun"] = max_tsun
        all_weather_data["max_tavg"] = max_tavg

        pickle.dump(all_weather_data, open("data/raw/weather_fast_lookup", "wb"))
        print "---saved weather data"

    # ------------------- ASYLUM DATA ------------------------
    print "---loading asylum data"
    asylum_data = pd.read_csv("data/raw/asylum_clean_full.csv")
    print "---loaded asylum data"


# Need to get these values play_team_nba, play_team_nfl, play_team_mlb, play_team_nhl from map object
def calculate_sports_score(data, play_date, judge_states):
    play_team_mlb = []
    play_team_nba = []
    play_team_nfl = []
    play_team_nhl = []

    for judge_state in judge_states:
        if judge_state in data.mlb_team_by_state.keys():
            play_team_mlb = play_team_mlb+data.mlb_team_by_state[judge_state]
        if judge_state in data.nba_team_by_state.keys():
            play_team_nba = play_team_nba+data.nba_team_by_state[judge_state]
        if judge_state in data.nfl_team_by_state.keys():
            play_team_nfl = play_team_nfl+data.nfl_team_by_state[judge_state]
        if judge_state in data.nhl_team_by_state.keys():
            play_team_nhl = play_team_nhl+data.nhl_team_by_state[judge_state]

    days_range = []
    win_percent = 0.0
    sport_count = 0
    for i in range(0,5):
        days_range.append(dt.datetime.strptime(play_date, '%m/%d/%y').date() - timedelta(days= i))
    filtered_data = []
    total_games = 0
    won_games = 0

    for nba_team in play_team_nba:
        for day in days_range:
            key = str(day) + nba_team.lower() + "nba"
            if key in data.all_sports_data:
                filtered_data.append(data.all_sports_data[key])

    for nfl_team in play_team_nfl:
        for day in days_range:
            key = str(day) + nfl_team.lower() + "nfl"
            if key in data.all_sports_data:
                filtered_data.append(data.all_sports_data[key])

    for mlb_team in play_team_mlb:
        for day in days_range:
            key = str(day) + mlb_team.lower() + "mlb"
            if key in data.all_sports_data:
                filtered_data.append(data.all_sports_data[key])

    for nhl_team in play_team_nhl:
        for day in days_range:
            key = str(day) + nhl_team.lower() + "nhl"
            if key in data.all_sports_data:
                filtered_data.append(data.all_sports_data[key])

    total_games = len(filtered_data)
    won_games = [x for x in filtered_data if x == 'W']

    if total_games != 0:
        return len(won_games)/float(total_games)
    else:
        return 0.5

def sports_weather_handler(data):
    all_sports_scores = []
    all_weather_scores = []
    all_twitter_scores = []
    for i in range(len(data.asylum_data)):
        all_sports_scores.append(sports_handler(data, i))
        all_weather_scores.append(weather_handler(data, i))
        all_twitter_scores.append(twitter_handler(data, i))
        if i % 10000 == 0:
            print "index at:", i
    return all_sports_scores, all_weather_scores, all_twitter_scores

def sports_handler(data, i):
    try:
        date_of_interest = getCompletionDate(data.asylum_data['comp_date'][i].astype(int)).strftime('%m/%d/%y')
        locations_of_interest = set()
        if isinstance(data.asylum_data['JudgeUndergradLocation'][i], basestring):
            locations_of_interest.add(data.asylum_data['JudgeUndergradLocation'][i].split(',')[1].strip())
        if isinstance(data.asylum_data['JudgeLawSchoolLocation'][i], basestring):
            locations_of_interest.add(data.asylum_data['JudgeLawSchoolLocation'][i].split(',')[1].strip())
        if isinstance(data.asylum_data['Bar'][i], basestring):
            locations_of_interest |= set(map(str.strip, re.split(';|,', data.asylum_data['Bar'][i])))
        if locations_of_interest is not None and len(locations_of_interest) != 0 and date_of_interest is not None:
            return calculate_sports_score(data, date_of_interest, locations_of_interest)
    except:
        return 0.5
    return 0.5

def twitter_handler(data, i):
    try:
        date_of_interest = getCompletionDate(data.asylum_data['comp_date'][i].astype(int)).strftime('%Y-%m-%d')
        city = data.asylum_data['Court_SLR'][i]
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

    for date in allDates:
        # Check if date and city values exist
        if type(date) is str and type(city) is str:
            dateCount += 1
            key = date + city
            if key in data.all_weather_data:
                weather_data = data.all_weather_data[key]
                prcp = 1.0 - safe_divide(weather_data[0], data.all_weather_data["max_prcp"]) # precipitation in 10ths of mm
                snow = 1.0 - safe_divide(weather_data[1], data.all_weather_data["max_snow"]) # snowfall in mm
                snwd = 1.0 - safe_divide(weather_data[2], data.all_weather_data["max_snwd"]) # depth of snow in mm
                tmax = weather_data[3]                                                                                      # highest day temperature in 10th of celcius
                tmin = weather_data[4]                                                                                      # lowest day temperature in 10th of celcius
                tavg = (float(tmax) + float(tmin)) / 2.0                                                # average day temperature in 10th of celcius
                tscore = temp_score(tavg)
                tsun = safe_divide(weather_data[5], data.all_weather_data["max_tsun"]) # daily total sunshine in minutes

                calculations = prcp, snow, snwd, tscore, tsun
                weather_scores.append(applyWeights(weights(calculations), calculations))
    if dateCount != 0:
        return compute_score(weather_scores)
    return 0.5

def main():
    # load sports, weather, and asylum data
    data = LoadData()
    sports_data, weather_data, twitter_score = sports_weather_handler(data)

    data.asylum_data["sports_score"] = sports_data
    data.asylum_data["weather_score"] = weather_data
    data.asylum_data["twitter_score"] = twitter_score

    data.asylum_data.to_csv('data/raw/asylum_clean_full_cluster.csv', index=False, index_label=False)
    print "written to file. Done!"


if __name__ == '__main__':
  main()












