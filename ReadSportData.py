import pandas as pd
import datetime as dt
from datetime import date, timedelta
import numpy as np
from pandas import Series
import sys
import re
import cPickle as pickle
import os.path

class LoadData:

    def format_date(row):
        date_list = row.split()
        play_date = date_list[len(date_list)-1]
        return dt.datetime.strptime(play_date, '%m/%d/%y').date()

    all_sports_data = {}
    print "loading sports data"
    if os.path.isfile("data/raw/sports_fast_lookup"):
        all_sports_data = pickle.load(open("data/raw/sports_fast_lookup", "rb"))
        print "loaded sports data"
    else:
        print "sports data not on file, loading"
        nba_data_matrix = pd.read_csv('data/sportsData/NBA.csv', index_col=False).as_matrix()
        nfl_data_matrix = pd.read_csv('data/sportsData/NFL.csv', index_col=False).as_matrix()
        mlb_data_matrix = pd.read_csv('data/sportsData/MLB.csv', index_col=False).as_matrix()
        nhl_data_matrix = pd.read_csv('data/sportsData/NHL.csv', sep=',', index_col=False).as_matrix()

        for row in nba_data_matrix:
            date = format_date(row[2])
            team_name = row[0]
            result = row[9]
            all_sports_data[str(date) + team_name + "nba"] = result
        for row in nfl_data_matrix:
            date = format_date(row[2])
            team_name = row[0]
            result = row[9]
            all_sports_data[str(date) + team_name + "nfl"] = result
        for row in mlb_data_matrix:
            date = format_date(row[2])
            team_name = row[0]
            result = row[10]
            all_sports_data[str(date) + team_name + "mlb"] = result
        for row in nhl_data_matrix:
            date = format_date(row[2])
            team_name = row[0]
            result = row[10]
            all_sports_data[str(date) + team_name + "nhl"] = result
        pickle.dump(all_sports_data, open("data/raw/sports_fast_lookup", "wb"))
        print "sports data saved"

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

# Need to get these values play_team_nba, play_team_nfl, play_team_mlb, play_team_nhl from map object
def win_score(data, play_date, judge_states):
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
            key = str(day) + nba_team + "nba"
            if key in data.all_sports_data:
                filtered_data.append(data.all_sports_data[key])

    for nfl_team in play_team_nfl:
        for day in days_range:
            key = str(day) + nfl_team + "nfl"
            if key in data.all_sports_data:
                filtered_data.append(data.all_sports_data[key])

    for mlb_team in play_team_mlb:
        for day in days_range:
            key = str(day) + mlb_team + "mlb"
            if key in data.all_sports_data:
                filtered_data.append(data.all_sports_data[key])

    for nhl_team in play_team_nhl:
        for day in days_range:
            key = str(day) + nhl_team + "nhl"
            if key in data.all_sports_data:
                filtered_data.append(data.all_sports_data[key])

    total_games = len(filtered_data)
    won_games = [x for x in filtered_data if x == 'W']

    if total_games != 0:
        return len(won_games)/float(total_games)
    else:
        return 0.5

def getCompletionDate(stata_date):
	date1960Jan1 = dt.datetime(1960,01,01)
	return date1960Jan1.date() + dt.timedelta(days=stata_date)


if __name__ == '__main__':
    data = LoadData();
    asy_data = pd.read_csv('data/raw/asylum_clean_full_small.csv')
    all_scores = []
    lowest_score = sys.float_info.max
    highest_score = sys.float_info.min
    score = 0.5
    for s in range(len(asy_data)):
        date_of_interest = getCompletionDate(asy_data['comp_date'][s].astype(int)).strftime('%m/%d/%y')
        locations_of_interest = set()
        if isinstance(asy_data['JudgeUndergradLocation'][s], basestring):
            locations_of_interest.add(asy_data['JudgeUndergradLocation'][s].split(',')[1].strip())
        if isinstance(asy_data['JudgeLawSchoolLocation'][s], basestring):
            locations_of_interest.add(asy_data['JudgeLawSchoolLocation'][s].split(',')[1].strip())
        if isinstance(asy_data['Bar'][s], basestring):
            locations_of_interest |= set(map(str.strip, re.split(';|,', asy_data['Bar'][s])))
        if locations_of_interest is not None and len(locations_of_interest) != 0 and date_of_interest is not None:
            score = win_score(data, date_of_interest, locations_of_interest)
        all_scores.append(score)
        if score > highest_score:
            highest_score = score
        if score < lowest_score:
            lowest_score = score

    print "highest_score sports:", highest_score
    print "lowest_score sports:", lowest_score

    asy_data["sports_score"] = all_scores
    asy_data.to_csv('data/raw/asylum_clean_full_sports.csv', index=False, index_label=False)




