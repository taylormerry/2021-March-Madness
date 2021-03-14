import pandas as pd
import re
import numpy as np
from bs4 import BeautifulSoup
import requests
from scipy.stats import zscore, norm
import pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression


# File from Kaggle, all possible team spellings (to get to the TeamID)
spellings = pd.read_csv('ncaam-march-mania-2021/MTeamSpellings.csv', encoding = 'ISO-8859-1')


# Put the team names in the same format (lowercase no punctuation) for joins
spellings['TeamNameSpelling'] = spellings['TeamNameSpelling'].str.replace('[^a-zA-Z&.()\' ]+',' ').str.lower()
spellings['TeamNameSpelling'] = spellings['TeamNameSpelling'].str.replace('[^a-z&.()\' ]+','')

# returns name of team that is in the spellings csv
def fix_name(row):
    if row['Team'] == 'st marys':
        return 'st marys ca'
    elif row['Team'] == 'wins salem' or row['Team'] == 'winston salem st.':
        return 'winston salem'
    elif row['Team'] == 'w virginia':
        return 'west virginia'
    elif row['Team'] == 'n carolina':
        return 'north carolina'
    elif row['Team'] == 'tx christian':
        return 'tcu'
    elif row['Team'] == 'va tech':
        return 'virginia tech'
    elif row['Team'] == 'miss state':
        return 'mississippi st'
    elif row['Team'] == 'st bonavent':
        return 'st bonaventure'
    elif row['Team'] == 'loyola chi':
        return 'loyola chicago'
    elif row['Team'] == 's methodist':
        return 'smu'
    elif row['Team'] == 'n mex state':
        return 'new mexico st'
    elif row['Team'] == 's carolina':
        return 'south carolina'
    elif row['Team'] == 'boston col':
        return 'boston college'
    elif row['Team'] == 'e tenn st':
        return 'etsu'
    elif row['Team'] == 'nc grnsboro':
        return 'unc greensboro'
    elif row['Team'] == 'central fl':
        return 'ucf'
    elif row['Team'] == 'utah val st':
        return 'utah valley state'
    elif row['Team'] == 'northeastrn':
        return 'northeastern'
    elif row['Team'] == 'ga tech':
        return 'georgia tech'
    elif row['Team'] == 'col charlestn':
        return 'college of charleston'
    elif row['Team'] == 'st josephs':
        return 'st josephs pa'
    elif row['Team'] == 'u penn':
        return 'penn'
    elif row['Team'] == 'ste f austin':
        return 'stephen f austin'
    elif row['Team'] == 'fla gulf cst':
        return 'florida gulf coast'
    elif row['Team'] == 'grd canyon':
        return 'grand canyon'
    elif row['Team'] == 'tx arlington':
        return 'ut arlington'
    elif row['Team'] == 'n iowa':
        return 'northern iowa'
    elif row['Team'] == 'la tech':
        return 'louisiana tech'
    elif row['Team'] == 'wm & mary':
        return 'william & mary'
    elif row['Team'] == 'jksnville st':
        return 'jacksonville st'
    elif row['Team'] == 'app state':
        return 'appalachian st'
    elif row['Team'] == 'san fransco':
        return 'san francisco'
    elif row['Team'] == 'e washingtn':
        return 'eastern washington'
    elif row['Team'] == 'geo wshgtn':
        return 'george washington'
    elif row['Team'] == 'u mass':
        return 'umass'
    elif row['Team'] == 'maryland bc':
        return 'umbc'
    elif row['Team'] == 'wash state':
        return 'washington st'
    elif row['Team'] == 'tx san ant':
        return 'utsa'
    elif row['Team'] == 'st fran (pa)' or row['Team'] == 'st. francis pa':
        return 'st francis pa'
    elif row['Team'] == 'miami oh':
        return 'miami ohio'
    elif row['Team'] == 'geo mason':
        return 'george mason'
    elif row['Team'] == 'wi milwkee':
        return 'milwaukee'
    elif row['Team'] == 'tn state':
        return 'tennessee st'
    elif row['Team'] == 'tn tech':
        return 'tennessee tech'
    elif row['Team'] == 'nc wilmgton':
        return 'unc wilmington'
    elif row['Team'] == 's alabama':
        return 'south alabama'
    elif row['Team'] == 'lg beach st':
        return 'long beach st'
    elif row['Team'] == 'james mad':
        return 'james madison'
    elif row['Team'] == 'sam hous st':
        return 'sam houston st'
    elif row['Team'] == 'cs bakersfld' or row['Team'] == 'cal st. bakersfield':
        return 'cal state bakersfield'
    elif row['Team'] == 'loyola mymt':
        return 'loyola marymount'
    elif row['Team'] == 's mississippi':
        return 'southern miss'
    elif row['Team'] == 'bowling grn':
        return 'bowling green'
    elif row['Team'] == 'tx el paso':
        return 'utep'
    elif row['Team'] == 'n hampshire':
        return 'new hampshire'
    elif row['Team'] == 'rob morris':
        return 'robert morris'
    elif row['Team'] == 'wi grn bay':
        return 'green bay'
    elif row['Team'] == 'charl south':
        return 'charleston southern'
    elif row['Team'] == 'abl christian':
        return 'abilene christian'
    elif row['Team'] == 'gard webb':
        return 'gardner webb'
    elif row['Team'] == 'tx pan am':
        return 'texas pan american'
    elif row['Team'] == 'se missouri' or row['Team'] == 'southeast missouri st.':
        return 'se missouri st'
    elif row['Team'] == 'neb omaha':
        return 'omaha'
    elif row['Team'] == 's florida':
        return 'south florida'
    elif row['Team'] == 'mass lowell':
        return 'umass lowell'
    elif row['Team'] == 'e carolina':
        return 'east carolina'
    elif row['Team'] == 'tx a&m cc' or row['Team'] == 'texas a&m corpus chris':
        return 'a&m corpus chris'
    elif row['Team'] == 's utah':
        return 'southern utah'
    elif row['Team'] == 'n florida':
        return 'north florida'
    elif row['Team'] == 'sacred hrt':
        return 'sacred heart'
    elif row['Team'] == 'st fran (ny)':
        return 'st francis ny'
    elif row['Team'] == 'ar lit rock':
        return 'arkansas little rock'
    elif row['Team'] == 'beth cook':
        return 'bethune cookman'
    elif row['Team'] == 'sac state':
        return 'sacramento st'
    elif row['Team'] == 'siu edward':
        return 'southern illinois'
    elif row['Team'] == 'youngs st':
        return 'youngstown st'
    elif row['Team'] == 'nw state':
        return 'northwestern st'
    elif row['Team'] == 'cal st nrdge':
        return 'cal state northridge'
    elif row['Team'] == 'ark pine bl':
        return 'arkansas pine bluff'
    elif row['Team'] == 'va military':
        return 'vmi'
    elif row['Team'] == 'incar word':
        return 'incarnate word'
    elif row['Team'] == 'n arizona':
        return 'northern arizona' 
    elif row['Team'] == 's car state':
        return 'south carolina state'
    elif row['Team'] == 'nw st':
        return 'northwestern st'
    elif row['Team'] == 'miss val st' or row['Team'] == 'mississippi valley st.':
        return 'mississippi valley state'
    elif row['Team'] == 'maryland es':
        return 'umes'
    elif row['Team'] == 'alab a&m':
        return 'alabama a&m' 
    elif row['Team'] == 'n alabama':
        return 'north alabama'
    elif row['Team'] == 'la lafayette':
        return 'louisiana lafayette'
    elif row['Team'] == 'grambling st':
        return 'grambling state'
    elif row['Team'] == 'ut rio grande valley':
        return 'texas rio grande valley'
    elif row['Team'] == 'liu brooklyn ( )' or row['Team'] == 'liu':
        return 'liu brooklyn'
    elif row['Team'] == 'tarleton state' or row['Team'] == 'tarleton st.':
        return 'tarleton st'
    elif row['Team'] == 'dixie state' or row['Team'] == 'dixie st.':
        return 'dixie st'
    else:
        return row['Team']
    
nans = lambda df: df[df.isnull().any(axis=1)]  # Function to print out rows with null values

# scrape each 2021's ratings
team_list = []
rating_list = []
teamrank_url = 'https://www.teamrankings.com/ncaa-basketball/ranking/predictive-by-other'
teamrank_page = requests.get(teamrank_url)
teamrank_soup = BeautifulSoup(teamrank_page.content, 'lxml')
teamrank_rows = teamrank_soup.select('tbody tr')
for row in teamrank_rows:
    anchor = row.select('.nowrap')[0].select('a')
    if not anchor and not row.select('.nowrap')[0].get_text().lower().startswith('liu'):
        continue
    if anchor:
        team_list.append(anchor[0].get_text())
    else:
        team_list.append(row.select('.nowrap')[0].get_text())
    rating_list.append(row.find_all('td')[2].get_text())  # magic number
teamrank = pd.DataFrame({'Team': team_list, 'TeamrankRating': rating_list})

# Put the team names in the same format (lowercase no punctuation) for joins
teamrank['Team'] = teamrank['Team'].str.replace('[^a-zA-Z&.()\' ]+',' ').str.lower()
teamrank['Team'] = teamrank['Team'].str.replace('[^a-z&.()\' ]+','')

# fix the names in order to join to get the team id
teamrank['Team'] = teamrank.apply(fix_name, axis = 1)

# merge to spellings to get team id
teamrank_teams = pd.merge(teamrank, spellings, how = 'left', left_on = 'Team', right_on = 'TeamNameSpelling')

# drop duplicate spellings
teamrank_teams = teamrank_teams[['TeamID', 'TeamrankRating']].drop_duplicates()

# Read trank ratings
trank_url = 'http://barttorvik.com/2021_team_results.csv'
trank = pd.read_csv(trank_url)

trank = trank[['rank', 'record', 'oe Rank', 'de Rank', 'Fun Rk, adjt']]
trank = trank.rename(columns = {'rank': 'Team', 'record': 'TrankRating', 'oe Rank': 'OE', 'de Rank': 'DE', 'Fun Rk, adjt': 'Tempo'})

# Calculate season average tempo
avg_tempo = trank['Tempo'].mean()
trank['AvgTempo'] = avg_tempo

# Put the team names in the same format (lowercase no punctuation) for joins
trank['Team'] = trank['Team'].str.replace('[^a-zA-Z&.()\' ]+',' ').str.lower()
trank['Team'] = trank['Team'].str.replace('[^a-z&.()\' ]+','')

# fix the names in order to join to get the team id
trank['Team'] = trank.apply(fix_name, axis = 1)

# merge to spellings to get team ID
trank_teams = pd.merge(trank, spellings, how = 'left', left_on = 'Team', right_on = 'TeamNameSpelling')

# remove duplicate spellings
trank_teams = trank_teams[['TeamID', 'TrankRating', 'OE', 'DE', 'Tempo', 'AvgTempo']].drop_duplicates()

# merge trank and teamrank
teams = pd.merge(teamrank_teams, trank_teams, how = 'inner', on = ['TeamID'])

# set ratings as floats for calculations
teams['TeamrankRating'] = teams['TeamrankRating'].astype(float)
teams['TrankRating'] = teams['TrankRating'].astype(float)

# Calculate Z score of ratings to put them on same scale
teams['TeamrankZScore'] = zscore(teams['TeamrankRating'])
teams['TrankZScore'] = zscore(teams['TrankRating'])

# Take a weighted average of ratings
teams['WeightedRating'] = 0.55 * teams['TrankZScore'] + 0.45 * teams['TeamrankZScore']
teams = teams.drop(columns = ['TrankZScore', 'TeamrankZScore'])

# TODO: Read in 2021 seeds
seeds = pd.read_csv('ncaam-march-mania-2021/MNCAATourneySeeds.csv')

# merge seeds with team data
teams = pd.merge(teams, seeds, on = ['Season', 'TeamID'], how = 'inner')

# extract just the seed number, no need for region here
def clean_seeds(row):
    return int(row['Seed'][1:3])

teams['Seed'] = teams.apply(clean_seeds, axis = 1)

# TODO: Read in regular season stats, might have to query for 2021 only and drop the season column
reg_season = pd.read_csv('ncaam-march-mania-2021/MRegularSeasonDetailedResults.csv')

# Score differential per posseassion
reg_season['ScoreDiffPerPoss'] = 2 * (reg_season['WScore'] - reg_season['LScore']) / (reg_season['WFGA'] + reg_season['WTO'] + 0.44 * reg_season['WFTA'] - reg_season['WOR'] + reg_season['LFGA'] + reg_season['LTO'] + 0.44 * reg_season['LFTA'] - reg_season['LOR'])

# Adjust score differential for home court
def adj_score_for_location(row):
    if row['WLoc'] == 'H':
        return row['ScoreDiffPerPoss'] - 0.02 # home court is worth less in 2021
    elif row['WLoc'] == 'A':
        return row['ScoreDiffPerPoss'] + 0.02 
    else:
        return row['ScoreDiffPerPoss']
    
reg_season['AdjScoreDiffPerPoss'] = reg_season.apply(adj_score_for_location, axis = 1)

# Get trank offensive efficiency and defensive efficiency
trank_ratings = trank_teams[['TeamID', 'OE', 'DE']]

# Get stats I need from regrular season stats
my_data = reg_season[['WTeamID', 'LTeamID', 'WFGM', 'LFGM', 'WFGA', 'LFGA', 'WFGM3', 'LFGM3', 'WFGA3', 'LFGA3', 'WFTM', 'LFTM', 'WFTA', 'LFTA', 'WAst', 'LAst', 'WTO', 'LTO', 'WOR', 'LOR', 'WDR', 'LDR', 'AdjScoreDiffPerPoss']]

# join stats and trank ratings for winning team
my_data = pd.merge(my_data, trank_ratings, left_on = ['WTeamID'], right_on = ['TeamID']).rename(columns = {'OE': 'WOE', 'DE': 'WDE'})

# join stats and trank ratings for losing team
my_data = pd.merge(my_data, trank_ratings, how = 'outer', left_on = ['LTeamID'], right_on = ['TeamID']).rename(columns = {'OE': 'LOE', 'DE': 'LDE'})

my_data = my_data.drop(columns = ['TeamID_x', 'TeamID_y'])

# What Trank predicted the adjusted score differential per possession would be for each game
my_data['PredictedAdjScoreDiffPerPoss'] = (my_data['WOE'] + my_data['LDE'] - (my_data['WDE'] + my_data['LOE'])) / 100

# w_data is data for games in which the team won, and l_data is data for the games in which the team lost
w_data = my_data.groupby(['WTeamID']).sum().drop(columns = ['LTeamID', 'AdjScoreDiffPerPoss', 'PredictedAdjScoreDiffPerPoss', 'WOE', 'WDE', 'LOE', 'LDE']).reset_index()
l_data = my_data.groupby(['LTeamID']).sum().drop(columns = ['WTeamID', 'AdjScoreDiffPerPoss', 'PredictedAdjScoreDiffPerPoss', 'WOE', 'WDE', 'LOE', 'LDE']).reset_index()

# join to get data for both wins and losses
wl_data = pd.merge(w_data, l_data, left_on = ['WTeamID'], right_on = ['LTeamID'], how = 'outer')
wl_data = wl_data.fillna(0)
# must do a outer join and fill NaNs with zeros due to undefeated teams

# Caculate season stats
stats = pd.DataFrame()
stats['Season'] = wl_data['Season']
stats['TeamID'] = wl_data['WTeamID']
stats['3ptRate'] = (wl_data['WFGA3_x'] + wl_data['LFGA3_y']) / (wl_data['WFGA_x'] + wl_data['LFGA_y'])
stats['Ast%'] = (wl_data['WAst_x'] + wl_data['LAst_y']) / (wl_data['WFGM_x'] + wl_data['LFGM_y'])
stats['FT%'] = (wl_data['WFTM_x'] + wl_data['LFTM_y']) / (wl_data['WFTA_x'] + wl_data['LFTA_y'])
stats['OppFT%'] = (wl_data['WFTM_y'] + wl_data['LFTM_x']) / (wl_data['WFTA_y'] + wl_data['LFTA_x'])
stats['Opp3ptRate'] = (wl_data['WFGA3_y'] + wl_data['LFGA3_x']) / (wl_data['WFGA_y'] + wl_data['LFGA_x'])
stats['OppAst%'] = (wl_data['WAst_y'] + wl_data['LAst_x']) / (wl_data['WFGM_y'] + wl_data['LFGM_x'])
stats['EFG%'] = (wl_data['WFGM_x'] + wl_data['LFGM_y'] + .5 * wl_data['WFGM3_x'] + .5 * wl_data['LFGM3_y']) / (wl_data['WFGA_x'] + wl_data['LFGA_y'])
stats['EFGD%'] = (wl_data['WFGM_y'] + wl_data['LFGM_x'] + .5 * wl_data['WFGM3_y'] + .5 * wl_data['LFGM3_x']) / (wl_data['WFGA_y'] + wl_data['LFGA_x'])
stats['TOR%'] = (wl_data['WTO_x'] + wl_data['LTO_y']) / (wl_data['WFGA_x'] + wl_data['LFGA_y'] - wl_data['WOR_x'] - wl_data['LOR_y'] + wl_data['WTO_x'] + wl_data['LTO_y'] + .44 * (wl_data['WFTA_x'] + wl_data['LFTA_y']))
stats['TORD%'] = (wl_data['WTO_y'] + wl_data['LTO_x']) / (wl_data['WFGA_y'] + wl_data['LFGA_x'] - wl_data['WOR_y'] - wl_data['LOR_x'] + wl_data['WTO_y'] + wl_data['LTO_x'] + .44 * (wl_data['WFTA_y'] + wl_data['LFTA_x']))
stats['ORB%'] = (wl_data['WOR_x'] + wl_data['LOR_y']) / (wl_data['WOR_x'] + wl_data['LOR_y'] + wl_data['WDR_y'] + wl_data['LDR_x'])
stats['OppORB%'] = 1 - (wl_data['WDR_x'] + wl_data['LDR_y']) / (wl_data['WOR_y'] + wl_data['LOR_x'] + wl_data['WDR_x'] + wl_data['LDR_y'])
stats['FTR'] = (wl_data['WFTA_x'] + wl_data['LFTA_y']) / (wl_data['WFGA_x'] + wl_data['LFGA_y'])
stats['FTRD'] = (wl_data['WFTA_y'] + wl_data['LFTA_x']) / (wl_data['WFGA_y'] + wl_data['LFGA_x'])
stats['2P%'] = (wl_data['WFGM_x'] + wl_data['LFGM_y'] - (wl_data['WFGM3_x'] + wl_data['LFGM3_y'])) / (wl_data['WFGA_x'] + wl_data['LFGA_y'] - (wl_data['WFGA3_x'] + wl_data['LFGA3_y']))
stats['2P%D'] = (wl_data['WFGM_y'] + wl_data['LFGM_x'] - (wl_data['WFGM3_y'] + wl_data['LFGM3_x'])) / (wl_data['WFGA_y'] + wl_data['LFGA_x'] - (wl_data['WFGA3_y'] + wl_data['LFGA3_x']))
stats['3P%'] = (wl_data['WFGM3_x'] + wl_data['LFGM3_y']) / (wl_data['WFGA3_x'] + wl_data['LFGA3_y'])
stats['3P%D'] = (wl_data['WFGM3_y'] + wl_data['LFGM3_x']) / (wl_data['WFGA3_y'] + wl_data['LFGA3_x'])

# compute season variance stats for each team
stats2 = pd.DataFrame()
for team in teams['TeamID'].unique():
    if len(teams[(teams['TeamID'] == team) & (teams['Seed'] > 0)]) > 0:
        season_data = my_data[my_data['Season'] == season]
        w_data = season_data[season_data['WTeamID'] == team]
        w_data.columns = ['Season', 'TeamID', 'OppTeamID', 'FGM', 'OppFGM', 'FGA', 'OppFGA', 'FGM3', 'OppFGM3', 'FGA3', 'OppFGA3', 'FTM', 'OppFTM', 'FTA', 'OppFTA', 'Ast', 'OppAst', 'TO', 'OppTO', 'OR', 'OppOR', 'DR', 'OppDR', 'AdjScoreDiffPerPoss', 'OE', 'DE', 'OppOE', 'OppDE', 'PredictedAdjScoreDiffPerPoss']
        l_data = season_data[season_data['LTeamID'] == team]
        l_data = l_data[['Season', 'LTeamID', 'WTeamID', 'LFGM', 'WFGM', 'LFGA', 'WFGA', 'LFGM3', 'WFGM3', 'LFGA3', 'WFGA3', 'LFTM', 'WFTM', 'LFTA', 'WFTA', 'LAst', 'WAst', 'LTO', 'WTO', 'LOR', 'WOR', 'LDR', 'WDR', 'AdjScoreDiffPerPoss', 'LOE', 'LDE', 'WOE', 'WDE', 'PredictedAdjScoreDiffPerPoss']]
        l_data['AdjScoreDiffPerPoss'] = -1 * l_data['AdjScoreDiffPerPoss']
        l_data['PredictedAdjScoreDiffPerPoss'] = -1 * l_data['PredictedAdjScoreDiffPerPoss']
        l_data.columns = w_data.columns
        team_data = pd.concat([w_data, l_data])
        team_data['3ptRate'] = team_data['FGA3'] / team_data['FGA']
        team_data['Opp3ptRate'] = team_data['OppFGA3'] / team_data['OppFGA']
        team_data['Ast%'] = team_data['Ast'] / team_data['FGM']
        team_data['OppAst%'] = team_data['OppAst'] / team_data['OppFGM']
        team_data['eFG%'] = (team_data['FGM'] + .5 * team_data['FGM3']) / team_data['FGA']
        team_data['OppeFG%'] = (team_data['OppFGM'] + .5 * team_data['OppFGM3']) / team_data['OppFGA']
        team_data['3pt%'] = team_data['FGM3'] / team_data['FGA3']
        team_data['Opp3pt%'] = team_data['OppFGM3'] / team_data['OppFGA3']
        team_data['FT%'] = team_data['FTM'] / team_data['FTA']
        team_data['FTR'] = team_data['FTA'] / team_data['FGA']
        team_data['OppFTR'] = team_data['OppFTA'] / team_data['OppFGA']
        team_data['OR%'] = team_data['OR'] / (team_data['OR'] + team_data['OppDR'])
        team_data['OppOR%'] = team_data['OppOR'] / (team_data['OppOR'] + team_data['DR'])
        team_data['TO%'] = team_data['TO'] / (team_data['TO'] + team_data['FGA'] - team_data['OR'] + .44 * team_data['FTA'])
        team_data['OppTO%'] = team_data['OppTO'] / (team_data['OppTO'] + team_data['OppFGA'] - team_data['OppOR'] + .44 * team_data['OppFTA'])
        team_data['TotalPoss'] = team_data['TO'] + team_data['OppTO'] + team_data['FGA'] + team_data['OppFGA'] - team_data['OR'] - team_data['OR'] + .44 * (team_data['FTA'] + team_data['OppFTA'])
        team_data['GameScore'] = team_data['AdjScoreDiffPerPoss'] - team_data['PredictedAdjScoreDiffPerPoss']
        stats2 = pd.concat([stats2, pd.DataFrame({'Season': [season],
                                                 'TeamID': [team],
                                                 '3ptRateVar': [np.var(team_data['3ptRate'])],
                                                 'Opp3ptRateVar': [np.var(team_data['Opp3ptRate'])],
                                                 'eFG%Var': [np.var(team_data['eFG%'])],
                                                 'OppeFG%Var': [np.var(team_data['OppeFG%'])],
                                                 '3pt%Var': [np.var(team_data['3pt%'])],
                                                 'Opp3pt%Var': [np.var(team_data['Opp3pt%'])],
                                                 'Ast%Var': [np.var(team_data['Ast%'])],
                                                 'OppAst%Var': [np.var(team_data['OppAst%'])],
                                                 'FT%Var': [np.var(team_data['FT%'])],
                                                 'FTRVar': [np.var(team_data['FTR'])],
                                                 'OppFTRVar': [np.var(team_data['OppFTR'])],
                                                 'OR%Var': [np.var(team_data['OR%'])],
                                                 'OppOR%Var': [np.var(team_data['OppOR%'])],
                                                 'TO%Var': [np.var(team_data['TO%'])],
                                                 'OppTO%Var': [np.var(team_data['OppTO%'])],
                                                 'TotalPossVar': [np.var(team_data['TotalPoss'])],
                                                 'GameScoreVar': [np.var(team_data['GameScore'])]})])

# Merge stats and then merge with teams
stats_merge = pd.merge(stats, stats2, on = ['TeamID'])
teams = pd.merge(teams, stats_merge, on = ['TeamID'])

# Submission file for 2021
# TODO: Update file name
data21 = pd.read_csv('DataFiles/SampleSubmissionStage2.csv')

# Weighted ratings for 2021
ratings = teams[['TeamID', 'WeightedRating']]

data21['TeamID_x'] = -1
data21['TeamID_y'] = -1
for i in range(len(data21.ID)):
    idstring = data21.iloc[i, 0]  # Game ID in the form year_teamID1_teamID2
    infolist = idstring.split('_')
    data21.iloc[i, 2] = int(infolist[1]) # TeamID_x
    data21.iloc[i, 3] = int(infolist[2]) # TeamID_y
    

# merge submission with ratings for both teams
game_data = pd.merge(data21, ratings, left_on = ['TeamID_x'], right_on = ['TeamID'])
game_data = pd.merge(game_data, ratings, left_on = ['TeamID_y'], right_on = ['TeamID'])
game_data = game_data.loc[:,~game_data.columns.duplicated()]  # Deletes duplicate columns

# To make team x be the team with the higher rating and team y be the team with the lower rating
def switch_teams(row):
    # if rating x is less than rating y
    if row['WeightedRating_x'] < row['WeightedRating_y']:
        underdog = row['TeamID_x']  # "Worse" team's ID
        favorite = row['TeamID_y']  # "Better" team's ID
        row['TeamID_x'] = favorite
        row['TeamID_y'] = underdog
        underdog = row['WeightedRating_x']  # "Worse" team's rating
        favorite = row['WeightedRating_y']  # "Better" team's rating
        row['WeightedRating_x'] = favorite
        row['WeightedRating_y'] = underdog
    return row
                       
game_data = game_data.apply(switch_teams, axis = 1)

# merge games with teams
game_data = pd.merge(game_data, teams, left_on = ['TeamID_x'], right_on = ['TeamID'])
game_data = pd.merge(game_data, teams, left_on = ['TeamID_y'], right_on = ['TeamID'])
game_data = game_data.loc[:,~game_data.columns.duplicated()]  # Removes duplicate columns



# Start with the stats for each team
matchups = game_data.drop(columns = ['TeamID_x', 'TeamID_y', 'AvgTempo_x', 'AvgTempo_y', 'WeightedRating_x', 'WeightedRating_y', 'ID', 'Pred'])

# Predictors

# Difference in NCAA tournament Seeds
matchups['SeedDiff'] = matchups['Seed_x'] - matchups['Seed_y']

# Trank Predicted Spread
matchups['TrankPredictedSpreadPerPoss'] = (game_data['OE_x'] +  game_data['DE_y'] - game_data['OE_y'] - game_data['DE_x']) / 100
matchups['TrankPredictedPoss'] = game_data['Tempo_x'] * game_data['Tempo_y'] / game_data['AvgTempo_x']
matchups['TrankPredictedSpread'] = matchups['TrankPredictedPoss'] * matchups['TrankPredictedSpreadPerPoss']

# Teamrank Predicted Spread
matchups['TeamrankPredictedSpread'] = game_data['TeamrankRating_x'] - game_data['TeamrankRating_y']
matchups['TeamrankPredictedSpreadPerPoss'] = matchups['TeamrankPredictedSpread'] / matchups['TrankPredictedPoss']

# tempo difference and absolute value of tempo difference for Trank tempo
matchups['TrankTempoDiff'] = game_data['Tempo_x'] - game_data['Tempo_y']
matchups['AbsTrankTempoDiff'] = abs(matchups['TrankTempoDiff'])

# Offensive vs defensive EFG% averages and differences
matchups['xOffyDefEFGAvg'] = (game_data['EFG%_x'] + game_data['EFGD%_y']) / 2
matchups['yOffxDefEFGAvg'] = (game_data['EFG%_y'] + game_data['EFGD%_x']) / 2

# Offensive vs defensive turnover rate averages and differences
matchups['xOffyDefTOAvg'] = (game_data['TOR%_x'] + game_data['TORD%_y']) / 2
matchups['yOffxDefTOAvg'] = (game_data['TOR%_y'] + game_data['TORD%_x']) / 2
matchups['xOffyOffTODiff'] = matchups['xOffyDefTOAvg'] - matchups['yOffxDefTOAvg']

# Offensive vs defensive rebound rate averages and differences
matchups['xOffRebAvg'] = (game_data['ORB%_x'] + game_data['OppORB%_y']) / 2
matchups['yOffRebAvg'] = (game_data['ORB%_y'] + game_data['OppORB%_x']) / 2
matchups['xOffyOffRebDiff'] = matchups['xOffRebAvg'] - matchups['yOffRebAvg']

# Offensive vs defensive FT rate averages and differences
matchups['xOffyDefFTRateAvg'] = (game_data['FTR_x'] + game_data['FTRD_y']) / 2
matchups['yOffxDefFTRateAvg'] = (game_data['FTR_y'] + game_data['FTRD_x']) / 2

# Offensive vs defensive assist rate averages and differences
matchups['AbsxOffyDefAstDiff'] = abs(game_data['Ast%_x'] - game_data['OppAst%_y'])
matchups['AbsyOffxDefAstDiff'] = abs(game_data['Ast%_y'] - game_data['OppAst%_x'])
matchups['xOffyDefAstAvg'] = (game_data['Ast%_x'] + game_data['OppAst%_y']) / 2
matchups['yOffxDefAstAvg'] = (game_data['Ast%_y'] + game_data['OppAst%_x']) / 2

# Sum of the variance in game possession of both teams
matchups['TotalPossVarSum'] = game_data['TotalPossVar_x'] + game_data['TotalPossVar_y']

# Sum of the variance in game performance of both teams (weighted by ratio of tempo and predicted tempo, ie sample sizes)
matchups['GameScoreVarSum'] = (game_data['Tempo_x'] / matchups['TrankPredictedPoss']) * game_data['GameScoreVar_x'] + (game_data['Tempo_y'] / matchups['TrankPredictedPoss']) * game_data['GameScoreVar_y']

# Naive upset probability using predicted spread and sum of variance
matchups['TrankNaiveUpsetProbability'] = norm.cdf(0, loc = matchups['TrankPredictedSpreadPerPoss'], scale = (0.5 * matchups['GameScoreVarSum']) ** 0.5)
matchups['TeamrankNaiveUpsetProbability'] = norm.cdf(0, loc = matchups['TeamrankPredictedSpreadPerPoss'], scale = (0.5 * matchups['GameScoreVarSum']) ** 0.5)


# Load models for predictions
prob_model = pickle.load(open('models/MProb.sav', 'rb'))
spread_rf_model = pickle.load(open('models/MSpreadRF.sav', 'rb'))
spread_xgb_model = pickle.load(open('models/MSpreadXGB.sav', 'rb'))

# features to use for probabilities model
features_to_use = ['TrankRating_x', 'ORB%_x', 'FTRVar_x', 'OR%Var_y', 'SeedDiff', 'TrankPredictedSpread', 'xOffyOffTODiff', 'xOffyDefAstAvg', 'TrankNaiveUpsetProbability', 'TeamrankNaiveUpsetProbability']

# Make predictions for probabilities
prob_df = pd.DataFrame(prob_model.predict_proba(matchups[features_to_use]))
prob_df.columns = ['Pred', 'Ignore']

# Make submission df for probabilities
prob_submission = game_data[['ID', 'Pred', 'TeamID_x', 'TeamID_y']]
prob_submission['Pred'] = prob_df['Pred']


# Make predictions for spreads
rf_df = pd.DataFrame(spread_rf_model.predict(matchups))
rf_df.columns = ['Pred']
xgb_df = pd.DataFrame(spread_xgb_model.predict(matchups))
xgb_df.columns = ['Pred']

# Make submission df for probabilities
spread_submission = game_data[['ID', 'Pred', 'TeamID_x', 'TeamID_y']]
spread_submission['Pred'] = matchups['TrankPredictedPoss'] * (0.65 * rf_df['Pred'] + 0.35 * xgb_df['Pred'])# multiply expected spread per poss by expected poss

# Write predictions to csv
prob_submission.to_csv('mydata/mens/original_probabilities.csv', index = False)
spread_submission.to_csv('mydata/mens/original_spreads.csv', index = False)