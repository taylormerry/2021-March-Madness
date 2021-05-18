import pandas as pd
import re
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# team spellings df from kaggle
spellings = pd.read_csv('ncaaw-march-mania-2021/WTeamSpellings.csv', encoding = "ISO-8859-1")

# Put the team names in the same format (lowercase no punctuation) for joins later
spellings['TeamNameSpelling'] = spellings['TeamNameSpelling'].str.replace('[^a-z&. ]+', ' ')
spellings['TeamNameSpelling'] = spellings['TeamNameSpelling'].str.replace('[^a-z& ]+', '')


# Load the Moore ratings for this season
moore_txt = pd.read_csv('mydata/womens/moore21.txt', sep = "\n", header = None)

# lists to store team ratings
teams = []
ratings = []

# for each row in the text file
for i in range(len(moore_txt)):
    row = moore_txt.iloc[i, 0].split(' ')

    # list of actual elements in row
    actual = []

    # for each element in the row, add it to the actuals if it's not an empty string
    for element in row:
        if element != '':
            actual.append(element)

    # magic number for team name element
    teams.append(" ".join(actual[1:len(actual) - 5]))

    # ratings is always the last element
    ratings.append(actual[-1])

# append the seasons data
moore_data = pd.DataFrame({'Team': teams,
                            'MooreRating': ratings})


# Put the team names in the same format (lowercase no punctuation) for joins later
moore_data['Team'] = moore_data.Team.str.replace('[^a-zA-Z&.() ]+',' ').str.lower()
moore_data['Team'] = moore_data.Team.str.replace('[^a-z& ]+','').str.rstrip()

# function to change team names to match what's in the spelling csv
def fix_name(row):
    if row['Team'] == 'purdue ft wayne':
        return 'pfw'
    elif row['Team'] == 'mass lowell':
        return 'massachusetts lowell'
    elif row['Team'] == 'nj tech':
        return 'new jersey tech'
    elif row['Team'] == 'presbyterian college':
        return 'presbyterian'
    elif row['Team'] == 'loyola illinois':
        return 'loyola chicago'
    elif row['Team'] == 'central connecticut st':
        return 'central conn'
    elif row['Team'] == 'mt st mary s md':
        return 'mt st mary s'
    elif row['Team'] == 'iupu ft wayne':
        return 'pfw'
    elif row['Team'] == 'mississippi valley st':
        return 'ms valley st'
    elif row['Team'] == 'oakland mi':
        return 'oakland'
    elif row['Team'] == 'towson st':
        return 'towson'
    elif row['Team'] == 'ohio university':
        return 'ohio'
    elif row['Team'] == 's f austin':
        return 'stephen f austin'
    elif row['Team'] == 'southern cal':
        return 'usc'
    elif row['Team'] == 'tarleton':
        return 'tarleton st'
    elif row['Team'] == 'california san diego':
        return 'uc san diego'
    else:
        return row['Team']
    

# fix the names for the join
moore_data['Team'] = moore_data.apply(fix_name, axis = 1)

# join moore ratings to spellings to get team id
moore_teams = pd.merge(moore_data, spellings, how = 'left', left_on = 'Team', right_on = 'TeamNameSpelling')

# drop unnecessary columns and duplicate rows, make sure rating is a float
moore_teams = moore_teams.drop(columns = ['Team', 'TeamNameSpelling']).drop_duplicates()
moore_teams['MooreRating'] = moore_teams['MooreRating'].astype(float)

# Read in 2021's tournament seeds
seeds = pd.read_csv('ncaaw-march-mania-2021/WNCAATourneySeeds.csv').query('Season == 2021')

# merge seeds and moore ratings
teams = pd.merge(moore_teams, seeds, on = ['TeamID'], how = 'inner').drop_duplicates()

# return just the seed number, no need for region for this use case
def clean_seeds(row):
    return int(row['Seed'][1:3])

# get seed number for each team
teams['Seed'] = teams.apply(clean_seeds, axis = 1)

# Submission file for 2021
data21 = pd.read_csv('ncaaw-march-mania-2021/WSampleSubmissionStage2.csv')

# Weighted ratings for 2021
ratings = teams[['TeamID', 'MooreRating']]

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
    if row['MooreRating_x'] < row['MooreRating_y']:
        underdog = row['TeamID_x']  # "Worse" team's ID
        favorite = row['TeamID_y']  # "Better" team's ID
        row['TeamID_x'] = favorite
        row['TeamID_y'] = underdog
        underdog = row['MooreRating_x']  # "Worse" team's rating
        favorite = row['MooreRating_y']  # "Better" team's rating
        row['MooreRating_x'] = favorite
        row['MooreRating_y'] = underdog
    return row
                       
game_data = game_data.apply(switch_teams, axis = 1)

# merge games with teams
game_data = pd.merge(game_data, teams, left_on = ['TeamID_x'], right_on = ['TeamID'])
game_data = pd.merge(game_data, teams, left_on = ['TeamID_y'], right_on = ['TeamID'])
game_data = game_data.loc[:,~game_data.columns.duplicated()]  # Removes duplicate columns

# Start with the stats for each team
matchups = game_data.drop(columns = ['TeamID_x', 'TeamID_y', 'Season_x', 'Season_y', 'ID', 'Pred'])

# Predictors

# Difference in NCAA tournament Seeds
matchups['SeedDiff'] = matchups['Seed_x'] - matchups['Seed_y']

# Difference in Moore Rating
matchups['MoorePredictedSpread'] = matchups['MooreRating_x'] - matchups['MooreRating_y']

# Load models for predictions
prob_model = pickle.load(open('models/WProb.sav', 'rb'))
spread_rf_model = pickle.load(open('models/WSpreadRF.sav', 'rb'))
spread_xgb_model = pickle.load(open('models/WSpreadXGB.sav', 'rb'))

features_to_use = ['MooreRating_x', 'SeedDiff', 'MoorePredictedSpread']

# read in training data to scale X columns
scale = StandardScaler()
X_train = pd.read_csv('mydata/womens/matchups_no_stats.csv')[features_to_use]
scale.fit(X_train)
X_prob = pd.DataFrame(scale.transform(matchups[features_to_use]), columns = features_to_use)

# Make predictions for probabilities
prob_df = pd.DataFrame(prob_model.predict_proba(X_prob))
prob_df.columns = ['Pred', 'Ignore']

# Make submission df for probabilities
prob_submission = game_data[['ID', 'Pred', 'TeamID_x', 'TeamID_y']]
prob_submission['Pred'] = prob_df['Pred']
prob_submission['SeedDiff'] = matchups['SeedDiff']
prob_submission['Seed_x'] = matchups['Seed_x']
prob_submission['Seed_y'] = matchups['Seed_y']


# Make predictions for spreads
rf_df = pd.DataFrame(spread_rf_model.predict(matchups))
rf_df.columns = ['RFPred']
xgb_df = pd.DataFrame(spread_xgb_model.predict(matchups))
xgb_df.columns = ['XGBPred']

# Make submission df for probabilities
spread_submission = game_data[['ID', 'Pred', 'TeamID_x', 'TeamID_y']]
spread_submission['Pred'] = (0.5 * rf_df['RFPred'] + 0.5 * xgb_df['XGBPred']) # avg the random forest and xgboost predictions

# Write predictions to csv
prob_submission.to_csv('mydata/womens/original_probabilities.csv', index = False)
spread_submission.to_csv('mydata/womens/original_spreads.csv', index = False)