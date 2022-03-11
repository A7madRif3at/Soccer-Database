# Soccer-Database
This soccer database comes from Kaggle and is well suited for data analysis and machine learning. It contains data for soccer matches, players, and teams from several European countries from 2008 to 2016. This dataset is quite extensive, and we encourage you to read more about it here.  The database is stored in a SQLite database. You can access database files using software like DB Browser. This dataset will help you get good practice with your SQL joins. Make sure to look at how the different tables relate to each other. Some column titles should be self-explanatory, and others you’ll have to look up on Kaggle.
###Project: Investigate a Dataset - [Soccer Database]

##Introduction:-
#This is data for soccer matches, players, and teams from several European countries from 2008 to 2016. 

##project questions:-
# What teams improved the most over the time period?
# What team attributes lead to the most victories?
# Which players had the most penalties?
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import datetime
%matplotlib inline
import seaborn as sns
# Load your data and print out a few lines. 
engine = ('sqlite:///C:/Users/admin/Downloads/Compressed/Soccer Database/database.sqlite/')
df = pd.read_sql('SELECT * FROM master', engine)
#Perform operations to inspect data
df.head(20)
#biggest data set found in Matches table which help us two questions 
Match = pd.read_sql("""SELECT * FROM Match;""",engine)
# Assessing and Building Intuition
Match.info
#Cleaning Data from Match Table & change data type for date
Match.fillna(Match.mean(), inplace=True)
Match.info()
sum(Match.duplicated())
Match['date'] = pd.to_datetime(Match['date'])
Match['date'] = Match['date'].dt.year
#Exploratory Data Analysis
match_col = Match[['season','stage','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal','date']]
match_col['result'] = match_col['home_team_goal'] - match_col['away_team_goal']
match_col['winner_home'] = match_col['result'] >0
match_col['winner_away'] = match_col['result'] <0
match_col.winner_home = match_col.winner_home.replace({True: 1, False: 0})
match_col.winner_away = match_col.winner_away.replace({True: 1, False: 0})
match_col
#Merging between home & away matches in order to calculate total wins for each team
def home_away():
    home = match_col[['date','home_team_api_id','home_team_goal','winner_home','stage']]
    home = home.groupby(['home_team_api_id','date',]).agg({'winner_home':'sum','home_team_goal': 'sum'}).reset_index().sort_values(['winner_home','home_team_goal'], ascending =False)
    away = match_col[['date','away_team_api_id','away_team_goal','winner_away','stage']]
    away = away.groupby(['away_team_api_id','date',]).agg({'winner_away':'sum','away_team_goal': 'sum'}).reset_index().sort_values(['winner_away','away_team_goal'], ascending =False)
    home.rename(columns = {'home_team_api_id' : 'Team id', 'winner_home' : 'Win.h'}, inplace = True)
    away.rename(columns = {'away_team_api_id' : 'Team id', 'winner_away' : 'Win.a'}, inplace = True)
home, away
total_wins = pd.merge(left=home, right=away, left_on=['date', 'Team id'], right_on=['date', 'Team id'])
total_wins['win'] = total_wins['Win.h'] + total_wins['Win.a']
total_wins['goals'] = total_wins['home_team_goal'] + total_wins['home_team_goal']

total = total_wins[['Team id','date','win','goals']]
total
# merging between two tables (Matches (total) and Team ,in order to identify our results with the team name)
Team = pd.read_sql("""SELECT * FROM Team;""",engine)
Teams = pd.merge(left=total, right=Team, left_on='Team id', right_on='team_api_id')
Teams = Teams.groupby(['date','team_long_name']).agg({'win':'sum','goals':'sum'}).reset_index().sort_values(['date','win'], ascending =False)
Teams = Teams.groupby('date').head(5).reset_index(drop=True).sort_values(['date','win'], ascending =False)
Teams['Team id'] = total['Team id']
Teams
#Ploting results to follow up the changes over the years
plt.figure(figsize=(24,6))
plt.bar(Teams['date'], Teams['team_long_name']);
plt.xlabel("date")
plt.ylabel("team_long_name")
plt.show()
#Ploting with Pie shape to have more clear visualize
Teams['team_long_name'].value_counts('win').plot(kind='pie', figsize=(8, 8));
# What team attributes lead to the most victories?
# Data Cleaning from team_attributes
team_attributes = pd.read_sql("""SELECT * FROM Team_Attributes;""",engine)
team_attributes.fillna(team_attributes.mean(), inplace=True)
sum(team_attributes.duplicated())
team_attributes.info()
#Mergine between tables and sort it ,with given chance for more visualize exploring
Winner_Atribute = pd.merge(left=Teams, right=team_attributes, left_on='Team id', right_on='team_api_id')
del Winner_Atribute["date_x"]
Winner_Atribute.head(40)
#Which players had the most penalties?
## Data Cleaning from Player_Attributes
Player_Attributes = pd.read_sql("""SELECT * FROM Player_Attributes;""",engine)
Player_Attributes.fillna(Player_Attributes.mean(), inplace=True)
Player_Attributes.info()
sum(Player_Attributes.duplicated())
# investigate Player_Attributes Table
Player_Attributes = pd.read_sql("""SELECT * FROM Player_Attributes;""",engine)
# investigate Player Table
Player = pd.read_sql("""SELECT * FROM Player;""",engine)
# investigate Penalties
Penalties = Player_Attributes.groupby(['player_api_id']).agg({'penalties':'count'}).reset_index().sort_values(['penalties'], ascending =False)             
# Players had the most penalties
players_Penalties = pd.merge(left=Penalties, right=Player, left_on='player_api_id', right_on='player_api_id')
top10 = players_Penalties[['penalties','player_name']]
top10 = players_Penalties.iloc[0:10]
top10
#Ploting results to follow up the players with the most penalties
plt.figure(figsize=(18,6))
plt.plot(top10['player_name'], top10['penalties']);
plt.xlabel("player_name")
plt.ylabel("penalties")
plt.show()
#Conclusions
#Results:
    #1. Teams improved the most over the time period ,we can follow up their performance more closely starting from season of 2013 ,with excluding some results from 2014 , as these teams their performace has been decreased for the next two years
    #2. Team attributes lead to the most victories,we should concentrate on defenceDefenderLineClass & chanceCreationShootingClass as there's correlations between this element and team performance 
    #3. players had the most penalties are Roberto Pereyra & Alessio Cerci
#Limitations :
    #1. Big matches need to be in seperate column with sorting the values as the follow (final or semi final) as this would be more helpful ,as stages are not enough for such info
    #2. As soccer fan there's some indicators are missing over here such as xG (The possibility of scoring goals) which is measuring the time between each goal for each player ,and this showing the stability in scoring for each player more than number of penalties ,at the end soccer is a group game not indiviual game.
    
    

