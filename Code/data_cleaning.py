import numpy as np
import pandas as pd
import os
import team_mapping
import re

month_mapping = {"Jan": "1", "Feb": "2", "Mar": "3", "Apr": "4", "May": "5", "Jun": "6", "Jul": "7", "Aug": "8", "Sep": "9", "Oct": "10", "Nov": "11", "Dec": "12"}
player_mapping = {
    "K. Porziņģis": "K. Porzingis",
    "N. Vučević": "N. Vucevic",
    "V. Micić": "V. Micic",
    "L. Dončić": "L. Doncic",
    "N. Jokić": "N. Jokic",
    "A. Şengün": "A. Sengun",
    "N. Jović": "N. Jovic",
    "J. Valančiūnas": "J. Valanciunas",
    "J. Nurkić": "J. Nurkic",
    "D. Schröder": "D. Schroder"
}

files = ["2017-2018_Season.csv", "2018-2019_Season.csv", "2019-2020_Season.csv", "2020-2021_Season.csv", "2021-2022_Season.csv", "2022-2023_Season.csv"]

def remove_non_ascii(val):
    if isinstance(val, str):
        return re.sub(r'[^\x00-\x7F]+', '', val)
    return val

for file in files:
    # Get the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_path = os.path.join(script_dir, os.pardir, "Data/Raw/Team Stats", file)
    new_path = os.path.abspath(new_path)
    df = pd.read_csv(new_path)

    # Clean the data
    df = df.iloc[:,1:3]
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df[["Wins", "Losses"]] = df["Overall"].str.split('-', expand=True)
    df = df.drop(columns="Overall")
    df["Team"] = df["Team"].apply(lambda x: team_mapping.map_team(x))
    df = df.apply(lambda col: col.replace(month_mapping))
    df = df.sort_values(by=["Team"])

    # Export the cleaned csv
    new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", file)
    new_path = os.path.abspath(new_path)
    df.to_csv(new_path, index=False)

months = ["Oct","Nov","Dec","Jan","Feb","Mar","Apr"]
years = ["2021-2022_","2022-2023_","2023-2024_"]

for year in years:
    new_df = pd.DataFrame(columns=["Month","Day","Away","Away_Pts","Home","Home_Pts"])
    for month in months:
        # Get the dataset
        script_dir = os.path.dirname(os.path.abspath(__file__))
        new_path = os.path.join(script_dir, os.pardir, "Data/Raw/Team Stats", year + month + ".csv")
        new_path = os.path.abspath(new_path)
        df = pd.read_csv(new_path)

        # Clean the data
        df = df.iloc[:,0:6]
        df = df.drop(columns="Start (ET)")
        df = df.rename(columns={"Visitor/Neutral": "Away", "PTS": "Away_Pts", "Home/Neutral": "Home", "PTS.1": "Home_Pts"})
        df["Away"] = df["Away"].apply(lambda x: team_mapping.map_team(x))
        df["Home"] = df["Home"].apply(lambda x: team_mapping.map_team(x))
        df[["Week_Day", "Month", "Day", "Year"]] = df["Date"].str.split(' ', expand=True)
        df = df.drop(columns=["Date", "Week_Day", "Year"])
        new_df = pd.concat([new_df, df])
        new_df["Game_Outcome"] = new_df.apply(lambda row: "Home" if row["Home_Pts"] > row["Away_Pts"] else "Away", axis=1)

    # Export the cleaned csv
    new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", year + "Games.csv")
    new_path = os.path.abspath(new_path)
    new_df.to_csv(new_path, index=False)

months = ["Dec","Jan","Feb","Mar","Apr","May"]
new_df = pd.DataFrame(columns=["Month","Day","Away","Away_Pts","Home","Home_Pts"])

for month in months:
    # Get the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_path = os.path.join(script_dir, os.pardir, "Data/Raw/Team Stats", "2020-2021_" + month + ".csv")
    new_path = os.path.abspath(new_path)
    df = pd.read_csv(new_path)

    # Clean the data
    df = df.iloc[:,0:6]
    df = df.drop(columns="Start (ET)")
    df = df.rename(columns={"Visitor/Neutral": "Away", "PTS": "Away_Pts", "Home/Neutral": "Home", "PTS.1": "Home_Pts"})
    df["Away"] = df["Away"].apply(lambda x: team_mapping.map_team(x))
    df["Home"] = df["Home"].apply(lambda x: team_mapping.map_team(x))
    df[["Week_Day", "Month", "Day", "Year"]] = df["Date"].str.split(' ', expand=True)
    df = df.drop(columns=["Date", "Week_Day", "Year"])
    new_df = pd.concat([new_df, df])
    new_df["Game_Outcome"] = new_df.apply(lambda row: "Home" if row["Home_Pts"] > row["Away_Pts"] else "Away", axis=1)

# Export the cleaned csv
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2020-2021_" + "Games.csv")
new_path = os.path.abspath(new_path)
new_df.to_csv(new_path, index=False)

teams = ["ATL","BOS","BRK","CHI","CHO","CLE","DAL","DEN","DET","GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK","OKC","ORL","PHI","PHO","POR","SAC","SAS","TOR","UTA","WAS"]

for team in teams:
    # Get the dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_path = os.path.join(script_dir, os.pardir, "Data/Raw/Lineups", team + "_Lineups.csv")
    new_path = os.path.abspath(new_path)
    df = pd.read_csv(new_path)

    # Clean the data
    df = df[["G","Starting Lineup"]]
    df[["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"]] = df["Starting Lineup"].str.split(' · ', expand=True)
    df = df.drop(columns="Starting Lineup")
    df = df.apply(lambda col: col.replace(player_mapping))

    # Function to manually check if starting players are spelled differently
    '''
    unique_values = set()
    for col in ["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"]:
        unique_values.update(df[col].dropna().unique())
    '''

    # Export the cleaned csv
    new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Lineups", team + "_Lineups.csv")
    new_path = os.path.abspath(new_path)
    df.to_csv(new_path, index=False)