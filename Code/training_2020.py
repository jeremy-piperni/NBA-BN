import numpy as np
import pandas as pd
import os
from season_parser import parse_season

# Get the 2017-2018 season dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2017-2018_Season.csv")
new_path = os.path.abspath(new_path)
df_2017 = pd.read_csv(new_path)

normalized_wins = round(df_2017.Wins / (df_2017.Wins + df_2017.Losses) * 82)
df_2017["Wins"] = normalized_wins.astype(int)

# Get the 2018-2019 season dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2018-2019_Season.csv")
new_path = os.path.abspath(new_path)
df_2018 = pd.read_csv(new_path)

normalized_wins = round(df_2018.Wins / (df_2018.Wins + df_2018.Losses) * 82)
df_2018["Wins"] = normalized_wins.astype(int)

# Get the 2019-2020 season dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2019-2020_Season.csv")
new_path = os.path.abspath(new_path)
df_2019 = pd.read_csv(new_path)

normalized_wins = round(df_2019.Wins / (df_2019.Wins + df_2019.Losses) * 82)
df_2019["Wins"] = normalized_wins.astype(int)

# Get the 2020-2021 games dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2020-2021_Games.csv")
new_path = os.path.abspath(new_path)
df_games_2020 = pd.read_csv(new_path)

# Prepare the 2020-2021 training data
df_games_2020 = df_games_2020.rename(columns={"Away":"Team"})
df_train_2020 = df_games_2020.merge(df_2019, on="Team", how="left")
df_train_2020 = df_train_2020.rename(columns={
    "Team": "Away",
    "Wins": "Away_Wins_Last"
})
df_train_2020 = df_train_2020.rename(columns={"Home":"Team"})
df_train_2020 = df_train_2020.merge(df_2019, on="Team", how="left")
df_train_2020 = df_train_2020.rename(columns={
    "Team": "Home",
    "Wins": "Home_Wins_Last"
})
df_train_2020 = df_train_2020[["Home","Away","Home_Wins_Last","Away_Wins_Last","Game_Outcome"]]
df_train_2020 = df_train_2020.rename(columns={"Away":"Team"})
df_train_2020 = df_train_2020.merge(df_2018, on="Team", how="left")
df_train_2020 = df_train_2020.rename(columns={
    "Team": "Away",
    "Wins": "Away_Wins_Second_Last"
})
df_train_2020 = df_train_2020.rename(columns={"Home":"Team"})
df_train_2020 = df_train_2020.merge(df_2018, on="Team", how="left")
df_train_2020 = df_train_2020.rename(columns={
    "Team": "Home",
    "Wins": "Home_Wins_Second_Last"
})
df_train_2020 = df_train_2020[["Home","Away","Home_Wins_Last","Away_Wins_Last","Home_Wins_Second_Last","Away_Wins_Second_Last","Game_Outcome"]]
df_train_2020 = df_train_2020.rename(columns={"Away":"Team"})
df_train_2020 = df_train_2020.merge(df_2017, on="Team", how="left")
df_train_2020 = df_train_2020.rename(columns={
    "Team": "Away",
    "Wins": "Away_Wins_Third_Last"
})
df_train_2020 = df_train_2020.rename(columns={"Home":"Team"})
df_train_2020 = df_train_2020.merge(df_2017, on="Team", how="left")
df_train_2020 = df_train_2020.rename(columns={
    "Team": "Home",
    "Wins": "Home_Wins_Third_Last"
})
df_train_2020 = df_train_2020[["Home","Away","Home_Wins_Last","Away_Wins_Last","Home_Wins_Second_Last","Away_Wins_Second_Last","Home_Wins_Third_Last","Away_Wins_Third_Last","Game_Outcome"]]

df_train_2020["Game_Outcome"] = df_train_2020.apply(lambda row: 1 if row["Game_Outcome"] == "Home" else 0, axis=1)

df_train_2020 = parse_season(df_train_2020, "2020-2021")