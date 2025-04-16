import numpy as np
import pandas as pd
import os
from season_parser import parse_season

# Get the 2020-2021 season dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2020-2021_Season.csv")
new_path = os.path.abspath(new_path)
df_2020 = pd.read_csv(new_path)

normalized_wins = round(df_2020.Wins / (df_2020.Wins + df_2020.Losses) * 82)
df_2020["Wins"] = normalized_wins.astype(int)

# Get the 2021-2022 season dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2021-2022_Season.csv")
new_path = os.path.abspath(new_path)
df_2021 = pd.read_csv(new_path)

normalized_wins = round(df_2021.Wins / (df_2021.Wins + df_2021.Losses) * 82)
df_2021["Wins"] = normalized_wins.astype(int)

# Get the 2022-2023 season dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2022-2023_Season.csv")
new_path = os.path.abspath(new_path)
df_2022 = pd.read_csv(new_path)

normalized_wins = round(df_2022.Wins / (df_2022.Wins + df_2022.Losses) * 82)
df_2022["Wins"] = normalized_wins.astype(int)

# Get the 2023-2024 games dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", "2023-2024_Games.csv")
new_path = os.path.abspath(new_path)
df_games_2023 = pd.read_csv(new_path)

# Prepare the 2023-2024 training data
df_test_2023 = df_games_2023.rename(columns={"Away":"Team"})
df_test_2023 = df_test_2023.merge(df_2022, on="Team", how="left")
df_test_2023 = df_test_2023.rename(columns={
    "Team": "Away",
    "Wins": "Away_Wins_Last"
})
df_test_2023 = df_test_2023.rename(columns={"Home":"Team"})
df_test_2023 = df_test_2023.merge(df_2022, on="Team", how="left")
df_test_2023 = df_test_2023.rename(columns={
    "Team": "Home",
    "Wins": "Home_Wins_Last"
})
df_test_2023 = df_test_2023[["Home","Away","Home_Wins_Last","Away_Wins_Last","Game_Outcome"]]
df_test_2023 = df_test_2023.rename(columns={"Away":"Team"})
df_test_2023 = df_test_2023.merge(df_2021, on="Team", how="left")
df_test_2023 = df_test_2023.rename(columns={
    "Team": "Away",
    "Wins": "Away_Wins_Second_Last"
})
df_test_2023 = df_test_2023.rename(columns={"Home":"Team"})
df_test_2023 = df_test_2023.merge(df_2021, on="Team", how="left")
df_test_2023 = df_test_2023.rename(columns={
    "Team": "Home",
    "Wins": "Home_Wins_Second_Last"
})
df_test_2023 = df_test_2023[["Home","Away","Home_Wins_Last","Away_Wins_Last","Home_Wins_Second_Last","Away_Wins_Second_Last","Game_Outcome"]]
df_test_2023 = df_test_2023.rename(columns={"Away":"Team"})
df_test_2023 = df_test_2023.merge(df_2020, on="Team", how="left")
df_test_2023 = df_test_2023.rename(columns={
    "Team": "Away",
    "Wins": "Away_Wins_Third_Last"
})
df_test_2023 = df_test_2023.rename(columns={"Home":"Team"})
df_test_2023 = df_test_2023.merge(df_2020, on="Team", how="left")
df_test_2023 = df_test_2023.rename(columns={
    "Team": "Home",
    "Wins": "Home_Wins_Third_Last"
})
df_test_2023 = df_test_2023[["Home","Away","Home_Wins_Last","Away_Wins_Last","Home_Wins_Second_Last","Away_Wins_Second_Last","Home_Wins_Third_Last","Away_Wins_Third_Last","Game_Outcome"]]

df_test_2023["Game_Outcome"] = df_test_2023.apply(lambda row: 1 if row["Game_Outcome"] == "Home" else 0, axis=1)

df_test_2023 = parse_season(df_test_2023, "2023-2024")