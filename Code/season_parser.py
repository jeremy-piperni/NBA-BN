import numpy as np
import pandas as pd
import os

def parse_season(df, year):
    # Get the games dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    new_path = os.path.join(script_dir, os.pardir, "Data/Cleaned/Team Stats", year + "_Games.csv")
    new_path = os.path.abspath(new_path)
    df_games = pd.read_csv(new_path)

    teams = ["ATL","BOS","BRK","CHI","CHO","CLE","DAL","DEN","DET","GSW","HOU","IND","LAC","LAL","MEM","MIA","MIL","MIN","NOP","NYK","OKC","ORL","PHI","PHO","POR","SAC","SAS","TOR","UTA","WAS"]

    streak = {team: [0] for team in df_games["Home"].unique()}
    fatigue = {team: [] for team in df_games["Home"].unique()}
    win_counts = {team: [0] for team in df_games["Home"].unique()}
    head_to_head = pd.DataFrame([[[0]] * len(teams) for _ in range(len(teams))], index=teams, columns=teams)

    for team in teams:
        days_last_game = 5
        cur_streak = 0
        for index, row in df_games.iterrows():
            if index == 0:
                date = row["Day"]
            else:
                new_date = row["Day"]
                if new_date != date:
                    date = new_date
                    days_last_game = days_last_game + 1
            if row["Home"] == team:
                fatigue[team].append(days_last_game)
                days_last_game = 0
                opp_team = row["Away"]
                game_outcome = row["Game_Outcome"]
                current_list = head_to_head.loc[team,opp_team].copy()
                if game_outcome == "Home":
                    win_counts[team].append(win_counts[team][-1] + 1)
                    new_value = current_list[-1] + 1
                    current_list.append(new_value)
                    head_to_head.loc[team,opp_team] = current_list
                    if cur_streak >= 0:
                        cur_streak = cur_streak + 1
                    else:
                        cur_streak = 1
                else:
                    win_counts[team].append(win_counts[team][-1])
                    new_value = current_list[-1]
                    current_list.append(new_value)
                    head_to_head.loc[team,opp_team] = current_list
                    if cur_streak >= 0:
                        cur_streak = -1
                    else:
                        cur_streak = cur_streak - 1
                streak[team].append(cur_streak)
            elif row["Away"] == team:
                fatigue[team].append(days_last_game)
                days_last_game = 0
                opp_team = row["Home"]
                game_outcome = row["Game_Outcome"]
                current_list = head_to_head.loc[team,opp_team].copy()
                if game_outcome == "Away":
                    win_counts[team].append(win_counts[team][-1] + 1)
                    new_value = current_list[-1] + 1
                    current_list.append(new_value)
                    head_to_head.loc[team,opp_team] = current_list
                    if cur_streak >= 0:
                        cur_streak = cur_streak + 1
                    else:
                        cur_streak = 1
                else:
                    win_counts[team].append(win_counts[team][-1])
                    new_value = current_list[-1]
                    current_list.append(new_value)
                    head_to_head.loc[team,opp_team] = current_list
                    if cur_streak >= 0:
                        cur_streak = -1
                    else:
                        cur_streak = cur_streak - 1
                streak[team].append(cur_streak)

    game_counts = {team: 0 for team in df_games["Home"].unique()}
    head_to_head_counts = pd.DataFrame(index=df_games["Home"].unique(), columns=df_games["Home"].unique())
    head_to_head_counts[:] = 0

    df["Home_Fatigue"] = 0
    df["Away_Fatigue"] = 0
    df["Home_Streak"] = 0
    df["Away_Streak"] = 0
    df["Home_Wins_Against"] = 0.0
    df["Away_Wins_Against"] = 0.0
    df["Home_Current_Wins"] = 0.0
    df["Away_Current_Wins"] = 0.0
    for index, row in df.iterrows():
        home_team = row["Home"]
        away_team = row["Away"]
        df.loc[index,"Home_Fatigue"] = fatigue[home_team][game_counts[home_team]]
        df.loc[index,"Away_Fatigue"] = fatigue[away_team][game_counts[away_team]]
        df.loc[index,"Home_Streak"] = streak[home_team][game_counts[home_team]]
        df.loc[index,"Away_Streak"] = streak[away_team][game_counts[away_team]]
        if game_counts[home_team] == 0:
            df.loc[index,"Home_Current_Wins"] = 0
        else:
            df.loc[index,"Home_Current_Wins"] = win_counts[home_team][game_counts[home_team]] / game_counts[home_team]
        if game_counts[away_team] == 0:
            df.loc[index,"Away_Current_Wins"] = 0
        else:
            df.loc[index,"Away_Current_Wins"] = win_counts[away_team][game_counts[away_team]] / game_counts[away_team]
        home_wins_against = head_to_head.loc[home_team,away_team][head_to_head_counts.loc[home_team,away_team]]
        away_wins_against = head_to_head.loc[away_team,home_team][head_to_head_counts.loc[away_team,home_team]]
        if home_wins_against == 0:
            df.loc[index,"Home_Wins_Against"] = 0
            df.loc[index,"Away_Wins_Against"] = 0
        else:
            df.loc[index,"Home_Wins_Against"] = home_wins_against / head_to_head_counts.loc[home_team,away_team]
            df.loc[index,"Away_Wins_Against"] = away_wins_against / head_to_head_counts.loc[away_team,home_team]
        head_to_head_counts.loc[home_team,away_team] += 1
        head_to_head_counts.loc[away_team,home_team] += 1
        game_counts[home_team] += 1
        game_counts[away_team] += 1

    return df

    