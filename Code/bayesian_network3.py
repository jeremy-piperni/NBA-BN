import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from training_2022 import df_train_2022
from training_2021 import df_train_2021
from training_2020 import df_train_2020
from testing_2023 import df_test_2023

num_bins = 3

def bin_previous_wins(wins, bins):
    if bins == 3:
        if wins < 36:
            return "Low"
        elif 36 <= wins < 46:
            return "Medium"
        else:
            return "High"
    else:
        if wins < 32:
            return "Very Low"
        elif 32 <= wins < 36:
            return "Low"
        elif 36 <= wins < 40:
            return "Medium Low"
        elif 40 <= wins < 46:
            return "Medium High"
        elif 46 <= wins < 51:
            return "High"
        else:
            return "Very High"

def bin_current_strength(form_perc, bins):
    if bins == 3:
        if form_perc < 0.333:
            return "Low"
        elif 0.333 <= form_perc < 0.482:
            return "Medium"
        else:
            return "High"

def bin_team_form(form, bins):
    if bins == 3:
        if form < 0:
            return "Low"
        elif 0 <= form < 0.17:
            return "Medium"
        else:
            return "High"

def bin_fatigue(fatigue):
    if fatigue == 1:
        return "High"
    elif fatigue == 2:
        return "Medium"
    else:
        return "Low"

def average_wins(w1, w2, w3):
    return round((0.5 * w1) + (0.3 * w2) + (0.2 * w3))

def compute_cur_str(win_perc_current, win_perc_against, streak):
    streak = streak / 10
    return (0.7 * win_perc_current) + (0.2 * win_perc_against) + (0.1 * streak)


train_df = pd.concat([df_train_2020, df_train_2021, df_train_2022])
train_df = train_df.reset_index(drop=True)

train_df["Home_Average_Past_Wins"] = train_df.apply(lambda row: average_wins(row["Home_Wins_Last"], row["Home_Wins_Second_Last"], row["Home_Wins_Third_Last"]), axis=1)
train_df["Away_Average_Past_Wins"] = train_df.apply(lambda row: average_wins(row["Away_Wins_Last"], row["Away_Wins_Second_Last"], row["Away_Wins_Third_Last"]), axis=1)

train_df["Home_Current_Strength"] = train_df.apply(lambda row: compute_cur_str(row["Home_Current_Wins"], row["Home_Wins_Against"], row["Home_Streak"]), axis=1)
train_df["Away_Current_Strength"] = train_df.apply(lambda row: compute_cur_str(row["Away_Current_Wins"], row["Away_Wins_Against"], row["Away_Streak"]), axis=1)

bins = pd.qcut(train_df["Home_Current_Strength"], q=3)
print(bins)

train_df = train_df[["Game_Outcome","Home_Average_Past_Wins","Away_Average_Past_Wins","Home_Current_Strength","Away_Current_Strength","Home_Fatigue","Away_Fatigue"]]
cols_to_map = ["Home_Average_Past_Wins","Away_Average_Past_Wins"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_previous_wins(x, num_bins))
cols_to_map = ["Home_Current_Strength", "Away_Current_Strength"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_current_strength(x, num_bins))
cols_to_map = ["Home_Fatigue", "Away_Fatigue"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_fatigue(x))

print(train_df)

model = DiscreteBayesianNetwork([
    ("Home_Average_Past_Wins", "Game_Outcome"),
    ("Home_Current_Strength", "Game_Outcome"),
    ("Away_Average_Past_Wins", "Game_Outcome"),
    ("Away_Current_Strength", "Game_Outcome"),
    ("Home_Fatigue", "Game_Outcome"),
    ("Away_Fatigue", "Game_Outcome")
])

model.fit(data=train_df, estimator=MaximumLikelihoodEstimator)
results = pd.DataFrame(columns=["Game_Outcome","Probability_Home","Probability_Away"])

infer = VariableElimination(model)

for index, row in df_test_2023.iterrows():
    home_wins_last = row["Home_Wins_Last"]
    home_wins_second_last = row["Home_Wins_Second_Last"]
    home_wins_third_last = row["Home_Wins_Third_Last"]
    away_wins_last = row["Away_Wins_Last"]
    away_wins_second_last = row["Away_Wins_Second_Last"]
    away_wins_third_last = row["Away_Wins_Third_Last"]
    home_wins_against = row["Home_Wins_Against"]
    away_wins_against = row["Away_Wins_Against"]
    home_current_wins = row["Home_Current_Wins"]
    away_current_wins = row["Away_Current_Wins"]
    home_fatigue = row["Home_Fatigue"]
    away_fatigue = row["Away_Fatigue"]
    home_streak = row["Home_Streak"]
    away_streak = row["Away_Streak"]

    home_average_past_wins = average_wins(home_wins_last,home_wins_second_last,home_wins_third_last)
    away_average_past_wins = average_wins(away_wins_last,away_wins_second_last,away_wins_third_last)
    home_current_strength = compute_cur_str(home_current_wins,home_wins_against,home_streak)
    away_current_strength = compute_cur_str(away_current_wins,away_wins_against,away_streak)

    home_average_past_wins = bin_previous_wins(home_average_past_wins, num_bins)
    away_average_past_wins = bin_previous_wins(away_average_past_wins, num_bins)
    home_current_strength = bin_current_strength(home_current_strength, num_bins)
    away_current_strength = bin_current_strength(away_current_strength, num_bins)

    home_fatigue = bin_fatigue(home_fatigue)
    away_fatigue = bin_fatigue(away_fatigue)

    query_result = infer.query(
        variables=['Game_Outcome'],
        evidence={
            "Home_Average_Past_Wins": home_average_past_wins,
            "Away_Average_Past_Wins": away_average_past_wins,
            "Home_Current_Strength": home_current_strength,
            "Away_Current_Strength": away_current_strength,
            "Home_Fatigue": home_fatigue,
            "Away_Fatigue": away_fatigue
        }
    )
    probabilities = query_result.values
    probability_home = probabilities[1]
    probability_away = probabilities[0]
    game_outcome = row["Game_Outcome"]
    new_row = {"Game_Outcome": game_outcome, "Probability_Home": probability_home, "Probability_Away": probability_away}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

print(results)
results["Correct_Prediction"] = results.apply(lambda row: 1 if (row["Probability_Home"] >= 0.5 and row["Game_Outcome"] == 1) or (row["Probability_Away"] > 0.5 and row["Game_Outcome"] == 0) else 0, axis=1)
print("50% Accuracy: " + str(results["Correct_Prediction"].mean() * 100) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.55) | (results["Probability_Away"] >= 0.55)]
print("55% Accuracy: " + str(results["Correct_Prediction"].mean() * 100) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.6) | (results["Probability_Away"] >= 0.6)]
print("60% Accuracy: " + str(results["Correct_Prediction"].mean() * 100) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.65) | (results["Probability_Away"] >= 0.65)]
print("65% Accuracy: " + str(results["Correct_Prediction"].mean() * 100) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.7) | (results["Probability_Away"] >= 0.7)]
print("70% Accuracy: " + str(results["Correct_Prediction"].mean() * 100) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.75) | (results["Probability_Away"] >= 0.75)]
print("75% Accuracy: " + str(results["Correct_Prediction"].mean() * 100) + ", Count: " + str(results["Correct_Prediction"].count()))