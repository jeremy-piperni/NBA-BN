import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from training_2022 import df_train_2022
from training_2021 import df_train_2021
from training_2020 import df_train_2020
from testing_2023 import df_test_2023

prev_bins = 4
cur_bins = 4
streak_bins = 4
fatigue_bins = 4

def bin_previous_wins(wins, bins):
    if bins == 2:
        if wins < 0:
            return "Low"
        else:
            return "High"
    if bins == 3:
        if wins < -6:
            return "Low"
        elif -6 <= wins < 6:
            return "Medium"
        else:
            return "High"
    if bins == 4:
        if wins < -10:
            return "Low"
        elif -10 <= wins < 0:
            return "Medium Low"
        elif 0 <= wins < 10:
            return "Medium High"
        else:
            return "High"

def bin_current_strength(form_perc, bins):
    if bins == 2:
        if form_perc < 0.019:
            return "Low"
        else:
            return "High"
    if bins == 3:
        if form_perc < -0.055:
            return "Low"
        elif -0.055 <= form_perc < 0.127:
            return "Medium"
        else:
            return "High"
    if bins == 4:
        if form_perc < -0.107:
            return "Low"
        elif -0.107 <= form_perc < 0.019:
            return "Medium Low"
        elif 0.019 <= form_perc < 0.187:
            return "Medium High"
        else:
            return "High"

def bin_streak(streak, bins):
    if bins == 2:
        if streak < 0:
            return "Low"
        else:
            return "High"
    if bins == 3:
        if streak < -2:
            return "Low"
        elif -2 <= streak < 2:
            return "Medium"
        else:
            return "High"
    if bins == 4:
        if streak < -3:
            return "Low"
        elif -3 <= streak < 0:
            return "Medium Low"
        elif 0 <= streak < 3:
            return "Medium High"
        else:
            return "High"

def bin_fatigue(fatigue, bins):
    if bins == 2:
        if fatigue < 0:
            return "High"
        else:
            return "Low"
    if bins == 3:
        if fatigue < 0:
            return "High"
        elif 0 <= fatigue < 1:
            return "Medium"
        else:
            return "Low"
    if bins == 4:
        if fatigue < 2:
            return "High"
        elif 2 <= fatigue < 0:
            return "Medium High"
        elif 0 <= fatigue < 2:
            return "Medium Low"
        else:
            return "Low"

def average_wins(w1, w2, w3):
    return round((0.5 * w1) + (0.3 * w2) + (0.2 * w3))

def compute_cur_str(win_perc_current, win_perc_against):
    return (0.8 * win_perc_current) + (0.2 * win_perc_against)


train_df = pd.concat([df_train_2020, df_train_2021, df_train_2022])
train_df = train_df.reset_index(drop=True)

train_df["Home_Average_Past_Wins"] = train_df.apply(lambda row: average_wins(row["Home_Wins_Last"], row["Home_Wins_Second_Last"], row["Home_Wins_Third_Last"]), axis=1)
train_df["Away_Average_Past_Wins"] = train_df.apply(lambda row: average_wins(row["Away_Wins_Last"], row["Away_Wins_Second_Last"], row["Away_Wins_Third_Last"]), axis=1)

train_df["Home_Current_Strength"] = train_df.apply(lambda row: compute_cur_str(row["Home_Current_Wins"], row["Home_Wins_Against"]), axis=1)
train_df["Away_Current_Strength"] = train_df.apply(lambda row: compute_cur_str(row["Away_Current_Wins"], row["Away_Wins_Against"]), axis=1)

train_df["Dif_Average_Past_Wins"] = train_df["Home_Average_Past_Wins"] - train_df["Away_Average_Past_Wins"]
train_df["Dif_Current_Strength"] = train_df["Home_Current_Strength"] - train_df["Away_Current_Strength"]
train_df["Dif_Streak"] = train_df["Home_Streak"] - train_df["Away_Streak"]
train_df["Dif_Fatigue"] = train_df["Home_Fatigue"] - train_df["Away_Fatigue"]

train_df = train_df[["Game_Outcome","Dif_Average_Past_Wins","Dif_Current_Strength","Dif_Streak","Dif_Fatigue"]]
cols_to_map = ["Dif_Average_Past_Wins"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_previous_wins(x, prev_bins))
cols_to_map = ["Dif_Current_Strength"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_current_strength(x, cur_bins))
cols_to_map = ["Dif_Streak"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_streak(x, streak_bins))
cols_to_map = ["Dif_Fatigue"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_fatigue(x, fatigue_bins))

print(train_df)

model = DiscreteBayesianNetwork([
    ("Dif_Average_Past_Wins", "Game_Outcome"),
    ("Dif_Current_Strength", "Game_Outcome"),
    ("Dif_Streak", "Game_Outcome"),
    ("Dif_Fatigue", "Game_Outcome")
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
    home_current_strength = compute_cur_str(home_current_wins,home_wins_against)
    away_current_strength = compute_cur_str(away_current_wins,away_wins_against)

    dif_average_past_wins = home_average_past_wins - away_average_past_wins
    dif_current_strength = home_current_strength - away_current_strength
    dif_streak = home_streak - away_streak
    dif_fatigue = home_fatigue - away_fatigue

    dif_average_past_wins = bin_previous_wins(dif_average_past_wins, prev_bins)
    dif_current_strength = bin_current_strength(dif_current_strength, cur_bins)
    dif_streak = bin_streak(dif_streak, streak_bins)
    dif_fatigue = bin_fatigue(dif_fatigue, fatigue_bins)

    query_result = infer.query(
        variables=['Game_Outcome'],
        evidence={
            "Dif_Average_Past_Wins": dif_average_past_wins,
            "Dif_Current_Strength": dif_current_strength,
            "Dif_Streak": dif_streak,
            "Dif_Fatigue": dif_fatigue
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
print("50% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.55) | (results["Probability_Away"] >= 0.55)]
print("55% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.6) | (results["Probability_Away"] >= 0.6)]
print("60% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.65) | (results["Probability_Away"] >= 0.65)]
print("65% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.7) | (results["Probability_Away"] >= 0.7)]
print("70% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.75) | (results["Probability_Away"] >= 0.75)]
print("75% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.80) | (results["Probability_Away"] >= 0.80)]
print("80% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.85) | (results["Probability_Away"] >= 0.85)]
print("85% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.90) | (results["Probability_Away"] >= 0.90)]
print("90% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))

results = results[(results["Probability_Home"] >= 0.95) | (results["Probability_Away"] >= 0.95)]
print("95% Accuracy: " + str(round(results["Correct_Prediction"].mean() * 100, 2)) + ", Count: " + str(results["Correct_Prediction"].count()))