import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from training_2022 import df_train_2022
from training_2021 import df_train_2021
from training_2020 import df_train_2020
from testing_2023 import df_test_2023

num_bins = 2

def bin_previous_wins(wins, bins):
    if bins == 2:
        if wins < 43:
            return "Low"
        else:
            return "High"
    if bins == 3:
        if wins < 35:
            return "Low"
        elif 35 <= wins < 49:
            return "Medium"
        else:
            return "High"

def bin_fatigue(days, bins):
    if bins == 2:
        if days < 2:
            return "High"
        else:
            return "Low"
    if bins == 3:
        if days == 1:
            return "High"
        elif days == 2:
            return "Medium"
        else:
            return "Low"

def bin_streak(streak, bins):
    if bins == 2:
        if streak < 0:
            return "Bad"
        else:
            return "Good"
    if bins == 3:
        if streak < -1:
            return "Bad"
        elif -1 <= streak < 2:
            return "Medium"
        else:
            return "Good"
        
def bin_head_to_head(win_perc, bins):
    if bins == 2:
        if win_perc < 0.5:
            return "Bad"
        else:
            return "Good"
    if bins == 3:
        if win_perc < .34:
            return "Bad"
        elif .33 <= win_perc < .66:
            return "Medium"
        else:
            return "Good"
        
def bin_current_wins(win_perc, bins):
    if bins == 2:
        if win_perc < 0.5:
            return "Bad"
        else:
            return "Good"
    if bins == 3:
        if win_perc < 0.448:
            return "Bad"
        elif .448 <= win_perc < 0.56:
            return "Medium"
        else:
            return "Good"

train_df = pd.concat([df_train_2020, df_train_2021, df_train_2022])
train_df = train_df.reset_index(drop=True)

cols_to_map = ["Home_Wins_Last","Away_Wins_Last","Home_Wins_Second_Last","Away_Wins_Second_Last","Home_Wins_Third_Last","Away_Wins_Third_Last"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_previous_wins(x, num_bins))
cols_to_map = ["Home_Fatigue", "Away_Fatigue"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_fatigue(x, num_bins))
cols_to_map = ["Home_Streak", "Away_Streak"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_streak(x, num_bins))
cols_to_map = ["Home_Wins_Against", "Away_Wins_Against"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_head_to_head(x, num_bins))
cols_to_map = ["Home_Current_Wins", "Away_Current_Wins"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_current_wins(x, num_bins))

model = DiscreteBayesianNetwork([
    ("Home_Wins_Last", "Game_Outcome"),
    ("Away_Wins_Last", "Game_Outcome"),
    ("Home_Wins_Second_Last", "Game_Outcome"),
    ("Away_Wins_Second_Last", "Game_Outcome"),
    ("Home_Wins_Third_Last", "Game_Outcome"),
    ("Away_Wins_Third_Last", "Game_Outcome"),
    ("Home_Fatigue", "Game_Outcome"),
    ("Away_Fatigue", "Game_Outcome"),
    ("Home_Streak", "Game_Outcome"),
    ("Away_Streak", "Game_Outcome"),
    ("Home_Wins_Against", "Game_Outcome"),
    ("Away_Wins_Against", "Game_Outcome"),
    ("Home_Current_Wins", "Game_Outcome"),
    ("Away_Current_Wins", "Game_Outcome")
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
    home_fatigue = row["Home_Fatigue"]
    away_fatigue = row["Away_Fatigue"]
    home_streak = row["Home_Streak"]
    away_streak = row["Away_Streak"]
    home_wins_against = row["Home_Wins_Against"]
    away_wins_against = row["Away_Wins_Against"]
    home_current_wins = row["Home_Current_Wins"]
    away_current_wins = row["Away_Current_Wins"]

    home_wins_last = bin_previous_wins(home_wins_last,num_bins)
    away_wins_last = bin_previous_wins(away_wins_last,num_bins)
    home_wins_second_last = bin_previous_wins(home_wins_second_last,num_bins)
    away_wins_second_last = bin_previous_wins(away_wins_second_last,num_bins)
    home_wins_third_last = bin_previous_wins(home_wins_third_last,num_bins)
    away_wins_third_last = bin_previous_wins(away_wins_third_last,num_bins)
    home_fatigue = bin_fatigue(home_fatigue, num_bins)
    away_fatigue = bin_fatigue(away_fatigue, num_bins)
    home_streak = bin_streak(home_streak, num_bins)
    away_streak = bin_streak(away_streak, num_bins)
    home_wins_against = bin_head_to_head(home_wins_against, num_bins)
    away_wins_against = bin_head_to_head(away_wins_against, num_bins)
    home_current_wins = bin_current_wins(home_current_wins, num_bins)
    away_current_wins = bin_current_wins(away_current_wins, num_bins)

    query_result = infer.query(
        variables=['Game_Outcome'],
        evidence={
            "Home_Wins_Last": home_wins_last,
            "Away_Wins_Last": away_wins_last,
            "Home_Wins_Second_Last": home_wins_second_last,
            "Away_Wins_Second_Last": away_wins_second_last,
            "Home_Wins_Third_Last": home_wins_third_last,
            "Away_Wins_Third_Last": away_wins_third_last,
            "Home_Fatigue": home_fatigue,
            "Away_Fatigue": away_fatigue,
            "Home_Streak": home_streak,
            "Away_Streak": away_streak,
            "Home_Wins_Against": home_wins_against,
            "Away_Wins_Against": away_wins_against,
            "Home_Current_Wins": home_current_wins,
            "Away_Current_Wins": away_current_wins
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
