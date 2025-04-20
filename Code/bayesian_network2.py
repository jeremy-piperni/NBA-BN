import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from training_2022 import df_train_2022
from training_2021 import df_train_2021
from training_2020 import df_train_2020
from testing_2023 import df_test_2023

prev_bins = 2
cur_bins = 4

def bin_previous_wins(wins, bins):
    if bins == 2:
        if wins < 40:
            return "Low"
        else:
            return "High"
    if bins == 3:
        if wins < 36:
            return "Low"
        elif 36 <= wins < 46:
            return "Medium"
        else:
            return "High"
    if bins == 4:
        if wins < 34:
            return "Low"
        elif 34 <= wins < 40:
            return "Medium Low"
        elif 40 <= wins < 49:
            return "Medium High"
        else:
            return "High"
    if bins == 5:
        if wins < 33:
            return "Very Low"
        elif 33 <= wins < 36:
            return "Low"
        elif 36 <= wins < 45:
            return "Medium"
        elif 45 <= wins < 50:
            return "High"
        else:
            return "Very High"
    if bins == 6:
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
    if bins == 2:
        if form_perc < 0.43:
            return "Low"
        else:
            return "High"
    if bins == 3:
        if form_perc < 0.378:
            return "Low"
        elif 0.378 <= form_perc < 0.5:
            return "Medium"
        else:
            return "High"
    if bins == 4:
        if form_perc < 0.34:
            return "Low"
        elif 0.34 <= form_perc < 0.43:
            return "Medium Low"
        elif 0.34 <= form_perc < 0.54:
            return "Medium High"
        else:
            return "High"
    if bins == 5:
        if form_perc < 0.312:
            return "Very Low"
        elif 0.312 <= form_perc < 0.395:
            return "Low"
        elif 0.395 <= form_perc < 0.475:
            return "Medium"
        elif 0.475 <= form_perc < 0.56:
            return "High"
        else:
            return "Very High"
    if bins == 6:
        if form_perc < 0.288:
            return "Very Low"
        elif 0.288 <= form_perc < 0.378:
            return "Low"
        elif 0.378 <= form_perc < 0.43:
            return "Medium Low"
        elif 0.43 <= form_perc < 0.5:
            return "Medium High"
        elif 0.5 <= form_perc < 0.58:
            return "High"
        else:
            return "Very High"

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

bins = pd.qcut(train_df["Home_Average_Past_Wins"], q=6)
print(bins)
bins = pd.qcut(train_df["Away_Average_Past_Wins"], q=6)
print(bins)

bins = pd.qcut(train_df["Home_Current_Strength"], q=6)
print(bins)
bins = pd.qcut(train_df["Away_Current_Strength"], q=6)
print(bins)

train_df = train_df[["Game_Outcome","Home_Average_Past_Wins","Away_Average_Past_Wins","Home_Current_Strength","Away_Current_Strength"]]
cols_to_map = ["Home_Average_Past_Wins","Away_Average_Past_Wins"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_previous_wins(x, prev_bins))
cols_to_map = ["Home_Current_Strength", "Away_Current_Strength"]
train_df[cols_to_map] = train_df[cols_to_map].map(lambda x: bin_current_strength(x, cur_bins))

print(train_df)

model = DiscreteBayesianNetwork([
    ("Home_Average_Past_Wins", "Game_Outcome"),
    ("Home_Current_Strength", "Game_Outcome"),
    ("Away_Average_Past_Wins", "Game_Outcome"),
    ("Away_Current_Strength", "Game_Outcome")
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

    home_average_past_wins = bin_previous_wins(home_average_past_wins, prev_bins)
    away_average_past_wins = bin_previous_wins(away_average_past_wins, prev_bins)
    home_current_strength = bin_current_strength(home_current_strength, cur_bins)
    away_current_strength = bin_current_strength(away_current_strength, cur_bins)

    query_result = infer.query(
        variables=['Game_Outcome'],
        evidence={
            "Home_Average_Past_Wins": home_average_past_wins,
            "Away_Average_Past_Wins": away_average_past_wins,
            "Home_Current_Strength": home_current_strength,
            "Away_Current_Strength": away_current_strength
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


# Queries
# Upset Query, if home is currently stronger what are the chances that the away team can cause an upset
query_result = infer.query(
        variables=['Game_Outcome'],
        evidence={
            "Home_Current_Strength": "High",
            "Away_Current_Strength": "Low"
        }
    )
print("Upset Query")
print(query_result)

# Query, how strong is the away team when the home team wins
query_result = infer.query(
        variables=['Away_Current_Strength'],
        evidence={
            "Game_Outcome": 1
        }
    )
print("Away Current Strength when Home Team Wins Query")
print(query_result)

# Query, how strong is the home team when the away team wins
query_result = infer.query(
        variables=['Home_Current_Strength'],
        evidence={
            "Game_Outcome": 0
        }
    )

print("Home Current Strength when Away Team Wins Query")
print(query_result)

# Query, win prediction if the home team was strong in the past, but is currently weak
query_result = infer.query(
        variables=['Game_Outcome'],
        evidence={
            "Home_Average_Past_Wins": "High",
            "Home_Current_Strength": "Low"
        }
    )

print("Home Team Previously Strong but Currently Weak Query")
print(query_result)
