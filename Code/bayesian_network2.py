import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from training_2022 import df_train_2022
from training_2021 import df_train_2021
from training_2020 import df_train_2020


def bin_wins(wins):
    if wins < 32:
        return "Very Low"
    elif 32 <= wins < 36:
        return "Low"
    elif 36 <= wins < 40:
        return "Medium Low"
    elif 40 <= wins < 47:
        return "Medium High"
    elif 47 <= wins < 51:
        return "High"
    else:
        return "Very High"

def average_wins(w1, w2, w3):
    average = (w1 + w2 + w3) / 3
    return average

train_df = pd.concat([df_train_2020, df_train_2021, df_train_2022])
train_df = train_df.reset_index(drop=True)

print(train_df)

train_df["Home_Average_Past_Wins"] = train_df.apply(lambda row: average_wins(row["Home_Wins_Last"], row["Home_Wins_Second_Last"], row["Home_Wins_Third_Last"]), axis=1)
train_df["Away_Average_Past_Wins"] = train_df.apply(lambda row: average_wins(row["Away_Wins_Last"], row["Away_Wins_Second_Last"], row["Away_Wins_Third_Last"]), axis=1)

bins =  pd.qcut(train_df["Home_Average_Past_Wins"], q=6)
print(bins)

train_df = train_df[["Game_Outcome","Home_Average_Past_Wins","Away_Average_Past_Wins"]]
cols_to_transform = train_df.columns[train_df.columns != "Game_Outcome"]
train_df[cols_to_transform] = train_df[cols_to_transform].applymap(bin_wins)

print(train_df)

model = DiscreteBayesianNetwork([
    ("Home_Average_Past_Wins", "Game_Outcome"),
    ("Away_Average_Past_Wins", "Game_Outcome")
])

model.fit(data=train_df, estimator=MaximumLikelihoodEstimator)

infer = VariableElimination(model)

query_result = infer.query(
    variables=['Game_Outcome'],
    evidence={
        "Home_Average_Past_Wins": "High",
        "Away_Average_Past_Wins": "High"
    }
)

print(query_result)