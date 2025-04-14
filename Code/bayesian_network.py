import numpy as np
import pandas as pd
import os
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination
from training_2022 import df_train_2022
from training_2021 import df_train_2021
from training_2020 import df_train_2020

'''
def bin_wins(wins):
    if wins < 21:
        return "Very Low"
    elif 21 <= wins < 31:
        return "Low"
    elif 31 <= wins < 41:
        return "Medium Low"
    elif 41 <= wins < 51:
        return "Medium High"
    elif 51 <= wins < 61:
        return "High"
    else:
        return "Very High"
'''
def bin_wins(wins):
    if wins < 21:
        return 0
    elif 21 <= wins < 31:
        return 1
    elif 31 <= wins < 41:
        return 2
    elif 41 <= wins < 51:
        return 3
    elif 51 <= wins < 61:
        return 4
    else:
        return 5


train_df = pd.concat([df_train_2020, df_train_2021, df_train_2022])
train_df = train_df.reset_index()

'''
assert set(["Home_Wins_Last", "Home_Wins_Second_Last", "Home_Wins_Third_Last", "Away_Wins_Last", "Away_Wins_Second_Last", "Away_Wins_Third_Last", "Game_Outcome"]).issubset(train_df.columns)

train_df["Home_Wins_Last"]= train_df["Home_Wins_Last"].astype(pd.CategoricalDtype(categories=range(83)))
train_df["Home_Wins_Second_Last"]= train_df["Home_Wins_Second_Last"].astype(pd.CategoricalDtype(categories=range(83)))
train_df["Home_Wins_Third_Last"]= train_df["Home_Wins_Third_Last"].astype(pd.CategoricalDtype(categories=range(83)))
train_df["Away_Wins_Last"]= train_df["Away_Wins_Last"].astype(pd.CategoricalDtype(categories=range(83)))
train_df["Away_Wins_Second_Last"]= train_df["Away_Wins_Second_Last"].astype(pd.CategoricalDtype(categories=range(83)))
train_df["Away_Wins_Third_Last"]= train_df["Away_Wins_Third_Last"].astype(pd.CategoricalDtype(categories=range(83)))
train_df["Game_Outcome"] = train_df["Game_Outcome"].astype("category")
'''

cols_to_transform = train_df.columns[train_df.columns != "Game_Outcome"]
train_df[cols_to_transform] = train_df[cols_to_transform].applymap(bin_wins)

print(train_df)

state_names = {
    "Home_Wins_Last": list(range(6)),
    "Home_Wins_Second_Last": list(range(6)),
    "Home_Wins_Third_Last": list(range(6)),
    "Away_Wins_Last": list(range(6)),
    "Away_Wins_Second_Last": list(range(6)),
    "Away_Wins_Third_Last": list(range(6)),
    "Game_Outcome": [0, 1]
}

model = DiscreteBayesianNetwork([
    ("Home_Wins_Last", "Game_Outcome"),
    ("Home_Wins_Second_Last", "Game_Outcome"),
    ("Home_Wins_Third_Last", "Game_Outcome"),
    ("Away_Wins_Last", "Game_Outcome"),
    ("Away_Wins_Second_Last", "Game_Outcome"),
    ("Away_Wins_Third_Last", "Game_Outcome")
])

model.fit(data=train_df, estimator=BayesianEstimator, prior_type="dirichlet", pseudo_counts=1, state_names=state_names)

infer = VariableElimination(model)

query_result = infer.query(
    variables=['Game_Outcome'],
    evidence={
        "Home_Wins_Last": 5,
        "Home_Wins_Second_Last": 5,
        "Home_Wins_Third_Last": 5,
        "Away_Wins_Last": 2,
        "Away_Wins_Second_Last": 2,
        "Away_Wins_Third_Last": 2
    }
)

print(query_result)