import numpy as np
import pandas as pd
from testing_2023 import df_test_2023

# Calculate Accuracy based on predicting the home team will win
print("Home only Accuracy: " + str(round(df_test_2023["Game_Outcome"].mean() * 100, 2)))

# Calculate Accuracy based solely on what team has a better current win %
df_test_2023["Win_Only_Temp"] = df_test_2023.apply(lambda row: 1 if row["Home_Current_Wins"] >= row["Away_Current_Wins"] else 0, axis=1)
df_test_2023["Win_Only"] = df_test_2023.apply(lambda row: 1 if row["Game_Outcome"] == row["Win_Only_Temp"] else 0, axis=1)
print("Win only Accuracy: " + str(round(df_test_2023["Win_Only"].mean() * 100, 2)))