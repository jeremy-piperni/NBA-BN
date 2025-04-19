import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from training_2022 import df_train_2022
from training_2021 import df_train_2021
from training_2020 import df_train_2020
from testing_2023 import df_test_2023

train_df = pd.concat([df_train_2020, df_train_2021, df_train_2022])
train_df = train_df.reset_index(drop=True)

X_train = train_df.drop(columns=["Home","Away","Game_Outcome"])
y_train = train_df["Game_Outcome"]

scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)

X_test = df_test_2023.drop(columns=["Home","Away","Game_Outcome"])
y_true = df_test_2023["Game_Outcome"]

X_test_scaled = scaler.transform(X_test)

clf = LogisticRegression(random_state=0, solver="liblinear", penalty="l2").fit(X_train_scaled,y_train)
y_pred = clf.predict(X_test_scaled)

print("Logistic Regression Accuracy: " + str(round(accuracy_score(y_true, y_pred) * 100, 2)))

xgb_model = xgb.XGBClassifier(eval_metric='logloss')

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)

grid_search.fit(X_train_scaled, y_train)

results = pd.DataFrame(grid_search.cv_results_)
results = results[['params', 'mean_test_score', 'std_test_score']]

print(results.to_string(index=False))


print("Best parameters:", grid_search.best_params_)
print("Best cross-validation accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
print("XGBoost Accuracy: " + str(round(accuracy_score(y_true, y_pred) * 100, 2)))


