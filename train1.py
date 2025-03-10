import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv('2023-2024 NBA Player Stats - Playoffs.csv', delimiter=';')

# Features and target
X = data[['PTS', 'TRB', 'AST', 'FG%', '3P%', 'FT%']]
y = data['PTS']  # Predicting PTS as an example

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest models
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the models
joblib.dump(model, 'player_stats_model.pkl')