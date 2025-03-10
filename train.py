import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load the dataset
data = pd.read_csv('2023-2024 NBA Player Stats - Playoffs.csv', delimiter=';')

# Group by team and aggregate player stats
team_stats = data.groupby('Tm').agg({
    'PTS': 'sum',
    'TRB': 'sum',
    'AST': 'sum',
    'FG%': 'mean',
    '3P%': 'mean',
    'FT%': 'mean'
}).reset_index()

# Assuming we have a target variable 'Rank' (this would need to be provided or calculated)
# For demonstration, let's create a dummy rank based on total points
team_stats['Rank'] = team_stats['PTS'].rank(ascending=False)

# Features and target
X = team_stats[['PTS', 'TRB', 'AST', 'FG%', '3P%', 'FT%']]
y = team_stats['Rank']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest models
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the models
joblib.dump(model, 'team_rank_model.pkl')