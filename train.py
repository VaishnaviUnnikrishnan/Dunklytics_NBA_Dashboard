import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load dataset
file_path = "play_off_totals_2010_2024.csv"
df = pd.read_csv(file_path)

# Convert GAME_DATE to datetime format
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")

# Select relevant features for training
features = ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FTM", "FTA", "REB", "AST", "TOV"]
target = "PTS"  # Points scored

# Drop rows with missing values in selected columns
df = df.dropna(subset=features + [target])

# Scale the features and target
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features + [target]])

# Define sequence length (last 10 games for LSTM)
sequence_length = 10

# Prepare data for LSTM
X, y = [], []

for i in range(len(df_scaled) - sequence_length):
    X.append(df_scaled[i : i + sequence_length, :-1])  # Features (Exclude target)
    y.append(df_scaled[i + sequence_length, -1])  # Target (PTS)

X, y = np.array(X), np.array(y)

# Split into training (80%) and testing (20%) sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, len(features))),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation="relu"),
    Dense(1)  # Output layer (Predict PTS)
])

# Compile the model
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save("lstm_playoff_points.h5")

# Plot training loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("LSTM Training Loss")
plt.show()

print("✅ Model training complete! Saved as 'lstm_playoff_points.h5'")

