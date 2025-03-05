import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load datasets
df = pd.read_csv("C:/Users/jonat/OneDrive/Desktop/Python ML project/Database/nba_team_stats_00_to_23.csv")
dfPlayers = pd.read_csv("C:/Users/jonat/OneDrive/Desktop/Python ML project/Database/NBA_Player_Stats.csv")

# Divide dataset into decades
df2000s = df[(df['season'].str[2:4].astype(int) >= 0) & (df['season'].str[2:4].astype(int) < 10)].copy()
df2010s = df[(df['season'].str[2:4].astype(int) >= 10) & (df['season'].str[2:4].astype(int) < 20)].copy()
df2020s = df[(df['season'].str[2:4].astype(int) >= 20) & (df['season'].str[2:4].astype(int) < 30)].copy()

# Convert stats to "per game"
df2020s[['three_pointers_attempted', 'turnovers', 'rebounds', 'steals', 
         'field_goals_attempted', 'plus_minus']] = (
    df2020s[['three_pointers_attempted', 'turnovers', 'rebounds', 'steals', 
             'field_goals_attempted', 'plus_minus']] / 82).round(2)

# Define features (X) and target (y)
X = df2020s[['field_goal_percentage', 'three_pointers_attempted', 'turnovers', 
             'rebounds', 'steals', 'field_goals_attempted', 'three_point_percentage', 'blocks_attempted','plus_minus']]
y = df2020s['win_percentage']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train RandomForestRegressor
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predictions
pred1 = model.predict(X_test)

# Model evaluation
mse = mean_squared_error(y_test, pred1)
r2 = r2_score(y_test, pred1)
mape = np.mean(np.abs((y_test - pred1) / y_test)) * 100
accuracy = 100 - mape

# Store test data and predictions as module-level variables
actual_y = y_test.values  

importances = model.feature_importances_
feature_names = X.columns

# Convert to DataFrame for better visualization
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the importance scores
print(importance_df)

print("Mean Squared Error:", mse)
print("R² Score:", r2)
print(f"Prediction Accuracy: {accuracy:.2f}%")


