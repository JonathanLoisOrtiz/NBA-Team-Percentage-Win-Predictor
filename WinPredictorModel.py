# Importación de librerias

import pandas as pd #procesar datos
import numpy as np #procesar datos
from sklearn.ensemble import RandomForestRegressor #modelo de machine learning
from sklearn.metrics import mean_squared_error, r2_score #metricas estadisticas para evaluar modelo
from sklearn.model_selection import train_test_split #entrenamiento y pruebas al modelo

# Leer archivos de estadisticas de equipos desde el 2000 hasta el 2023
df = pd.read_csv("nba_team_stats_00_to_23.csv")


# Se dividen los datos entre decadas o "eras"
df2000s = df[(df['season'].str[2:4].astype(int) >= 0) & (df['season'].str[2:4].astype(int) < 10)].copy()
df2010s = df[(df['season'].str[2:4].astype(int) >= 10) & (df['season'].str[2:4].astype(int) < 20)].copy()
df2020s = df[(df['season'].str[2:4].astype(int) >= 20) & (df['season'].str[2:4].astype(int) < 30)].copy()

# Se convierten datos de totales por temporada a estadisticas "por juego"
df2020s[['three_pointers_attempted', 'turnovers', 'rebounds', 'steals', 
         'field_goals_attempted', 'plus_minus']] = (
    df2020s[['three_pointers_attempted', 'turnovers', 'rebounds', 'steals', 
             'field_goals_attempted', 'plus_minus']] / 82).round(2)

# Se definen los atributos (X) y lo que queremos predecir (y)
X = df2020s[['field_goal_percentage', 'three_pointers_attempted', 'turnovers', 
             'rebounds', 'steals', 'field_goals_attempted', 'three_point_percentage','plus_minus']]

y = df2020s['wins']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Se inicializa y se entrena el modelo Random Forest Regressor
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Predictions
pred1 = model.predict(X_test)


# Se evalua el modelo
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


# Se puede interactuar con el modelo 
FGP = input("field_goal_percentage: ")
TPA = input("three_pointers_attempted: ")
TO = input("turnovers: ")
REB = input("rebounds: ")
STL = input("steals: ")
FGA = input("field_goal_attempted: ")
TPP = input("three_point_percentage: ")
PLM = input("plus_minus: ")



# Se almacena la data entregada
new_data = pd.DataFrame({
    'field_goal_percentage': [FGP],  
    'three_pointers_attempted': [TPA],
    'turnovers': [TO],
    'rebounds': [REB],
    'steals': [STL],
    'field_goals_attempted': [FGA],
    'three_point_percentage': [TPP],
    'plus_minus': [PLM]
})

# Nueva Predicción 
Newpred = model.predict(new_data)

print("\nYour team will win approximately" , round(Newpred) , "games!")


