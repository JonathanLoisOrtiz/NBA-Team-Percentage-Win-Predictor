import matplotlib.pyplot as plt
import NBADATABASE as nba  # Import the model script

# Create the plot
plt.figure(figsize=(8, 5))
plt.barh(nba.importance_df["Feature"], nba.importance_df["Importance"], color='blue')

# Add labels and title
plt.xlabel("Feature Importance Score")
plt.ylabel("Feature")
plt.title("NBA Feature Importance in Win Prediction")

# Invert Y-axis to show most important features on top
plt.gca().invert_yaxis()

# Display the plot
plt.show()
