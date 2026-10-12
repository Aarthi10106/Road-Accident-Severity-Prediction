# %% [markdown]
# # Road Accident Data Science Project (UK 2025)
# Goal: explore accident patterns and predict if an accident is SEVERE or SLIGHT.
# Run each block with the "Run Cell" button above it (or Shift+Enter).

# %% BLOCK 1: Import libraries
# Install once in terminal:  pip install pandas matplotlib scikit-learn ipykernel
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs("project_output", exist_ok=True)   # folder where results are saved
print("Libraries imported")

# %% BLOCK 2: Load the data (CSV must be in the same folder)
df = pd.read_csv("dft-road-casualty-statistics-collision-2025.csv")
print("Rows and columns:", df.shape)
df.head()

# %% BLOCK 3: Understand the data
print(df.columns.tolist())
print()
print("Missing values (only columns that have some):")
print(df.isnull().sum()[df.isnull().sum() > 0])
print()
print("Severity counts (1=Fatal, 2=Serious, 3=Slight):")
print(df["collision_severity"].value_counts())
df.describe()

# %% BLOCK 4: Select useful columns
df = df[["collision_severity", "number_of_vehicles", "number_of_casualties",
         "date", "day_of_week", "time", "speed_limit", "road_type",
         "junction_detail", "light_conditions", "weather_conditions",
         "road_surface_conditions", "urban_or_rural_area"]].copy()
print("Selected columns:", df.shape)

# %% BLOCK 5: Create new columns (feature engineering)
df["hour"] = df["time"].str[:2].astype(int)                        # "17:30" -> 17
df["date"] = df["date"].astype(str).str.replace("-", "/")        # works for 04-02-2025 and 04/02/2025
df["month"] = pd.to_datetime(df["date"], format="%d/%m/%Y").dt.month

df["is_weekend"] = 0
df.loc[df["day_of_week"].isin([1, 7]), "is_weekend"] = 1           # 1=Sun, 7=Sat

df["is_night"] = 0
df.loc[(df["hour"] >= 22) | (df["hour"] <= 5), "is_night"] = 1     # 10pm to 5am

df = df.drop(columns=["date", "time"])
df[["hour", "month", "is_weekend", "is_night"]].head()

# %% BLOCK 6: Clean missing values
before = len(df)
df = df[(df != -1).all(axis=1)]                      # -1 means "missing" in this dataset
df = df[df["urban_or_rural_area"].isin([1, 2])]      # 1=Urban, 2=Rural (3 = unallocated)
print("Rows removed:", before - len(df))
print("Rows left:", len(df))

# %% BLOCK 7: Create the target column
# severe = 1 for Fatal or Serious, 0 for Slight
df["severe"] = 0
df.loc[df["collision_severity"] != 3, "severe"] = 1
print("Severe accidents:", round(df["severe"].mean() * 100, 1), "%")
df["severe"].value_counts()

# %% BLOCK 8: Chart 1 - Severity
labels = {1: "Fatal", 2: "Serious", 3: "Slight"}
counts = df["collision_severity"].value_counts().sort_index()
counts.index = [labels[i] for i in counts.index]
counts.plot(kind="bar", color=["red", "orange", "green"])
plt.title("Accident Severity")
plt.ylabel("Number of accidents")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("project_output/1_severity.png")
plt.show()

# %% BLOCK 9: Chart 2 - Accidents by hour
df["hour"].value_counts().sort_index().plot(kind="bar", color="steelblue")
plt.title("Accidents by Hour of Day")
plt.xlabel("Hour (0-23)")
plt.ylabel("Number of accidents")
plt.tight_layout()
plt.savefig("project_output/2_hour.png")
plt.show()

# %% BLOCK 10: Chart 3 - Accidents by day of week
by_day = df["day_of_week"].value_counts().sort_index()
by_day.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
by_day.plot(kind="bar", color="orange")
plt.title("Accidents by Day of Week")
plt.ylabel("Number of accidents")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("project_output/3_day.png")
plt.show()

# %% BLOCK 11: Chart 4 - Accidents by month
df["month"].value_counts().sort_index().plot(kind="bar", color="teal")
plt.title("Accidents by Month")
plt.xlabel("Month (1-12)")
plt.ylabel("Number of accidents")
plt.tight_layout()
plt.savefig("project_output/4_month.png")
plt.show()

# %% BLOCK 12: Chart 5 - % severe by speed limit
(df.groupby("speed_limit")["severe"].mean() * 100).plot(kind="bar", color="red")
plt.title("% Severe Accidents by Speed Limit")
plt.xlabel("Speed limit (mph)")
plt.ylabel("% severe")
plt.tight_layout()
plt.savefig("project_output/5_speed_limit.png")
plt.show()

# %% BLOCK 13: Chart 6 - Urban vs Rural
area = df.groupby("urban_or_rural_area")["severe"].mean() * 100
area.index = ["Urban", "Rural"]
area.plot(kind="bar", color=["gray", "green"])
plt.title("% Severe Accidents: Urban vs Rural")
plt.ylabel("% severe")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("project_output/6_urban_rural.png")
plt.show()

# %% BLOCK 14: Chart 7 - Day vs Night
night = df.groupby("is_night")["severe"].mean() * 100
night.index = ["Day", "Night"]
night.plot(kind="bar", color=["gold", "navy"])
plt.title("% Severe Accidents: Day vs Night")
plt.ylabel("% severe")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("project_output/7_day_night.png")
plt.show()

# %% BLOCK 15: Split data into train and test
X = df.drop(columns=["severe", "collision_severity"])   # inputs
y = df["severe"]                                        # what we predict

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

# %% BLOCK 16: Train Model 1 - Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
log_model.fit(X_train_scaled, y_train)
log_pred = log_model.predict(X_test_scaled)
print("Logistic Regression accuracy:", round(accuracy_score(y_test, log_pred), 3))

# %% BLOCK 17: Train Model 2 - Decision Tree
tree_model = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
print("Decision Tree accuracy:", round(accuracy_score(y_test, tree_pred), 3))

# %% BLOCK 18: Train Model 3 - Random Forest
forest_model = RandomForestClassifier(n_estimators=150, max_depth=12,
                                      class_weight="balanced", random_state=42)
forest_model.fit(X_train, y_train)
forest_pred = forest_model.predict(X_test)
print("Random Forest accuracy:", round(accuracy_score(y_test, forest_pred), 3))

# %% BLOCK 19: Compare the models
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Decision Tree", "Random Forest"],
    "Accuracy": [accuracy_score(y_test, p) for p in [log_pred, tree_pred, forest_pred]],
    "Precision": [precision_score(y_test, p) for p in [log_pred, tree_pred, forest_pred]],
    "Recall": [recall_score(y_test, p) for p in [log_pred, tree_pred, forest_pred]],
    "F1": [f1_score(y_test, p) for p in [log_pred, tree_pred, forest_pred]],
}).round(3)
print(results.to_string(index=False))

results.set_index("Model").plot(kind="bar", figsize=(9, 5))
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("project_output/8_model_comparison.png")
plt.show()

# %% BLOCK 20: Confusion matrix of Random Forest
cm = confusion_matrix(y_test, forest_pred)
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xticks([0, 1], ["Slight", "Severe"])
plt.yticks([0, 1], ["Slight", "Severe"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
plt.colorbar()
plt.tight_layout()
plt.savefig("project_output/9_confusion_matrix.png")
plt.show()
print(classification_report(y_test, forest_pred, target_names=["Slight", "Severe"]))

# %% BLOCK 21: Feature importance (what matters most?)
importance = pd.Series(forest_model.feature_importances_, index=X.columns)
importance.sort_values().plot(kind="barh", color="purple", figsize=(8, 6))
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.savefig("project_output/10_feature_importance.png")
plt.show()
print(importance.sort_values(ascending=False).head(5))

# %% BLOCK 22: Save results
df.to_csv("project_output/cleaned_data.csv", index=False)
results.to_csv("project_output/model_results.csv", index=False)

with open("project_output/report.txt", "w") as f:
    f.write(classification_report(y_test, forest_pred, target_names=["Slight", "Severe"]))

print("Everything saved in the 'project_output' folder")

# %% BLOCK 23: Predict a new accident (try your own values!)
new_accident = pd.DataFrame([{
    "number_of_vehicles": 2, "number_of_casualties": 1, "day_of_week": 6,
    "speed_limit": 60, "road_type": 6, "junction_detail": 0,
    "light_conditions": 4, "weather_conditions": 1, "road_surface_conditions": 2,
    "urban_or_rural_area": 2, "hour": 23, "month": 11,
    "is_weekend": 0, "is_night": 1}])
new_accident = new_accident[X.columns]       # same column order as training
answer = forest_model.predict(new_accident)[0]
print("Prediction:", "SEVERE" if answer == 1 else "SLIGHT")