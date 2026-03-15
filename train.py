import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Sample synthetic data
data = {
    "project_type": ["Web App", "API", "ML App", "Real-time App", "Web App"],
    "team_size": [3, 2, 5, 6, 1],
    "perf_need": ["Medium", "Low", "Medium", "High", "Low"],
    "experience": ["Intermediate", "Beginner", "Expert", "Expert", "Beginner"],
    "stack": ["Django + PostgreSQL", "Flask + SQLite", "FastAPI + TensorFlow", "Node.js + Redis", "Django + SQLite"]
}

df = pd.DataFrame(data)

# Encode categorical variables
encoders = {}
for col in ["project_type", "perf_need", "experience", "stack"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Train model
X = df[["project_type", "team_size", "perf_need", "experience"]]
y = df["stack"]
model = DecisionTreeClassifier()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save encoders for use in app
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)
