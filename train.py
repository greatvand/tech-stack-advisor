# =============================================================================
# train.py — Tech Stack Advisor: Model Training Script
# =============================================================================
# This script does ONE job: learn from sample data and save the trained model.
# It is run only once (or whenever you want to retrain with new data).
# The output is two .pkl files: model.pkl and encoders.pkl
# Your app.py will load these files to make predictions for real users.
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# pandas — lets us work with data in a table format (like Excel but in Python)
import pandas as pd

# LabelEncoder — converts text values to numbers because ML models only
# understand numbers, not words like "Beginner" or "Web App"
from sklearn.preprocessing import LabelEncoder

# DecisionTreeClassifier — the ML algorithm we are using.
# It learns a set of if/else rules from the training data.
# Example rule it might learn:
#   "If experience == Expert AND perf_need == High → recommend Node.js + Redis"
from sklearn.tree import DecisionTreeClassifier

# pickle — saves Python objects (like our trained model) to a file on disk
# so we can reload and reuse them later without retraining from scratch
import pickle


# -----------------------------------------------------------------------------
# SAMPLE TRAINING DATA
# -----------------------------------------------------------------------------
# This is the data the model will learn from.
# Each row represents one example project with known inputs and a known answer.
# In a real project, this would come from a database or CSV file with
# hundreds or thousands of rows for better accuracy.
#
# Columns:
#   project_type — what kind of project it is
#   team_size    — how many developers are on the team
#   perf_need    — how much performance the project requires
#   experience   — skill level of the team
#   stack        — the correct tech stack recommendation (this is the answer
#                  the model will learn to predict)

data = {
    "project_type": ["Web App", "API", "ML App", "Real-time App", "Web App"],
    "team_size":    [3, 2, 5, 6, 1],
    "perf_need":    ["Medium", "Low", "Medium", "High", "Low"],
    "experience":   ["Intermediate", "Beginner", "Expert", "Expert", "Beginner"],
    "stack":        ["Django + PostgreSQL", "Flask + SQLite", "FastAPI + TensorFlow",
                     "Node.js + Redis", "Django + SQLite"]
}

# Convert the dictionary above into a DataFrame — a table structure like this:
#
#   project_type  team_size  perf_need    experience     stack
#   Web App       3          Medium       Intermediate   Django + PostgreSQL
#   API           2          Low          Beginner       Flask + SQLite
#   ML App        5          Medium       Expert         FastAPI + TensorFlow
#   Real-time App 6          High         Expert         Node.js + Redis
#   Web App       1          Low          Beginner       Django + SQLite

df = pd.DataFrame(data)


# -----------------------------------------------------------------------------
# ENCODING — Convert Text Values to Numbers
# -----------------------------------------------------------------------------
# ML models cannot process words — they only work with numbers.
# LabelEncoder assigns a unique integer to each unique text value.
#
# Example for the "experience" column:
#   "Beginner"     → 0
#   "Expert"       → 1
#   "Intermediate" → 2
#
# Example for the "project_type" column:
#   "API"           → 0
#   "ML App"        → 1
#   "Real-time App" → 2
#   "Web App"       → 3
#
# We store each encoder in a dictionary called `encoders` so we can reuse
# them later in app.py to:
#   1. Encode new user inputs (text → numbers) before predicting
#   2. Decode the model's output (numbers → text) to show the user

encoders = {}

for col in ["project_type", "perf_need", "experience", "stack"]:
    le = LabelEncoder()           # create a new encoder for this column
    df[col] = le.fit_transform(df[col])  # learn the mapping and apply it
    encoders[col] = le            # save the encoder so we can reuse it later

# Note: team_size is already a number so it does not need encoding


# -----------------------------------------------------------------------------
# DEFINING INPUTS (X) AND OUTPUT (y)
# -----------------------------------------------------------------------------
# X = the features — the information we know about a project (the inputs)
# y = the label  — what we want the model to predict (the output)
#
# Think of it as:
#   "Given these 4 things about a project → what stack should we recommend?"
#
# X contains: project_type, team_size, perf_need, experience
# y contains: stack (the correct answer for each row)

X = df[["project_type", "team_size", "perf_need", "experience"]]  # inputs
y = df["stack"]                                                     # output


# -----------------------------------------------------------------------------
# TRAINING THE MODEL
# -----------------------------------------------------------------------------
# DecisionTreeClassifier builds a tree of if/else rules by looking at all
# the rows in X and y. It figures out which questions to ask about the inputs
# to most accurately arrive at the correct stack recommendation.
#
# Example of what the tree might learn internally:
#
#   Is experience == Expert?
#   ├── Yes → Is perf_need == High?
#   │         ├── Yes → Node.js + Redis
#   │         └── No  → FastAPI + TensorFlow
#   └── No  → Is team_size <= 2?
#             ├── Yes → Flask + SQLite
#             └── No  → Django + PostgreSQL
#
# .fit(X, y) is where the actual learning happens.
# After this line, the model knows the rules — it is trained.

model = DecisionTreeClassifier()
model.fit(X, y)


# -----------------------------------------------------------------------------
# SAVING THE MODEL AND ENCODERS
# -----------------------------------------------------------------------------
# We use pickle to serialize (convert to bytes) and save both the trained
# model and the encoders to .pkl files on disk.
#
# Why save them?
#   - Training takes time. We don't want to retrain every time a user visits.
#   - app.py will load these files and use them to make instant predictions.
#
# "wb" means "write in binary mode" — required for pickle files.

# Save the trained Decision Tree model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save all the LabelEncoders (one per text column)
# app.py needs these to encode user inputs before passing them to the model
# and to decode the model's numeric output back to a readable stack name
with open("encoders.pkl", "wb") as f:
    pickle.dump(encoders, f)

print("Training complete. model.pkl and encoders.pkl saved successfully.")

# =============================================================================
# SUMMARY — What this script produces:
#
#   model.pkl    — the trained Decision Tree, ready to make predictions
#   encoders.pkl — the text-to-number mappings for all categorical columns
#
# Next step: run app.py, which loads these files and uses them to recommend
# a tech stack based on real user inputs.
# =============================================================================