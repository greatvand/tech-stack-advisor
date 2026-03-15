# =============================================================================
# app.py — Tech Stack Advisor: Web Application
# =============================================================================
# This script does ONE job: load the trained model and serve it as a
# web application that real users can interact with through a browser.
#
# It depends on train.py having been run first — it needs model.pkl and
# encoders.pkl to exist before this script can work.
#
# Flow:
#   User fills in the form in browser
#         ↓
#   recommend_stack() encodes the inputs (text → numbers)
#         ↓
#   model.predict() returns a number
#         ↓
#   encoders decode the number back to a stack name (numbers → text)
#         ↓
#   Result is displayed to the user
# =============================================================================


# -----------------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------------
# gradio — a library that lets you build web UIs for ML models with very
# little code. It handles the browser interface, form inputs, buttons,
# and displaying results — all without writing any HTML or JavaScript.
import gradio as gr

# pickle — used to load the saved model and encoders from disk.
# We saved these in train.py using pickle.dump() — now we reload them
# with pickle.load() to use them here without retraining.
import pickle

# numpy — used to create the input array that the model expects.
# The model was trained on a numpy array, so predictions must also
# be passed as a numpy array in the same shape.
import numpy as np


# -----------------------------------------------------------------------------
# LOADING THE SAVED MODEL AND ENCODERS
# -----------------------------------------------------------------------------
# These two files were created by train.py.
# We load them once at startup so they are ready in memory for every
# user request — much faster than loading from disk on every prediction.
#
# "rb" means "read in binary mode" — required for pickle files.

# Load the trained Decision Tree model
model = pickle.load(open("model.pkl", "rb"))

# Load the LabelEncoders for all categorical columns
# (project_type, perf_need, experience, stack)
encoders = pickle.load(open("encoders.pkl", "rb"))


# -----------------------------------------------------------------------------
# PREDICTION FUNCTION
# -----------------------------------------------------------------------------
# This is the core function — it receives the user's inputs from the form,
# processes them, runs them through the model, and returns a recommendation.
#
# Parameters (all come directly from the Gradio UI form):
#   project_type — string e.g. "Web App", "API", "ML App", "Real-time App"
#   team_size    — integer e.g. 3 (comes from the slider)
#   perf_need    — string e.g. "Low", "Medium", "High"
#   experience   — string e.g. "Beginner", "Intermediate", "Expert"
#
# Returns:
#   A formatted string with the recommended tech stack e.g.
#   "🔧 Recommended Tech Stack: Django + PostgreSQL"

def recommend_stack(project_type, team_size, perf_need, experience):

    # STEP 1 — Encode text inputs to numbers
    # The model was trained on numbers, not text. So we must convert
    # the user's text choices to the same numeric codes used during training.
    # .transform([value])[0] — wraps value in a list (required by sklearn),
    # transforms it, then takes the first (and only) result with [0].
    #
    # Example:
    #   encoders["project_type"].transform(["Web App"])[0]  → 3
    #   encoders["perf_need"].transform(["High"])[0]        → 0
    #   encoders["experience"].transform(["Expert"])[0]     → 1

    pt = encoders["project_type"].transform([project_type])[0]  # encode project type
    pn = encoders["perf_need"].transform([perf_need])[0]        # encode performance need
    ex = encoders["experience"].transform([experience])[0]      # encode experience level

    # Note: team_size is already a number from the slider — no encoding needed


    # STEP 2 — Build the input array
    # The model expects input in the same format it was trained on:
    # a 2D numpy array where each row is one prediction request.
    #
    # Shape must be [[pt, team_size, pn, ex]] — a list inside a list
    # because the model supports batch predictions (multiple rows at once).
    # We only have one row here, but the 2D shape is still required.
    #
    # Example result:
    #   np.array([[3, 4, 0, 1]])   → shape (1, 4)

    input_data = np.array([[pt, team_size, pn, ex]])


    # STEP 3 — Make the prediction
    # model.predict() returns an array of predictions — one per input row.
    # Since we only passed one row, we take [0] to get the single result.
    # The result is a number — the encoded stack label e.g. 2
    #
    # Example:
    #   model.predict([[3, 4, 0, 1]]) → [2]
    #   pred_encoded = 2

    pred_encoded = model.predict(input_data)[0]


    # STEP 4 — Decode the prediction back to a readable stack name
    # .inverse_transform() reverses the encoding done in train.py.
    # It converts the number back to the original text label.
    #
    # Example:
    #   encoders["stack"].inverse_transform([2]) → ["Django + PostgreSQL"]
    #   [0] takes the string out of the list     → "Django + PostgreSQL"

    stack_name = encoders["stack"].inverse_transform([pred_encoded])[0]


    # STEP 5 — Return the result as a formatted string
    # Gradio will display whatever this function returns in the output box.
    return f"🔧 Recommended Tech Stack: {stack_name}"


# -----------------------------------------------------------------------------
# BUILDING THE GRADIO USER INTERFACE
# -----------------------------------------------------------------------------
# gr.Interface() creates the full web UI — it wires up the input form,
# the output display, and the function that connects them.
#
# fn        — the function to call when the user clicks Submit
# inputs    — list of UI components, one per parameter of recommend_stack()
# outputs   — how to display the return value of recommend_stack()
# title     — heading shown at the top of the page
# description — subheading shown below the title

demo = gr.Interface(
    fn=recommend_stack,      # function to run when user submits the form

    inputs=[
        # Radio buttons for project type — user picks one of four options
        # label= is the text shown above the buttons in the UI
        gr.Radio(["Web App", "API", "ML App", "Real-time App"], label="Project Type"),

        # Slider for team size — user drags between 1 and 10
        # step=1 means only whole numbers (1, 2, 3 ... not 1.5, 2.3)
        gr.Slider(1, 10, step=1, label="Team Size"),

        # Radio buttons for performance requirement
        gr.Radio(["Low", "Medium", "High"], label="Performance Need"),

        # Radio buttons for team experience level
        gr.Radio(["Beginner", "Intermediate", "Expert"], label="Experience Level")
    ],

    # Output type — "text" tells Gradio to display the returned string
    # in a plain text box below the form
    outputs="text",

    title="Tech Stack Advisor",
    description="Get a recommended tech stack based on your project and team!"
)


# -----------------------------------------------------------------------------
# LAUNCHING THE APP
# -----------------------------------------------------------------------------
# demo.launch() starts a local web server and opens the app in the browser.
#
# server_name="0.0.0.0" — makes the app accessible from any IP address,
#   not just localhost. This is required when running inside Docker so the
#   app is reachable from outside the container.
#   If you were running locally only, you could use "127.0.0.1" instead.
#
# server_port=7860 — the port the web server listens on.
#   Access the app at: http://localhost:7860
#   This is Gradio's default port — must also be exposed in the Dockerfile
#   and mapped with -p 7860:7860 in docker run.

demo.launch(server_name="0.0.0.0", server_port=7860)

# =============================================================================
# SUMMARY — What this script does:
#
#   1. Loads model.pkl and encoders.pkl saved by train.py
#   2. Defines recommend_stack() — encodes inputs → predicts → decodes output
#   3. Builds a Gradio web UI with radio buttons and a slider
#   4. Launches a web server on port 7860 accessible from any IP
#
# To run:    uv run app.py   or   python app.py
# To access: http://localhost:7860
# =============================================================================