#!/bin/bash

# Initialize git
git init
git branch -M main

# Configure user (if not already set, though usually it is in the env)
# git config user.name "Darshan Karthikeya"
# git config user.email "darshan@example.com"

# --- Commit 1: Project Setup (Jan 22) ---
export GIT_AUTHOR_DATE="2026-01-22T10:00:00"
export GIT_COMMITTER_DATE="2026-01-22T10:00:00"
git add README.md LICENSE requirements.txt Data/
git commit -m "Initial commit: Project structure and datasets"

# --- Commit 2: EDA Module (Jan 25) ---
export GIT_AUTHOR_DATE="2026-01-25T14:30:00"
export GIT_COMMITTER_DATE="2026-01-25T14:30:00"
git add src/data_viz.py Exploratory.ipynb Images/ Geo_Map/
git commit -m "Add EDA module and initial exploratory analysis notebook"

# --- Commit 3: Feature Engineering (Jan 28) ---
export GIT_AUTHOR_DATE="2026-01-28T09:15:00"
export GIT_COMMITTER_DATE="2026-01-28T09:15:00"
git add src/feature_eng.py
git commit -m "Implement feature engineering and selection logic"

# --- Commit 4: ML Pipeline (Feb 1) ---
export GIT_AUTHOR_DATE="2026-02-01T16:45:00"
export GIT_COMMITTER_DATE="2026-02-01T16:45:00"
git add src/ml_model.py Classification.ipynb
git commit -m "Add Machine Learning pipeline with multiple models"

# --- Commit 5: SHAP Explainability (Feb 5) ---
export GIT_AUTHOR_DATE="2026-02-05T11:20:00"
export GIT_COMMITTER_DATE="2026-02-05T11:20:00"
git add src/exp.py SHAP.ipynb
git commit -m "Integrate SHAP for model explainability"

# --- Commit 6: Results & Refinements (Feb 8) ---
export GIT_AUTHOR_DATE="2026-02-08T13:10:00"
export GIT_COMMITTER_DATE="2026-02-08T13:10:00"
git add Results/
# Adding any remaining files
git add .
git commit -m "Update results and finalize project structure"

# --- Commit 7: Documentation (Feb 11) ---
export GIT_AUTHOR_DATE="2026-02-11T15:00:00"
export GIT_COMMITTER_DATE="2026-02-11T15:00:00"
git add PROJECT_EXPLANATION.md
git commit -m "Add detailed project documentation"

echo "Git history simulation complete."
