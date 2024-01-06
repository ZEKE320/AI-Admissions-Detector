# AI-Admissions-Detector

Code for application and training algorithms described in:

Yijun Zhao, Fernando Martinez, Haoran Xue, Gary M. Weiss (2024) "Admissions in the Age of AI: Detecting AI-Generated Application Materials in Higher Education"

## Repository Structure

This repository is organized as follows:

### `src` Directory
The `src` folder contains scripts used for the generation of prompts and the training and analysis of AI models.

- `LORPromptsMaker.py`: Generates prompts for letters of recommendation.
- `SOIPromptsMaker.py`: Generates prompts for statements of intent.
- `TrainingAndAnalysis.py`: Handles the training and analysis of models.

### `app` Directory
The `app` directory encompasses all the necessary components to run the application.

- `app.py`: Main application entry point.
- `custom_models.py`: Contains custom transformer-based models.
- `requirements.txt`: Lists all dependencies required to run the application.
- `Dockerfile`: Dockerfile for building the application container.
- `.streamlit`: Contains Streamlit configuration files (if applicable).

#### `models` Subdirectory
The `models` subdirectory within `app` contains baseline models for machine learning operations.

- `baseline_model_lr.joblib`: Baseline logistic regression model.
- `baseline_model_lr2.joblib`: Second logistic regression baseline model.
- `baseline_model_nb.joblib`: Baseline Naive Bayes model.
- `baseline_model_nb2.joblib`: Second Naive Bayes baseline model.