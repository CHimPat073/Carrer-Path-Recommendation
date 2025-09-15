#🤖 Career Path Recommendation System

An interactive web application that leverages a machine learning model to suggest suitable career paths based on an individual's skills and interests. The project provides transparent and explainable predictions using SHAP (SHapley Additive exPlanations).

📋 Table of Contents
About The Project

##✨ Features

🚀 Getting Started

Prerequisites

Installation

🛠️ Built With

##📈 Model Development


About The Project
Choosing a career path can be a daunting task. This project aims to simplify the process by using a data-driven approach. It utilizes a Random Forest Classifier trained on a dataset of skills and their corresponding career roles.

The user interacts with a simple web interface built with Streamlit, where they can rate their proficiency across 17 different technical and soft skills. The model then predicts the most suitable career path and, most importantly, explains why it made that decision by showing which skills had the most positive and negative impact on the outcome.

##✨ Features
Interactive Skill Assessment: User-friendly interface with sliders and dropdowns to input skill levels.

ML-Powered Predictions: Employs a trained Scikit-learn model to predict the best-suited career role.

Prediction Probability: Shows the confidence score for the predicted career path.

Personalized Prediction Explanations: Utilizes SHAP to visualize the impact of each skill for the individual user's prediction.

Overall Skill Importance: Displays a global SHAP summary plot to show which skills are the most important across all predictions.

##🚀 Getting Started
To get a local copy up and running, follow these simple steps.

Prerequisites
Python 3.8 or higher

pip package manager

Installation
Clone the repository:

Bash

git clone https://github.com/CHimPat073/Carrer-Path-Recommendation.git
cd Carrer-Path-Recommendation
Create a requirements.txt file with the following content:

streamlit
pandas
numpy
scikit-learn
joblib
shap
matplotlib
Install the required packages:

Bash

pip install -r requirements.txt
Organize your project folder. Your model files should be in a models directory. The structure should look like this:

.
├── models/
│   ├── carrer_predictor (1).pkl
│   └── role_encoder.pkl
├── app.py
├── GroupActivity.ipynb
└── README.md
Important: In your app.py file, make sure the model paths are relative, not absolute. Change these lines:

##Python

# From this:
model = joblib.load("E:/.../carrer_predictor (1).pkl")
label_encoder = joblib.load("E:/.../role_encoder.pkl")

# To this:
model = joblib.load("models/carrer_predictor (1).pkl")
label_encoder = joblib.load("models/role_encoder.pkl")
Run the Streamlit application:

Bash

streamlit run app.py
Open your browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

#🛠️ Built With

Python: Core programming language.

Scikit-learn: For machine learning model training and evaluation.

Streamlit: For building the interactive web application.

SHAP: For model explainability and feature importance visualization.

Pandas: For data manipulation and processing.

Joblib: For saving and loading the trained model.

#📈 Model Development

The predictive model was developed in the GroupActivity.ipynb Jupyter Notebook. The key steps included:

Data Preprocessing: Cleaning the dataset and encoding categorical features using LabelEncoder.

Model Selection: Several classification algorithms were tested, including  Decision Trees, and Random Forest.

Hyperparameter Tuning: GridSearchCV was used to find the optimal parameters for the Random Forest Classifier to maximize accuracy.

Evaluation: The final model was evaluated based on its accuracy score and a detailed classification report.

Serialization: The tuned Random Forest model and the label encoder were saved to .pkl files using joblib for use in the Streamlit application.

📸 Demo Screenshot