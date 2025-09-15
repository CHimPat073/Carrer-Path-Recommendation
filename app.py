import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ============================
# Load Model and Encoder
# ============================
model = joblib.load("E:/Sem 6 Fall 2025-26/CSA4008 Applied Machine Learning/GA/Research Paper/Final project/Models/carrer_predictor (1).pkl")
label_encoder = joblib.load("E:/Sem 6 Fall 2025-26/CSA4008 Applied Machine Learning/GA/Research Paper/Final project/Models/role_encoder.pkl")

# ============================
# Define Feature Inputs
# ============================
feature_names = [
    "Database Fundamentals", "Computer Architecture", "Distributed Computing Systems",
    "Cyber Security", "Networking", "Software Development", "Programming Skills",
    "Project Management", "Computer Forensics Fundamentals", "Technical Communication",
    "AI ML", "Software Engineering", "Business Analysis", "Communication skills",
    "Data Science", "Troubleshooting skills", "Graphics Designing"
]

# Map text inputs to numbers
skill_mapping = {
    "Not Interested":4 ,
    "Beginner": 1,
    "Poor": 5,
    "Average": 0,
    "Intermediate": 3,
    "Excellent": 2,
    "Professional": 6
    
}

# ============================
# Streamlit App
# ============================
st.title("🎓 Career Path Prediction App")
st.write("Fill in your skills to get a recommended career path!")

# Collect user inputs
user_input = []
for feature in feature_names:
    value = st.selectbox(f"{feature}", list(skill_mapping.keys()))
    user_input.append(skill_mapping[value])

# Predict button
if st.button("🔍 Predict Career Path"):
    input_df = pd.DataFrame([user_input], columns=feature_names)

    # Predict with model
    prediction = model.predict(input_df)

    # Decode back to actual role
    decoded_prediction = label_encoder.inverse_transform(prediction)

    st.success(f"✅ Recommended Career Path: *{decoded_prediction[0]}*")

    # Show input summary
    with st.expander("📊 See Your Input Details"):
        st.write(input_df)



explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_df) 

st.subheader("📊 How each skill contributed to this prediction")


class_idx = prediction[0] 




# Handle the two main output types from shap.TreeExplainer
if isinstance(shap_values, list):
    # Case 1: shap_values is a list of arrays (e.g., for scikit-learn models)
    # Each item in the list corresponds to a class.
    shap_values_for_class = shap_values[class_idx]
else:
    # Case 2: shap_values is a single 3D array (e.g., for XGBoost, LightGBM)
    # Shape is (n_samples, n_features, n_classes). We need to slice it.
    shap_values_for_class = shap_values[:, :, class_idx]


shap_values_for_sample = shap_values_for_class[0] 




# Create dataframe of features & SHAP values
shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_values_for_sample
})

# Sort by the absolute SHAP value to find the most impactful features
shap_df_sorted = shap_df.sort_values(by="SHAP Value", key=abs, ascending=False)

st.write("Top features influencing this decision (positive values push the prediction higher, negative values push it lower):")

st.bar_chart(shap_df.set_index("Feature"), y="SHAP Value")


# Global explanation (importance across all features)
st.subheader("🌍 Overall Skill Importance (Across All Career Paths)")
st.write(
        "This plot shows the most important skills for the model's predictions overall. "
        "The colors show which career path a skill is most important for."
    )


class_names = label_encoder.classes_
fig_global, ax_global = plt.subplots(figsize=(12, 15))
shap.summary_plot(
        shap_values,
        input_df,
        feature_names=feature_names,
        plot_type="bar",
        class_names=class_names, 
        show=False
    )
st.pyplot(fig_global, use_container_width=True)