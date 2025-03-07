import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import joblib  # Use joblib to load the trained model
import pickle

# Load trained model
best_grid_search = joblib.load("best_grid_search_model.pkl")  # Ensure the model is saved in this file

# Load dataset for reference
data_path = "C://Users//KAVYASRI//OneDrive//Documents//Performance.csv"
userinpdata = pd.read_csv(data_path)

# Define Feature Processing Functions
def preprocess_input(data):
    """ Preprocess user input to match training data format """

    # Gender Encoding with stripping extra spaces
    data['Gender'] = data['Gender'].astype(str).str.strip().replace({'Female': 0, 'Male': 1})

    # One-Hot Encoding
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(userinpdata[['Course Name', 'Certification obtained']])
    encoded_features = ohe.transform(data[['Course Name', 'Certification obtained']])
    
    encoded_df = pd.DataFrame(encoded_features, columns=ohe.get_feature_names_out(['Course Name', 'Certification obtained']))
    
    data = data.drop(['Course Name', 'Certification obtained'], axis=1)
    data = pd.concat([data, encoded_df], axis=1)

    # Label Encoding for class_Type
    label_encoder = LabelEncoder()
    label_encoder.fit(userinpdata['class_Type'].astype(str).str.strip())
    data["class_Type"] = label_encoder.transform(data["class_Type"].astype(str).str.strip())

    # Convert Duration with stripping spaces
    data['Duration'] = data['Duration'].astype(str).str.replace(" months", "").str.strip().astype(float)

    # Ensure feature order matches trained model
    trained_features = best_grid_search.feature_names_in_
    for feature in trained_features:
        if feature not in data.columns:
            data[feature] = 0  # Add missing features with default value
    
    data = data[trained_features]  # Ensure correct order
    return data

# Streamlit UI
st.set_page_config(page_title="Student Performance Prediction", layout="wide")
st.title("Student Performance Prediction App")
st.markdown("### Enter the details below to predict student performance.")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    In_name = st.selectbox("Institute Name", userinpdata['Institute Name'].unique())
    gen = st.selectbox("Gender", userinpdata.Gender.unique())
    cn = st.selectbox("Course Name", userinpdata['Course Name'].unique())
    ct = st.selectbox("Class Type", userinpdata['class_Type'].unique())
    dt = st.selectbox("Duration (months)", userinpdata.Duration.unique())
    age = st.slider("Age", int(userinpdata.Age.min()), int(userinpdata.Age.max()))

with col2:
    tot = st.selectbox("Certification obtained", userinpdata['Certification obtained'].unique())
    fd = st.slider("Assignments Submitted", int(userinpdata['Assignments submitted'].min()), int(userinpdata['Assignments submitted'].max()))
    obr = st.slider("Projects Submitted", int(userinpdata['projects submitted'].min()), int(userinpdata['projects submitted'].max()))
    scr = st.slider("Mock Test Score", int(userinpdata['Mock_Test'].min()), int(userinpdata['Mock_Test'].max()))
    bhr = st.slider("Attendance (%)", int(userinpdata['Attendance(in %)'].min()), int(userinpdata['Attendance(in %)'].max()))
    clr = st.slider("Final Score (%)", int(userinpdata['Final score(in %)'].min()), int(userinpdata['Final score(in %)'].max()))

# Prepare Data for Prediction
input_data = pd.DataFrame({
    'Gender': [gen],
    'Course Name': [cn],
    'class_Type': [ct],
    'Duration': [dt],
    'Age': [age],
    'Certification obtained': [tot],
    'Assignments submitted': [fd],
    'projects submitted': [obr],
    'Mock_Test': [scr],
    'Attendance(in %)': [bhr],
    'Final score(in %)': [clr],
    'Institute Name': [In_name]
})

if st.button("Predict Performance"):
    # Preprocess user input
    processed_data = preprocess_input(input_data)
    
    # Make Prediction
    prob = best_grid_search.predict_proba(processed_data)[0]
    prediction = best_grid_search.predict(processed_data)[0]

    # Display Results
    st.success(f"Prediction: **{prediction}**")
    st.write(f"Predicted Probabilities:")
    st.write(f"- **{best_grid_search.classes_[0]}**: {round(prob[0], 2)}")
    st.write(f"- **{best_grid_search.classes_[1]}**: {round(prob[1], 2)}")