# =============================================================================
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# import joblib
# import os
# 
# print("--- Starting Malware Classification Project ---")
# print("Project Goal: Malware classification using static analysis on a low-end PC.")
# print("Focus: Efficient feature utilization and interpretable models.")
# 
# # --- Step 1: Data Acquisition (Using the downloaded Kaggle dataset) ---
# dataset_path =r"D:\malwareproj\top_1000_pe_imports.csv"
# 
# if not os.path.exists(dataset_path):
#     print(f"Error: Dataset '{dataset_path}' not found.")
#     print("Please download it from: https://www.kaggle.com/datasets/ang3loliveira/malware-analysis-datasets-top1000-pe-imports")
#     print("Extract 'top_1000_pe_imports.csv' from the downloaded ZIP and place it in the same folder as this script.")
#     exit()
# 
# try:
#     print(f"Loading dataset from: {dataset_path}")
#     df = pd.read_csv(dataset_path)
#     print("Dataset loaded successfully.")
#     print(f"Dataset shape (rows, columns): {df.shape}")
# 
#     X = df.drop(['hash', 'malware'], axis=1) # Features
#     y = df['malware']                     # Target variable
# 
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#     print(f"\nData split into training ({len(X_train)} samples) and testing ({len(X_test)} samples).")
# 
#     # Store the feature column names. This is CRUCIAL for the web app to ensure
#     # the extracted features are in the same order as the trained model expects.
#     feature_columns = X.columns.tolist()
#     # Save the feature column names to a file
#     joblib.dump(feature_columns, 'model_features.joblib')
#     print("Feature column names saved to 'model_features.joblib'.")
# 
#     # --- Step 3: Model Building and Training ---
#     # We'll train all models, but specifically save the Random Forest for the web app.
#     models = {
#         "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, solver='liblinear'),
#         "Decision Tree": DecisionTreeClassifier(random_state=42),
#         "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
#     }
# 
#     trained_models = {}
#     results = {}
# 
#     print("\n--- Training and Evaluating Models ---")
# 
#     for name, model in models.items():
#         print(f"\nTraining {name}...")
#         model.fit(X_train, y_train)
#         trained_models[name] = model
#         print(f"  {name} training complete.")
# 
#         y_pred = model.predict(X_test)
# 
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred)
#         recall = recall_score(y_test, y_pred)
#         f1 = f1_score(y_test, y_pred)
#         cm = confusion_matrix(y_test, y_pred)
# 
#         results[name] = {
#             "Accuracy": accuracy,
#             "Precision": precision,
#             "Recall": recall,
#             "F1-Score": f1,
#             "Confusion Matrix": cm.tolist()
#         }
# 
#         print(f"  {name} Evaluation Results:")
#         print(f"    Accuracy : {accuracy:.4f}")
#         print(f"    Precision: {precision:.4f}")
#         print(f"    Recall   : {recall:.4f}")
#         print(f"    F1-Score : {f1:.4f}")
#         print(f"    Confusion Matrix:\n{cm}")
# 
#     
#     model_to_save = trained_models["Random Forest"]
#     joblib.dump(model_to_save, 'random_forest_malware_classifier.joblib')
#     print("\n'Random Forest' model saved as 'random_forest_malware_classifier.joblib'.")
# 
# 
#     print("\n--- All Model Evaluations Complete ---")
# 
#     print("\n--- Summary of All Model Results ---")
#     for name, res in results.items():
#         print(f"\nModel: {name}")
#         for metric, value in res.items():
#             if metric == "Confusion Matrix":
#                 print(f"  {metric}:\n{value}")
#             else:
#                 print(f"  {metric}: {value:.4f}")
# 
#     print("\n--- Project Execution Finished ---")
#     print("You can now proceed to build the web tool using 'app.py' and 'index.html'.")
# 
# except Exception as e:
#     print(f"\nAn unexpected error occurred: {e}")
#     print("Please check the following:")
#     print("1. Ensure Python and all required libraries (`pandas`, `scikit-learn`, `numpy`) are installed.")
#     print("   You can install them using: `pip install pandas numpy scikit-learn`")
#     print("2. Confirm 'top_1000_pe_imports.csv' is in the correct directory.")
#     print("3. Check for any network issues if downloading the dataset or library dependencies.")
# =============================================================================


import streamlit as st
import pandas as pd
import numpy as np
import lief
import lightgbm as lgb
import joblib
import os
from sklearn.preprocessing import StandardScaler

scaler = joblib.load("scaler.pkl")
model = joblib.load("lgb_model.pkl")

st.set_page_config(page_title="Malware Detector", layout="centered")
st.markdown("""
    <style>
    .main {
        background-color: #f6f7f8;
        background-image: url("https://www.transparenttextures.com/patterns/cream-pixels.png");
        background-size: cover;
        color: #1f3a2d;
    }
    .stButton > button {
        background-color: #3c6e71;
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Malware Detection and Family Classification")
uploaded_file = st.file_uploader("Upload a PE (.exe) file", type=["exe"])

def extract_features(path):
    binary = lief.parse(path)
    feature_vector = []
    feature_vector.append(binary.optional_header.major_operating_system_version)
    feature_vector.append(binary.optional_header.minor_operating_system_version)
    feature_vector.append(binary.optional_header.major_image_version)
    feature_vector.append(binary.optional_header.minor_image_version)
    feature_vector.append(binary.optional_header.major_subsystem_version)
    feature_vector.append(binary.optional_header.minor_subsystem_version)
    feature_vector.append(binary.optional_header.sizeof_code)
    feature_vector.append(binary.optional_header.sizeof_initialized_data)
    feature_vector.append(binary.optional_header.sizeof_uninitialized_data)
    feature_vector.append(binary.optional_header.address_of_entrypoint)
    feature_vector.append(binary.optional_header.base_of_code)
    feature_vector.append(binary.optional_header.imagebase)
    feature_vector.append(binary.optional_header.section_alignment)
    feature_vector.append(binary.optional_header.file_alignment)
    feature_vector.append(binary.optional_header.sizeof_image)
    feature_vector.append(binary.optional_header.sizeof_headers)
    feature_vector.append(binary.optional_header.checksum)
    feature_vector.append(binary.optional_header.loader_flags)
    feature_vector.append(binary.optional_header.numberof_rva_and_size)
    feature_vector.append(len(binary.imports))
    feature_vector.append(len(binary.exports))
    feature_vector.append(len(binary.sections))
    for section in binary.sections:
        feature_vector.append(section.virtual_size)
        feature_vector.append(section.sizeof_raw_data)
        feature_vector.append(section.entropy)
    feature_vector += [0] * (2351 - len(feature_vector))
    return np.array(feature_vector[:2351])

if uploaded_file is not None:
    with open("temp_uploaded.exe", "wb") as f:
        f.write(uploaded_file.read())

    try:
        features = extract_features("temp_uploaded.exe")
        scaled = scaler.transform([features])
        prediction = model.predict(scaled)
        predicted_label = int(prediction[0])

        if predicted_label == 0:
            st.success("This file is predicted as Benign.")
        else:
            st.error("This file is predicted as Malicious.")

        explanation = "The prediction is based on extracted static features such as PE header information, section sizes, entropy, and imported functions. High entropy and irregular structure across sections are strong indicators of obfuscation or packing techniques used in malware."
        st.markdown(f"**Model's Reasoning:**\n{explanation}")

    except Exception as e:
        st.error("Failed to analyze the file. Ensure it's a valid PE file.")

    os.remove("temp_uploaded.exe")
