import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import lief
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


def extract_features_from_pe(file_path):
    try:
        binary = lief.parse(file_path)
        features = {}

        
        features["has_debug"] = int(binary.has_debug)
        features["has_signature"] = int(binary.has_signature)
        features["sizeof_headers"] = binary.optional_header.sizeof_headers
        features["sizeof_image"] = binary.optional_header.sizeof_image
        features["numberof_sections"] = len(binary.sections)
        features["entropy"] = np.mean([s.entropy for s in binary.sections]) if binary.sections else 0

        
        for i in range(6, 2351):
            features[f"f{i}"] = 0

        return pd.DataFrame([features])

    except Exception as e:
        st.error(f"Feature extraction failed: {str(e)}")
        return None


@st.cache_data
def load_ember_dataset(x_path, y_path):
    X = np.fromfile(x_path, dtype=np.float32).reshape(-1, 2351)
    y = np.fromfile(y_path, dtype=np.int8)
    return X, y


@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = lgb.LGBMClassifier(objective='binary', n_estimators=100)
    model.fit(X_train, y_train)
    return model, X_test, y_test