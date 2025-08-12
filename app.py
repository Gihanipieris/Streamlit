import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import classification_report, confusion_matrix


def looks_like_datetime(series, threshold=0.8):
    """Return True if >=threshold fraction of values parse as datetimes."""
    try:
        parsed = pd.to_datetime(series, errors='coerce')
        return parsed.notna().sum() >= int(len(series) * threshold)
    except Exception:
        return False

def safe_label_map(series, classes):
    """Map values in series to indices using classes list. Unseen -> -1."""
    mapping = {v: i for i, v in enumerate(classes)}
    return series.map(mapping).fillna(-1).astype(int)


@st.cache_data
def load_data():
    return pd.read_csv('data/academicStress.csv')

@st.cache_resource
def load_model():
    if not os.path.exists("model.pkl"):
        st.error("❌ model.pkl not found. Please train and save the model first.")
        st.stop()

    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        st.error("❌ model.pkl is corrupted or not a valid pickle file.")
        st.stop()

df = load_data()
model_data = load_model()


model = model_data.get('model')
scaler = model_data.get('scaler')
preprocessor = model_data.get('preprocessor')
encoders_dict = model_data.get('encoders')  
feature_names = model_data.get('feature_names')  
target_encoder = model_data.get('target_encoder')  
saved_metrics = model_data.get('metrics')

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualisations", "Model Prediction"])

TARGET_COLUMN_DEFAULT = df.columns[-1]


if page == "Home":
    st.title("Academic Stress Prediction App")
    st.write("This app predicts academic stress levels based on survey data.")
    if os.path.exists('images.jpeg'):
        st.image('images.jpeg', caption="Understanding Academic Stress", use_container_width=True)

elif page == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write(df.dtypes)
    st.dataframe(df.head())
    column = st.selectbox("Filter by column", df.columns)
    val = st.selectbox("Select value", df[column].unique())
    st.write(df[df[column] == val])

elif page == "Visualisations":
    st.subheader("Visualisations")
    TARGET_COLUMN = st.selectbox("Target column (for plotting)", df.columns, index=len(df.columns)-1)
    fig, ax = plt.subplots()
    sns.countplot(data=df, x=TARGET_COLUMN, ax=ax)
    ax.set_title(f"Countplot of {TARGET_COLUMN}")
    st.pyplot(fig)

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=TARGET_COLUMN, y=numeric_cols[0], ax=ax)
        ax.set_title(f"Boxplot of {numeric_cols[0]} by {TARGET_COLUMN}")
        st.pyplot(fig)
    else:
        st.info("No numeric columns available for boxplot visualization.")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)

elif page == "Model Prediction":
    st.subheader("Make a Prediction")

    if feature_names:
        FEATURE_COLUMNS = feature_names
    else:
        TARGET_COLUMN = df.columns[-1]
        FEATURE_COLUMNS = [col for col in df.columns if col not in [TARGET_COLUMN, 'Timestamp']]

    input_data = []
    input_widgets = {}
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            st.error(f"Feature '{col}' missing from the data!")
            st.stop()

        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            vals = df[col].dropna().unique().tolist()
            default_index = 0
            val = st.selectbox(f"Select {col}", vals, index=default_index)
            
            if encoders_dict and col in encoders_dict:
                enc = encoders_dict[col]
                if hasattr(enc, 'classes_'):
                    if val in enc.classes_:
                        val_encoded = int(np.where(enc.classes_ == val)[0][0])
                    else:
                        val_encoded = -1
                else:
                    try:
                        val_encoded = enc.transform([val])[0]
                    except Exception:
                        val_encoded = -1
            else:
                val_encoded = pd.factorize(df[col])[1].tolist().index(val) if val in pd.factorize(df[col])[1] else -1
            input_data.append(val_encoded)
        else:
            val = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()), value=float(df[col].median()))
            input_data.append(val)

    if st.button("Predict"):
        try:
            input_array = np.array([input_data])
            if scaler is not None and hasattr(scaler, 'transform'):
                input_array = scaler.transform(input_array)
            prediction = model.predict(input_array)[0]
            if target_encoder is not None and hasattr(target_encoder, 'inverse_transform'):
                try:
                    prediction_label = target_encoder.inverse_transform([prediction])[0]
                except Exception:
                    prediction_label = prediction
            else:
                prediction_label = prediction

            prob = model.predict_proba(input_array)[0] if hasattr(model, 'predict_proba') else None
            st.success(f"Prediction: {prediction_label}")
            if prob is not None:
                st.info(f"Confidence: {max(prob)*100:.2f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
