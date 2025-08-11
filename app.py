import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


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

TARGET_COLUMN = df.columns[-1]

st.write(f"Using target column for prediction and visualization: `{TARGET_COLUMN}`")

model_data = load_model()
model = model_data['model']
scaler = model_data['scaler']
encoder = model_data['encoder']


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Exploration", "Visualisations", "Model Prediction", "Model Performance"]
)


if page == "Home":
    st.title("Academic Stress Prediction App")
    st.write("This app predicts academic stress levels based on survey data.")
    
    image_path = 'images.jpeg'  
    if os.path.exists(image_path):
        st.image(image_path, caption="Understanding Academic Stress", use_container_width=True)
    else:
        st.warning("Stress image not found at path: " + image_path)


elif page == "Data Exploration":
    st.subheader("Dataset Overview")
    st.write(f"Shape: {df.shape}")
    st.write(df.dtypes)
    st.dataframe(df.head())

    column = st.selectbox("Filter by column", df.columns)
    unique_vals = df[column].unique()
    val = st.selectbox("Select value", unique_vals)
    st.write(df[df[column] == val])


elif page == "Visualisations":
    st.subheader("Visualisations")

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

    
    FEATURE_COLUMNS = [col for col in df.columns if col not in [TARGET_COLUMN, 'timestamp']]

    input_data = []

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            st.error(f"Feature '{col}' missing from the data!")
            st.stop()

        if df[col].dtype == 'object':
            val = st.selectbox(f"Select {col}", df[col].unique())
            try:
                if val in encoder.classes_:
                    val_encoded = encoder.transform([val])[0]
                else:
                    
                    val_encoded = 0
            except Exception as e:
                st.error(f"Encoding error for column '{col}': {e}")
                st.stop()
            input_data.append(val_encoded)
        else:
            val = st.number_input(f"Enter {col}", float(df[col].min()), float(df[col].max()))
            input_data.append(val)

    if st.button("Predict"):
        try:
            input_array = np.array([input_data])
            input_array = scaler.transform(input_array)
            prediction = model.predict(input_array)[0]
            prob = model.predict_proba(input_array)[0] if hasattr(model, 'predict_proba') else None

            st.success(f"Prediction: {prediction}")
            if prob is not None:
                st.info(f"Confidence: {max(prob)*100:.2f}%")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Model Performance":
    st.subheader("Model Performance Metrics")
    st.write("Check `model_training.ipynb` for detailed evaluation.")
