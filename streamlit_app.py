import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Title
st.title("🧠 Customer Segment Predictor")

# Load the trained classifier and scaler using joblib
try:
    model = joblib.load("customer_segment_model.joblib")  # Trained model (e.g., RandomForest)
    scaler = joblib.load("scaler.joblib")                 # StandardScaler used during training
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please make sure both 'customer_segment_model.joblib' and 'scaler.joblib' exist.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("📁 Upload your customer data (unscaled CSV)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Drop extra index column if exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])

        st.subheader("🔍 Preview of Uploaded Data")
        st.dataframe(df.head())

        # Option to apply scaling
        scale_checkbox = st.checkbox("🔄 Apply Standard Scaling (check if data is unscaled)", value=True)

        if scale_checkbox:
            scaled_data = scaler.transform(df)
        else:
            scaled_data = df.values  # Assume already scaled

        # Predict cluster segments
        predicted_segments = model.predict(scaled_data)

        # Append predictions to original dataframe
        df['Predicted_Segment'] = predicted_segments

        st.subheader("📊 Predicted Customer Segments")
        st.dataframe(df)

        # Download results
        csv_output = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download predictions as CSV",
            data=csv_output,
            file_name='predicted_segments.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"⚠️ An error occurred while processing: {e}")
else:
    st.info("📁 Please upload a CSV file to get started.")
