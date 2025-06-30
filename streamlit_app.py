import streamlit as st
import pandas as pd
import joblib

# Load saved model and scaler
model = joblib.load("logistic_model.pkl")  # Trained logistic regression model
scaler = joblib.load("scaler.pkl")         # StandardScaler used during training

st.title("Customer Cluster Predictor 🚀")

# Upload CSV file (unscaled/original data)
uploaded_file = st.file_uploader("Upload your CSV file (unscaled data)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load uploaded data
        data = pd.read_csv(uploaded_file)

        # Drop 'Unnamed: 0' if present (from saved CSVs with index)
        if 'Unnamed: 0' in data.columns:
            data = data.drop(columns=['Unnamed: 0'])

        st.subheader("Uploaded Data Preview:")
        st.dataframe(data.head())

        # Scale the unscaled input using the original training scaler
        scaled_data = scaler.transform(data)
        scale_checkbox = st.checkbox("Is your data unscaled? (check this to apply scaling)", value=True)

        if scale_checkbox:
           scaled_data = scaler.transform(data)
        else:
           scaled_data = data  # Already scaled

        # Predict cluster labels using trained model
        predictions = model.predict(scaled_data)

        # Append predictions to original (unscaled) data
        data['Predicted_Cluster'] = predictions

        st.subheader("Predicted Clusters:")
        st.dataframe(data)

        # Download button
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download predictions as CSV",
            data=csv,
            file_name='cluster_predictions.csv',
            mime='text/csv',
        )
    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")
else:
    st.info("📁 Please upload a CSV file to begin.")
