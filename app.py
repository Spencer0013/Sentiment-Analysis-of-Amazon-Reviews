# app.py
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from sentimentanalyzer.utils.common import load_fasttext_file, convert_labels

# App title and description
st.title("Amazon Reviews Sentiment Analyzer")
st.markdown("""
Analyze sentiment of product reviews using our NLP model
""")

# Load data and model
@st.cache_resource
def load_data():
    data_path = 'artifacts/data_transformation/test_ft.txt'
  # Update with your data path
    return load_fasttext_file(data_path)

@st.cache_resource
def load_model():
    model_path = Path("artifacts/model_trainer/sentiment_model")  # Update with your model path
    return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

# Load data and model
texts, labels = load_data()
model = load_model()

# Display sample data
if st.checkbox("Show sample data"):
    sample_df = pd.DataFrame({
        "Review": texts[:10],
        "Label": convert_labels(labels[:10])
    })
    st.dataframe(sample_df)

# Sentiment analysis section
st.header("Sentiment Analysis")
user_input = st.text_area("Enter a review to analyze:", "This product is amazing!")

if st.button("Analyze Sentiment"):
    # Preprocess and predict
    prediction = model.predict([user_input])
    sentiment = "Positive" if np.argmax(prediction) == 1 else "Negative"
    confidence = np.max(prediction)
    
    # Display results
    st.subheader("Results")
    st.metric("Sentiment", sentiment)
    st.metric("Confidence", f"{confidence:.2%}")
    st.progress(float(confidence))
    
    # Show probabilities
    prob_df = pd.DataFrame({
        "Sentiment": ["Negative", "Positive"],
        "Probability": prediction[0]
    })
    st.bar_chart(prob_df.set_index("Sentiment"))

# Batch processing section
st.header("Batch Processing")
uploaded_file = st.file_uploader("Upload reviews file (CSV or TXT)", type=["csv", "txt"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        batch_df = pd.read_csv(uploaded_file)
        # Assuming CSV has 'review' column
        reviews = batch_df["review"].tolist()
    else:  # TXT file
        reviews = [line.decode("utf-8").strip() for line in uploaded_file.readlines()]
    
    # Process in batches
    results = []
    for review in reviews:
        prediction = model.predict([review])
        sentiment = "Positive" if np.argmax(prediction) == 1 else "Negative"
        confidence = np.max(prediction)
        results.append({
            "Review": review,
            "Sentiment": sentiment,
            "Confidence": confidence
        })
    
    results_df = pd.DataFrame(results)
    st.dataframe(results_df)
    
    # Download button
    st.download_button(
        label="Download Results",
        data=results_df.to_csv(index=False),
        file_name="sentiment_results.csv",
        mime="text/csv"
    )