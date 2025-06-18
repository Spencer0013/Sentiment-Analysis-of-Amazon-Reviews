# Sentiment Analysis of Amazon Reviews
# 📌 Overview
This project provides a comprehensive solution for analyzing sentiment in Amazon product reviews using advanced NLP techniques. The system features an end-to-end MLOps pipeline for data processing, model training, and deployment, with a user-friendly Streamlit web interface for real-time sentiment analysis.

# 🚀 Key Features
End-to-end MLOps pipeline with modular stages

BERT-based sentiment classification with 94.96% accuracy

Streamlit web application for real-time analysis

Batch processing for TXT files

CI/CD integration with GitHub Actions

Azure deployment ready

# 📊 Model Performance

{
    "loss": 0.1364,
    "accuracy": 0.9496
}
The model achieves 94.96% accuracy with low loss (0.1364) on test data.

## 🛠️ Technical Stack
Core Framework: TensorFlow 2.13

Models: CNN

MLOps Tools: DVC, MLflow, Streamlit

Cloud Deployment: Azure Web Apps

Containerization: Docker

CI/CD: GitHub Actions

## ⚙️ Installation
Clone the repository:

git clone https://github.com/opeyemi/Sentiment-Analysis-of-Amazon-Reviews.git
cd Sentiment-Analysis-of-Amazon-Reviews

# Create virtual environment:
conda create -p sentA python=3.10 -y

# Install dependencies:
pip install -r requirements.txt

# 🏃‍♂️ Running the Pipeline
python main.py

# Pipeline stages:

- Data Ingestion

- Data Preprocessing

- Data Transformation

- Model Training

- Model Evaluation

# 🌐 Running the Web Application
streamlit run app.py

# 🔧 Configuration
Modify parameters in params.yaml:

max_tokens: 11470
output_sequence_length: 163
input_dim: 11470
output_dim: 107
batch_size: 512
epochs: 2
label_col: "target"
classes: 2
learning_rate : 0.001
input_dtype: int
num_labels: 2
max_length: 163
random_state: 42
dropout_rate: 0.1
dense_units: 64

# 🚢 Deployment
The CI/CD pipeline automatically builds and deploys to Azure on push to main:

# Docs for the Azure Web Apps Deploy action: https://github.com/Azure/webapps-deploy
# More GitHub Actions for Azure: https://github.com/Azure/actions

name: Build and deploy container app to Azure Web App - sentimentanalysis

on:
  push:
    branches:
      - main
  workflow_dispatch:

jobs:
  build:
    runs-on: 'ubuntu-latest'

    steps:
    - uses: actions/checkout@v2

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to registry
      uses: docker/login-action@v2
      with:
        registry: https://sentimentanalysis.azurecr.io/
        username: ${{ secrets.AzureAppService_ContainerUsername_9dfbfe55afa749d7a327ebf7131b60d9 }}
        password: ${{ secrets.AzureAppService_ContainerPassword_36ede34ecd8d4cb88e99b8992f2cf92d }}

    - name: Build and push container image to registry
      uses: docker/build-push-action@v3
      with:
        push: true
        tags: sentimentanalysis.azurecr.io/${{ secrets.AzureAppService_ContainerUsername_9dfbfe55afa749d7a327ebf7131b60d9 }}/sentimentanalysis:${{ github.sha }}
        file: ./Dockerfile

  deploy:
    runs-on: ubuntu-latest
    needs: build
    environment:
      name: 'production'
      url: ${{ steps.deploy-to-webapp.outputs.webapp-url }}

    steps:
    - name: Deploy to Azure Web App
      id: deploy-to-webapp
      uses: azure/webapps-deploy@v2
      with:
        app-name: 'sentimentanalysis'
        slot-name: 'production'
        publish-profile: ${{ secrets.AzureAppService_PublishProfile_68506586f0c141a08933bc31c0b0696e }}
        images: 'sentimentanalysis.azurecr.io/${{ secrets.AzureAppService_ContainerUsername_9dfbfe55afa749d7a327ebf7131b60d9 }}/sentimentanalysis:${{ github.sha }}'

# 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.



