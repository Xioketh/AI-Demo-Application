# Machine Learning Projects Showcase

## Overview
This repository contains a unified Streamlit web application featuring 5 key machine learning projects that demonstrate various fundamental techniques in data science and machine learning. The application provides an interactive interface for users to explore different ML models and their applications.

## Live Demo
Access the live application at: https://ai-demo-application-zpovajianqftbfbmswcb58.streamlit.app

## Featured Projects

### 1. Housing Price Predictor
- **Technique**: Multiple Linear Regression
- **Features**: One-hot encoding for categorical variables
- **Functionality**: Predicts housing prices based on various features like area, location, bedrooms, etc.

### 2. Iris Flower Classifier
- **Technique**: Logistic Regression (Multiclass)
- **Features**: Feature scaling for optimal performance
- **Functionality**: Classifies iris flowers into three species based on sepal and petal measurements

### 3. Customer Segmentation
- **Technique**: K-Means Clustering
- **Features**: PCA for dimensionality reduction and visualization
- **Functionality**: Groups customers into clusters based on their characteristics

### 4. Email Spam Classifier
- **Technique**: Binary Logistic Regression
- **Features**: TF-IDF vectorization for text processing
- **Functionality**: Classifies emails as spam or not spam

### 5. Movie Review Sentiment Analyzer
- **Technique**: Multiple Classifiers (SVM, Random Forest, Logistic Regression)
- **Features**: GridSearchCV for hyperparameter tuning
- **Functionality**: Analyzes movie reviews to determine positive or negative sentiment

## Key ML Techniques Applied
- Regression (Linear Regression)
- Classification (Logistic Regression, SVM, Random Forest)
- Clustering (K-Means)
- Regularization
- Cross-validation
- Dimensionality Reduction (PCA)
- Text Vectorization (TF-IDF)

## Technical Implementation
- Streamlit for web application interface
- Scikit-learn for machine learning models
- Pipeline for streamlined workflows
- StandardScaler for feature scaling
- TfidfVectorizer for text processing
- GridSearchCV for hyperparameter tuning
- Joblib for model serialization

## Installation
To run this application locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/Xioketh/AI-Demo-Application.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Select a project from the sidebar menu
2. Interact with the input parameters (if applicable)
3. View the model's predictions or analysis
4. Explore different settings to see how they affect the results

## Contribution
Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
