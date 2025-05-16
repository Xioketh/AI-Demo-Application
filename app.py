import streamlit as st
import pandas as pd
from joblib import load

# Sidebar navigation
st.sidebar.title("Choose a Predictor")
option = st.sidebar.radio("Go to", [
    "Housing Price Predictor",
    "Iris Flower Classification",
    "Customer Segmentation",
    "Email Spam Classification",
    "Movie Review Analysis"
])

if option == "Housing Price Predictor":
    model = load('trained_models/house_price_predict_model.joblib')
    st.title('🏠 Housing Price Predictor')
    st.markdown("Predict the price of a house based on its features.")

    st.header('House Features')
    col1, col2 = st.columns(2)
    with col1:
        area = st.slider('Area (sqft)', 500, 5000, 1000, step=50)
        bedrooms = st.slider('Number of Bedrooms', 1, 10, 3)
    with col2:
        location = st.selectbox('Location', ['Jaela', 'Colombo', 'Kadawatha', 'Nittambuwa', 'Minuwangoda'])

    bedroom_size = area / bedrooms if bedrooms > 0 else 0
    input_data = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Location': [location.lower()],
        'Bedroom_Size': [bedroom_size]
    })

    if st.button('Predict Price'):
        try:
            prediction = model.predict(input_data)[0]
            st.success(f'### Predicted Price: ${prediction:,.2f}')
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    st.markdown("""
    ### 🛠️ Technologies Used:
    - **Linear Regression (Multiple Variables)**: To predict house prices.
    - **One Hot Encoding**: Used `OneHotEncoder` to handle the `Location` feature.
    - **Training and Testing Split**: Data was split using `train_test_split`.
    - **Pipeline**: Combined preprocessing and model using `Pipeline`.
    - **Model Deployment**: App built using **Streamlit**.
    """)

elif option == "Iris Flower Classification":
    model = load("trained_models/iris_model.pkl")
    class_names = ["Setosa", "Versicolor", "Virginica"]

    st.title("🌸 Iris Flower Classification")
    st.write("Enter flower measurements to predict its species.")
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    if st.button("Predict"):
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"])
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted species is: **{class_names[prediction]}**")

    st.markdown("""
    ### 🛠️ Technologies Used:
    - **Logistic Regression (Multiclass Classification)**: For classifying Iris-setosa, Iris-versicolor, Iris-virginica.
    - **StandardScaler**: Used to normalize feature values.
    - **Pipeline**: For seamless integration of scaling and modeling.
    - **Training and Testing Split**: To evaluate model performance.
    - **Model Saving**: Exported using `joblib`.
    - **Model Deployment**: Interactive interface with **Streamlit**.
    """)

elif option == "Customer Segmentation":
    model_data = load("trained_models/customer_segmentation_model.pkl")
    scaler = model_data["scaler"]
    kmeans = model_data["kmeans"]
    pca = model_data["pca"]

    st.title("🧠 Customer Segmentation")
    st.write("Predict the customer segment based on income and spending score.")
    income = st.slider("Annual Income (k$)", 10.0, 150.0, 60.0)
    score = st.slider("Spending Score (1-100)", 1.0, 100.0, 50.0)

    if st.button("Segment Customer"):
        input_df = pd.DataFrame([[income, score]], columns=["Annual Income (k$)", "Spending Score"])
        scaled = scaler.transform(input_df)
        cluster = kmeans.predict(scaled)[0]
        st.success(f"🧾 The customer belongs to **Segment {cluster}**")

    st.markdown("""
    ### 🛠️ Technologies Used:
    - **K Means Clustering Algorithm**: Grouped customers into clusters based on income and spending.
    - **StandardScaler**: Standardized data before clustering.
    - **PCA (Principal Component Analysis)**: Reduced features to 2D for visualization.
    - **Model Saving**: Saved using `joblib` for reuse.
    - **Model Deployment**: Visualized and interacted using **Streamlit**.
    """)

elif option == "Email Spam Classification":
    model = load("trained_models/spam_classifier.pkl")

    st.title("📧 Email Spam Classification")
    email = st.text_area("Enter email content:")

    if st.button("Check Spam"):
        pred = model.predict([email])[0]
        st.success("This is SPAM!" if pred == 1 else "This is NOT spam.")

    st.markdown("""
    ### 🛠️ Technologies Used:
    - **Logistic Regression (Binary Classification)**: To classify messages as spam or ham.
    - **TF-IDF Vectorization**: Converted text to numerical values.
    - **Pipeline**: Combined TF-IDF and classifier.
    - **Training and Testing Split**: Ensured accurate model evaluation.
    - **Model Saving**: Used `joblib` for persistence.
    - **Model Deployment**: Interactive UI created with **Streamlit**.
    """)

elif option == "Movie Review Analysis":
    model = load("trained_models/sentiment_model.pkl")

    st.title("🎬 Movie Review Sentiment Analysis")
    review = st.text_area("Enter movie review:")

    if st.button("Analyze"):
        prediction = model.predict([review])[0]
        st.success("Positive Review 🎉" if prediction == 1 else "Negative Review 😞")

    st.markdown("""
    ### 🛠️ Technologies Used:
    - **Logistic Regression, SVM, Random Forest**: Tested multiple models for comparison.
    - **TF-IDF & Count Vectorizer**: Text vectorization methods.
    - **Text Preprocessing**: Cleaned and lemmatized reviews using NLTK.
    - **Hyperparameter Tuning (GridSearchCV)**: Optimized model performance.
    - **L1 and L2 Regularization**: Applied in logistic regression.
    - **Evaluation**: Used accuracy score and classification report.
    - **Deployment**: Built and served using **Streamlit**.
    """)

# Footer
st.markdown("---")
st.markdown("*Note: The underlying models in this application were trained using demonstration datasets for illustrative purposes.*")
