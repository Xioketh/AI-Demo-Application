import streamlit as st
import pandas as pd
import pickle
from joblib import load



tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Housing Price Predictor",
                            "🌸 Iris Flower Classification",
                            "🧠 Customer Segmentation",
                            "📦 Email Spam Classification",
                            "🎬 Movie Review Analysis"
                            ])

with tab1:
    # Load the saved model
    model = load('trained_models/house_price_predict_model.joblib')


    # Streamlit app
    # st.title('🏠 Housing Price Predictor')

    st.markdown("""
    Predict the price of a house based on its features.
    Adjust the parameters using the sliders and dropdown below.
    """)

    # User input section
    st.header('House Features')

    col1, col2 = st.columns(2)

    with col1:
        area = st.slider('Area (sqft)', min_value=500, max_value=5000, value=1000, step=50)
        bedrooms = st.slider('Number of Bedrooms', min_value=1, max_value=10, value=3)

    with col2:
        location = st.selectbox('Location', ['Jaela', 'Colombo', 'Kadawatha', 'Nittambuwa', 'Minuwangoda'])

    # Calculate bedroom size automatically
    bedroom_size = area / bedrooms if bedrooms > 0 else 0

    # Create input dataframe
    input_data = pd.DataFrame({
        'Area': [area],
        'Bedrooms': [bedrooms],
        'Location': [location.lower()],
        'Bedroom_Size': [bedroom_size]
    })

    # Make prediction
    if st.button('Predict Price'):
        try:
            prediction = model.predict(input_data)[0]
            st.success(f'### Predicted Price: ${prediction:,.2f}')

            # Show the calculated bedroom size
            # st.info(f'Calculated Bedroom Size: {bedroom_size:.2f} sqft per bedroom')

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add some explanation
    # st.markdown("""
    # ### How It Works
    # 1. Adjust the house features using the sliders and dropdown
    # 2. Click the "Predict Price" button
    # 3. The model will calculate the price based on:
    #    - Area (square footage)
    #    - Number of bedrooms
    #    - Location
    #    - Bedroom Size (automatically calculated as Area/Bedrooms)
    # """)

with tab2:
    model = load("trained_models/iris_model.pkl")

    # Class names
    class_names = ["Setosa", "Versicolor", "Virginica"]
    st.write("Enter flower measurements to predict its species.")

    # Input sliders
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    # Predict
    if st.button("Predict"):
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                                  columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)",
                                           "petal width (cm)"])
        prediction = model.predict(input_data)[0]
        st.success(f"The predicted species is: **{class_names[prediction]}**")
with tab3:
    model_data = load("trained_models/customer_segmentation_model.pkl")
    scaler = model_data["scaler"]
    kmeans = model_data["kmeans"]
    pca = model_data["pca"]

    st.write("Predict the customer segment based on income and spending score.")

    # Input sliders
    income = st.slider("Annual Income (k$)", 10.0, 150.0, 60.0)
    score = st.slider("Spending Score (1-100)", 1.0, 100.0, 50.0)

    # Predict
    if st.button("Segment Customer"):
        input_df = pd.DataFrame([[income, score]], columns=["Annual Income (k$)", "Spending Score"])
        scaled = scaler.transform(input_df)
        cluster = kmeans.predict(scaled)[0]
        st.success(f"🧾 The customer belongs to **Segment {cluster}**")

with tab4:
    st.header('Null')

with tab5:
    model = load("trained_models/sentiment_model.pkl")
    review = st.text_area("Enter movie review:")

    if st.button("Analyze"):
        prediction = model.predict([review])[0]
        st.success("Positive Review 🎉" if prediction == 1 else "Negative Review 😞")

# Optional: Add a footer
st.markdown("---")
st.markdown("*Note: This is a demo application using a pre-trained model.*")