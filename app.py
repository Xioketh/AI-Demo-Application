import streamlit as st
import pandas as pd
import pickle


tab1, tab2, tab3 = st.tabs(["🏠 Housing Price Predictor", "Dog", "Owl"])

with tab1:
    # Load the saved model
    with open('house_price_predict_model', 'rb') as f:
        model = pickle.load(f)

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
    st.header("A dog")
    st.image("https://static.streamlit.io/examples/dog.jpg", width=200)
with tab3:
    st.header("An owl")
    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)


# Optional: Add a footer
st.markdown("---")
st.markdown("*Note: This is a demo application using a pre-trained model.*")