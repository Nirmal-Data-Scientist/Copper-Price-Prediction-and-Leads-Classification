# Copper Price Prediction and Leads Classification Web App
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://industrial-copper-modelling.streamlit.app/)

The Copper Price Prediction and Leads Classification project is a web application that allows users to predict the price of copper and classify leads as "Won" or "Lost" based on various features related to copper products. The application is powered by machine learning models, including a Random Forest Regressor for price prediction and a Random Forest Classifier for leads classification.

## Prerequisites

Before you begin, ensure you have the following tools installed:

- Python 3.7 or higher
- scikit-learn
- pandas
- numpy
- streamlit

## Features

### Copper Price Prediction

- Predictive Analysis: The app utilizes a Random Forest Regressor model to predict the price of copper based on various features such as quantity, customer ID, country code, item type, application, thickness, and width.
- User Input: Users can enter their own copper product details, including quantity, customer ID, country code, leads, item type, application, thickness, and width, to generate personalized price predictions.
- Scalable Data: The app includes a data scaling function that standardizes the input data, ensuring accurate predictions by bringing the features to a similar scale.

### Leads Classification

- Predictive Analysis: The app utilizes a Random Forest Classifier model to classify leads as "Won" or "Lost" based on various features such as quantity, customer ID, country code, item type, application, thickness, width, and selling price.
- User Input: Users can enter their own copper product details, including quantity, customer ID, country code, item type, application, thickness, width, and selling price, to receive personalized leads classification.
- Scalable Data: The app includes a data scaling function that standardizes the input data, ensuring accurate classification by bringing the features to a similar scale.

## User Guide

1. Go to the web app URL in your web browser.
2. For Copper Price Prediction:
   - Enter the copper product details, including quantity, customer ID, country code, leads, item type, application, thickness, and width in the provided text input fields.
   - Click the "Predict Price" button to generate the price prediction based on the provided input.

3. For Leads Classification:
   - Enter the copper product details, including quantity, customer ID, country code, item type, application, thickness, width, and selling price in the provided text input fields.
   - Click the "Predict Leads" button to classify the leads as "Won" or "Lost" based on the provided input.

## Developer Guide

To run the app, follow these steps:

1. Clone the repository to your local machine using the following command: `git clone [repository_url]`.
2. Install the required libraries by running the following command: `pip install -r requirements.txt`.
3. Open a terminal window and navigate to the directory where the app is located using the following command: `cd [app_directory]`.
4. Run the command `streamlit run app.py` to start the app.
5. The app should now be running on a local server. If it doesn't start automatically, you can access it by going to either:
   - Local URL: [http://localhost:8501]
   - Network URL: [http://192.168.0.1:8501] (replace with your machine's IP address)

## Web App Snap
![image](https://github.com/Nirmal-Data-Scientist/Copper-Price-Prediction-and-Leads-Classification/assets/123751119/0930d61f-ea68-4bcf-9805-1908581f2c44)

## Streamlit web URL

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://industrial-copper-modelling.streamlit.app/)

## Disclaimer

The Copper Price Prediction and Leads Classification App serves as a tool for preliminary analysis and should not be considered as financial advice or professional guidance. Users should conduct in-depth research and consult financial experts for critical decision-making.

## Contact

If you have any questions, comments, or suggestions for the app, please feel free to contact me at [nirmal.works@outlook.com].
