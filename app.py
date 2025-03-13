import streamlit as st
import joblib
import numpy as np

# Function to set full-page background image
def set_background(image_file):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url(data:image/jpg;base64,{image_file});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    /* Darker select boxes & number inputs */
    .stSelectbox, .stNumberInput {{
        background-color: rgba(0, 0, 0, 0.7) !important;  
        color: white !important;  
        border-radius: 10px;
        padding: 8px;
    }}
    /* Styling sub-headers */
    .subheader1 {{
        font-size: 20px;
        font-weight: bold;
        color: #ffffff; /* White text */
        text-align: center;
        margin-top: -10px;
    }}
    .subheader2 {{
        font-size: 18px;
        font-weight: normal;
        color: #ffffff; /* White text */
        text-align: center;
        margin-bottom: 20px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Convert image to Base64
import base64
def get_base64_image(image_path):
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode()

# Apply background image
set_background(get_base64_image("shipping.jpg"))

# Load CO₂ model, scaler, and PCA
co2_model = joblib.load("model_co2.joblib")
co2_scaler = joblib.load("scaler_co2.joblib")
co2_pca = joblib.load("pca_co2.joblib")

# Load Fuel model, scaler, and PCA
fuel_model = joblib.load("model_fuel.joblib")
fuel_scaler = joblib.load("scaler_fuel.joblib")
fuel_pca = joblib.load("pca_fuel.joblib")

# Streamlit UI
st.title("Ship Fuel & CO₂ Emission Predictor")

# Sub-headers with explanation and instructions
st.markdown('<p class="subheader1">Predicting fuel consumption and CO₂ emissions to optimize efficiency and reduce environmental impact.</p>', unsafe_allow_html=True)

st.markdown('<p class="subheader2">Enter the required details below and click "Predict Fuel & CO₂" to get accurate estimations.</p>', unsafe_allow_html=True)


# Ship Type Selection
ship_types = ['Oil Service Boat', 'Fishing Trawler', 'Surfer Boat', 'Tanker Ship']
ship_type = st.selectbox("Select Ship Type", ship_types)

# Route Selection
routes = ['Warri-Bonny', 'Port Harcourt-Lagos', 'Lagos-Apapa', 'Escravos-Lagos']
route = st.selectbox("Select Route", routes)

# Month Selection
months = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]
month = st.selectbox("Select Month", months)

# Distance Input
distance = st.number_input("Enter Distance (km)", min_value=1.0, step=0.1)

# Fuel Type Selection
fuel_types = ["HFO", "Diesel"]
fuel_type = st.selectbox("Select Fuel Type", fuel_types)

# Weather Condition
weather_conditions = ["Stormy", "Moderate", "Calm"]
weather = st.selectbox("Select Weather Condition", weather_conditions)

# Engine Efficiency (Number Input)
engine_efficiency = st.number_input("Enter Engine Efficiency (%)", min_value=0.0, max_value=100.0, step=0.1)

# Encode Categorical Variables
ship_type_encoded = ship_types.index(ship_type)
route_encoded = routes.index(route)
month_encoded = months.index(month)
fuel_type_encoded = fuel_types.index(fuel_type)
weather_encoded = weather_conditions.index(weather)

# Prepare Input Data
input_data = np.array([[ship_type_encoded, route_encoded, month_encoded, distance, fuel_type_encoded, weather_encoded, engine_efficiency]])

# Apply StandardScaler and PCA for both models
input_scaled_fuel = fuel_scaler.transform(input_data)
input_pca_fuel = fuel_pca.transform(input_scaled_fuel)

input_scaled_co2 = co2_scaler.transform(input_data)
input_pca_co2 = co2_pca.transform(input_scaled_co2)

# Prediction Button
if st.button("Predict Fuel & CO₂"):
    fuel_prediction = fuel_model.predict(input_pca_fuel)[0]
    co2_prediction = co2_model.predict(input_pca_co2)[0]

    st.success(f" Estimated Fuel Consumption: **{fuel_prediction:.2f} liters**")
    st.success(f" Estimated CO₂ Emission: **{co2_prediction:.2f} g/km**")

    # Disclaimer Note
    st.markdown('<p class="disclaimer">Note: Predictions are based on historical data and may not be fully accurate for all cases. Consult experts for precise evaluations.</p>', unsafe_allow_html=True)