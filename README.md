# Ship Fuel Efficiency and CO₂ Emissions Optimization

## Project Overview
The maritime industry is a significant contributor to global fuel consumption and CO₂ emissions. Optimizing fuel usage can lead to cost savings and reduced environmental impact. This project leverages machine learning to predict fuel consumption based on ship attributes, route details, and weather conditions. The best-performing model is deployed as a web application for real-time predictions.

## Problem Statement
Fuel consumption and CO₂ emissions are critical concerns for the shipping industry. Inefficient fuel usage increases operational costs and environmental impact. The challenge is to develop a predictive model that accurately estimates fuel consumption, allowing companies to optimize their shipping operations and minimize emissions.

## Project Objectives
- Analyze factors influencing fuel consumption and CO₂ emissions.
- Develop machine learning models to predict fuel usage based on ship characteristics, routes, and weather conditions.
- Identify the most influential factors affecting fuel efficiency.
- Deploy the best-performing model as a web application for easy accessibility.

## Dataset
The dataset, `ship_fuel_efficiency.csv`, contains the following columns:
- **ship_id**: Unique identifier for each ship.
- **ship_type**: Type of ship.
- **route_id**: Identifier for the route taken.
- **month**: Month of the voyage.
- **distance**: Distance traveled (in nautical miles).
- **fuel_type**: Type of fuel used.
- **fuel_consumption**: Amount of fuel consumed (in liters).
- **CO2_emissions**: CO₂ emissions (in kg).
- **weather_conditions**: Weather conditions during the voyage.
- **engine_efficiency**: Efficiency of the ship’s engine.

## Methodology
1. **Data Preprocessing**
   - Categorical variables were encoded using Label Encoding.
   - The `ship_id` column was dropped as it was not relevant for prediction.
   - The dataset was split into training and test sets.
   - Feature scaling was applied using StandardScaler.
   - PCA was used for dimensionality reduction.

2. **Exploratory Data Analysis (EDA)**
   - A correlation matrix was plotted to understand feature relationships and identify key factors influencing fuel consumption.

3. **Model Training & Evaluation**
   - Various machine learning models were trained and evaluated:
     - Linear Regression
     - Random Forest
     - Gradient Boosting
     - XGBoost
     - LightGBM
   - The models were assessed based on Mean Squared Error (MSE) and R² Score.

   | Model               | MSE          | R² Score  |
   |--------------------|--------------|----------|
   | Linear Regression  | 5.25e+06     | 0.776    |
   | Random Forest      | 2.03e+06     | 0.913    |
   | Gradient Boosting  | 1.93e+06     | 0.917    |
   | XGBoost           | 1.91e+06     | 0.918    |
   | LightGBM          | 1.89e+06     | 0.919    |

4. **Cross-Validation & Model Selection**
   - The top three models (LightGBM, XGBoost, Gradient Boosting) were further validated using cross-validation.
   - Hyperparameter tuning was performed for Gradient Boosting to improve performance.
   - The best model was selected based on cross-validation results and generalization performance.

5. **Model Deployment**
   - The optimized Gradient Boosting model, StandardScaler, and PCA were saved using joblib.
   - The model was deployed as a web application using Streamlit.

## Deployment
### Dependencies
Ensure the following dependencies are installed:
```
joblib
numpy
scikit-learn==1.5.1
streamlit
```
### Running the App
To run the Streamlit application, use:
```bash
streamlit run app.py
```
The deployed application can be accessed here: [Ship Fuel and CO2 Emission Model Optimization](https://ship-fuel-and-co2-emission-model-optimization.streamlit.app/)

## Conclusion
This project successfully developed and deployed a machine learning model to predict ship fuel consumption. By leveraging real-world data, we provided an accessible solution for optimizing fuel usage, reducing costs, and minimizing environmental impact. Future improvements could include incorporating real-time weather data and refining models with additional feature engineering techniques.

