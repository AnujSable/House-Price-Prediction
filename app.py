import streamlit as st
import joblib
import pandas as pd

# Load the saved model artifacts
# These files were created by your train_model.py script
try:
    model = joblib.load("house_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    model_columns = joblib.load("model_columns.pkl")
except FileNotFoundError:
    st.error("Model files not found. Please run 'train_model.py' first.")
    st.stop()

st.title("🏠 House Price Prediction")

# --- Inputs ---
area = st.number_input("Area (sq ft)", 300, 8000, 1200)
bed = st.slider("Bedrooms", 1, 6, 2)
bath = st.slider("Bathrooms", 1, 4, 2)
floor = st.slider("Floors", 1, 3, 1)
age = st.slider("House Age", 0, 50, 10)

# Options match your CSV data exactly
location = st.selectbox("Location", ["Suburban", "Rural", "Downtown", "Urban"])
condition = st.selectbox("Condition", ["Poor", "Fair", "Good", "Excellent"])
garage = st.selectbox("Garage", ["No", "Yes"])

if st.button("Predict Price"):
    # 1. Prepare Base Features
    areaperbedroom = area / bed
    
    # 2. Create input dictionary with all columns initialized to 0
    input_data = {col: 0 for col in model_columns}
    
    # 3. Fill Numeric Values
    input_data['area'] = area
    input_data['bedrooms'] = bed
    input_data['bathrooms'] = bath
    input_data['floors'] = floor
    input_data['houseage'] = age
    input_data['areaperbedroom'] = areaperbedroom
    
    # 4. Fill Categorical (One-Hot Encoded)
    # The keys must match how get_dummies names them: column_Value
    
    # Condition
    cond_col = f"condition_{condition}"
    if cond_col in input_data:
        input_data[cond_col] = 1
        
    # Garage
    garage_col = f"garage_{garage}"
    if garage_col in input_data:
        input_data[garage_col] = 1
        
    # Location
    loc_col = f"location_{location}"
    if loc_col in input_data:
        input_data[loc_col] = 1

    # 5. Create DataFrame
    df_input = pd.DataFrame([input_data])
    
    # 6. Scale Numeric Columns
    # Use the exact same columns used in training
    num_cols = ['area', 'bedrooms', 'bathrooms', 'floors', 'houseage', 'areaperbedroom']
    
    # Transform only if columns exist
    if all(col in df_input.columns for col in num_cols):
        df_input[num_cols] = scaler.transform(df_input[num_cols])
    
    # 7. Predict
    prediction = model.predict(df_input)[0]
    
    st.success(f"Estimated House Price: ₹ {int(prediction):,}")