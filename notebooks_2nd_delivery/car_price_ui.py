import streamlit as st
import pandas as pd
import joblib

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Cars 4 You - Price Estimator")

st.title("Cars 4 You – Car Price Estimator")
st.write("Enter the car details to estimate its resale price.")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load("models/models_organized/hgb_params.pkl")

model_pipeline = load_model()

# ---------------- MODELS BY BRAND ----------------
models_by_brand = {
    "Audi": ['A', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q', 'Q2', 'Q3', 'Q5', 'Q7', 'Q8',
             'R8', 'Rs', 'Rs3', 'Rs4', 'Rs5', 'Rs6', 'S3', 'S4', 'S5', 'S8', 'Sq5', 'Sq7', 'T', 'Tt'],
    "BMW": ['1 series', '2 serie', '4 series', '6 series', '7 series', '8 series', 'I', 'I3', 'I8',
            'M', 'M2', 'M3', 'M4', 'M5', 'M6', 'X', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7',
            'Z', 'Z3', 'Z4'],
    "Ford": ['C-ma', 'Ecosport', 'Edge', 'Escort', 'Fiesta', 'Focus', 'Fusion', 'Galaxy',
             'Grand c-max', 'K', 'Ka', 'Kuga', 'Mondeo', 'Mustang', 'Puma', 'Ranger',
             'S-max', 'Streetka', 'Tourneo connect', 'Tourneo custom'],
    "Hyundai": ['Accent', 'Getz', 'I1', 'I10', 'I2', 'I3', 'I30', 'I40', 'I800', 'Ioniq',
                'Ix20', 'Ix35', 'Kona', 'Santa fe', 'Terracan', 'Tucson', 'Veloste'],
    "Mercedes": ['200', '220', '230', 'A clas', 'A class', 'B clas', 'C clas', 'C class',
                 'Cl clas', 'Clk', 'Cls clas', 'Cls class', 'E clas', 'G class', 'Gla clas',
                 'Glb class', 'Glc clas', 'Gle clas', 'Gls clas', 'M clas', 'M class',
                 'S clas', 'Sl', 'Sl clas', 'Sl class', 'Slk', 'V clas', 'X-clas', 'X-class'],
    "Opel": ['Adam', 'Agila', 'Ampera', 'Antara', 'Astra', 'Cascada', 'Combo life', 'Corsa',
             'Crossland x', 'Grandland x', 'Gtc', 'Insignia', 'Kadjar', 'Meriva', 'Mokk',
             'Mokka x', 'Tigra', 'Vectra', 'Viva', 'Zafira', 'Zafira tourer'],
    "Skoda": ['Citigo', 'Fabia', 'Kamiq', 'Karoq', 'Kodiaq', 'Octavia', 'Rapid', 'Roomster',
              'Scala', 'Superb', 'Yeti', 'Yeti outdoor'],
    "Toyota": ['Auri', 'Avensis', 'Aygo', 'C-hr', 'Camry', 'Corolla', 'Gt86', 'Hilux', 'Iq',
               'Land cruiser', 'Prius', 'Proace verso', 'Rav4', 'Supra', 'Verso', 'Yaris'],
    "VW": ['Amarok', 'Arteon', 'Beetle', 'Caddy', 'Caddy maxi life', 'California', 'Caravelle',
           'Cc', 'Eos', 'Fox', 'Golf', 'Golf sv', 'Jetta', 'Passat', 'Polo', 'Scirocco',
           'Sharan', 'Shuttle', 'T-cross', 'T-roc', 'Tiguan', 'Tiguan allspace', 'Touareg',
           'Touran', 'U', 'Up']
}

# ---------------- INPUTS ----------------
brand = st.selectbox("Brand", sorted(models_by_brand.keys()))
model_name = st.selectbox("Model", models_by_brand[brand])

car_age = st.number_input("Car Age (years)", min_value=0)
transmission = st.selectbox("Transmission", ['Automatic', 'Manual', 'Other', 'Semi-Auto', 'Unknown'])
mileage = st.number_input("Mileage (km)", 0, 999999)
fuel_type = st.selectbox("Fuel Type", ['Diesel', 'Hybrid', 'Other', 'Petrol'])
tax = st.number_input("Tax", min_value=0)
mpg = st.number_input("MPG", min_value=0.0)
engine_size = st.number_input("Engine Size (L)", min_value=0.5)
previous_owners = st.number_input("Previous Owners", 0, 1000)

# ---------------- PREDICTION ----------------
if st.button("Estimate Price"):

    input_df = pd.DataFrame([{
        "Brand": brand,
        "model": model_name,
        "car_age": car_age,
        "transmission": transmission,
        "mileage": mileage,
        "fuelType": fuel_type,
        "tax": tax,
        "mpg": mpg,
        "engineSize": engine_size,
        "previousOwners": previous_owners
    }])

    predicted_price = model_pipeline.predict(input_df)[0]

    st.success(f"Estimated resale price: £{predicted_price:,.0f}")
