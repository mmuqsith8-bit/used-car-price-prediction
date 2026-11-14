
import json
import joblib
import pandas as pd
import streamlit as st

# Load the trained model and the allowed options
model = joblib.load("used_car_price_model.pkl")

with open("car_options.json") as file:
    options = json.load(file)

st.title("Used Car Price Prediction")
st.write("Sri Lankan used-car market - Machine Learning Demonstration")

st.warning(
    "This application is for academic purposes only. "
    "It estimates the asking price from past advertisements and is not a valuation."
)

# User inputs
brand = st.selectbox("Brand", sorted(options["models_by_brand"].keys()))
car_model = st.selectbox("Model", options["models_by_brand"][brand])

year = st.number_input(
    "Year of Manufacture",
    min_value=options["yom_min"],
    max_value=options["yom_max"],
    value=2015
)

engine = st.number_input(
    "Engine Capacity (cc)",
    min_value=options["engine_min"],
    max_value=options["engine_max"],
    value=1500
)

gear = st.selectbox("Gear", options["gear"])
fuel = st.selectbox("Fuel Type", options["fuel_type"])
town = st.selectbox("Town", options["towns"])
leasing = st.selectbox("Leasing", options["leasing"])
condition = st.selectbox("Condition", options["condition"], index=options["condition"].index("USED"))

st.write("Options")
ac = st.checkbox("Air Conditioning", value=True)
power_steering = st.checkbox("Power Steering", value=True)
power_mirror = st.checkbox("Power Mirrors", value=True)
power_window = st.checkbox("Power Windows", value=True)

st.caption("Mileage is estimated as 11,000 km per year of age, the same rule used in the dataset.")

def available(flag):
    return "Available" if flag else "Not_Available"

if st.button("Predict Price"):

    age = options["reference_year"] - year

    # Build one row with every column the model could need
    row = {
        "Brand": brand,
        "Model": car_model,
        "Brand_Model": brand + " " + car_model,
        "YOM": year,
        "Car_Age": age,
        "Engine (cc)": engine,
        "Millage(KM)": 11000 * age,
        "KM_per_Year": 11000,
        "Gear": gear,
        "Fuel Type": fuel,
        "Town": town,
        "Leasing": leasing,
        "Condition": condition,
        "AIR CONDITION": available(ac),
        "POWER STEERING": available(power_steering),
        "POWER MIRROR": available(power_mirror),
        "POWER WINDOW": available(power_window)
    }

    # Keep only the columns used when the model was trained
    input_data = pd.DataFrame([row])[list(model.feature_names_in_)]

    price = float(model.predict(input_data)[0])

    st.success(f"Estimated price: {price:,.1f} lakhs (about Rs {price * 100000:,.0f})")
