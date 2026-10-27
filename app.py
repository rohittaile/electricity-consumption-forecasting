
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Electricity Forecasting System",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Electricity Consumption Forecasting System")
st.write(
    "A machine learning project comparing electricity "
    "forecasting models."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

linear_model = joblib.load(
    os.path.join(BASE_DIR, "linear_regression.joblib")
)

rf_model = joblib.load(
    os.path.join(BASE_DIR, "random_forest.joblib")
)

results_path = os.path.join(
    BASE_DIR, "model_comparison.csv"
)

st.subheader("Forecast Electricity Consumption")

previous_power = st.number_input(
    "Previous hour's average power (kW)",
    min_value=0.0,
    max_value=20.0,
    value=1.0,
    step=0.1
)

hour = st.slider(
    "Hour of the day",
    min_value=0,
    max_value=23,
    value=12
)

day = st.selectbox(
    "Day of the week",
    options=list(range(7)),
    format_func=lambda x: [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ][x]
)

model_choice = st.selectbox(
    "Select forecasting model",
    ["Linear Regression", "Random Forest"]
)

if st.button("Forecast Next Hour"):
    input_data = pd.DataFrame(
        [[previous_power, hour, day]],
        columns=[
            "previous_hour_power",
            "hour",
            "day_of_week"
        ]
    )

    if model_choice == "Linear Regression":
        prediction = linear_model.predict(input_data)[0]
    else:
        prediction = rf_model.predict(input_data)[0]

    st.success(
        f"Predicted average active power: {prediction:.3f} kW"
    )

    st.caption(
        "This is a one-hour-ahead estimate based on the "
        "inputs provided. It is not a utility bill or "
        "a guaranteed measurement."
    )

st.divider()
st.subheader("Model Performance Comparison")

if os.path.exists(results_path):
    results = pd.read_csv(results_path)
    st.dataframe(results, use_container_width=True)

    st.bar_chart(
        results.set_index("Model")[["MAE (kW)", "RMSE (kW)"]]
    )
else:
    st.info("Model comparison results are not available yet.")
