import streamlit as st
import pandas as pd
import numpy as np
import requests

from pathlib import Path
from datetime import date
from sklearn.linear_model import LinearRegression


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Solar Energy Optimizer",
    page_icon="☀️",
    layout="wide"
)


# ==================================================
# SESSION STATE
# ==================================================

DEFAULT_STATE = {
    "daily_results": None,
    "city_results": [],
    "weather_data": None,
    "weather_city": "Ulhasnagar, Maharashtra, India",
    "selected_city_id": None,
    "last_search": "",
    "weather_coordinates": None
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ==================================================
# DATASET CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "solardata.csv"

FEATURES = [
    "temperature_c",
    "humidity_pct",
    "cloud_cover_pct",
    "solar_irradiance_w_m2"
]

TARGET = "solar_generation_kwh"


# ==================================================
# LOAD DATASET
# ==================================================

@st.cache_data
def load_dataset():

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Make sure solardata.csv is in the project folder."
        )

    data = pd.read_csv(DATA_FILE)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in dataset: {missing_columns}"
        )

    for column in required_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.dropna(
        subset=required_columns
    ).copy()

    data = data[
        (data[TARGET] >= 0)
        & (data["solar_irradiance_w_m2"] >= 0)
    ].copy()

    if len(data) < 20:
        raise ValueError(
            "Not enough valid rows to train the model."
        )

    return data


# ==================================================
# TRAIN MACHINE LEARNING MODEL
# ==================================================

@st.cache_resource
def train_model():

    data = load_dataset()

    split_index = int(len(data) * 0.80)

    X_train = data[FEATURES].iloc[:split_index]
    y_train = data[TARGET].iloc[:split_index]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    return model, data


try:

    model, df = train_model()

except Exception as error:

    st.error(
        f"Could not load the dataset or train the model: {error}"
    )

    st.stop()


# ==================================================
# SOLAR CAPACITY
# ==================================================

if "solar_capacity_kw" in df.columns:

    capacities = pd.to_numeric(
        df["solar_capacity_kw"],
        errors="coerce"
    ).dropna()

    capacity_kw = (
        float(capacities.max())
        if not capacities.empty
        else 5.0
    )

else:

    capacity_kw = 5.0

if not np.isfinite(capacity_kw) or capacity_kw <= 0:
    capacity_kw = 5.0


# ==================================================
# PREDICT SOLAR GENERATION
# ==================================================

def predict_generation(weather):

    X = weather[FEATURES].copy()

    predictions = model.predict(X)

    predictions = np.clip(
        predictions,
        0,
        capacity_kw
    )

    night_mask = (
        weather["solar_irradiance_w_m2"].to_numpy() <= 1
    )

    predictions[night_mask] = 0

    return predictions


# ==================================================
# INDIA-ONLY CITY SEARCH
# ==================================================

@st.cache_data(ttl=3600)
def search_city(city_name):

    city_name = city_name.strip()

    if not city_name:
        return []

    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": city_name,
            "count": 100,
            "language": "en",
            "format": "json",
            "countryCode": "IN"
        },
        timeout=20
    )

    response.raise_for_status()

    response_data = response.json()

    # Extra country check for safety.
    results = response_data.get("results", [])

    indian_results = [
        city
        for city in results
        if city.get("country_code", "").upper() == "IN"
    ]

    return indian_results


# ==================================================
# WEATHER FORECAST API
# ==================================================

@st.cache_data(ttl=1800)
def get_weather(latitude, longitude):

    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "cloud_cover,"
                "shortwave_radiation"
            ),
            "timezone": "Asia/Kolkata",
            "forecast_days": 7
        },
        timeout=30
    )

    response.raise_for_status()

    response_data = response.json()

    if "hourly" not in response_data:
        raise ValueError(
            "Weather data was not returned by the API."
        )

    hourly = response_data["hourly"]

    weather = pd.DataFrame({
        "Time": pd.to_datetime(hourly["time"]),
        "temperature_c": hourly["temperature_2m"],
        "humidity_pct": hourly["relative_humidity_2m"],
        "cloud_cover_pct": hourly["cloud_cover"],
        "solar_irradiance_w_m2": hourly["shortwave_radiation"]
    })

    weather = weather.dropna(
        subset=FEATURES
    ).copy()

    return weather


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def format_city(city):

    city_name = city.get(
        "name",
        "Unknown city"
    )

    region = city.get(
        "admin1",
        ""
    )

    country = city.get(
        "country",
        "India"
    )

    parts = [
        part
        for part in [city_name, region, country]
        if part
    ]

    return ", ".join(parts)


def reset_weather_and_results():

    st.session_state.weather_data = None
    st.session_state.daily_results = None
    st.session_state.weather_coordinates = None


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("☀️ Solar Optimizer")

page = st.sidebar.radio(
    "Navigate",
    [
        "Dashboard",
        "Solar Prediction",
        "Manual Prediction",
        "Investment Calculator"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Solar generation prediction using "
    "weather data and machine learning."
)

st.sidebar.caption(
    "📍 Location search restricted to India."
)

st.sidebar.caption(
    f"Model capacity limit: {capacity_kw:.1f} kW"
)


# ==================================================
# DASHBOARD
# ==================================================

if page == "Dashboard":

    st.title("☀️ Solar Energy Dashboard")

    st.caption(
        "Monitor solar generation, electricity consumption "
        "and estimated savings."
    )

    results = st.session_state.daily_results or {}

    generation = results.get("generation", 0.0)
    consumption = results.get("consumption", 0.0)
    grid_import = results.get("grid_import", 0.0)
    savings = results.get("savings", 0.0)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "☀️ Solar Generation",
        f"{generation:.2f} kWh"
    )

    col2.metric(
        "🏠 Energy Consumption",
        f"{consumption:.2f} kWh"
    )

    col3.metric(
        "⚡ Grid Import",
        f"{grid_import:.2f} kWh"
    )

    col4.metric(
        "💰 Estimated Savings",
        f"₹{savings:.2f}"
    )

    st.divider()

    st.subheader("Generation vs Consumption")

    comparison = pd.DataFrame({
        "Energy Type": [
            "Solar Generation",
            "Home Consumption"
        ],
        "Energy (kWh)": [
            generation,
            consumption
        ]
    })

    st.bar_chart(
        comparison.set_index("Energy Type")
    )

    st.divider()

    st.subheader("Electricity Bill Comparison")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Bill Without Solar",
        f"₹{results.get('bill_without_solar', 0):.2f}"
    )

    col2.metric(
        "Estimated Bill With Solar",
        f"₹{results.get('bill_with_solar', 0):.2f}"
    )

    col3.metric(
        "Potential Export",
        f"{results.get('export_energy', 0):.2f} kWh"
    )

    if results:

        st.success(
            f"Latest calculation: "
            f"{results.get('city', 'Unknown')} | "
            f"{results.get('date', '')}"
        )

    else:

        st.info(
            "No calculation available yet. "
            "Open Solar Prediction and calculate your daily savings."
        )

    st.caption(
        "Figures are estimates from the selected dataset and "
        "weather forecast, not actual meter readings."
    )


# ==================================================
# SOLAR PREDICTION
# ==================================================

elif page == "Solar Prediction":

    st.title("🌞 Solar Generation Prediction")

    st.caption(
        "Search Indian cities, fetch their weather forecast "
        "and estimate hourly solar generation."
    )

    with st.form("city_search_form"):

        city_name = st.text_input(
            "📍 Enter Indian City Name",
            value="Ulhasnagar",
            placeholder="Mumbai, Pune, Delhi, Nashik..."
        )

        search_button = st.form_submit_button(
            "🔍 Search Indian City",
            type="primary"
        )

    if search_button:

        if not city_name.strip():

            st.warning(
                "Please enter an Indian city name."
            )

        else:

            try:

                with st.spinner(
                    f"Searching India for {city_name}..."
                ):

                    cities = search_city(city_name)

                st.session_state.city_results = cities
                st.session_state.last_search = city_name.strip()

                reset_weather_and_results()

                if cities:

                    st.success(
                        f"Found {len(cities)} matching Indian location(s)."
                    )

                else:

                    st.warning(
                        "No matching Indian locations found. "
                        "Try Mumbai, Pune, Delhi, Ulhasnagar or Thane."
                    )

            except requests.RequestException as error:

                st.error(
                    f"City search failed: {error}"
                )

            except Exception as error:

                st.error(
                    f"Unexpected error: {error}"
                )

    cities = st.session_state.city_results

    if cities:

        st.subheader("📍 Indian Search Results")

        labels = [
            format_city(city)
            for city in cities
        ]

        selected_index = st.selectbox(
            "Select your Indian city or location",
            options=range(len(cities)),
            format_func=lambda index: labels[index],
            key="selected_city_index"
        )

        selected_city = cities[selected_index]

        st.caption(
            f"Latitude: {selected_city['latitude']:.4f} | "
            f"Longitude: {selected_city['longitude']:.4f} | "
            f"Country: India"
        )

        fetch_weather = st.button(
            "🌦️ Fetch Weather",
            type="primary"
        )

        if fetch_weather:

            try:

                with st.spinner(
                    f"Fetching weather for {selected_city['name']}..."
                ):

                    weather = get_weather(
                        selected_city["latitude"],
                        selected_city["longitude"]
                    )

                st.session_state.weather_data = (
                    weather.to_dict("records")
                )

                st.session_state.weather_city = (
                    labels[selected_index]
                )

                st.session_state.weather_coordinates = (
                    selected_city["latitude"],
                    selected_city["longitude"]
                )

                st.session_state.daily_results = None

                st.success(
                    f"Weather loaded for {labels[selected_index]}!"
                )

            except requests.RequestException as error:

                st.error(
                    f"Weather API error: {error}"
                )

            except Exception as error:

                st.error(
                    f"Could not load weather: {error}"
                )

    records = st.session_state.weather_data

    if records:

        weather = pd.DataFrame(records)

        weather["Time"] = pd.to_datetime(
            weather["Time"]
        )

        st.divider()

        st.subheader(
            f"Weather Forecast: {st.session_state.weather_city}"
        )

        st.caption(
            "Forecast provided by Open-Meteo. "
            "Solar output is estimated by your trained model."
        )

        available_dates = sorted(
            weather["Time"]
            .dt.strftime("%Y-%m-%d")
            .unique()
        )

        today = date.today().strftime("%Y-%m-%d")

        default_index = (
            available_dates.index(today)
            if today in available_dates
            else 0
        )

        selected_date = st.selectbox(
            "📅 Select Forecast Date",
            available_dates,
            index=default_index,
            key="forecast_date"
        )

        day_data = weather[
            weather["Time"].dt.strftime("%Y-%m-%d")
            == selected_date
        ].copy()

        day_data = day_data.sort_values("Time")

        if len(day_data) != 24:

            st.warning(
                f"This date has {len(day_data)} hourly records. "
                "A complete day normally contains 24 records."
            )

        if not day_data.empty:

            day_data["Generation (kWh)"] = (
                predict_generation(day_data)
            )

            total_generation = float(
                day_data["Generation (kWh)"].sum()
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Predicted Generation",
                f"{total_generation:.2f} kWh"
            )

            col2.metric(
                "Average Temperature",
                f"{day_data['temperature_c'].mean():.1f} °C"
            )

            col3.metric(
                "Peak Solar Radiation",
                f"{day_data['solar_irradiance_w_m2'].max():.0f} W/m²"
            )

            st.subheader("Hourly Solar Generation")

            st.bar_chart(
                day_data.set_index("Time")[
                    ["Generation (kWh)"]
                ]
            )

            st.subheader("Weather Conditions")

            weather_chart = day_data.set_index("Time")[
                [
                    "temperature_c",
                    "humidity_pct",
                    "cloud_cover_pct",
                    "solar_irradiance_w_m2"
                ]
            ]

            weather_chart.columns = [
                "Temperature (°C)",
                "Humidity (%)",
                "Cloud Cover (%)",
                "Solar Radiation (W/m²)"
            ]

            st.line_chart(weather_chart)

            with st.expander(
                "View hourly weather and prediction table"
            ):

                display = day_data.copy()

                display.columns = [
                    "Time",
                    "Temperature (°C)",
                    "Humidity (%)",
                    "Cloud Cover (%)",
                    "Solar Radiation (W/m²)",
                    "Generation (kWh)"
                ]

                st.dataframe(
                    display,
                    width="stretch",
                    hide_index=True
                )

            st.divider()

            st.subheader("🏠 Daily Electricity Consumption")

            st.caption(
                "Enter your home's total electricity consumption "
                "for the selected day."
            )

            col1, col2 = st.columns(2)

            with col1:

                consumption = st.number_input(
                    "Daily Consumption (kWh)",
                    min_value=0.0,
                    value=10.0,
                    step=0.5,
                    key="daily_consumption_input"
                )

            with col2:

                rate = st.number_input(
                    "Electricity Rate (₹/unit)",
                    min_value=0.0,
                    value=8.0,
                    step=0.5,
                    key="electricity_rate_input"
                )

            export_rate = st.number_input(
                "Solar Export Rate (₹/unit)",
                min_value=0.0,
                value=0.0,
                step=0.5,
                key="export_rate_input"
            )

            st.caption(
                "Set the export rate to the compensation applicable "
                "to your solar connection. Keep it at ₹0 if unknown."
            )

            if st.button(
                "💰 Calculate My Savings",
                type="primary"
            ):

                self_consumed = min(
                    consumption,
                    total_generation
                )

                grid_import = max(
                    consumption - self_consumed,
                    0
                )

                export_energy = max(
                    total_generation - self_consumed,
                    0
                )

                bill_without_solar = consumption * rate

                bill_with_solar = (
                    grid_import * rate
                    - export_energy * export_rate
                )

                savings = (
                    bill_without_solar - bill_with_solar
                )

                st.session_state.daily_results = {
                    "generation": total_generation,
                    "consumption": consumption,
                    "grid_import": grid_import,
                    "export_energy": export_energy,
                    "bill_without_solar": bill_without_solar,
                    "bill_with_solar": bill_with_solar,
                    "savings": savings,
                    "city": st.session_state.weather_city,
                    "date": selected_date
                }

                st.success(
                    "Calculation complete! Your Dashboard is updated."
                )

            results = st.session_state.daily_results

            if (
                results
                and results.get("date") == selected_date
                and results.get("city") == st.session_state.weather_city
            ):

                st.divider()

                st.subheader("💰 Estimated Savings")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Bill Without Solar",
                    f"₹{results['bill_without_solar']:.2f}"
                )

                col2.metric(
                    "Estimated Bill With Solar",
                    f"₹{results['bill_with_solar']:.2f}"
                )

                col3.metric(
                    "Estimated Savings",
                    f"₹{results['savings']:.2f}"
                )

                st.caption(
                    "This is a simplified estimate. Real savings depend "
                    "on hourly household usage, solar system size, "
                    "electricity tariff and export rules."
                )


# ==================================================
# MANUAL PREDICTION
# ==================================================

elif page == "Manual Prediction":

    st.title("🌤️ Manual Solar Prediction")

    st.caption(
        "Enter weather conditions manually to estimate "
        "solar generation for one hour."
    )

    col1, col2 = st.columns(2)

    with col1:

        temperature = st.slider(
            "Temperature (°C)",
            -5.0,
            55.0,
            30.0
        )

        humidity = st.slider(
            "Humidity (%)",
            0,
            100,
            60
        )

    with col2:

        cloud_cover = st.slider(
            "Cloud Cover (%)",
            0,
            100,
            20
        )

        irradiance = st.slider(
            "Solar Radiation (W/m²)",
            0,
            1200,
            700
        )

    if st.button(
        "Predict Generation",
        type="primary"
    ):

        manual_data = pd.DataFrame({
            "temperature_c": [temperature],
            "humidity_pct": [humidity],
            "cloud_cover_pct": [cloud_cover],
            "solar_irradiance_w_m2": [irradiance]
        })

        prediction = float(
            predict_generation(manual_data)[0]
        )

        st.metric(
            "Estimated Generation This Hour",
            f"{prediction:.3f} kWh"
        )

        st.caption(
            "Prediction is based on the training dataset "
            "and is not a guaranteed real-world output."
        )


# ==================================================
# INVESTMENT CALCULATOR
# ==================================================

elif page == "Investment Calculator":

    st.title("💰 Solar Investment Calculator")

    st.caption(
        "Estimate annual financial benefits and simple payback time."
    )

    cost = st.number_input(
        "Solar Installation Cost (₹)",
        min_value=0.0,
        value=100000.0,
        step=5000.0
    )

    daily_generation = st.number_input(
        "Estimated Daily Generation (kWh)",
        min_value=0.0,
        value=10.0,
        step=0.5
    )

    self_use = st.slider(
        "Solar Energy Used Directly at Home (%)",
        0,
        100,
        80
    )

    rate = st.number_input(
        "Electricity Rate (₹/unit)",
        min_value=0.0,
        value=8.0,
        step=0.5
    )

    export_rate = st.number_input(
        "Export Rate (₹/unit)",
        min_value=0.0,
        value=0.0,
        step=0.5
    )

    annual_generation = daily_generation * 365

    directly_used = annual_generation * self_use / 100

    exported = annual_generation - directly_used

    annual_benefit = (
        directly_used * rate
        + exported * export_rate
    )

    annual_benefit = max(
        annual_benefit,
        0
    )

    payback = (
        cost / annual_benefit
        if annual_benefit > 0
        else None
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Annual Generation",
        f"{annual_generation:.2f} kWh"
    )

    col2.metric(
        "Estimated Annual Benefit",
        f"₹{annual_benefit:.2f}"
    )

    col3.metric(
        "Simple Payback Period",
        f"{payback:.1f} years"
        if payback is not None
        else "N/A"
    )

    st.caption(
        "This estimate excludes maintenance, degradation, "
        "financing costs and changes in electricity prices."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Solar Energy Optimization System | "
    "Python • Streamlit • Scikit-learn • Open-Meteo"
)