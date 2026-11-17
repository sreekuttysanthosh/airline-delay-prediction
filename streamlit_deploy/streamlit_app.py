
import json
from pathlib import Path

import numpy as np
import streamlit as st


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Airline Delay Prediction",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ==================================================
# MODEL LOADING
# ==================================================

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "deployment_artifacts"
    / "airline_delay_model.json"
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    with open(MODEL_PATH, "r") as file:
        return json.load(file)


try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load the prediction model: {error}")
    st.stop()


# ==================================================
# MODEL PARAMETERS
# ==================================================

try:
    carrier_labels = model["categorical_features"]["OP_CARRIER"]
    origin_labels = model["categorical_features"]["ORIGIN"]
    dest_labels = model["categorical_features"]["DEST"]

    std_values = np.asarray(
        model["standard_scaler"]["std"],
        dtype=float
    )

    coefficients = np.asarray(
        model["logistic_regression"]["coefficients"],
        dtype=float
    )

    intercept = float(
        model["logistic_regression"]["intercept"]
    )

    threshold = float(
        model["logistic_regression"]["threshold"]
    )

except (KeyError, TypeError, ValueError) as error:
    st.error(f"Model artifact is missing required parameters: {error}")
    st.stop()


# ==================================================
# MODEL INTEGRITY CHECKS
# ==================================================

if len(coefficients) != 641:
    st.error("Model integrity error: expected 641 coefficients.")
    st.stop()

if len(std_values) != 8:
    st.error("Model integrity error: expected 8 scaler values.")
    st.stop()

if not np.all(np.isfinite(coefficients)):
    st.error("Model contains invalid coefficients.")
    st.stop()

if not np.all(np.isfinite(std_values)) or np.any(std_values == 0):
    st.error("Model contains invalid scaling values.")
    st.stop()


# ==================================================
# CATEGORY ENCODING
# ==================================================

def encode_category(value, labels, output_size):

    vector = np.zeros(output_size, dtype=float)

    try:
        index = labels.index(value)
    except ValueError:
        # Matches Spark StringIndexer(handleInvalid="keep")
        index = len(labels)

    if index < output_size:
        vector[index] = 1.0

    return vector


# ==================================================
# PREDICTION FUNCTION
# ==================================================

def predict_flight(flight):

    carrier_vector = encode_category(
        flight["OP_CARRIER"],
        carrier_labels,
        12
    )

    origin_vector = encode_category(
        flight["ORIGIN"],
        origin_labels,
        311
    )

    dest_vector = encode_category(
        flight["DEST"],
        dest_labels,
        310
    )

    numerical_values = np.array([
        flight["DEP_HOUR"],
        flight["DEP_MINUTE"],
        flight["ARR_HOUR"],
        flight["ARR_MINUTE"],
        flight["CRS_ELAPSED_TIME"],
        flight["DISTANCE"],
        flight["MONTH"],
        flight["DAY_OF_WEEK"]
    ], dtype=float)

    scaled_values = numerical_values / std_values

    features = np.concatenate([
        carrier_vector,
        origin_vector,
        dest_vector,
        scaled_values
    ])

    if len(features) != 641:
        raise ValueError("Unexpected feature vector dimension.")

    margin = float(np.dot(coefficients, features) + intercept)

    # Numerically stable sigmoid
    margin = np.clip(margin, -709, 709)

    delayed_probability = float(
        1.0 / (1.0 + np.exp(-margin))
    )

    prediction = int(
        delayed_probability >= threshold
    )

    return {
        "prediction": prediction,
        "delayed_probability": delayed_probability,
        "not_delayed_probability": 1.0 - delayed_probability,
        "margin": margin
    }


# ==================================================
# HEADER
# ==================================================

st.title("✈️ Airline Delay Prediction")

st.markdown(
    """
    Predict the likelihood of an airline flight arriving
    **more than 15 minutes late**, using a machine learning
    model trained with Apache Spark MLlib.
    """
)

st.caption(
    "Model: Logistic Regression | Training data: 2016–2018 | "
    "Inference: Validated NumPy implementation"
)

st.divider()


# ==================================================
# MODEL OVERVIEW
# ==================================================

st.subheader("About the Prediction Model")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Model", "Logistic Regression")

with metric2:
    st.metric("Input Features", "11")

with metric3:
    st.metric("Feature Vector", "641 dimensions")

st.info(
    "Enter the scheduled flight details below. "
    "The model uses airline, airport, departure and arrival "
    "schedule, flight duration, distance, month and weekday."
)

st.divider()


# ==================================================
# FLIGHT INPUT FORM
# ==================================================

st.subheader("Flight Information")

st.markdown("#### Route Details")

with st.form("flight_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:
        carrier = st.text_input(
            "Airline Code",
            value="9E",
            max_chars=3,
            help="Enter a two- or three-character airline code."
        ).strip().upper()

    with col2:
        origin = st.selectbox(
            "Origin Airport",
            options=origin_labels,
            index=origin_labels.index("CAE")
            if "CAE" in origin_labels else 0
        )

    with col3:
        destination_options = [
            airport for airport in dest_labels
            if airport != origin
        ]

        destination = st.selectbox(
            "Destination Airport",
            options=destination_options,
            index=destination_options.index("ATL")
            if "ATL" in destination_options else 0
        )

    st.divider()

    st.markdown("#### Scheduled Departure")

    col1, col2 = st.columns(2)

    with col1:
        dep_hour = st.number_input(
            "Departure Hour",
            min_value=0,
            max_value=23,
            value=9
        )

    with col2:
        dep_minute = st.number_input(
            "Departure Minute",
            min_value=0,
            max_value=59,
            value=30
        )

    st.markdown("#### Scheduled Arrival")

    col1, col2 = st.columns(2)

    with col1:
        arr_hour = st.number_input(
            "Arrival Hour",
            min_value=0,
            max_value=23,
            value=11
        )

    with col2:
        arr_minute = st.number_input(
            "Arrival Minute",
            min_value=0,
            max_value=59,
            value=15
        )

    st.divider()

    st.markdown("#### Flight Schedule Details")

    col1, col2 = st.columns(2)

    with col1:
        elapsed_time = st.number_input(
            "Scheduled Flight Duration (minutes)",
            min_value=1.0,
            max_value=1500.0,
            value=105.0,
            step=5.0
        )

    with col2:
        distance = st.number_input(
            "Distance (miles)",
            min_value=1.0,
            max_value=10000.0,
            value=500.0,
            step=10.0
        )

    col1, col2 = st.columns(2)

    with col1:
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            index=5,
            format_func=lambda x: [
                "January", "February", "March",
                "April", "May", "June",
                "July", "August", "September",
                "October", "November", "December"
            ][x - 1]
        )

    with col2:
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(1, 8)),
            index=1,
            format_func=lambda x: [
                "Sunday", "Monday", "Tuesday",
                "Wednesday", "Thursday", "Friday",
                "Saturday"
            ][x - 1]
        )

    st.divider()

    submitted = st.form_submit_button(
        "✈️ Predict Flight Delay",
        type="primary",
        use_container_width=True
    )


# ==================================================
# PREDICTION RESULTS
# ==================================================

if submitted:

    if not carrier:
        st.error("Please enter an airline code.")

    elif origin == destination:
        st.error("Origin and destination airports must be different.")

    else:

        flight = {
            "OP_CARRIER": carrier,
            "ORIGIN": origin,
            "DEST": destination,
            "DEP_HOUR": dep_hour,
            "DEP_MINUTE": dep_minute,
            "ARR_HOUR": arr_hour,
            "ARR_MINUTE": arr_minute,
            "CRS_ELAPSED_TIME": elapsed_time,
            "DISTANCE": distance,
            "MONTH": month,
            "DAY_OF_WEEK": day_of_week
        }

        try:
            result = predict_flight(flight)

            st.divider()
            st.subheader("Prediction Results")

            col1, col2 = st.columns(2)

            with col1:
                if result["prediction"] == 1:
                    st.error("⚠️ Likely Delayed")
                else:
                    st.success("✅ Likely Not Delayed")

            with col2:
                st.metric(
                    "Delayed Probability",
                    f"{result['delayed_probability'] * 100:.2f}%"
                )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Not Delayed Probability",
                    f"{result['not_delayed_probability'] * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Classification Threshold",
                    f"{threshold * 100:.0f}%"
                )

            st.markdown("#### Probability Distribution")

            st.progress(
                result["delayed_probability"],
                text=(
                    f"Delayed: "
                    f"{result['delayed_probability'] * 100:.2f}%"
                )
            )

            st.progress(
                result["not_delayed_probability"],
                text=(
                    f"Not delayed: "
                    f"{result['not_delayed_probability'] * 100:.2f}%"
                )
            )

            st.markdown("#### Submitted Flight Details")

            st.dataframe(
                {
                    "Parameter": [
                        "Airline",
                        "Origin",
                        "Destination",
                        "Departure time",
                        "Arrival time",
                        "Scheduled duration",
                        "Distance",
                        "Month",
                        "Day of week"
                    ],
                    "Value": [
                        carrier,
                        origin,
                        destination,
                        f"{dep_hour:02d}:{dep_minute:02d}",
                        f"{arr_hour:02d}:{arr_minute:02d}",
                        f"{elapsed_time:.0f} minutes",
                        f"{distance:.0f} miles",
                        month,
                        [
                            "Sunday", "Monday", "Tuesday",
                            "Wednesday", "Thursday", "Friday",
                            "Saturday"
                        ][day_of_week - 1]
                    ]
                },
                hide_index=True,
                use_container_width=True
            )

            if carrier not in carrier_labels:
                st.warning(
                    "This airline code was not present in the training "
                    "categories. It was handled using the model's "
                    "unseen-category encoding."
                )

            st.caption(
                "A delayed flight is defined as one with an arrival "
                "delay greater than 15 minutes. This is a statistical "
                "prediction, not a guaranteed flight outcome."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Academic project | Apache Spark MLlib | "
    "Logistic Regression | NumPy inference"
)

st.caption(
    "The prediction interface uses the exported parameters "
    "of the trained Spark model. Spark does not need to run "
    "during individual predictions."
)
