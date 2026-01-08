import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==========================
# LOAD MODELS & ENCODERS
# ==========================
price_model = joblib.load("price_model_xgb.pkl")
availability_model = joblib.load("availability_model_rf.pkl")
encoders = joblib.load("price_encoders.pkl")

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Airbnb Price & Availability Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Airbnb Price & Availability Prediction System")
st.markdown("Predict **nightly price** and **availability** based on property details.")

# ==========================
# INPUT FORM
# ==========================
with st.form("prediction_form"):
    st.subheader("🏘 Property Information")

    property_type = st.selectbox(
        "Property Type",
        encoders["property_type"].classes_
    )

    room_type = st.selectbox(
        "Room Type",
        encoders["room_type"].classes_
    )

    neighbourhood = st.selectbox(
        "Neighborhood",
        encoders["neighbourhood_cleansed"].classes_
    )

    accommodates = st.slider("Guests Accommodated", 1, 16, 2)
    bedrooms = st.number_input("Bedrooms", 0, 10, 1)
    beds = st.number_input("Beds", 0, 10, 1)
    bathrooms = st.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5)

    st.subheader("🧺 Amenities")
    amenities = st.multiselect(
        "Select Amenities",
        [
            "Wi-Fi", "Kitchen", "Free parking", "Washer",
            "Dryer", "Air conditioning", "Heating", "TV",
            "Pool", "Gym", "Workspace", "Elevator"
        ]
    )

    st.subheader("👤 Host & Booking")

    host_superhost = st.radio("Superhost", ["Yes", "No"])
    instant_bookable = st.radio("Instant Bookable", ["Yes", "No"])

    month = st.selectbox(
        "Month",
        [
            ("January", 1), ("February", 2), ("March", 3),
            ("April", 4), ("May", 5), ("June", 6),
            ("July", 7), ("August", 8), ("September", 9),
            ("October", 10), ("November", 11), ("December", 12)
        ],
        format_func=lambda x: x[0]
    )[1]

    submitted = st.form_submit_button("🔮 Predict")

# ==========================
# PREDICTION
# ==========================
if submitted:

    # Encode categorical variables - base input data
    input_data = {
        "property_type": encoders["property_type"].transform([property_type])[0],
        "room_type": encoders["room_type"].transform([room_type])[0],
        "neighbourhood_cleansed": encoders["neighbourhood_cleansed"].transform([neighbourhood])[0],
        "accommodates": accommodates,
        "bedrooms": bedrooms,
        "beds": beds,
        "bathrooms": bathrooms,
        "amenities_count": len(amenities),
        "host_is_superhost": encoders["host_is_superhost"].transform(
            ["t" if host_superhost == "Yes" else "f"]
        )[0],
        "instant_bookable": encoders["instant_bookable"].transform(
            ["t" if instant_bookable == "Yes" else "f"]
        )[0],
        "month": month,
    }

    df_input = pd.DataFrame([input_data])

    # ======================
    # PRICE PREDICTION
    # ======================
    # Get the features the price model was trained on
    price_features = list(price_model.feature_names_in_)

    # Create dataframe with only the features needed for price model
    df_price = df_input.copy()

    # Add any missing features with default values
    for feat in price_features:
        if feat not in df_price.columns:
            df_price[feat] = 0

    # Select only the features the model expects, in the correct order
    df_price = df_price[price_features]

    price_log_pred = price_model.predict(df_price)[0]
    price_pred = np.expm1(price_log_pred)

    # ======================
    # AVAILABILITY PREDICTION
    # ======================
    # Get the features the availability model was trained on
    avail_features = list(availability_model.feature_names_in_)

    # Create dataframe with base input + additional features
    df_avail = df_input.copy()
    df_avail["price_log"] = price_log_pred
    df_avail["avg_availability"] = 0.5

    # Add any missing features with default values
    for feat in avail_features:
        if feat not in df_avail.columns:
            df_avail[feat] = 0

    # Select only the features the model expects, in the correct order
    df_avail = df_avail[avail_features]

    avail_logit = availability_model.predict(df_avail)[0]
    availability_pred = 1 / (1 + np.exp(-avail_logit))

    # ======================
    # OUTPUT
    # ======================
    st.success("✅ Prediction Complete!")

    col1, col2 = st.columns(2)

    col1.metric("💰 Estimated Price per Night", f"${price_pred:,.2f}")
    col2.metric("📅 Expected Availability", f"{availability_pred * 100:.1f}%")

    st.caption("Predictions are based on historical Airbnb New Brunswick data.")

    # Optional: Show what features each model uses (for debugging)
    with st.expander("🔍 Debug: Model Features"):
        st.write("**Price Model Features:**", price_features)
        st.write("**Availability Model Features:**", avail_features)