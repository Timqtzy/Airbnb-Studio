import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

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
    layout="wide"
)

# ==========================
# SIDEBAR INPUTS
# ==========================
with st.sidebar:
    st.header("🏠 Property Details")

    with st.form("prediction_form"):
        st.subheader("🏘 Property Info")

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

        host_superhost = st.radio("Superhost", ["Yes", "No"], horizontal=True)
        instant_bookable = st.radio("Instant Bookable", ["Yes", "No"], horizontal=True)

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

        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

# ==========================
# MAIN CONTENT AREA
# ==========================
st.title("🏠 Airbnb Price & Availability Prediction")
st.markdown("Predict **nightly price** and **availability** using trained ML models.")

if not submitted:
    # Welcome state
    st.info("👈 Enter your property details in the sidebar and click **Predict** to see results.")

    st.markdown("---")
    st.markdown("### How It Works")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 1️⃣ Enter Details")
        st.markdown("Fill in your property information including type, location, amenities, and host details.")

    with col2:
        st.markdown("#### 2️⃣ Get Predictions")
        st.markdown("Our ML models analyze your inputs to predict optimal pricing and expected availability.")

    with col3:
        st.markdown("#### 3️⃣ View Trends")
        st.markdown("Explore monthly price and availability trends to optimize your listing strategy.")

else:
    # ----------------------
    # BASE ENCODED FEATURES
    # ----------------------
    base_features = {
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
        "month": month
    }

    # ----------------------
    # PRICE MODEL INPUT
    # ----------------------
    price_input = pd.DataFrame([{
        **base_features,
        "reviews_per_month": 1.0,
        "review_scores_rating": 95.0,
        "minimum_nights": 2,
        "number_of_reviews": 10
    }])

    price_log_pred = price_model.predict(price_input)[0]
    price_pred = np.expm1(price_log_pred)

    # ----------------------
    # AVAILABILITY MODEL INPUT
    # ----------------------
    avail_input = pd.DataFrame([{
        **base_features,
        "price_log": price_log_pred,
        "avg_availability": 0.5,
        "reviews_per_month": 1.0
    }])

    avail_logit = availability_model.predict(avail_input)[0]
    availability_pred = 1 / (1 + np.exp(-avail_logit))

    # ======================
    # RESULTS SECTION
    # ======================
    st.success("✅ Prediction Complete!")

    # Main metrics
    st.markdown("### 📊 Prediction Results")

    month_names_full = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Estimated Price", f"${price_pred:,.2f}", "per night")
    col2.metric("📅 Expected Availability", f"{availability_pred * 100:.1f}%")
    col3.metric("📆 Selected Month", month_names_full[month - 1])

    st.markdown("---")

    # ======================
    # EXPLANATION SECTION
    # ======================
    st.markdown("### 💡 What This Means")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Price Interpretation")
        if price_pred < 75:
            st.markdown(
                "Your property falls in the **budget-friendly** range. This is typical for shared rooms or basic accommodations.")
        elif price_pred < 150:
            st.markdown(
                "Your property is in the **mid-range** pricing tier. This is competitive for most private rooms and smaller apartments.")
        elif price_pred < 250:
            st.markdown(
                "Your property commands a **premium** price. This reflects desirable features, location, or larger capacity.")
        else:
            st.markdown(
                "Your property is in the **luxury** segment. High-end amenities and prime locations justify this pricing.")

    with col2:
        st.markdown("#### Availability Interpretation")
        if availability_pred < 0.3:
            st.markdown(
                "**High demand** expected! Your property characteristics suggest strong booking potential with limited open dates.")
        elif availability_pred < 0.6:
            st.markdown(
                "**Moderate demand** predicted. A balanced mix of booked and available dates is typical for your property type.")
        else:
            st.markdown(
                "**Higher availability** expected. Consider adjusting pricing or enhancing amenities to increase bookings.")

    st.markdown("---")

    # ======================
    # MONTHLY TRENDS
    # ======================
    st.markdown("### 📈 Monthly Price & Availability Trends")
    st.markdown("See how predictions vary throughout the year for your property configuration.")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    monthly_prices = []
    monthly_availability = []

    for m in range(1, 13):
        monthly_features = base_features.copy()
        monthly_features["month"] = m

        monthly_price_input = pd.DataFrame([{
            **monthly_features,
            "reviews_per_month": 1.0,
            "review_scores_rating": 95.0,
            "minimum_nights": 2,
            "number_of_reviews": 10
        }])

        monthly_log_pred = price_model.predict(monthly_price_input)[0]
        monthly_prices.append(np.expm1(monthly_log_pred))

        monthly_avail_input = pd.DataFrame([{
            **monthly_features,
            "price_log": monthly_log_pred,
            "avg_availability": 0.5,
            "reviews_per_month": 1.0
        }])

        monthly_avail_logit = availability_model.predict(monthly_avail_input)[0]
        monthly_availability.append(1 / (1 + np.exp(-monthly_avail_logit)) * 100)

    # Create DataFrame for charts
    chart_data = pd.DataFrame({
        "Month": month_names,
        "Price ($)": monthly_prices,
        "Availability (%)": monthly_availability
    })

    # Two column layout for charts
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 💰 Price by Month")

        # Plotly price chart
        fig_price = go.Figure()

        fig_price.add_trace(go.Scatter(
            x=month_names,
            y=monthly_prices,
            mode='lines+markers',
            name='Price',
            line=dict(color='#FF5A5F', width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='%{x}<br>Price: $%{y:.2f}<extra></extra>'
        ))

        # Highlight selected month
        fig_price.add_trace(go.Scatter(
            x=[month_names[month - 1]],
            y=[monthly_prices[month - 1]],
            mode='markers',
            name='Selected',
            marker=dict(size=14, color='#FF5A5F', symbol='star', line=dict(width=2, color='white')),
            hovertemplate='%{x} (Selected)<br>Price: $%{y:.2f}<extra></extra>'
        ))

        fig_price.update_layout(
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_price, use_container_width=True)

        # Price insights
        min_price_idx = np.argmin(monthly_prices)
        max_price_idx = np.argmax(monthly_prices)

        st.markdown(f"""
        **Insights:**
        - 📉 Lowest: **${min(monthly_prices):,.2f}** in {month_names_full[min_price_idx]}
        - 📈 Highest: **${max(monthly_prices):,.2f}** in {month_names_full[max_price_idx]}
        - 📊 Range: **${max(monthly_prices) - min(monthly_prices):,.2f}**
        """)

    with col2:
        st.markdown("#### 📅 Availability by Month")

        # Plotly availability chart
        fig_avail = go.Figure()

        fig_avail.add_trace(go.Scatter(
            x=month_names,
            y=monthly_availability,
            mode='lines+markers',
            name='Availability',
            line=dict(color='#00A699', width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='%{x}<br>Availability: %{y:.1f}%<extra></extra>'
        ))

        # Highlight selected month
        fig_avail.add_trace(go.Scatter(
            x=[month_names[month - 1]],
            y=[monthly_availability[month - 1]],
            mode='markers',
            name='Selected',
            marker=dict(size=14, color='#00A699', symbol='star', line=dict(width=2, color='white')),
            hovertemplate='%{x} (Selected)<br>Availability: %{y:.1f}%<extra></extra>'
        ))

        fig_avail.update_layout(
            yaxis_title='Availability (%)',
            hovermode='x unified',
            showlegend=False,
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
            yaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_avail, use_container_width=True)

        # Availability insights
        min_avail_idx = np.argmin(monthly_availability)
        max_avail_idx = np.argmax(monthly_availability)

        st.markdown(f"""
        **Insights:**
        - 🔥 Busiest: **{min(monthly_availability):.1f}%** available in {month_names_full[min_avail_idx]}
        - 🌙 Slowest: **{max(monthly_availability):.1f}%** available in {month_names_full[max_avail_idx]}
        """)

    st.markdown("---")

    # ======================
    # DETAILED DATA TABLE
    # ======================
    with st.expander("📋 View Full Monthly Data"):
        display_df = chart_data.copy()
        display_df["Month"] = month_names_full
        display_df["Price ($)"] = display_df["Price ($)"].apply(lambda x: f"${x:,.2f}")
        display_df["Availability (%)"] = display_df["Availability (%)"].apply(lambda x: f"{x:.1f}%")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # ======================
    # RECOMMENDATIONS
    # ======================
    st.markdown("### 🎯 Recommendations")

    best_price_month = month_names_full[np.argmax(monthly_prices)]
    best_booking_month = month_names_full[np.argmin(monthly_availability)]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Best month for revenue:** {best_price_month}\n\nHighest predicted nightly rate.")

    with col2:
        st.info(
            f"**Peak demand month:** {best_booking_month}\n\nLowest availability indicates highest booking potential.")

    with col3:
        if host_superhost == "No":
            st.warning("**Tip:** Becoming a Superhost can increase bookings and justify higher prices.")
        else:
            st.success("**Great!** Superhost status helps attract more guests and premium pricing.")

    st.caption("Predictions are based on historical Airbnb New Brunswick data.")