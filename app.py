"""
app.py

Dashboard for the flight delay/demand predictor. Loads the models trained in
train_model.py and lets you play with predictions + see some charts.

Run locally with:
    streamlit run app.py
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Flight Delay & Demand Predictor", layout="wide")

# ---------------- Load data + models ----------------
@st.cache_data
def load_data():
    return pd.read_csv('data/flights.csv')

@st.cache_resource
def load_models():
    clf = joblib.load('models/delay_classifier.pkl')
    reg = joblib.load('models/demand_regressor.pkl')
    fi = pd.read_csv('models/feature_importance.csv')
    return clf, reg, fi

df = load_data()
clf, reg, fi = load_models()

ROUTE_DISTANCES = df.groupby('destination')['distance_km'].first().to_dict()
AIRLINES = sorted(df['airline'].unique())
DESTINATIONS = sorted(df['destination'].unique())
DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

st.title("✈️ Flight Delay & Demand Predictor")
st.caption("DXB hub routes — predicting delay likelihood and passenger demand from route, timing, and season.")

# ================= Sidebar: prediction inputs =================
st.sidebar.header("Predict a flight")
airline = st.sidebar.selectbox("Airline", AIRLINES)
destination = st.sidebar.selectbox("Destination", DESTINATIONS)
month = st.sidebar.slider("Month", 1, 12, 6)
day = st.sidebar.selectbox("Day of week", DAYS)
dep_hour = st.sidebar.slider("Departure hour (24h)", 0, 23, 18)
weather_score = st.sidebar.slider("Weather score (0=bad, 10=clear)", 0.0, 10.0, 6.0)
is_holiday = st.sidebar.checkbox("Holiday season (Jun/Jul/Dec/Jan)", value=month in [6, 7, 12, 1])

if st.sidebar.button("Predict", type="primary"):
    input_row = pd.DataFrame([{
        'airline': airline,
        'destination': destination,
        'day_of_week': day,
        'distance_km': ROUTE_DISTANCES.get(destination, 5000),
        'month': month,
        'dep_hour': dep_hour,
        'weather_score': weather_score,
        'is_holiday_season': is_holiday,
    }])

    delay_proba = clf.predict_proba(input_row)[0][1]
    demand_pred = reg.predict(input_row)[0]
    est_price = (1800 + ROUTE_DISTANCES.get(destination, 5000) * 0.15) * (1 + (demand_pred - 0.6) * 0.8)

    c1, c2, c3 = st.columns(3)
    c1.metric("Delay probability", f"{delay_proba*100:.1f}%")
    c2.metric("Predicted demand load", f"{demand_pred*100:.1f}%")
    c3.metric("Suggested fare (AED)", f"{est_price:,.0f}")

    if delay_proba > 0.35:
        st.warning("⚠️ Elevated delay risk for this slot — consider buffer time in connections or crew scheduling.")
    if demand_pred > 0.85:
        st.info("📈 High demand predicted — this is a candidate for dynamic pricing uplift.")

st.divider()

# ================= Exploratory charts =================
st.header("Business insights from historical data")

col1, col2 = st.columns(2)

with col1:
    delay_by_month = df.groupby('month')['delayed'].mean().reset_index()
    fig1 = px.line(delay_by_month, x='month', y='delayed', markers=True,
                    title="Delay rate by month", labels={'delayed': 'Delay rate', 'month': 'Month'})
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    delay_by_hour = df.groupby('dep_hour')['delayed'].mean().reset_index()
    fig2 = px.bar(delay_by_hour, x='dep_hour', y='delayed',
                   title="Delay rate by departure hour", labels={'delayed': 'Delay rate', 'dep_hour': 'Hour'})
    st.plotly_chart(fig2, use_container_width=True)

col3, col4 = st.columns(2)

with col3:
    demand_by_route = df.groupby('destination')['demand_ratio'].mean().reset_index().sort_values('demand_ratio', ascending=False)
    fig3 = px.bar(demand_by_route, x='destination', y='demand_ratio',
                   title="Average demand load by route", labels={'demand_ratio': 'Demand ratio'})
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    fig4 = px.bar(fi.head(10), x='importance', y='feature', orientation='h',
                   title="What drives delay predictions (top 10 features)")
    fig4.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Peak pricing windows")
pricing = df.groupby(['month', 'is_holiday_season'])['price_aed'].mean().reset_index()
fig5 = px.box(df, x='month', y='price_aed', color='is_holiday_season',
              title="Fare distribution by month (holiday vs non-holiday season)",
              labels={'price_aed': 'Fare (AED)', 'is_holiday_season': 'Holiday season'})
st.plotly_chart(fig5, use_container_width=True)

st.caption("Data: synthetic dataset modeling realistic DXB-hub delay/demand patterns. See README for how to swap in real Kaggle flight data.")
