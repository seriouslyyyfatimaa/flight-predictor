# Flight Delay & Demand Predictor ✈️

Machine learning project that predicts flight delay risk and passenger demand for Dubai (DXB) hub routes, with an interactive dashboard to play around with predictions.

🔗 **Live demo:** [Try it here](https://flight-predictor-upyjnyjzncjggsbd3qrjmx.streamlit.app/)

I built this as part of my AI portfolio in my final year of AI & Computer Science at Heriot-Watt University Dubai. It started out as a general sales-prediction idea and I reworked it to fit the airline/aviation space instead.

## What it does

- **Delay classifier** – predicts the probability a flight gets delayed, using airline, route, month, day, departure hour, weather, and whether it's holiday season
- **Demand regressor** – predicts how full the flight will be (as % of seats booked)
- **Dashboard** – pick your flight details in the sidebar and get live predictions, plus some charts on delay trends, demand by route, and what actually drives the delay predictions

## Screenshots

| Delay rate by month | Demand by route |
|---|---|
| ![Delay by month](screenshots/delay_by_month.png) | ![Demand by route](screenshots/demand_by_route.png) |

**Feature importance (what the model actually weighs most):**

![Feature importance](screenshots/feature_importance.png)

## Project structure

```
flight-predictor/
├── generate_data.py     # builds the dataset (data/flights.csv)
├── train_model.py       # trains + saves both models
├── app.py                # Streamlit dashboard
├── requirements.txt
├── LICENSE
├── .gitignore
├── data/
│   └── flights.csv
├── models/
│   ├── delay_classifier.pkl
│   ├── demand_regressor.pkl
│   └── feature_importance.csv
└── screenshots/
    ├── delay_by_month.png
    ├── demand_by_route.png
    └── feature_importance.png
```

## Running it locally

```bash
pip install -r requirements.txt
python generate_data.py     # regenerate the dataset if you want
python train_model.py       # retrain the models if you want
streamlit run app.py        # opens the dashboard at localhost:8501
```

## About the dataset

I used a synthetic dataset for this instead of pulling directly from Kaggle, since Kaggle needs an account + API key to download programmatically and I wanted the whole pipeline working end to end first. I built the generator so it follows patterns real flight data actually has — evening flights delay more, bad weather increases delay risk, holiday season pushes up both demand and price.

If I swap in a real Kaggle dataset later (e.g. the [Flight Delays dataset](https://www.kaggle.com/datasets/usdot/flight-delays)), the pipeline should mostly just work — I'd just need to match up the column names in `train_model.py`.

## A note on the model

The delay classifier uses `class_weight='balanced'` because only ~18% of flights in the dataset are delayed. Without balancing, the model just predicts "not delayed" for everything and still gets a decent-looking accuracy score while being basically useless. Balancing it drops the accuracy but makes the F1/ROC-AUC scores actually mean something.

## What I'd improve with more time

- Swap in real flight delay data instead of synthetic
- Try gradient boosting (XGBoost) instead of Random Forest
- Pull in real weather API data instead of a made-up weather score
