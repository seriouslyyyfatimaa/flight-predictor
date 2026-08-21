"""
generate_data.py

Builds a fake but realistic flight dataset for DXB routes since I didn't want
to deal with Kaggle API setup just to get a first version working.

The relationships are made up but based on how flight delays actually work in
real life - bad weather = more delays, evening flights = more congestion,
holiday season = more demand and higher prices.

If I want to use real data later:
1. Make a Kaggle account, get an API key (kaggle.json)
2. pip install kaggle
3. kaggle datasets download -d usdot/flight-delays
4. Match up the column names with what's in train_model.py
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_ROWS = 8000

AIRLINES = ['Emirates', 'Etihad', 'Qatar Airways', 'British Airways', 'Singapore Airlines', 'Turkish Airlines']
AIRLINE_WEIGHTS = [0.35, 0.15, 0.15, 0.10, 0.15, 0.10]

# (origin, destination, distance_km) - DXB hub routes
ROUTES = [
    ('DXB', 'LHR', 5500), ('DXB', 'JFK', 11000), ('DXB', 'SYD', 12000),
    ('DXB', 'BKK', 4900), ('DXB', 'SIN', 5900), ('DXB', 'CDG', 5200),
    ('DXB', 'FRA', 4800), ('DXB', 'JNB', 6400), ('DXB', 'BOM', 1900),
    ('DXB', 'NRT', 7900),
]

DAYS = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']


def generate():
    rows = []
    for _ in range(N_ROWS):
        airline = np.random.choice(AIRLINES, p=AIRLINE_WEIGHTS)
        origin, dest, distance = ROUTES[np.random.randint(len(ROUTES))]
        month = int(np.random.choice(range(1, 13)))
        day = np.random.choice(DAYS)
        dep_hour = int(np.random.randint(0, 24))

        is_holiday_season = month in [6, 7, 12, 1]

        # weather is better in shoulder months, worse in peak summer/winter
        weather_score = np.random.normal(7 if month in [3, 4, 10, 11] else 5, 1.5)
        weather_score = float(np.clip(weather_score, 0, 10))

        # delay probability - stacking up a few factors
        delay_prob = 0.12
        if dep_hour in [17, 18, 19, 20]:
            delay_prob += 0.08  # evening congestion
        if weather_score < 4:
            delay_prob += 0.15
        if is_holiday_season:
            delay_prob += 0.05
        if day in ['Fri', 'Sun']:
            delay_prob += 0.03

        delayed = int(np.random.binomial(1, min(delay_prob, 0.85)))
        delay_minutes = int(np.random.exponential(45) + 15) if delayed else 0

        # demand
        base_demand = 0.65
        if is_holiday_season:
            base_demand += 0.20
        if day in ['Fri', 'Sat', 'Sun']:
            base_demand += 0.08
        demand_ratio = float(np.clip(np.random.normal(base_demand, 0.12), 0.2, 1.0))

        seat_capacity = 350
        passengers_booked = int(demand_ratio * seat_capacity)

        # price goes up with demand
        base_price = 1800 + distance * 0.15
        price = base_price * (1 + (demand_ratio - 0.6) * 0.8)
        price = round(max(price, 800), 2)

        rows.append([
            airline, origin, dest, distance, month, day, dep_hour,
            round(weather_score, 2), is_holiday_season, delayed, delay_minutes,
            passengers_booked, seat_capacity, round(demand_ratio, 3), price,
        ])

    df = pd.DataFrame(rows, columns=[
        'airline', 'origin', 'destination', 'distance_km', 'month', 'day_of_week', 'dep_hour',
        'weather_score', 'is_holiday_season', 'delayed', 'delay_minutes',
        'passengers_booked', 'seat_capacity', 'demand_ratio', 'price_aed',
    ])
    return df


if __name__ == '__main__':
    df = generate()
    df.to_csv('data/flights.csv', index=False)
    print(f"Generated {len(df)} rows -> data/flights.csv")
    print(df.head())
    print("\nDelay rate:", df['delayed'].mean().round(3))
    print("Avg demand ratio:", df['demand_ratio'].mean().round(3))
