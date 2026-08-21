"""
train_model.py

Trains two models on the flight dataset:
  1. Classifier - predicts if a flight will be delayed
  2. Regressor  - predicts how full the flight will be (demand)

Saves both to models/ so app.py doesn't need to retrain every time it loads.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error
import joblib

CATEGORICAL = ['airline', 'destination', 'day_of_week']
NUMERIC = ['distance_km', 'month', 'dep_hour', 'weather_score', 'is_holiday_season']

def build_pipeline(model):
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL),
    ], remainder='passthrough')
    return Pipeline([
        ('prep', preprocessor),
        ('model', model),
    ])

def main():
    df = pd.read_csv('data/flights.csv')
    X = df[CATEGORICAL + NUMERIC]

    # ---------------- delay classifier ----------------
    y_delay = df['delayed']
    X_train, X_test, y_train, y_test = train_test_split(X, y_delay, test_size=0.2, random_state=42, stratify=y_delay)

    # class_weight='balanced' matters here - only ~18% of flights are delayed,
    # so without this the model just predicts "not delayed" every time and
    # still looks accurate while being useless
    clf = build_pipeline(RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'
    ))
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]
    print("=== Delay Classifier ===")
    print("Accuracy:", round(accuracy_score(y_test, preds), 3))
    print("F1:", round(f1_score(y_test, preds), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, proba), 3))

    joblib.dump(clf, 'models/delay_classifier.pkl')

    # ---------------- demand regressor ----------------
    y_demand = df['demand_ratio']
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y_demand, test_size=0.2, random_state=42)

    reg = build_pipeline(RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42))
    reg.fit(X_train2, y_train2)

    preds2 = reg.predict(X_test2)
    print("\n=== Demand Regressor ===")
    print("R2:", round(r2_score(y_test2, preds2), 3))
    print("MAE:", round(mean_absolute_error(y_test2, preds2), 4))

    joblib.dump(reg, 'models/demand_regressor.pkl')

    # ---------------- feature importance (used in the dashboard) ----------------
    ohe = clf.named_steps['prep'].named_transformers_['cat']
    cat_names = ohe.get_feature_names_out(CATEGORICAL)
    all_names = list(cat_names) + NUMERIC
    importances = clf.named_steps['model'].feature_importances_

    fi_df = pd.DataFrame({'feature': all_names, 'importance': importances})
    fi_df = fi_df.sort_values('importance', ascending=False)
    fi_df.to_csv('models/feature_importance.csv', index=False)

    print("\nSaved models to models/delay_classifier.pkl, models/demand_regressor.pkl, models/feature_importance.csv")

if __name__ == '__main__':
    main()
