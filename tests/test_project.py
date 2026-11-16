"""Checks that the simulated data keeps its known properties and the model still learns from it."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"
NUM = ["tenure", "MonthlyCharges", "TotalCharges"]


@pytest.fixture(scope="module")
def df():
    d = pd.read_csv(DATA)
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"].str.strip().replace("", np.nan)).fillna(0.0)
    d["Churn"] = (d["Churn"] == "Yes").astype(int)
    return d


def test_shape_and_ids():
    raw = pd.read_csv(DATA)
    assert raw.shape == (7043, 21)
    assert raw["customerID"].is_unique


def test_blank_total_charges_only_for_new_customers():
    raw = pd.read_csv(DATA)
    blank = raw["TotalCharges"].str.strip() == ""
    assert blank.sum() == 11
    assert (raw.loc[blank, "tenure"] == 0).all()


def test_churn_rate_and_contract_ordering(df):
    assert 0.25 < df["Churn"].mean() < 0.31
    rate = df.groupby("Contract")["Churn"].mean()
    assert rate["Month-to-month"] > rate["One year"] > rate["Two year"]


def test_logistic_regression_beats_chance_by_a_margin(df):
    X, y = df.drop(columns=["customerID", "Churn"]), df["Churn"]
    cat = [c for c in X.columns if c not in NUM]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), NUM),
                                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])),
        ("lr", LogisticRegression(max_iter=2000)),
    ]).fit(X_tr, y_tr)
    assert roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]) > 0.80
