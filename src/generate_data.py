"""Generate the simulated telecom customer dataset used in this repository.

Same 21 columns as the well-known IBM "Telco Customer Churn" sample (7,043 customers),
but every value is synthetic. Churn is drawn from a known logistic model (contract type,
tenure, fibre internet, electronic-check payment, missing tech support, monthly charges),
so the notebook's findings can be checked against the truth that generated the data.

Data-quality quirks of the original layout are reproduced: TotalCharges is text and is
blank for customers with tenure 0.

Run:  python src/generate_data.py      -> writes data/telco_customer_churn.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 11
N = 7043
OUT = Path(__file__).resolve().parents[1] / "data" / "telco_customer_churn.csv"


def pick(rng, options, probs, n):
    return rng.choice(options, size=n, p=probs)


def main():
    rng = np.random.default_rng(SEED)
    ids = [f"{rng.integers(1000, 9999)}-{''.join(rng.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 5))}" for _ in range(N)]

    gender = pick(rng, ["Female", "Male"], [0.495, 0.505], N)
    senior = (rng.random(N) < 0.16).astype(int)
    partner = np.where(rng.random(N) < 0.48, "Yes", "No")
    dependents = np.where((partner == "Yes") & (rng.random(N) < 0.6) | (rng.random(N) < 0.05), "Yes", "No")

    contract = pick(rng, ["Month-to-month", "One year", "Two year"], [0.55, 0.21, 0.24], N)
    base_tenure = {"Month-to-month": 18, "One year": 40, "Two year": 56}
    tenure = np.clip(np.round(np.array([rng.normal(base_tenure[c], 16) for c in contract])), 1, 72).astype(int)
    tenure[rng.choice(N, 11, replace=False)] = 0

    phone = np.where(rng.random(N) < 0.90, "Yes", "No")
    multiple = np.where(phone == "No", "No phone service", np.where(rng.random(N) < 0.47, "Yes", "No"))
    internet = pick(rng, ["Fiber optic", "DSL", "No"], [0.44, 0.34, 0.22], N)

    def addon(p):
        return np.where(internet == "No", "No internet service", np.where(rng.random(N) < p, "Yes", "No"))

    security, backup, protection = addon(0.37), addon(0.44), addon(0.44)
    tech, tv, movies = addon(0.37), addon(0.50), addon(0.50)
    paperless = np.where(rng.random(N) < 0.59, "Yes", "No")
    payment = pick(rng, ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                   [0.34, 0.23, 0.22, 0.21], N)

    monthly = (20 + (phone == "Yes") * 5 + (multiple == "Yes") * 5
               + np.select([internet == "Fiber optic", internet == "DSL"], [45, 25], 0)
               + 5 * sum((x == "Yes") for x in [security, backup, protection, tech])
               + 9 * sum((x == "Yes") for x in [tv, movies]) + rng.normal(0, 2.5, N))
    monthly = np.round(np.clip(monthly, 18.25, 118.75), 2)
    total = np.round(monthly * tenure * rng.normal(1, 0.03, N), 2)

    # Known churn model (log-odds)
    logit = (-1.95
             + np.select([contract == "Month-to-month", contract == "One year"], [1.25, 0.0], -1.30)
             - 0.032 * tenure
             + 0.70 * (internet == "Fiber optic") - 0.55 * (internet == "No")
             + 0.55 * (payment == "Electronic check")
             + 0.45 * (tech == "No") + 0.30 * (security == "No")
             + 0.30 * (paperless == "Yes") + 0.25 * senior
             + 0.008 * (monthly - 65))
    churn = np.where(rng.random(N) < 1 / (1 + np.exp(-logit)), "Yes", "No")

    df = pd.DataFrame({
        "customerID": ids, "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection, "TechSupport": tech,
        "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": np.where(tenure == 0, " ", total.astype(str)), "Churn": churn,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(df):,} rows, churn rate {(df['Churn'] == 'Yes').mean():.1%})")


if __name__ == "__main__":
    main()
