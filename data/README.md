# Data

`telco_customer_churn.csv` is **simulated** by `src/generate_data.py` (seed 11). It has the same 21 columns as the public IBM Telco Customer Churn sample, but no real customer data.

| Column group | Columns |
|---|---|
| Customer | customerID, gender, SeniorCitizen, Partner, Dependents |
| Account | tenure (months), Contract, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges |
| Services | PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies |
| Target | Churn (Yes / No) |

Churn is drawn from a known logistic model: month-to-month contracts, fibre internet, electronic-check payment, missing tech support or online security, paperless billing and senior customers raise the odds, while longer tenure and two-year contracts lower them. `TotalCharges` is text and blank for the 11 customers with tenure 0, as in the original layout.
