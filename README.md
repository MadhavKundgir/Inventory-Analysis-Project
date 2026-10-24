# Vendor Invoice Intelligence Portal

A machine learning app that predicts freight cost for vendor invoices and flags
risky invoices for manual approval, with a Streamlit interface.

## Features
- **Freight cost prediction:** regression model trained on invoice data
- **Invoice risk flagging:** Random Forest classifier tuned with GridSearchCV
- **SQL feature engineering:** CTEs, joins and aggregations on SQLite
- **Streamlit app** for real-time predictions

## Results
| Model | Metric | Value |
|---|---|---|
| Linear Regression (freight cost) | MAE / R² | $24.46 / 0.97 |
| Random Forest (invoice flag) | F1 (flagged class) | 0.82 |

Three models were compared for freight cost (Linear Regression, Decision Tree,
Random Forest); the one with the lowest MAE was saved.

## Tech Stack
Python, pandas, NumPy, scikit-learn, SQLite, Streamlit, joblib

## Project Structure
```
├── app.py                      # Streamlit app
├── data/                       # SQLite database
├── Freight_cost_prediction/    # training code for freight model
├── Invoice Flagging/           # training code for invoice flag model
├── inference/                  # prediction functions used by the app
└── models/                     # saved models
```

## How to Run
```bash
git clone https://github.com/YOUR-USERNAME/vendor-invoice-intelligence.git
cd vendor-invoice-intelligence
pip install -r requirements.txt

# train the models (creates files in models/)
python "Freight_cost_prediction/train.py"
python "Invoice Flagging/train.py"

# start the app
streamlit run app.py
```


## Author
Madhav Kundgir www.linkedin.com/in/madhav-kundgir
