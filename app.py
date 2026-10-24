import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from inference.predict_freight import predict_freight_cost
from inference.predict_invoice_flag import predict_flag_invoice

# --------------------------------------------------------------------
# Page Configuration
# --------------------------------------------------------------------

st.set_page_config(
    page_title = "Vendor Invoice Intelligence Portal",
    page_icon = "",
    layout = "wide"
)

# --------------------------------------------------------------------
# Header Section
# --------------------------------------------------------------------

st.markdown("""
# Vendor Invoice Intelligence Portal
### AI-Driven freight Cost prediction & Invoice Risk Flagging 

this internal analytics portal leverages machine learning to 
-**Forecast freight costs accurately**
-**Detect risky or abnormal vendor invoices**
-**Reduce financial leakage and manual workload**
-**Forecast freight costs accurately**
""")

st.divider()

# --------------------------------------------------------------------
# SideBar
# --------------------------------------------------------------------

st.sidebar.title(" Model Selection")
selected_model = st.sidebar.radio(
    "Choose Prediction Module",
    [ "Freight Cost Prediction",
      "Invoice Manual Approval Flag"
    ]
)

st.sidebar.markdown("""
-----
**Business Impact**
- Improved cost forecasting
- Reduced invoice fraud & anomalies
- Faster finance operations
""")

# --------------------------------------------------------------------
# Freight Cost Prediction
# --------------------------------------------------------------------

if selected_model == "Freight Cost Prediction":
    st.subheader("Freight Cost Prediction")

    st.markdown("""
    **Objective:**
    Predict freight cost for vendor invoice using **Quantity** and **Invoice Dollars**
    to support budgeting, forecasting and vendor negotiations.
    """)

    with st.form("freight_form"):
        col1,col2 = st.columns(2)

        with col1:
            quantity = st.number_input(
                "Quantity",
                min_value = 1,
                value = 1200
            )

        with col2:
            dollars = st.number_input(
                "Invoice Dollars",
                min_value = 1.0,
                value = 18500.0
            )

        submit_freight = st.form_submit_button("Predict Freight Cost")

    if submit_freight:
        input_data = {
            "Quantity" : [quantity],
            "Dollars" : [dollars]
        }
        prediction = predict_freight_cost(input_data)['predicted_Freight']

        st.success("Prediction completed successfully.")

        st.metric(
            label = "Estimated Feight Cost",
            value = f"${prediction[0]:,.2f}"
        )


# --------------------------------------------------------------------
# Invoice Flag Prediction
# --------------------------------------------------------------------
else:
    st.subheader("Invoice Manual Approval Prediction")

    st.markdown("""
    **Objective:**
    Predict whether a vendor should be  **flagged for manual approval** 
    based on abnormal cost, freight, or delivery patters.
    """)

    with st.form("invoice_flag_form"):
        col1,col2,col3 = st.columns(3)

        with col1:
            invoice_quantity = st.number_input(
                "Invoice Quantity",
                min_value = 1,
                value = 50
            )
            freight = st.number_input(
                "Freight Cost",
                min_value = 0.0,
                value = 1.73
            )
        with col2:
            invoice_dollars = st.number_input(
                "Invoice Dollars",
                min_value = 1.0,
                value = 352.95
            )
            total_item_quantity = st.number_input(
                "Total Item Quantity",
                min_value = 1,
                value = 162
            )
        with col3:
            total_item__dollars = st.number_input(
                "Total Item Dollars",
                min_value = 1.0,
                value = 2576.0
            )

        submit_freight = st.form_submit_button("Evaluate Invoice Risk")

    if submit_freight:
        input_data = {
            "invoice_quantity" : [invoice_quantity],
            "invoice_dollars" : [invoice_dollars],
            "Freight" :[freight],
            "total_item_quantity" : [total_item_quantity],
            "total_item_dollars" : [total_item__dollars]
        }
        
        flag_prediction = predict_flag_invoice(input_data)['predicted_invoice_flag']
        is_flagged = bool(flag_prediction[0])

        if is_flagged:
            st.error("Invoice requires **MANUAL APPROVAL**")
        else:
            st.success("Invoice is **SAFE for Auto-Approval**")