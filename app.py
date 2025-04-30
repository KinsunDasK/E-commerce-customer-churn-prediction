import streamlit as st
import pandas as pd
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

st.title("Customer Churn Prediction")

col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    Tenure = st.number_input("Tenure", min_value=0)
    PreferredLoginDevice = st.selectbox("Preferred Login Device", ["Mobile Phone", "Computer"])
    CityTier = st.selectbox("City Tier", [1,2,3])
    WarehouseToHome = st.number_input("Warehouse to Home", min_value=0)
with col2:
    PreferredPaymentMode = st.selectbox("Preferred payment mode", ["Debit Card","Credit Card","E wallet","UPI","Cash on Delivery"])
    gender = st.selectbox("Gender", ["Male","Female"])
    HourSpendOnApp = st.number_input("Hour Spend on App", min_value=0.0)
    NumberOfDeviceRegistered = st.selectbox("Number of Devices Registered", [1,2,3,4])
with col3:
    PreferedOrderCat = st.selectbox("Preferred Order Category", ["Laptop & Accessory","Mobile Phone","Fashion","Grocery","others"])
    SatisfactionScore = st.selectbox("Satisfaction Score", [1,2,3,4,5])
    MaritalStatus = st.selectbox("Marital Status", ["Single","Married","Divorced"])
    NumberOfAddress = st.number_input("Number of address", min_value=0)
with col4:
    Complain = st.number_input("No of complains", min_value=0)
    OrderAmountHikeFromlastYear = st.number_input("Order Amount Hike from last year", min_value=0)
    CouponUsed = st.number_input("Coupon Used", min_value=0)
with col5:
    OrderCount = st.number_input("Order Count", min_value=0)
    DaySinceLastOrder = st.number_input("Days Since Last Order", min_value=0)
    CashbackAmount = st.number_input("Cash back amount", min_value=0)

input_data = pd.DataFrame(
        {
            "Tenure": [Tenure],
            "PreferredLoginDevice": [PreferredLoginDevice],
            "CityTier": [CityTier],
            "WarehouseToHome": [WarehouseToHome],
            "PreferredPaymentMode": [PreferredPaymentMode],
            "Gender": [gender],
            "HourSpendOnApp": [HourSpendOnApp],
            "NumberOfDeviceRegistered": [NumberOfDeviceRegistered],
            "PreferedOrderCat": [PreferedOrderCat],
            "SatisfactionScore": [SatisfactionScore],
            "MaritalStatus": [MaritalStatus],
            "NumberOfAddress": [NumberOfAddress],
            "Complain": [Complain],
            "OrderAmountHikeFromlastYear": [OrderAmountHikeFromlastYear],
            "CouponUsed": [CouponUsed],
            "OrderCount": [OrderCount],
            "DaySinceLastOrder": [DaySinceLastOrder],
            "CashbackAmount": [CashbackAmount],
            
        })

cat_cols = ['PreferredLoginDevice',
 'PreferredPaymentMode',
 'Gender',
 'PreferedOrderCat',
 'MaritalStatus']

encoded_data = pd.DataFrame(
        encoder.transform(input_data[cat_cols]).toarray(),
        columns=encoder.get_feature_names_out(cat_cols),
    )
input_data = input_data.drop(cat_cols, axis=1)
input_data = pd.concat([input_data, encoded_data], axis=1)

    # Make prediction using the loaded model
prediction = model.predict(input_data)

    # Display the prediction
st.subheader("Prediction:")
if prediction[0] == 1:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is likely to stay.")


