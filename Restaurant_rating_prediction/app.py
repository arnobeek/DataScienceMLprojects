import streamlit as st
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib

st.set_page_config(layout="wide")

scaler = joblib.load("scaler.pkl")

st.title("Restaurant Rating Prediction App")

st.caption("This app predicts the rating of a restaurant")

st.divider()

averagecost = st.number_input("Please enter the estimated cost for two",min_value=50,max_value=9999999,value = 1000, step = 100)
tablebooking = st.selectbox("Restaurant has table booking?",["Yes","No"])
onlinedelivery = st.selectbox("Restaurant has online delivery?",["Yes","No"])

pricerange = st.selectbox("What is the price range(1 Cheapest, 4 Most Expensive)",[1,2,3,4])

predictbutton = st.button("Predict the review!")

st.divider()

model = joblib.load("mlmodel.pkl")

bookingstatus = 1 if tablebooking == "Yes" else 0
deliverystatus = 1 if onlinedelivery == "Yes" else 0

values = [[averagecost,bookingstatus,deliverystatus,pricerange]]
my_X_values = np.array(values)

X = scaler.transform(my_X_values)

if predictbutton:
    
    prediction = model.predict(X)

    if prediction < 2.5:
        st.write("Poor")
    elif prediction <3.5:
        st.write("Average")    
    elif prediction <4.0:
        st.write("Good")    
    elif prediction <4.5:
        st.write("Very Good")
    else:
        st.write("Excellent")        