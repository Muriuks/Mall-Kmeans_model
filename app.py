import pandas as pd
import numpy as np
import joblib
import streamlit as st

#load file

model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

st.title("Customer Clustering App")

st.write("This app uses a KMeans clustering model to segment customers based on their features.")

#create input fields for user to enter customer data
genre = st.selectbox("Genre", options=["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income (in $)", min_value=10000, max_value=1000000, value=50000)   
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)    

#encode categorical variable
genre_encoded = encoder.transform([[genre]])[0]

#created dataFrame for the input data
input_data = pd.DataFrame({
    'Genre': [genre_encoded],
    'Age': [age],
    'Annual Income (k$)': [income],
    'Spending Score (1-100)': [spending_score]
})

#scale the input data
input_data_scaled = scaler.transform(input_data)

#make prediction using the KMeans model
if st.button("Predict Cluster"):
    cluster = model.predict(input_data_scaled)
    if cluster[0] == 0:
        st.write("The customer belongs to Cluster 0: Send luxury promotions.")
    if cluster[0] == 1:
        st.write("The customer belongs to Cluster 1: Send budget-friendly promotions.")
    if cluster[0] == 2:
        st.write("The customer belongs to Cluster 2: Send high-value promotions.")
    if cluster[0] == 3:
        st.write("The customer belongs to Cluster 3: Send mid-range promotions.")
    if cluster[0] == 4:
        st.write("The customer belongs to Cluster 4: Send low-value promotions.")
    if cluster[0] == 5:
        st.write("The customer belongs to Cluster 5: Send exclusive promotions.")
    if cluster[0] == 6:
        st.write("The customer belongs to Cluster 6: Send seasonal promotions.")
    if cluster[0] == 7:
        st.write("The customer belongs to Cluster 7: Send personalized promotions.")
    if cluster[0] == 8:
        st.write("The customer belongs to Cluster 8: Send loyalty program promotions.")
    if cluster[0] == 9:
        st.write("The customer belongs to Cluster 9: Send referral program promotions.")
else:
    st.write("General Marketing")