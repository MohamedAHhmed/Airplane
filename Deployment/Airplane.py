import streamlit as st
import pickle
import numpy as np
import pandas as pd
from statistics import mode

# Load the model and scaler
SVC = pickle.load(open('notebook/SVC.pkl', 'rb'))
Bagging = pickle.load(open('notebook/Bagging.pkl', 'rb'))
RF = pickle.load(open('notebook/RF.pkl', 'rb'))
GB = pickle.load(open('notebook/Gradient Boosting.pkl', 'rb'))
KNN = pickle.load(open('notebook/KNN.pkl', 'rb'))
scaler = pickle.load(open('notebook/scaler.pkl', 'rb'))
pt = pickle.load(open('notebook/Powertransformer.pkl', 'rb'))
input_names = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
       'Flight Distance', 'Inflight wifi service',
       'Departure/Arrival time convenient', 'Gate location', 'Food and drink',
       'Online boarding', 'Seat comfort', 'Inflight entertainment',
       'On-board service', 'Leg room service', 'Baggage handling',
       'Checkin service', 'Inflight service', 'Departure Delay in Minutes'
]

cat_features =['Gender','Type of Travel','Class','Customer Type']



def user_input():
    features = {}

    # Create 3 columns for all inputs
    col1, col2, col3 = st.columns(3)

    # Distribute inputs across the 3 columns
    with col1:
        features['Gender'] = st.selectbox('Gender', options=('Male', 'Female'))
        features['Age'] = st.slider('Age', min_value=10, max_value=100, value=50)
        features['Inflight wifi service'] = st.selectbox('Inflight Wifi Service', options=(1, 2, 3, 4, 5))
        features['Online boarding'] = st.selectbox('Online Boarding', options=(1, 2, 3, 4, 5))
        features['On-board service'] = st.selectbox('On-board Service', options=(1, 2, 3, 4, 5))
        features['Checkin service'] = st.selectbox('Checkin Service', options=(1, 2, 3, 4, 5))

    with col2:
        features['Customer Type'] = st.selectbox('Customer Type', options=('Loyal Customer', 'disloyal Customer'))
        features['Flight Distance'] = st.slider('Flight Distance (km)', min_value=0,max_value=100000 ,value=50000)
        features['Departure/Arrival time convenient'] = st.selectbox('Departure/Arrival Time Convenient', options=(1, 2, 3, 4, 5))
        features['Seat comfort'] = st.selectbox('Seat Comfort', options=(1, 2, 3, 4, 5))
        features['Leg room service'] = st.selectbox('Leg Room Service', options=(1, 2, 3, 4, 5))
        features['Inflight service'] = st.selectbox('Inflight Service', options=(1, 2, 3, 4, 5))

    with col3:
        features['Type of Travel'] = st.selectbox('Type of Travel', options=('Business travel','Personal Travel'))
        features['Class'] = st.selectbox('Class', options=('Eco', 'Eco Plus', 'Business'))
        features['Gate location'] = st.selectbox('Gate Location', options=(1, 2, 3, 4, 5))
        features['Food and drink'] = st.selectbox('Food and Drink', options=(1, 2, 3, 4, 5))
        features['Inflight entertainment'] = st.selectbox('Inflight Entertainment', options=(1, 2, 3, 4, 5))
        features['Baggage handling'] = st.selectbox('Baggage Handling', options=(1, 2, 3, 4, 5))
        features['Departure Delay in Minutes'] = st.number_input('Departure Delay in Minutes', min_value=0, value=0)

    return features


user_features = user_input()

features_list = []
for col in input_names:
    value = user_features[col]

    if col in cat_features:
        le = pickle.load(open(f'notebook/{col}_le.pkl', 'rb'))
        transformed_value = le.transform(np.array([[value]]))
        features_list.append(transformed_value.item())
    else:
         features_list.append(value)
        
features_array = np.array(features_list).reshape(1,-1)
feature_trans=pt.transform(features_array)
features_scaled = scaler.transform(feature_trans)

col = st.columns(5)
y_pred = []

def predict_and_display(model,name):
    if st.button(name):
        # Make a prediction
        y_pred_model = model.predict(features_scaled)
        y_pred.append(y_pred_model)

        # Display the result
        if y_pred_model == 1:
            st.success('The User May Satisfied ')
        else:
            st.error('The User May Neutral or Dissatisfied ')
            

with col[0]:
    predict_and_display(KNN,'KNN')
with col[1]:
    predict_and_display(SVC,'SVC')
with col[2]:
    predict_and_display(GB,'GB')
with col[3]:
    predict_and_display(RF,'RF')
with col[4]:
    predict_and_display(Bagging,'Bagging')




