import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

# Load the model and preprocessing objects
model = joblib.load('D:/AML proj/random_forest_model.pkl')
label_encoder = joblib.load('E:/aml/label_encoder.pkl')
one_hot_encoder = joblib.load('E:/aml/one_hot_encoder.pkl')
imputer_num = joblib.load('E:/aml/imputer_num.pkl')
imputer_cat = joblib.load('E:/aml/imputer_cat.pkl')

# Define a function for prediction
def predict_vendor(size, purchase_price, dollars):
    data = {
        'Size': [size],
        'PurchasePrice': [purchase_price],
        'Dollars': [dollars]
    }
    df = pd.DataFrame(data)
    
    # Preprocess the input data
    df['Size'] = df['Size'].astype(str)
    df[numeric_features] = imputer_num.transform(df[numeric_features])
    df[categorical_features] = imputer_cat.transform(df[categorical_features])
    
    encoded_cat = one_hot_encoder.transform(df[categorical_features])
    encoded_cat_df = pd.DataFrame(encoded_cat.toarray(), columns=one_hot_encoder.get_feature_names_out(categorical_features))
    
    X = pd.concat([df[numeric_features], encoded_cat_df], axis=1)
    
    # Predict using the model
    prediction = model.predict(X)
    vendor_name = label_encoder.inverse_transform(prediction)
    
    return vendor_name[0]

# Streamlit app
st.title('Vendor Prediction App')

size = st.text_input('Size')
purchase_price = st.number_input('Purchase Price', min_value=0.0, format='%f')
dollars = st.number_input('Dollars', min_value=0.0, format='%f')

if st.button('Predict'):
    result = predict_vendor(size, purchase_price, dollars)
    st.write(f'The predicted vendor name is: {result}')
