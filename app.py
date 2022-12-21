import streamlit as st
import numpy as np
# import pickle
import pandas as pd
from pycaret.classification import load_model, predict_model


header = st.container()
dataset = st.container()
# features = st.container()
# model_prediction = st.container()

# print(loaded_model)


@st.cache(allow_output_mutation=True)
def get_data(filename):
    loaded_model_f = load_model(filename)
 	# ml_service = pickle.load(open(file_name,'rb'))
    return loaded_model_f
loaded_model = get_data("my_best_dia_pipline")


# Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
with st.form(key='my_form'):
    Pregnancies = st.number_input('Pregnancies')
    Glucose = st.number_input('Glucose')
    BloodPressure = st.number_input('BloodPressure')
    SkinThickness = st.number_input('SkinThickness')
    Insulin = st.number_input('Insulin')
    BMI = st.number_input('BMI')
    DiabetesPedigreeFunction = st.number_input('DiabetesPedigreeFunction')
    Age = st.number_input('Age')
    # sit_ups_counts = st.number_input('sit_ups_counts')
    # broad_jump_cm = st.number_input('broad_jump_cm')
    # gender_txt = st.text_input('M or F')
    predict = st.form_submit_button('Predict')
    
    if predict:
        # creating data frame for input
        data = dict(Pregnancies=Pregnancies,	Glucose=Glucose,	BloodPressure=BloodPressure,
        SkinThickness=SkinThickness,Insulin=Insulin,BMI=BMI,
        DiabetesPedigreeFunction=DiabetesPedigreeFunction,Age=Age,Outcome=1)
        df_3 = pd.DataFrame(data, index=[0])   
        # printing data entered     
        st.write(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
        # model preedict        
        df1 = predict_model(loaded_model,df_3)
        # removing one of the column
        df1.drop('Outcome',inplace=True, axis=1)
        st.write(df1)        
        st.header("Result")
        st.write(df1.loc[0, 'Label'])

    