# -*- coding: utf-8 -*-
"""
Created on Fri May 31 10:13:21 2024

@author: align
"""

import pickle
import streamlit as st

import pandas as pd

load= open('model.pkl','rb')
model=pickle.load(load)

load1= open('scaler.pkl','rb')
scaler=pickle.load(load1)


def main():   
    
    st.title('Titanic Survival Prediction')
    
    # Input fields
    Pclass = st.selectbox('Pclass', [1, 2, 3])
    Age = st.number_input('Age', value=30)
    SibSp = st.number_input('Siblings/Spouses Aboard', value=0)
    Parch = st.number_input('Parents/Children Aboard', value=0)
    Fare = st.number_input('Fare', value=32.0)
    Sex = st.selectbox('Sex', ['male', 'female'])
    Embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
    
    # Encode and scale the input data
    input_data = pd.DataFrame([[Pclass, Age, SibSp, Parch, Fare, Sex, Embarked]], columns=['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked'])
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    input_data = scaler.transform(input_data)

    # Prediction
    print(input_data)
    
    if st.button('Predict'):
        prediction = model.predict(input_data)[0]
        if prediction==1:
            st.success("Survived")
        else:
            st.success("Did not survive")

if __name__ == '__main__':
    main()
    
 