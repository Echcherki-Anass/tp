# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 10:27:19 2021

@author: ANASS ECHCHERKI
"""

import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
# RON  Prediction App

This app predicts the **Research Octane Number**!
""")
st.write('---')

# Loads the dataset

df= pd.read_excel("RON_Data.xlsx",index_col=0, sheet_name='Table 1')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df=df.drop('Ref.', axis=1)
df=df.dropna()
df = df.rename(columns = {
 'n-heptane':'n_heptane',
 'iso-octane':'iso_octane',
 '1-hexene':'One_hexene',
 'cyclopent':'cyclopent',
 'toluene':'toluene',
 'ethanol':'ethanol',
 'ETBE':'ETBE',
 'RON':'RON',
 })
X= df.drop('RON', axis=1)
Y=df['RON']


# Sidebar
# Header of Specify Input Parameters
st.sidebar.header('Specify Input Parameters')

def user_input_features():
    n_heptane = st.sidebar.slider('n_heptane', X.n_heptane.min(), X.n_heptane.max(), X.n_heptane.mean())
    iso_octane = st.sidebar.slider('iso_octane', X.iso_octane.min(), X.iso_octane.max(), X.iso_octane.mean())
    One_hexene = st.sidebar.slider('One_hexene', X.One_hexene.min(), X.One_hexene.max(), X.One_hexene.mean())
    cyclopent = st.sidebar.slider('cyclopent', X.cyclopent.min(), X.cyclopent.max(), X.cyclopent.mean())
    toluene = st.sidebar.slider('toluene', X.toluene.min(), X.toluene.max(), X.toluene.mean())
    ethanol = st.sidebar.slider('ethanol', X.ethanol.min(), X.ethanol.max(), X.ethanol.mean())
    ETBE = st.sidebar.slider('ETBE', X.ETBE.min(), X.ETBE.max(), X.ETBE.mean())
    data = {'n_heptane': n_heptane,
            'iso_octane': iso_octane,
            'One_hexene': One_hexene,
            'cyclopent': cyclopent,
            'toluene': toluene,
            'ethanol': ethanol,
            'ETBE': ETBE
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# Main Panel

# Print specified input parameters
st.header('Specified Input parameters')
st.write(df)
st.write('---')

# Build Regression Model
model = RandomForestRegressor()
model.fit(X, Y)
# Apply Model to Make Prediction
prediction = model.predict(df)

st.header('Prediction of RON')
st.write(prediction)
st.write('---')

# Explaining the model's predictions using SHAP values
# https://github.com/slundberg/shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

st.header('Feature Importance')
plt.title('Feature importance based on SHAP values')
shap.summary_plot(shap_values, X)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values (Bar)')
shap.summary_plot(shap_values, X, plot_type="bar")
st.pyplot(bbox_inches='tight')
