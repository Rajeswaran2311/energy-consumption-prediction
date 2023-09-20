import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import matplotlib.image as mpimg
from tensorflow import keras
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import numpy as np


from keras.models import load_model



# Define the home function
def home():
    st.header('AI Based Energy Monitoring')
    
    st.write("## Introduction")
    # imageha = mpimg.imread('stone.jpg')     
    # st.image(imageha)
    st.write("This app uses to monitoring energy consumptionof devcice and visualize flow of current over period of time .")
   
    data=pd.read_csv('datae.csv')
    st.markdown('**Glimpse of dataset**')
    st.write(data.head(5))
    st.write("Vrms ph-n L1N Min (Minimum voltage measured between phase and neutral)")
    st.write("Vrms ph-n L1N Avg (Average voltage measured between phase and neutral)")
    st.write("Vrms ph-n L1N Max (Maximum voltage measured between phase and neutral) ")
    st.write("Current L1 Min (Minimum current)")
    st.write("Current L1 Avg (Average current)")
    st.write("Current L1 Max (Maximum current)")

   
# Define the prediction function
def prediction():
    st.header('AI Based Energy Monitoring')
    st.subheader('Power Factor Predictor')
    
    st.write("Please fill in the following information to get a prediction:")
    
    # Define the input fields
    V1min = st.number_input("R Phase Voltage", step=1.,format="%.2f")
    V2Avg= st.number_input("Y Phase Voltage",step=1.,format="%.2f")
    V3max = st.number_input("B Phase Voltage", step=1.,format="%.2f")
    c1min = st.number_input("R Phase Current", step=1.,format="%.2f")
    c2avg = st.number_input("Y Phase Current", step=1.,format="%.2f")
    c3max = st.number_input("B Phase Current", step=1.,format="%.2f")
    
    

    
    data = pd.DataFrame([[V1min,V2Avg,V3max,c1min,c2avg,c3max]])
    

# Load the saved logistic regression model
    model =load_model('bestEEE.h5')
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = data.reshape(1, data.shape[1], 1)

# Get the model prediction
    
    prediction = model.predict(data)
    # if prediction>0.5:
    #     prediction=1
    # else:
    #     prediction=0
    

# Show the prediction result
    st.write("### Prediction Result")
    if st.button("Predict"): 
            st.success(f"{prediction[0][0]}")
def dashboard():
    
# Sample dataset

# Load the dataset into a DataFrame
    df = pd.read_csv('datae.csv')
    df['Time'] = pd.to_numeric(df['Time'].str.replace(':', '').str.replace('.', ''), errors='coerce')

    # Streamlit app
    st.title('Interactive Dashboard')

    # 4x4 grid layout
    col1, _ = st.columns(2)
    
    

    # Line Chart
    with col1:
        st.subheader('Line Chart')
        start_time, end_time = st.slider(
        'Select Time Range',
        min_value=df['Time'].min(),
        max_value=df['Time'].max(),
        value=(df['Time'].min(), df['Time'].max())
    )

# Filter the DataFrame based on the selected time range
        filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
        fig_line = px.line(filtered_df , x='Time',  y=['Current L1 Min', 'Current L1 Avg', 'Current L1 Max'], 
        color_discrete_sequence=['blue', 'green', 'red'])
        st.plotly_chart(fig_line)
    
    # Area Chart
    with col1:
        st.subheader('Area Chart')
        

# Filter the DataFrame based on the selected time range
        filtered_df = df[(df['Time'] >= start_time) & (df['Time'] <= end_time)]
        fig_area = px.area(filtered_df, x='Time', y='Current L1 Min')
        st.plotly_chart(fig_area)

    # Bar Chart
    # with col4:
    #     st.subheader('Bar Chart')
    #     fig_bar = px.bar(df.head(20), x='Time', y='Current L1 Min')
    #     st.plotly_chart(fig_bar)
    # with col5:
    #     pass

    # # Line and Bar Chart Combination
    # with col6:
    #     st.subheader('Side-by-Side Bar Chart')
    #     fig_bar_side_by_side = go.Figure()

    #     # Add the 'Vrms ph-n L1N Avg' bar
    #     fig_bar_side_by_side.add_trace(go.Bar(
    #         x=df['Time'].head(10),
    #         y=df['Vrms ph-n L1N Avg'].head(10),
    #         name='Vrms ph-n L1N Avg',
    #         marker_color='blue'  # Specify color for the first bar
    #     ))

    #     # Add the 'Current L1 Min' bar next to it
    #     fig_bar_side_by_side.add_trace(go.Bar(
    #         x=df['Time'].head(10),
    #         y=df['Current L1 Avg'].head(10),
    #         name='Current L1 Avg',
    #         marker_color='green'  # Specify color for the second bar
    #     ))

    #     fig_bar_side_by_side.update_layout(
    #         barmode='group',  # Use 'group' for side-by-side bars
    #         xaxis_title='Duration',
    #         yaxis_title='Values'
    #     )

    #     st.plotly_chart(fig_bar_side_by_side)



def main():
    st.set_page_config(page_title="AI Based Energy Monitoring", page_icon=":electric_plug:")
    st.markdown("<h1 style='text-align: center; color: white;'>AI Based Energy Monitoring</h1>", unsafe_allow_html=True)
    
    
# Create the tab layout
    tabs = ["Home", "Prediction",'Dashboard']
    page = st.sidebar.selectbox("Select a page", tabs)

# Show the appropriate page based on the user selection
    
    if page == "Home":
        home()
    elif page == "Prediction":
        prediction()
    elif page =='Dashboard':
        dashboard()
    
   
main()


























