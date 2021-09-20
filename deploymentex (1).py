import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters.model import SimpleExpSmoothing

import streamlit as st

html_temp = """
    <div style="background-color:#025246 ;padding:12px">
    <h2 style="color:black;text-align:center;">Forecasting Engine </h2>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

st.markdown(" **Operation** : *Forecasting* Availability of the beds ")

st.write("""
### Explore which model is best for the given dataset using the dropdown on the left
""")

forecasting_technique = st.sidebar.selectbox("select forecasting_methods", (
"ARIMA", "SARIMA", "SIMPLE_AVERAGE", "SIMPLE_EXPONENTIAL_SMOOTHING", "HOLTS_METHOD", "HOLTS_WINTER_METHOD"))

data = pd.read_csv(r"C:\Users\91770\Downloads\Beds_Occupied.csv", encoding="latin1")

data_set = st.sidebar.selectbox("select the dataset", ("beds_occupied", "none"))

data["total_beds"] = 900
data["Available_beds"] = data["total_beds"] - data["Total Inpatient Beds"]
data = data[["collection_date", "Available_beds"]]
data.shape
train_data = data[:186]
test_data = data[186:]
train_data["collection_date"] = pd.to_datetime(train_data["collection_date"])
train_data["collection_date"] = train_data["collection_date"].dt.strftime("%d-%m-%y")
train_data["collection_date"] = train_data["collection_date"].astype("datetime64[ns]")
# train_data["collection_date"].strftime("%Y-%m-%d")

train_data.index = train_data["collection_date"]


train_data = train_data[["Available_beds"]]

tr = train_data[:140]
test = train_data[140:]

val_tr = test_data[:120]
val_test = test_data[120:]

## Arima model:

model_arima = ARIMA(val_tr["Available_beds"].astype("float64"), order=(0,0,4))
model_arima_fit = model_arima.fit(disp=0)

val_test["forecasted"] = model_arima_fit.forecast(steps=60)[0]

arima_rmse = sqrt(mean_squared_error(val_test["forecasted"], val_test["Available_beds"]))

# arima_future forecasting:

arima_future = model_arima_fit.forecast(steps=31)[0]
date = pd.date_range("16-06-2021", "16-07-2021", freq="D")
Date = date.strftime("%Y-%m-%d")
final_data = pd.DataFrame(Date)
final_data["forecasting"] = arima_future
final_data.columns = ["date", "forecasted_result"]

import statsmodels.api as sm

sarima = sm.tsa.statespace.SARIMAX(val_tr["Available_beds"].values, order=(2, 0, 2), seasonal_order=(0, 1, 1, 7))

sarima_model = sarima.fit()

val_test["forecasted_sarima"] = sarima_model.predict(start=307, end=366, dynamic=True)

# train rmse sarima:

Sarima_rmse = sqrt(mean_squared_error(val_test["forecasted_sarima"], val_test["Available_beds"]))

sarima_future = sarima_model.predict(start=367, end=398)[0]
date = pd.date_range("16-06-2021", "16-07-2021", freq="D")
Date = date.strftime("%Y-%m-%d")
final_sarima = pd.DataFrame(Date)
final_sarima["forecasting"] = sarima_future
final_sarima.columns = ["date", "forecasted_result"]

## Simple _Average_method

val_test["forecasted_simpleaverage"] = val_tr["Available_beds"].mean()

simpleaverage_rmse = sqrt(mean_squared_error(val_test["forecasted_simpleaverage"], val_test["Available_beds"]))

sa_future = val_tr["Available_beds"].mean()
date = pd.date_range("16-06-2021", "16-07-2021", freq="D")
Date = date.strftime("%Y-%m-%d")
final_sa = pd.DataFrame(Date)
final_sa["forecasting"] = sarima_future
final_sa.columns = ["date", "forecasted_result"]

##Simple exponential smoothing:
model = SimpleExpSmoothing(val_tr["Available_beds"])

simple_model = model.fit()

val_test["SimpleExp"] = simple_model.forecast(steps=60)

simpleexp_rmse = sqrt(mean_squared_error(val_test["SimpleExp"], val_test["Available_beds"]))

simpleExp_future = simple_model.forecast(steps=31)
date = pd.date_range("16-06-2021", "16-07-2021", freq="D")
Date = date.strftime("%Y-%m-%d")
final_simexp = pd.DataFrame(simpleExp_future)
final_simexp.index= Date
# final_simexp["forecasting"] = simpleExp_future
final_simexp.columns = ["forecast"]

## Holts method
de_model = ExponentialSmoothing(val_tr["Available_beds"])
de_model = de_model.fit()
holt_forecasted = de_model.predict(start=val_test.index[0], end=val_test.index[-1])
holt_forecasted.index = val_test.index
# d= val_test["holt_forecasted"].astype("int64")
# d= val_test["Available_beds"].astype("float64")

holt_future = de_model.forecast(steps=31)
date = pd.date_range("16-06-2021", "16-07-2021", freq="D")
Date = date.strftime("%Y-%m-%d")
final_holt = pd.DataFrame(simpleExp_future)
final_holt.index= Date
# final_simexp["forecasting"] = simpleExp_future
final_holt.columns = ["forecast"]


Holt_rmse = sqrt(mean_squared_error(holt_forecasted, val_test["Available_beds"]))

## holts winter method:

hw_model = ExponentialSmoothing(val_tr["Available_beds"], seasonal="add", seasonal_periods=40).fit()
val_test["hw_forecasted"] = hw_model.forecast(len(val_test))

hw_test_rmse = sqrt(mean_squared_error(val_test["hw_forecasted"], val_test["Available_beds"]))

holtwinter_future = hw_model.forecast(steps=31)
date = pd.date_range("16-06-2021", "16-07-2021", freq="D")
Date = date.strftime("%Y-%m-%d")
final_holtwinter = pd.DataFrame(simpleExp_future)
final_holtwinter.index= Date
# final_simexp["forecasting"] = simpleExp_future
final_holtwinter.columns = ["forecast"]

if forecasting_technique == "ARIMA":
    if data_set == "beds_occupied":
        fig = plt.figure(figsize=(10, 5))
        plt.plot(val_tr["Available_beds"], label="train")
        plt.plot(val_test["Available_beds"], label="test")
        plt.plot(val_test["forecasted"], label="forecasted")
        plt.legend(loc="lower left")
        plt.show()
        st.pyplot(fig)
        st.write(f"ARIMA MODEL RMSE IS : ", arima_rmse)
        st.write("lets try forecasting future values")
        if st.button("predict"):
            st.write("The Result's are *Here*:")
            final_data

if forecasting_technique == "SARIMA":
    if data_set == "beds_occupied":
        fig = plt.figure(figsize=(10, 5))
        plt.plot(val_tr["Available_beds"], label="train")
        plt.plot(val_test["Available_beds"], label="test")
        plt.plot(val_test["forecasted_sarima"], label="forecasted")
        plt.legend(loc="lower left")
        plt.show()
        st.pyplot(fig)
        st.write(f"SARIMA MODEL RMSE IS : ", Sarima_rmse)
        if st.button("predict"):
            st.write("The Result's are *Here*:")
            final_sarima

if forecasting_technique == "SIMPLE_AVERAGE":
    if data_set == "beds_occupied":
        fig = plt.figure(figsize=(10, 5))
        plt.plot(val_tr["Available_beds"], label="train")
        plt.plot(val_test["Available_beds"], label="test")
        plt.plot(val_test["forecasted_simpleaverage"], label="forecasted")
        plt.legend(loc="lower left")
        plt.show()
        st.pyplot(fig)
        st.write(f"SIMPLE_AVERAGE MODEL RMSE IS : ", simpleaverage_rmse)
        if st.button("predict"):
            st.write("The Result's are *Here*:")
            final_sa

if forecasting_technique == "SIMPLE_EXPONENTIAL_SMOOTHING":
    if data_set == "beds_occupied":
        fig = plt.figure(figsize=(10, 5))
        plt.plot(val_tr["Available_beds"], label="train")
        plt.plot(val_test["Available_beds"], label="test")
        plt.plot(val_test["SimpleExp"], label="forecasted")
        plt.legend(loc="lower left")
        plt.title("SIMPLE_EXPONENTIAL")
        plt.show()
        st.pyplot(fig)
        st.write(f"Simple Exponential MODEL RMSE IS : ", simpleexp_rmse)
        if st.button("predict"):
            st.write("The Result's are *Here*")
            final_simexp

if forecasting_technique == "HOLTS_METHOD":
    if data_set == "beds_occupied":
        fig = plt.figure(figsize=(10, 5))
        plt.plot(val_tr["Available_beds"], label="train")
        plt.plot(val_test["Available_beds"], label="test")
        plt.plot(holt_forecasted, label="forecasted")
        plt.legend(loc="lower left")
        plt.title("Holt Method")
        plt.show()
        st.pyplot(fig)
        st.write(f"HOLT MODEL RMSE IS : ", Holt_rmse)
        if st.button("predict"):
            st.write("The Result's are *Here*")
            final_holt

if forecasting_technique == "HOLTS_WINTER_METHOD":
    if data_set == "beds_occupied":
        fig = plt.figure(figsize=(10, 5))
        plt.plot(val_tr["Available_beds"], label="train")
        plt.plot(val_test["Available_beds"], label="test")
        plt.plot(val_test["hw_forecasted"], label="forecasted")
        plt.legend(loc="lower left")
        plt.title("Holt Method")
        plt.show()
        st.pyplot(fig)
        st.write(f"HOLT WINTER MODEL RMSE IS : ", hw_test_rmse)
        # st.error("Greater than other models")
        if st.button("predict"):
            st.write("The Result's are *Here*")
            final_holtwinter

safe_html = """  
      <div style="background-color:#2AFE00;padding:3px >
       <h5 style="color:black;text-align:center;"> Great job!! </h5>
       </div>
    """



st.balloons()

st.caption(" Thank you !!")
