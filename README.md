# Ex.No: 07                                       AUTO REGRESSIVE MODEL




### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM

## Import necessary libraries :

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

## Read the CSV file into a DataFrame :

data = pd.read_csv('/content/drive/MyDrive/AirPassengers.csv',parse_dates=['Month'],index_col='Month')

## Perform Augmented Dickey-Fuller test :

result = adfuller(data['#Passengers']) 
print('ADF Statistic:', result[0])
print('p-value:', result[1])

## Split the data into training and testing sets : 

x=int(0.8 * len(data))
train_data = data.iloc[:x]
test_data = data.iloc[x:]

## Fit an AutoRegressive (AR) model with 13 lags :

lag_order = 13
model = AutoReg(train_data['#Passengers'], lags=lag_order)
model_fit = model.fit()

## Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :

plt.figure(figsize=(10, 6))
plot_acf(data['#Passengers'], lags=40, alpha=0.05)
plt.title('Autocorrelation Function (ACF)')
plt.show()
plt.figure(figsize=(10, 6))
plot_pacf(data['#Passengers'], lags=40, alpha=0.05)
plt.title('Partial Autocorrelation Function (PACF)')
plt.show()

## Make predictions using the AR model :

predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data))

## Compare the predictions with the test data :

mse = mean_squared_error(test_data['#Passengers'], predictions)
print('Mean Squared Error (MSE):', mse)

## Plot the test data and predictions :

plt.figure(figsize=(12, 6))
plt.plot(test_data['#Passengers'], label='Test Data - Number of passengers')
plt.plot(predictions, label='Predictions - Number of passengers',linestyle='--')
plt.xlabel('Date')
plt.ylabel('Number of passengers')
plt.title('AR Model Predictions vs Test Data')
plt.legend()
plt.grid()
plt.show()

### OUTPUT:

Dataset:

![image (1)](https://github.com/user-attachments/assets/88b66af5-1ccd-4ba1-8052-95902f0a97b7)

ADF test result:

![image](https://github.com/user-attachments/assets/8bce4a0f-5957-4cb0-bf78-92bffd2bcbcc)

PACF plot:

![image](https://github.com/user-attachments/assets/c844f8c3-0462-4fed-8844-02678dd9f811)

ACF plot:

![image](https://github.com/user-attachments/assets/18ed7868-63e6-4c78-84cd-d4163861e69e)

Accuracy:

![image](https://github.com/user-attachments/assets/f0fd8539-8e34-466a-833a-09c7edca5321)

Prediction vs test data:

![image](https://github.com/user-attachments/assets/d3e1a79d-b97c-4ee4-905e-3f2e61ba9f3f)


### RESULT:
Thus we have successfully implemented the auto regression function using python.
