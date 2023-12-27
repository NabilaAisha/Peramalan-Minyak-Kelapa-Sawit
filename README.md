# Palm Oil Forecasting
## Purpose of Analysis
This analysis aims to predict palm oil production and consumption using the ARIMA method

## Research Urgency
Palm oil production has been decreasing for the last 3 years (2020-2022), while palm oil consumption continues to increase every year. Theory says that production and consumption have a positive correlation, which indicates that if production increases, consumption will also increase. This is what underlies forecasting analysis for palm oil production and consumption

## Analysis
### Package
```
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
!pip install pandas_profiling
from pandas_profiling import ProfileReport
from plotly.offline import iplot
!pip install joypy
import joypy
import os
import glob
from sklearn.cluster import KMeans

plt.rcParams['figure.figsize'] = 8, 5
plt.style.use("fivethirtyeight")
```
This code is to input package for analysis used

### Import Data
```
from google.colab import files
uploaded = files.upload()
import io
data = pd.read_csv(io.BytesIO(uploaded['Data Prod & Konsum.csv']))
```
This code is to import original data as "data" in Google Colab

### Data Visualization By Chart and Descriptive Statistics
```
#### Chart
data.groupby('Tahun')['Produksi'].sum().plot.bar()
data.groupby('Tahun')['Produksi'].sum().plot.line()
data.groupby('Tahun')['Konsumsi'].sum().plot.bar()
data.groupby('Tahun')['Konsumsi'].sum().plot.line()
```
This code is to visualize "produksi" and "konsumsi" column with bar chart and line chart

```
#### Descriptive Statistics
data.groupby('Tahun')['Konsumsi'].sum().plot.line()
```
This code will produce descriptive statistics from original data, such as count, mean, std, min, 25% quartil, median, 75% quartil, and max

### Stationarity Test
#### Stationarity Test For Palm Production
H0 : Palm production is not stationary
H1 : Palm production is stationary
alfa : 0,05
Rejection area : Reject H0 if p-value < alfa
Statistics Test :
```
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Produksi'])

print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(key, value)
```
It will produce adf-statistics, p-value, critical value to conclude the result

```
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data['Produksi'], lags=len(data['Produksi'])-1)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('ACF Plot')
plt.show()
```
This code is to visualize the ACF plot

```
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(data['Produksi'], lags=3)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('PACF Plot')
plt.show()
plot_pacf(data['Produksi'], lags=3, method='ywm')
```
This code is to visualize PACF plot

#### Stationarity Test For Palm Consumption
H0 : Palm consumption is not stationary
H1 : Palm consumption is stationary
alfa : 0,05
Rejection area : Reject H0 if p-value < alfa
Statistics Test :
```
from statsmodels.tsa.stattools import adfuller

result = adfuller(data['Konsumsi'])

print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:')
for key, value in result[4].items():
    print(key, value)
```
It will produce adf-statistics, p-value, critical value to conclude the result

```
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(data['Konsumsi'], lags=len(data['Konsumsi'])-1)
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('ACF Plot')
plt.show()
```
This code is to visualize the ACF plot

```
from statsmodels.graphics.tsaplots import plot_pacf

plot_pacf(data['Konsumsi'], lags=3)
plt.xlabel('Lag')
plt.ylabel('Partial Autocorrelation')
plt.title('PACF Plot')
plt.show()
plot_pacf(data['Konsumsi'], lags=3, method='ywm')
```
This code is to visualize PACF plot
