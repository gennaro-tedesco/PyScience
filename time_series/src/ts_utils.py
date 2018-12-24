import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARMAResults
from pyramid.arima import auto_arima

plt.style.use('seaborn-dark')
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["axes.linewidth"] = 1

# ----------------------------- #
# define the object time series #
# ----------------------------- #
def get_ts():
    rng = pd.date_range('1/1/2018', periods=31, freq='H')
    ts = pd.Series(np.random.randn(len(rng)), index=rng)
    return ts


def read_ts(file_name):
    data = pd.read_csv(file_name)
    times_values = pd.to_datetime(data[data.columns[0]])
    num_values = data[data.columns[1]]
    ts = pd.Series(num_values.values, index=times_values)
    return ts


# ------------------- #
# check if stationary #
# ------------------- #

def plot_rolling_avg(ts):
    assert isinstance(ts, pd.Series)
    rol_mean = ts.rolling(window=14, center=False).mean()
    rol_std = ts.rolling(window=14, center=False).std()
    orig = plt.plot(ts, label='Original')
    mean = plt.plot(rol_mean, label='Rolling Mean')
    std = plt.plot(rol_std, label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()


def is_Dickey_Fuller_rejected(ts):
    assert isinstance(ts, pd.Series)
    test_param = adfuller(ts, autolag='AIC')
    return test_param[1] < 0.05


# --------------------------------- #
# make stationary by removing trend #
# --------------------------------- #

def remove_trend(ts):
    assert isinstance(ts, pd.Series)
    rol_mean = ts.rolling(window=14, center=False).mean()
    ts_diff = ts - rol_mean
    ts_diff.dropna(inplace=True)
    return ts_diff


# ------------------------------ #
# make stationary by differences #
# ------------------------------ #

def get_diff(ts):
    ts_diff = ts - ts.shift()
    ts_diff.dropna(inplace=True)
    return ts_diff


# -------------------------------- #
# make stationary by decomposition #
# -------------------------------- #

def get_decomposition(ts, plot=True):
    decomposition = seasonal_decompose(ts)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    if plot:
        plt.subplot(411)
        plt.plot(ts, label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(residual, label='Residuals')
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    dec_dict = {'trend': trend,
                'seasonal': seasonal,
                'residual': residual}
    return dec_dict


def get_train_test_split(ts, ratio=2/3):
    assert isinstance(ts, pd.Series)
    idx = int(ratio*len(ts.index))
    train = ts[0:idx]
    test = ts[idx+1: len(ts.index)]
    return train, test


def train_ARIMA_model(train_ts):
    assert isinstance(train_ts, pd.Series)
    stepwise_model = auto_arima(train_ts, start_p=1, start_q=1,
                                max_p=3, max_q=3, m=12,
                                start_P=0, seasonal=True,
                                d=1, D=1, trace=True,
                                error_action='ignore',
                                suppress_warnings=True,
                                stepwise=True)
    print(stepwise_model.summary())
    return stepwise_model


def test_ARIMA_model(test_ts, stepwise_model, plot=True):
    assert isinstance(test_ts, pd.Series)
    future_forecast = stepwise_model.predict(n_periods=len(test_ts.index))
    future_forecast = pd.DataFrame(
        future_forecast, index=test_ts.index, columns=['prediction'])
    test_forecast = pd.concat([test_ts, future_forecast], axis=1)
    test_forecast.rename(columns={0: 'original'}, inplace=True)
    if plot:
        test_forecast.plot()
        plt.show()
    return test_forecast


def append_forecast(ts, test_forecast, plot=True):
    assert isinstance(ts, pd.Series)
    assert isinstance(test_forecast, pd.DataFrame)
    complete_prediction_df = pd.concat([ts, test_forecast], axis=1)
    complete_prediction_df.rename(columns={0: 'train_set'}, inplace=True)
    if plot:
        complete_prediction_df.plot()
        plt.show()
    return complete_prediction_df
