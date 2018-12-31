from src.time_series.ts_utils import *

def main(input_file):
    ts = read_ts(input_file)
    train_ts, test_ts = get_train_test_split(ts, ratio=2/3)
    stepwise_model = train_ARIMA_model(train_ts)
    test_forecast = test_ARIMA_model(test_ts, stepwise_model, plot=False)
    append_forecast(ts, test_forecast, plot=True)

if __name__ == "__main__":
    input_file = 'time_series/files/energy.csv'
    main(input_file)
    