from traffic.services.forecast.arima import ArimaForecaster
from traffic.services.forecast.lstm import LstmForecaster
from traffic.services.forecast.tcn import TcnForecaster

FORECASTERS = {
    "arima": ArimaForecaster,
    "lstm": LstmForecaster,
    "tcn": TcnForecaster,
}
