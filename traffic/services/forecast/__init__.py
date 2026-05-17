from traffic.services.forecast.ets import EtsForecaster
from traffic.services.forecast.lstm import LstmForecaster
from traffic.services.forecast.tcn import TcnForecaster

FORECASTERS = {
    "ets": EtsForecaster,
    "lstm": LstmForecaster,
    "tcn": TcnForecaster,
}
