from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def build_lstm_model(window_size, forecast_horizon):
    """
    Cria e compila um modelo LSTM para previsão de séries temporais.

    Args:
        window_size (int): Tamanho da janela temporal
        forecast_horizon (int): Quantidade de passos futuros

    Returns:
        model (keras.Model): Modelo compilado
    """

    model = Sequential()
    model.add(
        LSTM(
            64,
            activation="relu",
            input_shape=(window_size, 1)
        )
    )
    model.add(Dense(forecast_horizon))

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    return model
