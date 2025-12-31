import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(
    file_path,
    window_size=30,
    forecast_horizon=7
):
    """
    Carrega e prepara os dados de vendas para séries temporais.

    Args:
        file_path (str): Caminho do arquivo CSV
        window_size (int): Tamanho da janela temporal
        forecast_horizon (int): Quantidade de dias a prever

    Returns:
        X (np.array): Dados de entrada
        y (np.array): Valores alvo
        scaler (MinMaxScaler): Scaler ajustado
    """

    df = pd.read_csv(file_path, parse_dates=["date"])
    sales = df["sales"].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    sales_scaled = scaler.fit_transform(sales)

    X, y = [], []

    for i in range(len(sales_scaled) - window_size - forecast_horizon):
        X.append(sales_scaled[i:i + window_size])
        y.append(
            sales_scaled[
                i + window_size:i + window_size + forecast_horizon
            ].flatten()
        )

    return np.array(X), np.array(y), scaler
