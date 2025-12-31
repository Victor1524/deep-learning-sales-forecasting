import numpy as np
from tensorflow.keras.models import load_model
from src.data_preprocessing import load_and_preprocess_data


def main():
    window_size = 30
    forecast_horizon = 7

    model = load_model("sales_forecast_model.h5")

    X, y, scaler = load_and_preprocess_data(
        file_path="data/sales.csv",
        window_size=window_size,
        forecast_horizon=forecast_horizon
    )

    last_window = X[-1].reshape(1, window_size, 1)
    prediction = model.predict(last_window)

    prediction_rescaled = scaler.inverse_transform(
        prediction.reshape(-1, 1)
    )

    print("📈 Previsão de vendas para os próximos 7 dias:")
    for i, value in enumerate(prediction_rescaled, start=1):
        print(f"Dia {i}: {int(value[0])}")


if __name__ == "__main__":
    main()
