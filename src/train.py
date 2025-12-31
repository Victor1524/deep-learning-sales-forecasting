from src.data_preprocessing import load_and_preprocess_data
from src.model import build_lstm_model


def main():
    window_size = 30
    forecast_horizon = 7

    X, y, scaler = load_and_preprocess_data(
        file_path="data/sales.csv",
        window_size=window_size,
        forecast_horizon=forecast_horizon
    )

    model = build_lstm_model(
        window_size=window_size,
        forecast_horizon=forecast_horizon
    )

    model.fit(
        X,
        y,
        epochs=30,
        batch_size=16,
        validation_split=0.2
    )

    model.save("sales_forecast_model.h5")
    print("Modelo treinado e salvo com sucesso!")


if __name__ == "__main__":
    main()
