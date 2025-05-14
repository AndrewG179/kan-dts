from data_pipeline import data_creation, preprocessing, dataset_order
from logs.logger import log_json, log_note

def main():
    df = data_creation.create_time_series(500, ["trend", "seasonality"], noise=True)
    X, y = preprocessing.create_windows(df)
    X_train, X_test, y_train, y_test = dataset_order.split_dataset(X, y)

    log_json({"x_train_shape": X_train.shape, "y_train_shape": y_train.shape}, "data_shapes", "data_pipeline")
    log_note("Created time series with trend+seasonality+noise and segmented into windows", "creation_summary", "data_pipeline")

if __name__ == "__main__":
    main()