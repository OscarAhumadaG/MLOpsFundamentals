import pandas as pd
from joblib import dump


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, Binarizer, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score

import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature



# Read and clean the data
def load_data(filepath):
    hotel_bookings = pd.read_csv(filepath)

    # Remove personal information of customers
    hotel_bookings = hotel_bookings.drop(['name', 'email', 'phone-number', 'credit_card'], axis=1)

    # Avoid data leakage
    hotel_bookings = hotel_bookings.drop(['reservation_status', 'reservation_status_date'], axis=1)

    # Convert objects to string
    object_columns = hotel_bookings.select_dtypes('object').columns
    hotel_bookings[object_columns] = hotel_bookings[object_columns].astype('str')

    return hotel_bookings



if __name__ == "__main__":
    # Split the dataset
    def split_dataset(bookings_dataset, train_proportion, test_proportion):
        is_canceled = bookings_dataset['is_canceled'].copy()
        hotel_data = bookings_dataset.drop(['is_canceled'], axis=1)

        original_count = len(bookings_dataset)
        training_size = int(original_count * train_proportion)
        test_size = int((1 - train_proportion) * test_proportion * training_size)

        train_x, rest_x, train_y, rest_y = train_test_split(hotel_data, is_canceled, train_size=training_size)
        test_x, validate_x, test_y, validate_y = train_test_split(rest_x, rest_y, train_size=test_size)

        mlflow.log_params({
            'dataset_size': original_count,
            'training_set_size': len(train_x),
            'validate_set_size': len(validate_x),
            'test_set_size': len(test_x)
        })

        return (train_x, train_y), (validate_x, validate_y), (test_x, test_y)

    # Creating a pipeline

    def build_pipeline(n_estimators):
        # One-hot encoder
        internal_one_hot_encoding = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        columns_to_encode = [
            "hotel",
            "meal",
            "distribution_channel",
            "reserved_room_type",
            "assigned_room_type",
            "customer_type"
        ]

        mlflow.log_param('one_hot_encoded_columns', columns_to_encode)
        encoder_params = internal_one_hot_encoding.get_params()
        mlflow.log_params({
            f"encoder__{key}": value for key, value in encoder_params.items()
        })

        one_hot_encoding = ColumnTransformer([
            (
                'one_hot_encode',
                internal_one_hot_encoding,
                columns_to_encode
            )
        ])

        # Binarizer
        internal_binarizer = Binarizer()
        columns_to_binarize = [
            "total_of_special_requests",
            "required_car_parking_spaces",
            "booking_changes",
            "previous_bookings_not_canceled",
            "previous_cancellations",
        ]
        internal_encoder_binarizer = OneHotEncoder(sparse_output=False, handle_unknown="ignore")

        binarizer = ColumnTransformer([
            (
                'binarizer',
                internal_binarizer,
                columns_to_binarize
            )
        ])

        one_hot_binarized = Pipeline([
            ("binarizer", binarizer),
            ("one_hot_encoder", internal_encoder_binarizer),
        ])

        # Scaler
        internal_scaler = RobustScaler()
        columns_to_scale = ["adr"]

        scaler = ColumnTransformer([
            ("scaler", internal_scaler, columns_to_scale)
        ])

        # Passthrough columns
        pass_columns = [
            "stays_in_week_nights",
            "stays_in_weekend_nights",
        ]

        passthrough = ColumnTransformer([
            (
                "pass_columns",
                "passthrough",
                pass_columns
            )
        ])

        # Full pipeline
        feature_engineering_pipeline = Pipeline([
            (
                "features",
                FeatureUnion([
                    ('categories', one_hot_encoding),
                    ('binaries', one_hot_binarized),
                    ('scaled', scaler),
                    ('passthrough', passthrough)
                ])
            )
        ])

        # Machine learning model
        model = RandomForestClassifier(n_estimators=n_estimators)

        model_params = model.get_params()
        mlflow.log_params({
            f"model__{key}": value for key, value in model_params.items()
        })

        # Full pipeline
        final_pipeline = Pipeline([
            ("feature_engineering", feature_engineering_pipeline),
            ("model", model)
        ])

        return final_pipeline

    # Model training and validation

    def model_training_validation(final_pipeline, train_x, train_y, validate_x, validate_y):
        final_pipeline.fit(train_x, train_y)

        train_pred_y = final_pipeline.predict(train_x)
        validate_pred_y = final_pipeline.predict(validate_x)

        train_accuracy = accuracy_score(train_pred_y, train_y)
        train_recall = recall_score(train_pred_y, train_y)

        validate_accuracy = accuracy_score(validate_pred_y, validate_y)
        validate_recall = recall_score(validate_pred_y, validate_y)

        print('Train accuracy', train_accuracy)
        print('Train recall', train_recall)

        print('Validate accuracy', validate_accuracy)
        print('Validate recall', validate_recall)

        metrics = {
            'train_accuracy': train_accuracy,
            'train_recall': train_recall,
            'validate_accuracy': validate_accuracy,
            'validate_recall': validate_recall,
        }

        mlflow.log_metrics(metrics)

        return final_pipeline, metrics



    # Full training run
    def full_training_run(file_path):
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment("/hotel-facilito/BookingCancellations")
        with mlflow.start_run() as run:
            raw_dataset = load_data(file_path)

            training_data, validate_data, test_data = split_dataset(raw_dataset, train_proportion=0.6, test_proportion=0.5)

            training_pipeline = build_pipeline(n_estimators=100)

            training_pipeline, metrics = model_training_validation(
                training_pipeline,
                train_x=training_data[0],
                train_y=training_data[1],
                validate_x=validate_data[0],
                validate_y=validate_data[1]
            )

            dump(training_pipeline, "data/inference_pipeline.joblib")

            mlflow.log_artifact('data/inference_pipeline.joblib')

            # Use .head(100) machine doesn't have too much resources
            signature = infer_signature(training_data[0].head(100), training_pipeline.predict(training_data[0]))


            # Log and register the model
            model_info = mlflow.sklearn.log_model(
                sk_model=training_pipeline,
                artifact_path="cancellations-model",
                signature=signature,
                registered_model_name="CancellationsModel"
            )

        return training_pipeline

    # Run the training
    data_path = "data/hotel_bookings_training.csv"
    print(full_training_run(data_path))