import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

# Set up the MLflow server URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set up the experiment name
experiment_name = "iris_random_forest_experiment"
mlflow.set_experiment(experiment_name)

# Start a new run
with mlflow.start_run(run_name="my_model") as run:
    # Load the data and split into train and test sets
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

    # Train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Log metrics and other artifacts
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Infer the model signature
    y_pred = model.predict(X_test)
    signature = infer_signature(X_test, y_pred)

    # Log the model with signature
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="sklearn-model",
        signature=signature,
        registered_model_name="sk-learn-random-forest-reg-model-iris",
    )


print(f"Run ID: {run.info.run_id}")

