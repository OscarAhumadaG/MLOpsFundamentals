from mlflow import log_metric, log_param, log_artifact


if __name__ == "__main__":
 log_param("threshold", 3)
 log_param("verbosity", "DEBUG")

 log_metric("timestamp", 10000)
 log_metric("TTC", 33)

 log_artifact("data/produced-dataset.csv")
