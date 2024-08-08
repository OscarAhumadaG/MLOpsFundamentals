import mlflow
from mlflow import log_metric
from random import choice

metric_names = ["cpu", "ram", "disk"]

percentages = [i for i in range(0, 100)]

experiment_id = "produce-metrics"
mlflow.set_experiment(experiment_id)

for i in range(40):
    log_metric(choice(metric_names), choice(percentages))