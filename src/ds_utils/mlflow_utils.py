def save_experiment(experiment_name, **params):
    import mlflow
    from datetime import datetime
    with mlflow.start_run(run_name=experiment_name + '_' + str(datetime.now())):
        for k, v in params.items():
            mlflow.log_param(k, v)
