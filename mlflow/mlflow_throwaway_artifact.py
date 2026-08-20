import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("smoke-test")

with mlflow.start_run() as run:
    with open("hello.txt", "w") as f:
        f.write("mlflow + s3 works")
    mlflow.log_artifact("hello.txt")
    print("run_id:", run.info.run_id)