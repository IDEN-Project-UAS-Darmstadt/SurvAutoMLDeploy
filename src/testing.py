import mlflow.pyfunc

MODELDIR = "/workspace/model"
model = mlflow.pyfunc.load_model(model_uri=f"file://{MODELDIR}")
unwrapped = model.unwrap_python_model()
print(unwrapped.timegrid)
