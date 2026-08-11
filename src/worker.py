import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from joblib import Memory
from redis import Redis
from rq import Queue, Worker

MODELDIR = os.getenv("MODEL_DIR")
model = mlflow.pyfunc.load_model(model_uri=f"file://{MODELDIR}")
unwrapped = model.unwrap_python_model()
unwrapped.validate_input = True
tmpdir = os.getenv("CACHEDIR", "/tmp")
cache_dir = Path(tmpdir) / "cache"
memory = Memory(str(cache_dir), verbose=10)
sizelimit = os.getenv("CACHELIMIT", "1G")


@memory.cache
def predict_survival_table(input):
    memory.reduce_size(bytes_limit=sizelimit)
    input_data = pd.DataFrame.from_records(input)
    survs = unwrapped.predict_survival_table(None, input_data)
    records = survs.to_dict("records")
    return records


@memory.cache
def explain_survival_table(input, timegrid=None, or_names=True):
    memory.reduce_size(bytes_limit=sizelimit)
    input_data = pd.DataFrame.from_records(input)
    shaps_dfs, prediction, baseline, timegrid_ret = unwrapped.explain_survival_table(
        None, input_data, timegrid=timegrid, or_names=or_names
    )
    timecols = [col for col in shaps_dfs.columns if col.startswith("t")]
    calc_preds = shaps_dfs.groupby("observation")[timecols].sum().values + baseline
    assert np.allclose(prediction, calc_preds), "Mismatch between pred and sums"
    return shaps_dfs.to_dict("records"), list(baseline), list(timegrid_ret)


@memory.cache
def predict_risk_score(input):
    memory.reduce_size(bytes_limit=sizelimit)
    input_data = pd.DataFrame.from_records(input)
    return unwrapped.predict_risk_score(None, input_data).to_dict("records")


if __name__ == "__main__":
    # Read the Redis connection string from the environment variable
    redis_conn_str = os.getenv("REDIS_CONNECTION")
    if not redis_conn_str:
        raise ValueError("REDIS_CONNECTION environment variable is not set")
    # Create a connection to Redis
    conn = Redis.from_url(redis_conn_str)
    # Start the worker
    queue = Queue(connection=conn)
    # Create a worker and specify the queues to listen on
    queues = [queue]
    worker = Worker(queues, connection=conn)
    worker.work()
