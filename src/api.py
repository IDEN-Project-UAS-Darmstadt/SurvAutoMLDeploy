import copy
import json
import os
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime
from enum import Enum
from statistics import median
from typing import Annotated, Dict, List, Optional, Union

import mlflow
import pandas as pd
import requests
from fastapi import APIRouter, Depends, FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, HttpUrl, create_model
from redis import ConnectionPool, Redis
from rq import Queue
from rq.exceptions import NoSuchJobError
from rq.job import Job as rqjob
from rq.registry import FinishedJobRegistry

#
# Models
#


class JobStatus(str, Enum):
    queued = "queued"
    started = "started"
    deferred = "deferred"
    finished = "finished"
    stopped = "stopped"
    scheduled = "scheduled"
    canceled = "canceled"
    failed = "failed"


class JobType(str, Enum):
    predict_survival_table = "predict_survival_table"
    explain_survival_table = "explain_survival_table"
    predict_risk_score = "predict_risk_score"
    unknown = "unknown"


class ModelInputFeature(BaseModel):
    variable: str = Field(description="Name of the model input")
    dtype: str = Field(description="Data type of the model input")
    allowmissing: bool = Field(description="Whether the model input can be missing (NA)")
    min: Optional[Union[int, float]] = Field(
        description="Minimum value for the model input, if applicable", default=None
    )
    max: Optional[Union[int, float]] = Field(
        description="Maximum value for the model input, if applicable", default=None
    )
    vals: Optional[List[str]] = Field(
        description="Allowed values for the model input, if applicable", default=None
    )
    multiple: Optional[bool] = Field(
        description="Whether this categorical model input can contain multiple values, "
        + "separated by ;;",
        default=None,
    )


class DataDictEntry(BaseModel):
    variable: str = Field(description="Name of the model input")
    human_name: str = Field(description="Human-readable name of the model input")
    unit: Optional[str] = Field(description="Unit of the model input, if applicable", default=None)
    group: Optional[str] = Field(
        description="Group of the model input, if applicable", default=None
    )
    description: Optional[str] = Field(description="Description of the model input", default=None)


class PredictSurvivalTableResultBase(BaseModel):
    observation: int = Field(description="ID of the observation, position in the input list")
    # to add: fields t0, t1, ... depending on the time grid,
    # here those are the survival probabilities at each time point


class SHAPBase(PredictSurvivalTableResultBase):
    variable: str = Field(description="Name of the model input")
    aggregated_change: float = Field(
        description="Aggregated change in survival probability for the observation"
    )

    # It will have additional fields t0, t1, ... depending on the time grid
    # passed, here those are the SHAP values at each time point
    class Config:
        extra = "allow"  # Allow additional fields dynamically


class PredictRiskScoreResult(PredictSurvivalTableResultBase):
    risk_score: float = Field(description="Risk score for the observation, a single value")


class Job(BaseModel):
    job_id: str = Field(description="The ID of the job in the queue")
    status: JobStatus = Field(description="The current status of the job")
    started_at: Optional[datetime] = Field(
        description="The time when the job was started", default=None
    )
    enqueued_at: datetime = Field(description="When was the job submitted?")
    finished_at: Optional[datetime] = Field(description="When was the job finished?", default=None)
    estimated_time: Optional[float] = Field(
        description="Estimated run time to finish the job in seconds", default=None
    )
    job_type: JobType = Field(description="The type of job that was submitted")


# Default estimated run times in seconds
RecordList = List[Dict[str, Union[str, int, float, bool, None]]]

#
# Constants
#

DEFAULT_RUN_TIMES = {
    JobType.predict_survival_table: 5,
    JobType.predict_risk_score: 5,
    JobType.explain_survival_table: 120,
    JobType.unknown: 5,
}
RESULT_TTL = 60 * 60 * 24

#
# Setup Metadata
#

model_metadata = {}


def load_model_metadata():
    MODELDIR = os.getenv("MODEL_DIR")
    model = mlflow.pyfunc.load_model(model_uri=f"file://{MODELDIR}")
    unwrapped = model.unwrap_python_model()
    global model_metadata
    # List of the columns that the model expects as input
    # Each is a dict with keys col, dtype, allowmissing, min, max, vals, multiple
    # NOTE: col gets renamed to variable when returned via the API
    inputs = list(unwrapped.inputs)
    model_metadata["original_input_order"] = {inp["col"]: i for i, inp in enumerate(inputs)}
    # Make required inputs come first
    inputs.sort(key=lambda x: x["allowmissing"])
    model_metadata["inputs"] = inputs
    # a list with the time values at which the model predicts
    model_metadata["time_grid"] = unwrapped.timegrid

    # Load the data dictionary for localization
    # csv cols: variable,human_name,unit,group,description
    script_file = os.path.realpath(__file__)
    par_dir = os.path.dirname(script_file)
    data_dict_path = os.path.join(par_dir, "data_dict.csv")
    data_dict = pd.read_csv(data_dict_path)
    model_metadata["data_dict"] = data_dict.to_dict(orient="records")

    # Load the input example
    input_example_path = os.path.join(par_dir, "serving_input_example.json")
    # fall back to the model input example, if the src folder has none
    if not os.path.exists(input_example_path):
        input_example_path = os.path.join(MODELDIR, "serving_input_example.json")
    with open(input_example_path) as indata:
        data = json.load(indata)["dataframe_split"]
    data = pd.DataFrame(data["data"], columns=data["columns"])

    # Bool columns need to be object (if they allow missing values)
    sel = data.select_dtypes("bool")
    data[sel.columns] = sel.astype("object")
    model_metadata["input_example"] = data

    # assert that the data_dict contains all the inputs
    input_vars = set([i["col"] for i in model_metadata["inputs"]])
    data_dict_vars = set([i["variable"] for i in model_metadata["data_dict"]])
    if not input_vars.issubset(data_dict_vars):
        missing_vars = input_vars - data_dict_vars
        raise ValueError(f"Data dictionary is missing variables: {missing_vars}")

    # Assert that the input example contains all the input
    # (if they are set to missing, check that allowmissing is True)
    example_vars = set(data.columns)
    for inp in model_metadata["inputs"]:
        var = inp["col"]
        if var not in example_vars:
            raise ValueError(f"Input example is missing variable: {var}")
        val = data[var]
        if val.isnull().all() and not inp.get("allowmissing", False):
            raise ValueError(f"Input example variable {var} is missing but allowmissing is False")


load_model_metadata()

#
# Dynamic Models
#

PredictSurvivalTableResultDyn = create_model(
    "PredictSurvivalTableResult",
    __base__=PredictSurvivalTableResultBase,
    **{
        f"t{i}": (float, Field(description=f"Survival probability at time t{i}"))
        for i in range(len(model_metadata["time_grid"]))
    },
)


class PredictSurvivalTableResult(PredictSurvivalTableResultDyn):
    """
    Model output for predicting survival probabilities at multiple time points.
    """


class SHAP(SHAPBase):
    """
    Model output for SHAP values, showing the contribution of each feature to the survival
    probabilities at multiple time points.

    The number of time point fields (t0, t1, t2, ...) is dynamic and depends on the timegrid used.
    """


class ExplainSurvivalTableResult(BaseModel):
    shaps_dfs: List[SHAP] = Field(
        description="List of SHAP values for each observation and variable"
    )
    baseline: List[float] = Field(description="Baseline survival probabilities at each time point")
    timegrid: List[float] = Field(
        description="Time points corresponding to the survival probabilities explained"
    )


JobResult = Union[
    List[PredictSurvivalTableResult], ExplainSurvivalTableResult, List[PredictRiskScoreResult]
]

#
# FastAPI app setup
#

redis_pool: ConnectionPool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_pool
    # Create Redis connection pool
    redis_pool = ConnectionPool.from_url(os.getenv("REDIS_CONNECTION"))
    yield
    # Cleanup
    if redis_pool:
        redis_pool.disconnect()


app = FastAPI(
    title="IDEN TxReg Model API",
    description="API for the models produced by the IDEN project.",
    version="0.1",
    lifespan=lifespan,
    openapi_tags=[
        {
            "name": "Tasks",
            "description": "Endpoints for submitting prediction and explanation jobs.",
        },
        {"name": "Jobs", "description": "Endpoints for job management and results."},
        {"name": "Metadata", "description": "Endpoints for model and API metadata."},
        {"name": "Data", "description": "Endpoints for example data."},
    ],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_redis_connection() -> Redis:
    r = Redis(connection_pool=redis_pool)
    return r


def get_queue() -> Queue:
    q = Queue(connection=get_redis_connection())
    return q


QueueDep = Annotated[Queue, Depends(get_queue)]
RedisDep = Annotated[Redis, Depends(get_redis_connection)]


@app.get("/health", response_class=JSONResponse, tags=["Metadata"])
async def health_check():
    """
    Health Check Endpoint

    Returns a simple status message to indicate that the service is running.
    """
    return JSONResponse(content={"status": "ok"})


@app.get("/myip", response_class=JSONResponse, tags=["Metadata"])
def get_my_ip(request: Request):
    """
    Get My IP

    Returns the IP address of the client.
    """
    client_host = request.client.host
    return JSONResponse(content={"my_ip": client_host})


@app.get("/allowed_inputs", response_model=List[ModelInputFeature], tags=["Metadata"])
async def get_allowed_inputs():
    """
    Get Allowed Inputs

    Returns a list of columns that the model expects as input.
    """
    # TODO this is a workaround, need to change this in the model
    res = copy.deepcopy(model_metadata["inputs"])
    for row in res:
        if "col" in row:
            row["variable"] = row.pop("col")
    return res


@app.get("/data_dict", response_model=List[DataDictEntry], tags=["Metadata"])
async def get_data_dict():
    """
    Get Data Dictionary

    Returns a data dictionary with human-readable names,
    units, groups, and descriptions for the model inputs.
    """
    res = model_metadata["data_dict"]
    for row in res:
        for k, v in row.items():
            if pd.isnull(v):
                row[k] = None
    return res


@app.get("/time_grid", response_model=List[float], tags=["Metadata"])
async def get_time_grid():
    """
    Get Time Grid

    Returns a list of time values at which the model predicts.
    """
    return model_metadata["time_grid"]


@app.get("/input_example", tags=["Data"])
async def get_input_example() -> RecordList:
    """
    Get Input Example

    Returns an example input that can be used to test the model.
    """
    res = model_metadata["input_example"].to_dict(orient="records")
    return res


def estimate_run_time(q: Queue, job_type: JobType, worker: Optional[str], n: int) -> float:
    jobreg = FinishedJobRegistry(queue=q)
    vals = []
    jobs = rqjob.fetch_many(jobreg.get_job_ids(), connection=q.connection)
    for job in jobs:
        if job.meta.get("job_type") == job_type:
            if worker is not None and job.worker_name != worker:
                continue
            if job.ended_at and job.started_at:
                job_n = float(job.meta.get("job_n", 1))
                vals.append((job.ended_at - job.started_at).total_seconds() / job_n)
    if vals:
        return median(vals) * n
    else:
        return DEFAULT_RUN_TIMES[job_type] * n


def rqjob_to_job(q: Queue, job: Job) -> Job:
    """
    Convert an RQ job to a Job model.
    """
    job_info = {
        "job_id": job.id,
        "status": job.get_status(),
        "started_at": job.started_at,
        "enqueued_at": job.enqueued_at,
        "finished_at": job.ended_at,
        "job_type": job.meta.get("job_type", "unknown"),
        "estimated_time": estimate_run_time(
            q,
            job.meta.get("job_type", "unknown"),
            job.worker_name if hasattr(job, "worker_name") else None,
            job.meta.get("job_n", 1),
        ),
    }
    return Job(**job_info)


# @app.get("/job/", response_model=List[Job], tags=["Jobs"])
# async def get_jobs(queue: QueueDep, offset: int = 0, length: int = -1):
#    """
#    Get Jobs
#
#    Returns a list of recent jobs in the queue.
#    """
#    jobs = []
#    for job in queue.get_jobs(offset=offset, length=length):
#        job_info = rqjob_to_job(queue, job)
#        jobs.append(job_info)
#    return jobs


@app.get("/job/{job_id}", response_model=Job, tags=["Jobs"])
async def get_job_status(job_id: str, redis: RedisDep, queue: QueueDep):
    """
    Get Job Status

    Returns the status of a job in the queue.
    """
    # This throws an exception if the job is not found, not returning None
    try:
        job = rqjob.fetch(job_id, connection=redis)
    except NoSuchJobError:
        return JSONResponse(status_code=404, content={"message": "Job not found"})

    job_info = rqjob_to_job(queue, job)
    return job_info


@app.post("/job/{job_id}/cancel", tags=["Jobs"])
async def cancel_job(job_id: str, redis: RedisDep):
    """
    Cancel Job

    Cancels a job in the queue.
    """
    job = rqjob.fetch(job_id, connection=redis)
    if not job:
        return JSONResponse(status_code=404, content={"message": "Job not found"})

    status = job.get_status()
    if status in {"finished", "stopped", "canceled", "failed"}:
        return JSONResponse(
            status_code=400, content={"message": "Job is already finished or failed"}
        )
    job.cancel()
    return JSONResponse(status_code=200, content={"message": "Job cancelled successfully"})


callback_router = APIRouter()


@callback_router.post("{$callback_url}/job/{$request.body.job_id}")
def job_notification(body: Job):
    """Processes a job notification callback.

    This callback is triggered on a job's completion, failure, or stop event.

    Args:
        body (Job): The job notification body containing job details.
    """
    pass


def do_callback(job, connection):
    callback_url = str(job.meta.get("callback_url"))

    if callback_url:
        # Safely join the callback_url and "/job/{job_id}"
        parsed = urllib.parse.urlparse(callback_url)
        base = parsed.geturl().rstrip("/")
        callback_url = f"{base}/job/{job.id}"
    if not callback_url:
        logger.warning(f"Job {job.id} has no callback URL set, skipping callback.")
        return
    try:
        queue_name = job.origin or "default"
        q = Queue(connection=connection, name=queue_name)
        body = rqjob_to_job(q, job)
        requests.post(callback_url, json=jsonable_encoder(body), timeout=5)
    except requests.RequestException as e:
        logger.error(f"Failed to send callback for job {job.id} to {callback_url}: {e}")


def report_success(job, connection, result, *args, **kwargs):
    do_callback(job, connection)


def report_failure(job, connection, type, value, traceback):
    do_callback(job, connection)


def report_stopped(job, connection):
    do_callback(job, connection)


def fix_input(input: RecordList) -> RecordList:
    allowed_inputs = model_metadata["inputs"]
    fields = {inp["col"]: inp for inp in allowed_inputs}
    field_ix = model_metadata["original_input_order"]

    new_input = []

    for record in input:
        # Complete this record with missing fields
        for field_name, field_info in fields.items():
            if field_name not in record:
                if field_info.get("allowmissing", False):
                    # print(f"Input missing field {field_name}, setting to None")
                    record[field_name] = None

        cur_pos = [(field_ix.get(col, 1e6), col) for col in record.keys()]
        cur_pos.sort(key=lambda x: x[0])
        new_record = {}
        for _, col in cur_pos:
            new_record[col] = record[col]
        new_input.append(new_record)
    return new_input


@app.post(
    "/predict_survival_table", response_model=Job, tags=["Tasks"], callbacks=callback_router.routes
)
async def predict_survival_table(
    input: RecordList, q: QueueDep, callback_url: Union[HttpUrl, None] = None
) -> Job:
    """
    Predict Survival Table

    Submits a job to predict the survival table for a given input.
    """
    callback_args = {}

    if callback_url:
        callback_args = {
            "on_success": report_success,
            "on_failure": report_failure,
            "on_stopped": report_stopped,
        }

    input = fix_input(input)

    job = q.enqueue(
        "worker.predict_survival_table",
        input,
        job_timeout=f"{60 * len(input)}s",
        result_ttl=RESULT_TTL,
        meta={
            "job_type": JobType.predict_survival_table,
            "job_n": len(input),
            "callback_url": callback_url,
        },
        **callback_args,
    )
    return rqjob_to_job(q, job)


@app.post(
    "/explain_survival_table", response_model=Job, tags=["Tasks"], callbacks=callback_router.routes
)
async def explain_survival_table(
    input: RecordList,
    q: QueueDep,
    timegrid: Annotated[
        Union[List[float], None],
        Query(
            description="Time points at which to explain survival"
            + "probabilities. If not provided, uses"
            + "threeequaldistant time points."
        ),
    ] = None,
    callback_url: Union[HttpUrl, None] = None,
) -> Job:
    """
    Explain Survival Table

    Submits a job to explain the survival table for a given input.
    """
    callback_args = {}

    if callback_url:
        callback_args = {
            "on_success": report_success,
            "on_failure": report_failure,
            "on_stopped": report_stopped,
        }

    input = fix_input(input)

    job = q.enqueue(
        "worker.explain_survival_table",
        input,
        or_names=True,
        timegrid=timegrid,
        job_timeout=f"{60 * 5 * len(input)}s",
        result_ttl=RESULT_TTL,
        meta={
            "job_type": JobType.explain_survival_table,
            "job_n": len(input),
            "callback_url": callback_url,
        },
        **callback_args,
    )
    return rqjob_to_job(q, job)


@app.post(
    "/predict_risk_score", response_model=Job, tags=["Tasks"], callbacks=callback_router.routes
)
async def predict_risk_score(
    input: RecordList, q: QueueDep, callback_url: Union[HttpUrl, None] = None
) -> Job:
    """
    Predict Risk Score

    Submits a job to predict the risk score for a given input.
    """

    callback_args = {}

    if callback_url:
        callback_args = {
            "on_success": report_success,
            "on_failure": report_failure,
            "on_stopped": report_stopped,
        }

    input = fix_input(input)

    input_data = pd.DataFrame.from_records(input)
    job = q.enqueue(
        "worker.predict_risk_score",
        input_data,
        job_timeout=f"{60 * len(input)}s",
        result_ttl=RESULT_TTL,
        meta={
            "job_type": JobType.predict_risk_score,
            "job_n": len(input),
            "callback_url": callback_url,
        },
        **callback_args,
    )
    return rqjob_to_job(q, job)


@app.get("/job/{job_id}/result", tags=["Jobs"], response_model=JobResult)
async def get_job_result(job_id: str, redis: RedisDep):
    """
    Get Job Result

    Returns the result of a job in the queue. Results expire after a certain time!
    The format of the result depends on the job type:

    - For `predict_survival_table`, returns a list of survival probabilities at
      multiple time points.
    - For `explain_survival_table`, returns SHAP values, baseline survival
      probabilities and the time points used for explanation.
    - For `predict_risk_score`, returns a single risk score value.
    """
    job = rqjob.fetch(job_id, connection=redis)
    if not job:
        return JSONResponse(status_code=404, content={"message": "Job not found"})

    if job.is_failed:
        return JSONResponse(status_code=500, content={"message": "Job failed"})

    if not job.is_finished:
        return JSONResponse(status_code=400, content={"message": "Job is not finished yet"})

    results = job.latest_result()
    # type - an enum of SUCCESSFUL, FAILED, RETRIED or STOPPED

    result_type = results.type

    if result_type == results.Type.STOPPED:
        return JSONResponse(status_code=500, content={"message": "Job stopped"})
    elif result_type == results.Type.FAILED:
        exc = results.exc_string
        return JSONResponse(status_code=500, content={"message": "Job failed: " + exc})

    results = results.return_value

    if not results:
        return JSONResponse(status_code=204, content={"message": "No results available"})

    if job.meta.get("job_type") == JobType.explain_survival_table:
        # If the job is an explanation job, return the results as a dictionary
        shaps_dfs, baseline, timegrid = results
        return {
            "shaps_dfs": shaps_dfs,
            "baseline": baseline,
            "timegrid": timegrid,
        }
    elif job.meta.get("job_type") == JobType.predict_survival_table:
        return results
    elif job.meta.get("job_type") == JobType.predict_risk_score:
        return results
    else:
        return JSONResponse(status_code=400, content={"message": "Unknown job type"})
