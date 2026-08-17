# 🚀 SurvAutoMLDeploy

An asynchronous, scalable REST API and model-serving framework built with
**FastAPI**, **Redis Queue (RQ)**, and **MLflow**. Designed specifically for
serving models from our pipeline
[SurvAutoML](https://github.com/IDEN-Project-UAS-Darmstadt/SurvAutoML/)
trained on medical registry datasets, it handles long-running survival table
predictions, risk scoring, and time-dependent SHAP explainability calculations
without blocking.

## 🌟 Key Features

* **⚡ Asynchronous Task Execution:** Heavy inference and time-dependent SHAP
    calculations are offloaded to background workers via **Redis & RQ**,
    preventing HTTP timeouts and providing live job status tracking with
    estimated completion times.
* **📦 Dynamic MLflow Model Loading:** Automatically inspects
    models and documentation at runtime to extract required feature schemas,
    validation limits, categorical domains, and model prediction time grids.
* **🧩 Dynamic Pydantic Response Schemas:** Dynamically builds Pydantic output
    models and documentation at runtime to mirror the exact target time grid
    ($t_0, t_1, \dots, t_k$) of the loaded survival model.
* **🔍 Explainable AI (XAI):** Built-in support for generating multi-horizon
    time-dependent SHAP values and baseline survival functions
    (`/explain_survival_table`).
* **🔔 Asynchronous Webhooks & Callbacks:** Supports `callback_url` parameters
    for POST notifications on job completion, failure, or cancellation.
* **🛠️ Clinical Frontend Integration:** Exposes metadata endpoints
    (`/allowed_inputs`, `/data_dict`, `/time_grid`) and serving examples
    (`/input_example`) to streamline UI form generation and data localization.
* **🐳 Containerized Deployment:** Fully dockerized ecosystem
    (`docker-compose`) including the FastAPI application, Redis server,
    background RQ workers, and an RQ monitoring dashboard.

## 🏗️ System Architecture

```txt
                                  +-------------------+
                                  |   Client / UI     |
                                  +---------+---------+
                                            |
                                  HTTP POST | (e.g., /predict_survival_table)
                                            v
                                  +-------------------+
                                  |    FastAPI App    |
                                  | (Job Management & |
                                  |  Dynamic Schemas) |
                                  +---------+---------+
                                            |
                                  Enqueue   | Job
                                            v
                                  +-------------------+
                                  |    Redis Queue    |
                                  +---------+---------+
                                            |
                                  Fetch     | Task
                                            v
                                  +-------------------+
                                  |   RQ Background   | <---> MLflow Model
                                  |     Workers       |       (Model Directory)
                                  +---------+---------+
                                            |
                                  HTTP POST | Callback (Optional)
                                            v
                                  +-------------------+
                                  | Client Webhook /  |
                                  | Callback Endpoint |
                                  +-------------------+
```

## 📑 API Endpoints Summary

### 1. Metadata & Discovery

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/health` | `GET` | Service status check. |
| `/allowed_inputs` | `GET` | List of expected input variables, dtypes, ranges, allowed values, and missingness rules. |
| `/data_dict` | `GET` | Human-readable variable labels, units, grouping, and clinical descriptions. |
| `/time_grid` | `GET` | Array of time horizons (in days) at which the survival model predicts. |
| `/input_example` | `GET` | Sample input record formatted for API testing. |

### 2. Task Submission (Asynchronous)

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/predict_survival_table` | `POST` | Enqueues a job to compute survival probabilities across all time grid points. |
| `/explain_survival_table` | `POST` | Enqueues a job to calculate time-dependent SHAP explanations and baseline curves. |
| `/predict_risk_score` | `POST` | Enqueues a job to calculate a single scalar risk score per observation. |

### 3. Job Management & Results

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/job/{job_id}` | `GET` | Fetches current status, execution timings, and estimated completion time. |
| `/job/{job_id}/cancel` | `POST` | Cancels a queued or executing job. |
| `/job/{job_id}/result` | `GET` | Retrieves the final inference/explanation results once the job is `finished`. |

## 💻 Quickstart & Development

### Setup Instructions

1. **Extract the Model Archive**

   Move the `export_model` output from
   [SurvAutoML](https://github.com/IDEN-Project-UAS-Darmstadt/SurvAutoML)
   to the the `model/` directory.

   For quick testing, use the included `example_model.tgz`:

   ```bash
   mkdir cox
   tar -xzf example_model.tgz -C cox/
   ```

2. **Configure Model Metadata**

   Update the extracted model configuration files:
   * `cox/src/data_dict.csv` — Variable definitions and clinical descriptions
   * `cox/src/serving_input_example.json` — Sample input record for example data

### Run with Docker Compose

Start all services (API, Redis, Worker, Dashboard) in detached mode:

```bash
docker-compose build
docker-compose up -d
```

### Available Local Services

* **REST API & Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
* **RQ Queue Dashboard:** [http://localhost:9181/](http://localhost:9181/)

### Devcontainer

The devcontainer uses the `./model/conda.yaml` from the exported artifact
and the existing docker-compose setup. It also has the API running in a
seperated container, but also starts a seperate dev server at
`8000` within it. You can contact the API container with `http://mapi:8000`
from within the devcontainer and the dev server with `http://localhost:8000`.

The devcontainer might assign a different port than `8000` to the dev server
for the host machine (e.g., `8001`). Check the port mapping to find
the correct port.

The output from the dev server is placed in the `nohup.out` file.

See the `docker-compose.yml` and the `.devcontainer/devcontainer.json`
filesfor more details.

## 🐍 Python Example

```python
import requests
import time

BASE_URL = "http://localhost:8000"

# 1. Fetch sample input data
example_input = requests.get(f"{BASE_URL}/input_example").json()

# 2. Submit an asynchronous prediction job
response = requests.post(
    f"{BASE_URL}/predict_survival_table", 
    json=example_input
).json()

job_id = response["job_id"]
print(f"Submitted Job ID: {job_id}")

# 3. Poll for status until completed
while True:
    status = requests.get(f"{BASE_URL}/job/{job_id}").json()
    if status["status"] == "finished":
        print("Job finished!")
        break
    elif status["status"] in ["failed", "canceled"]:
        raise RuntimeError(f"Job failed with status: {status['status']}")
    
    time.sleep(1)

# 4. Fetch prediction results
results = requests.get(f"{BASE_URL}/job/{job_id}/result").json()
print(results)
```

## 💰 Funding & Acknowledgements

This research project was funded by the Federal Ministry of Research, Technology
and Space (*Bundesministerium für Forschung, Technologie und Raumfahrt*, BMFTR)
under project grant **13FH019KX1** and the German federal state of Hesse.

Further details regarding the project can be found on the
[FORSCHUNG.HAW project page](https://www.forschung-haw.de/fachhochschulen/shareddocs/projekte/de/fh-kooperativ/iden.html).
