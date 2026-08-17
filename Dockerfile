ARG MICROMAMBA_VERSION=2.1.0-cuda12.6.3-ubuntu22.04
FROM mambaorg/micromamba:${MICROMAMBA_VERSION} AS base

USER root
RUN apt-get update && apt-get install -y rsync sudo && rm -rf /var/lib/apt/lists/*

# Install packages with caching
USER $MAMBA_USER


ENV MODEL_NAME=cox
ENV MODEL_DIR=/home/$MAMBA_USER/model

COPY --chown=$MAMBA_USER:$MAMBA_USER ./${MODEL_NAME}/conda.yaml /tmp/conda.yaml
RUN --mount=type=cache,target=/opt/pkg_cache,sharing=locked,uid=57439,gid=57439 \
    rsync -a /opt/pkg_cache/ /opt/conda/pkgs/ && \
    micromamba config set always_yes True && \
    micromamba install -y -n base \
        -c conda-forge/label/broken \
        -c conda-forge \
        -f /tmp/conda.yaml && \
    rsync -a /opt/conda/pkgs/ /opt/pkg_cache/ && \
    micromamba clean --all --yes
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /home/$MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER ./${MODEL_NAME} ${MODEL_DIR}
RUN echo 'export PYTHONPATH="${MODEL_DIR}/code:$PYTHONPATH"' >> /home/$MAMBA_USER/.bashrc

FROM base AS api
RUN micromamba install -y -v -n base fastapi-cli==0.0.11 fastapi==0.116.1 rq -c conda-forge && \
    micromamba clean --all --yes
ENV REDIS_CONNECTION=
COPY src/api.py /home/$MAMBA_USER/src/api.py
COPY src/*.csv /home/$MAMBA_USER/src/
COPY src/*.json /home/$MAMBA_USER/src/
#HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
#    CMD curl --fail http://localhost:8000/health || exit 1
CMD ["fastapi", "run", "src/api.py"]

FROM api AS worker
COPY src/worker.py /home/$MAMBA_USER/src/worker.py
CMD ["python", "src/worker.py"]

FROM worker AS rqdashboard
RUN micromamba install -y -v -n base rq-dashboard -c conda-forge && \
    micromamba clean --all --yes
RUN echo 'export PYTHONPATH="/home/{$MAMBA_USER}/src:$PYTHONPATH"' >> /home/$MAMBA_USER/.bashrc
CMD ["python3", "-m", "rq_dashboard"]

FROM api AS devcontainer
RUN micromamba install -y -v -n base ipykernel dvc dvc-webdav ruff just markdownlint-cli curlify && \
    micromamba clean --all --yes


