# syntax=docker/dockerfile:1.7

ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.11.6

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv-bin

FROM python:${PYTHON_VERSION}-slim AS build-base
COPY --from=uv-bin /uv /uvx /bin/
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/scienceflow
WORKDIR /build
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY scienceflow ./scienceflow

FROM build-base AS core-build
RUN uv sync --frozen --no-dev --no-editable

FROM core-build AS ml-build
RUN uv sync --frozen --no-dev --no-editable --extra ml

FROM core-build AS gpu-build
RUN uv sync --frozen --no-dev --no-editable --extra gpu

FROM core-build AS mlebench-build
RUN uv sync --frozen --no-dev --no-editable --extra mlebench

FROM core-build AS full-build
RUN uv sync --frozen --no-dev --no-editable --extra full

FROM python:${PYTHON_VERSION}-slim AS runtime-base
ENV PATH=/opt/scienceflow/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
RUN groupadd --gid 10001 scienceflow \
    && useradd --uid 10001 --gid scienceflow --create-home scienceflow \
    && mkdir -p /workspace /home/scienceflow/.config/scienceflow \
    && chown -R scienceflow:scienceflow /workspace /home/scienceflow/.config
WORKDIR /workspace
USER scienceflow
ENTRYPOINT ["scienceflow"]
CMD ["--help"]

FROM runtime-base AS task-runtime-base
USER root
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
USER scienceflow

FROM runtime-base AS core
COPY --from=core-build --chown=scienceflow:scienceflow /opt/scienceflow /opt/scienceflow

FROM task-runtime-base AS ml
COPY --from=ml-build --chown=scienceflow:scienceflow /opt/scienceflow /opt/scienceflow

FROM task-runtime-base AS gpu
COPY --from=gpu-build --chown=scienceflow:scienceflow /opt/scienceflow /opt/scienceflow

FROM task-runtime-base AS mlebench
COPY --from=mlebench-build --chown=scienceflow:scienceflow /opt/scienceflow /opt/scienceflow

FROM task-runtime-base AS full
COPY --from=full-build --chown=scienceflow:scienceflow /opt/scienceflow /opt/scienceflow
