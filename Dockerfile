# syntax=docker/dockerfile:1

FROM quay.io/jupyter/base-notebook
RUN pip install --no-cache-dir matplotlib scikit-learn torch
WORKDIR "work"
COPY pyproject.toml ./
COPY selfatt ./selfatt
RUN pip install -e .