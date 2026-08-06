# Binder Dockerfile: build a container that can run the full analysis
# Mirrors the project's Dockerfile but keeps the layers friendly to Binder
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /home/jovyan/work

COPY requirements.txt requirements-optional.txt ./
RUN python -m pip install --upgrade pip setuptools wheel
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
RUN if [ -f requirements-optional.txt ]; then pip install -r requirements-optional.txt; fi

COPY . ./

CMD ["/bin/bash"]
