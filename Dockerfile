FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build tools required by some scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    git \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install (including optional heavy deps)
COPY requirements.txt requirements-optional.txt /app/

RUN python -m pip install --upgrade pip setuptools wheel
RUN if [ -f requirements.txt ]; then pip install -r requirements.txt; fi
RUN if [ -f requirements-optional.txt ]; then pip install -r requirements-optional.txt; fi

# Copy the repository
COPY . /app

# Lightweight entrypoint: start a shell. Users can run the demo or full pipeline.
CMD ["/bin/bash"]
