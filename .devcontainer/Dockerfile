FROM public.ecr.aws/docker/library/python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_PYTHON_REFRESH=quiet

RUN apt-get update && apt-get install -y --no-install-recommends htop wget curl unzip git

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
