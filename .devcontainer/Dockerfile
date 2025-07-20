FROM public.ecr.aws/docker/library/python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_PYTHON_REFRESH=quiet

RUN apt-get update && apt-get install -y --no-install-recommends htop wget curl unzip git && rm -rf /var/lib/apt/lists/*

RUN sh -c "echo 'deb https://gitsecret.jfrog.io/artifactory/git-secret-deb git-secret main' >> /etc/apt/sources.list" && \
    wget -qO - 'https://gitsecret.jfrog.io/artifactory/api/gpg/key/public' | apt-key add - && \
    apt-get update  && \
    apt-get install -y git-secret

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

RUN pre-commit install-hooks
