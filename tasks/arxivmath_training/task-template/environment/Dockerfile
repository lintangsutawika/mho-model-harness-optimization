FROM python:3.11-slim
WORKDIR /workspace

# uvicorn/fastapi for harbor's sandbox server, agent tools. The sandbox shares host net
# (online), but harbor's bootstrap pip-installs uvicorn into a fresh /opt/harbor-server venv
# from /usr/bin/python3. That MUST stay python3.11 (non-PEP-668, venv+pip work) -- NOT Debian's
# python3.13 (externally-managed PEP-668, whose venv/pip fails). So we symlink /usr/bin/python3
# to the base 3.11 and do NOT apt-install Debian python3.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        python3-pip \
        python3-venv \
        build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/local/bin/python3 /usr/bin/python3

# uvicorn/fastapi native (so harbor's venv, which inherits nothing, still has a fallback;
# the real win is that /usr/bin/python3 is 3.11 so harbor's own pip-install works online).
RUN /usr/local/bin/python3 -m pip install --no-cache-dir \
        uvicorn fastapi

WORKDIR /workspace
