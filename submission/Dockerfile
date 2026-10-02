# Executive summary: Track-2 forecaster image. linux/amd64, offline, read-only-FS friendly.
# Build:  docker build --platform linux/amd64 -t t2-forecaster:dev .
FROM python:3.13-slim-bookworm

LABEL qfbench2.interface_version=2.0
LABEL qfbench2.track=forecasting
LABEL qfbench2.verb=forecast

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp \
    PYTHONPATH=/opt \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

RUN pip install --no-cache-dir \
        "numpy==2.1.3" \
        "pandas==2.2.3" \
        "pyarrow==18.1.0" \
        "jsonschema==4.23.0" \
        "qfbench2-common @ https://github.com/Agenthon-2026/Agenthon2026-public/archive/refs/tags/v2.4.4.tar.gz#subdirectory=common"

COPY qfbench2_track_forecasting /opt/qfbench2_track_forecasting
COPY t2agent /opt/t2agent
COPY textlayer /opt/textlayer
ARG CONFIG_JSON="{}"
RUN printf '%s' "$CONFIG_JSON" > /opt/t2agent/config.json
COPY ARTIFACT_PROVENANCE.md LICENSE LICENSE.third_party_qfbench2 /opt/

RUN printf '#!/bin/sh\nexec python3 -m t2agent.cli "$@"\n' > /usr/local/bin/forecast \
 && chmod 755 /usr/local/bin/forecast \
 && find /opt -type d -exec chmod 755 {} + && find /opt -type f -exec chmod 644 {} +

USER 65534:65534
WORKDIR /tmp
CMD ["forecast", "--help"]
