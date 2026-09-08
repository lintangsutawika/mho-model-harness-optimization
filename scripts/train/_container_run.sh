#!/usr/bin/env bash
# Runs INSIDE the training Singularity container, wrapping the real training cmd.
#
# Harbor's `singularity` trial environment starts an apptainer sandbox *nested*
# inside this container (one per rollout trial). The training SIF ships no
# apptainer binary or its support libs, so scripts/train/train_math_dapo.sh binds
# the host apptainer in and stages the shared libs the image lacks at
# $HOSTLIBS_DIR (bound read-only). We must register those on the loader *cache*
# here: apptainer strips LD_LIBRARY_PATH when it re-execs its (setuid/starter)
# helper, so an env var alone is not enough -- only ld.so.cache survives. Then
# exec the real command unchanged.
set -euo pipefail

: "${HOSTLIBS_DIR:=/opt/hostlibs}"
if [ -d "$HOSTLIBS_DIR" ]; then
  echo "$HOSTLIBS_DIR" > /etc/ld.so.conf.d/zz-nested-apptainer.conf 2>/dev/null || true
  ldconfig 2>/dev/null || true
fi

exec "$@"
