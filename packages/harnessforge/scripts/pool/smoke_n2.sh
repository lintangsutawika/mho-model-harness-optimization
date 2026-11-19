#!/bin/bash
# The recorded N=2 x rt_HG smoke run. Run inside an rt_HC job (no compute on the login node):
#   qsub -q <queue> -P <project> -v RTYPE=rt_HC -l select=1 -l walltime=2:00:00 -N hfpool-drv2 \
#        -j oe -o smoke.out -- /bin/bash <this dir>/smoke_n2.sh <spec.json> <perturbed-ckpt-dir>
# PYTHONPATH must point at packages/harnessforge/src; PY at a Python >= 3.12.
SPEC=$1; CKPT=$2; HERE=$(dirname "$0")
PY=${PY:-$HOME/.local/bin/python3.12}
export PYTHONPATH=${PYTHONPATH:-$(cd "$HERE/../.." && pwd)/src}
set -x
$PY -m harnessforge.pool stop --spec $SPEC || true
time $PY -m harnessforge.pool start --spec $SPEC --timeout 2400 || exit 1
$PY -m harnessforge.pool status --spec $SPEC
EP=$($PY -m harnessforge.pool endpoint --spec $SPEC)
echo "driver host $(hostname) -> endpoint $EP"
$PY $HERE/smoke_client.py $EP
$PY $HERE/bench.py throughput $EP policy 64 256
$PY $HERE/bench.py throughput $EP policy 128 512
$PY $HERE/swap_check.py $SPEC $CKPT
# node failure: kill node 1's job, repair, wait until the router serves both engines again
J1=$($PY -c "import sys,json;from harnessforge.pool import from_config,load_config;print(from_config(load_config(sys.argv[1])).dir.pool()['jobs']['1'])" $SPEC)
/opt/pbs/bin/qdel $J1
sleep 60
$PY -m harnessforge.pool status --spec $SPEC
$PY -m harnessforge.pool repair --spec $SPEC
time $PY -m harnessforge.pool wait --spec $SPEC --timeout 1800
$PY -m harnessforge.pool status --spec $SPEC
$PY $HERE/smoke_client.py $EP
$PY -m harnessforge.pool stop --spec $SPEC
