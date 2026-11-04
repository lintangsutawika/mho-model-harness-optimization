# shellcheck shell=bash
# Source <repo>/.env with MHO_* entries acting as DEFAULTS: any MHO_* variable already set in
# the environment wins over .env. The loop hands each job its config via env (e.g.
# MHO_MODEL=Qwen/Qwen3.5-9B); a plain `. .env` re-exported .env's MHO_MODEL over it, so every
# "9B" eval silently served 4B. Non-MHO vars (node-local paths such as
# BASE_DIR=${PBS_LOCALDIR:-...}) are still re-derived from .env on each node, as before.
# Usage:  . scripts/utils/load_env.sh && load_env [path/to/.env]
load_env() {
  local _f="${1:-.env}" _saved
  [ -f "$_f" ] || return 0
  _saved="$(export -p | grep -E '^declare -x MHO_[A-Za-z0-9_]+=' || true)"
  set -a; . "$_f"; set +a
  # declare inside a function is LOCAL; -g restores the values globally.
  eval "${_saved//declare -x /declare -gx }"
}
