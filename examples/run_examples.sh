#!/usr/bin/env bash
# Run every atlas case study headlessly (figures -> ../output/).
#
#   ./examples/run_examples.sh                 # run all
#   ./examples/run_examples.sh bh_shadow       # run one
#
# Finds a numpy-capable python: $PYTHON first, then this repo's pixi env, then
# python3/python on PATH.  (In this dev tree the env may live in a sibling
# project; point PYTHON at it if needed.)
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"

have() { "$1" -c "import numpy, matplotlib" >/dev/null 2>&1; }
pick() {
  local c
  if [[ -n "${PYTHON:-}" ]]; then
    if have "${PYTHON}"; then echo "${PYTHON}"; return 0; fi
  fi
  while IFS= read -r c; do [[ -n "$c" ]] && have "$c" && { echo "$c"; return 0; }; done \
      < <(compgen -g .pixi/envs/*/bin/python 2>/dev/null | sort)
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && have "$c"; then echo "$c"; return 0; fi
  done
  return 1
}

if ! PY="$(pick)"; then
  echo "error: no python with numpy+matplotlib found on PATH." >&2
  echo "       activate the environment (pixi shell) or set PYTHON=/path/to/python." >&2
  exit 1
fi
echo "using python: $PY"

run() { echo; echo "=== $1 ==="; "$PY" -m "examples.$1"; }

if [[ $# -gt 0 ]]; then
  for name in "$@"; do run "$name"; done
else
  for name in bh_shadow ising_fss gp_vortices ns2d_turb lattice_ft \
              pic_plasma graphene_tb wavepacket_scatt md_greenkubo bayesian_infer \
              fgh_bound_states nfe_bands atomic_hf kmt_dynamics; do
    run "$name"
  done
  echo
  echo "All examples complete. Figures are in ./output/"
fi
