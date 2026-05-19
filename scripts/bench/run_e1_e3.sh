#!/usr/bin/env bash
# Convenience runner for E1 (resume parity) and E3 (startup time).
# Run from repo root.
#
#   bash scripts/bench/run_e1_e3.sh 2>&1 | tee tmp/bench_e1_e3.log
#
# Each phase is launched in a fresh process so E3's cold-start timing is honest.

set -u

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"

echo "================================================================"
echo " E3 — Startup time (Grain)"
echo "================================================================"
LOADER_KIND=grain $PY scripts/bench/e3_startup_time.py

echo
echo "================================================================"
echo " E3 — Startup time (tf.data)"
echo "================================================================"
LOADER_KIND=tf $PY scripts/bench/e3_startup_time.py

echo
echo "================================================================"
echo " E1 — Resume parity (Grain)"
echo "================================================================"
LOADER_KIND=grain $PY scripts/bench/e1_resume_parity.py
grain_rc=$?

echo
echo "================================================================"
echo " E1 — Resume parity (tf.data, expected: cannot restore)"
echo "================================================================"
LOADER_KIND=tf $PY scripts/bench/e1_resume_parity.py
tf_rc=$?

echo
echo "Done. Grain E1 exit=$grain_rc  tf E1 exit=$tf_rc"
echo "(Grain should be 0=PASS; tf is informational — divergence is the expected result.)"
