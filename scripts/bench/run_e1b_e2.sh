#!/usr/bin/env bash
# Convenience runner for E1b (multi-cycle resume) and E2 (memory footprint).
#
#   bash scripts/bench/run_e1b_e2.sh 2>&1 | tee tmp/bench_e1b_e2.log

set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"

echo "================================================================"
echo " E2 — Memory footprint (Grain)"
echo "================================================================"
LOADER_KIND=grain $PY scripts/bench/e2_memory_footprint.py

echo
echo "================================================================"
echo " E2 — Memory footprint (tf.data)"
echo "================================================================"
LOADER_KIND=tf $PY scripts/bench/e2_memory_footprint.py

echo
echo "================================================================"
echo " E1b — Multi-cycle resume stress (Grain)"
echo "================================================================"
LOADER_KIND=grain $PY scripts/bench/e1b_resume_stress.py
grain_rc=$?

echo
echo "================================================================"
echo " E1b — Multi-cycle resume stress (tf.data, expected: only cycle 0 matches)"
echo "================================================================"
LOADER_KIND=tf $PY scripts/bench/e1b_resume_stress.py
tf_rc=$?

echo
echo "Done. Grain E1b exit=$grain_rc  tf E1b exit=$tf_rc"
