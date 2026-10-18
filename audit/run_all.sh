#!/usr/bin/env bash
# run_all.sh - re-run the whole audit in order and rebuild audit/REPORT.md.
#
# Usage (any directory):   ANKI_DB=<path to a COPY of collection.anki2> bash audit/run_all.sh
#
# The Anki collection is opened read-only. Without ANKI_DB the scripts fall back to the path in audit_lib.py.
# s2_placebo.py takes about 10 minutes on 16 cores; the rest take a few minutes in total.
set -euo pipefail
cd "$(dirname "$0")"          # the scripts import audit_lib from this directory

echo "== s0 replication";      python s0_replication.py   > results/00_replication_stdout.txt
echo "== s1 overlap";          python s1_overlap.py       > results/01_overlap_stdout.txt
echo "== s2 placebo (B=500)";  python s2_placebo.py 500   > results/02_placebo_stdout.txt
echo "== s3 leakage";          python s3_leakage.py       > results/03_leakage_stdout.txt
echo "== s4 history";          python s4_history.py       > results/04_history_stdout.txt
echo "== s5 RCT";              python s5_rct_checks.py    > results/05_rct_stdout.txt
echo "== s6 cloud branch";     python s6_cloud_branch.py  > results/06_cloud_branch_stdout.txt
echo "== build report";        python build_report.py
echo "done: audit/REPORT.md"
