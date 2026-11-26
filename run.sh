#!/usr/bin/env bash
# End-to-end driver: get data, build, run on the GPU, log, plot.
#
#   ./run.sh                 # SIPI images if reachable, else synthetic ones
#   ./run.sh --synthetic     # force synthetic images only
#   SIGMA=3 STREAMS=8 ./run.sh
#
# Everything that proves the run happened lands in results/:
#   results/run_log.txt, results/nvidia_smi.txt, results/output/metrics.csv,
#   results/output/*_{equalized,blurred,edges}.pgm and results/*.png
# and proof_of_execution.tar.gz bundles them (as PNG) for upload.

set -euo pipefail
cd "$(dirname "$0")"

SIGMA="${SIGMA:-2.0}"
STREAMS="${STREAMS:-4}"
INPUT_DIR="data/input"
RESULTS_DIR="results"

mkdir -p "${INPUT_DIR}" "${RESULTS_DIR}"

# 1. Input data.
if ! ls "${INPUT_DIR}"/*.pgm >/dev/null 2>&1; then
  if [[ "${1:-}" != "--synthetic" ]]; then
    python3 scripts/fetch_sipi.py --out "${INPUT_DIR}" ||
      echo "SIPI download failed; continuing with synthetic images only."
  fi
  python3 scripts/generate_synthetic.py --out "${INPUT_DIR}" --count 24
fi

# 2. Build (Makefile honours CUDA_PATH / ARCH from the environment).
make -j"$(nproc)"

# 3. Run, keeping a full log as proof of execution.
nvidia-smi > "${RESULTS_DIR}/nvidia_smi.txt" 2>&1 || true
{
  echo "Run started: $(date -u '+%Y-%m-%d %H:%M:%S UTC') on $(hostname)"
  ./bin/image_pipeline \
    --input_dir="${INPUT_DIR}" \
    --output_dir="${RESULTS_DIR}/output" \
    --sigma="${SIGMA}" \
    --streams="${STREAMS}"
  echo
  echo "Stream scaling experiment (GPU only, no CPU check):"
  for s in 1 2 4 8; do
    echo "--- streams=${s}"
    ./bin/image_pipeline --input_dir="${INPUT_DIR}" \
      --output_dir="${RESULTS_DIR}/streams_${s}" --streams="${s}" \
      --sigma="${SIGMA}" --no_cpu_check --no_save | grep -E "wall time"
  done
  rm -rf "${RESULTS_DIR}"/streams_*
} 2>&1 | tee "${RESULTS_DIR}/run_log.txt"

# 4. Figures (optional: needs matplotlib + Pillow).
python3 scripts/plot_results.py --results "${RESULTS_DIR}" \
  --input "${INPUT_DIR}" ||
  echo "Plotting skipped (pip install --user matplotlib pillow to enable)."

# 5. Upload-ready archive of logs + before/after images (needs Pillow).
python3 scripts/package_proof.py --results "${RESULTS_DIR}" \
  --input "${INPUT_DIR}" --out proof_of_execution.tar.gz ||
  echo "Packaging skipped (pip install --user pillow to enable)."

echo "Done. See ${RESULTS_DIR}/ and proof_of_execution.tar.gz"
