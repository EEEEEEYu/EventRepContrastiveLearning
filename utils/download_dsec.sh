#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# DSEC Full Dataset Downloader (Event + Calibration Only)
# ============================================================
# This script downloads all left/right event camera data and
# calibration files for both training and test sets, unzips
# them, and removes temporary ZIPs afterwards.
#
# Directory structure:
#   DSEC/
#     train/
#       zurich_city_02_a/
#         event/left/
#         event/right/
#         calibration/
#     test/
#       ...
# ============================================================

BASE_DIR="../dataset/DSEC"
TRAIN_DIR="${BASE_DIR}/train"
TEST_DIR="${BASE_DIR}/test"
mkdir -p "$TRAIN_DIR" "$TEST_DIR"

BASE_TRAIN_URL="https://download.ifi.uzh.ch/rpg/DSEC/train"
BASE_TEST_URL="https://download.ifi.uzh.ch/rpg/DSEC/test"

# ------------------------------------------------------------
# Training Sequences (from DSEC page)
# ------------------------------------------------------------
train_sequences=(
  interlaken_00_c interlaken_00_d interlaken_00_e interlaken_00_f interlaken_00_g
  thun_00_a
  zurich_city_00_a zurich_city_00_b
  zurich_city_01_a zurich_city_01_b zurich_city_01_c zurich_city_01_d zurich_city_01_e zurich_city_01_f
  zurich_city_02_a zurich_city_02_b zurich_city_02_c zurich_city_02_d zurich_city_02_e
  zurich_city_03_a
  zurich_city_04_a zurich_city_04_b zurich_city_04_c zurich_city_04_d zurich_city_04_e zurich_city_04_f
  zurich_city_05_a zurich_city_05_b
  zurich_city_06_a
  zurich_city_07_a
)

# ------------------------------------------------------------
# Test Sequences (subset from DSEC test set)
# ------------------------------------------------------------
test_sequences=(
  zurich_city_08_a zurich_city_09_a zurich_city_10_a zurich_city_11_a zurich_city_12_a
)

# ------------------------------------------------------------
# Helper Function: Download + Extract
# ------------------------------------------------------------
download_and_extract() {
  local seq="$1"
  local split="$2"
  local base_url="$3"
  local target_dir="$4"

  echo "==== Processing sequence: ${seq} (${split}) ===="
  mkdir -p "${target_dir}/${seq}/event/left" "${target_dir}/${seq}/event/right" "${target_dir}/${seq}/calibration"

  # --- Event Data (left/right) ---
  for side in left right; do
    ZIP_URL="${base_url}/${seq}/${seq}_events_${side}.zip"
    ZIP_PATH="${target_dir}/${seq}/event/${side}/${seq}_events_${side}.zip"

    echo "Downloading events_${side} → ${ZIP_PATH}"
    wget -c "${ZIP_URL}" -O "${ZIP_PATH}" || { echo "❌ Failed: ${ZIP_URL}"; continue; }

    echo "Extracting → ${target_dir}/${seq}/event/${side}/"
    unzip -q -o "${ZIP_PATH}" -d "${target_dir}/${seq}/event/${side}/"
    rm -f "${ZIP_PATH}"
  done

  # --- Calibration Data ---
  CALIB_URL="${base_url}/${seq}/${seq}_calibration.zip"
  CALIB_PATH="${target_dir}/${seq}/calibration/${seq}_calibration.zip"

  echo "Downloading calibration → ${CALIB_PATH}"
  wget -c "${CALIB_URL}" -O "${CALIB_PATH}" || { echo "❌ Failed: ${CALIB_URL}"; return; }

  echo "Extracting calibration..."
  unzip -q -o "${CALIB_PATH}" -d "${target_dir}/${seq}/calibration/"
  rm -f "${CALIB_PATH}"
}

# ------------------------------------------------------------
# Download all training sequences
# ------------------------------------------------------------
for seq in "${train_sequences[@]}"; do
  download_and_extract "$seq" "train" "$BASE_TRAIN_URL" "$TRAIN_DIR"
done

# ------------------------------------------------------------
# Download all test sequences
# ------------------------------------------------------------
for seq in "${test_sequences[@]}"; do
  download_and_extract "$seq" "test" "$BASE_TEST_URL" "$TEST_DIR"
done

echo "✅ All event and calibration data downloaded successfully."
echo "Structure:"
echo "${BASE_DIR}/train/<sequence>/event/{left,right}/"
echo "${BASE_DIR}/train/<sequence>/calibration/"
echo "${BASE_DIR}/test/<sequence>/event/{left,right}/"
echo "${BASE_DIR}/test/<sequence>/calibration/"
