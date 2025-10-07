#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Download Training Set
# -----------------------------
TRAIN_DIR="../dataset/N-ImageNet/train"
mkdir -p "${TRAIN_DIR}"

BASE_TRAIN_URL="https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training"

for i in $(seq 1 10); do
  PART="Part_${i}"
  ZIPNAME="${PART}.zip"
  ZIPPATH="${TRAIN_DIR}/${ZIPNAME}"
  URL="${BASE_TRAIN_URL}/${ZIPNAME}"

  echo "Downloading ${PART}..."
  curl -L --fail --retry 3 -o "${ZIPPATH}" "${URL}"

  echo "Extracting ${ZIPNAME}..."
  unzip -q "${ZIPPATH}" -d "${TRAIN_DIR}"

  echo "Removing ${ZIPNAME}..."
  rm -f "${ZIPPATH}"
done

echo "✅ Done downloading training set."

# -----------------------------
# Download Validation Set
# -----------------------------
VAL_DIR="../dataset/N-ImageNet/validation"
mkdir -p "${VAL_DIR}"

BASE_VAL_URL="https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/validation"

VAL_FILES=(
  "extracted_val.zip"
  "extracted_val_brightness_4.zip"
  "extracted_val_brightness_5.zip"
  "extracted_val_brightness_6.zip"
  "extracted_val_brightness_7.zip"
  "extracted_val_mode_1.zip"
  "extracted_val_mode_3.zip"
  "extracted_val_mode_5.zip"
  "extracted_val_mode_6.zip"
  "extracted_val_mode_7.zip"
)

for ZIPNAME in "${VAL_FILES[@]}"; do
  ZIPPATH="${VAL_DIR}/${ZIPNAME}"
  URL="${BASE_VAL_URL}/${ZIPNAME}"

  echo "Downloading ${ZIPNAME}..."
  curl -L --fail --retry 3 -o "${ZIPPATH}" "${URL}"

  echo "Extracting ${ZIPNAME}..."
  unzip -q "${ZIPPATH}" -d "${VAL_DIR}"

  echo "Removing ${ZIPNAME}..."
  rm -f "${ZIPPATH}"
done

echo "✅ Done downloading validation set."
echo "All training and validation data are downloaded and extracted successfully."
