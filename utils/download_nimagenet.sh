#!/usr/bin/env bash
set -euo pipefail

TRAIN_DIR="../dataset/N-ImageNet/train"
mkdir -p "${TRAIN_DIR}"

BASE_URL="https://huggingface.co/datasets/82magnolia/N-ImageNet/resolve/main/training"

for i in $(seq 1 10); do
  PART="Part_${i}"
  ZIPNAME="${PART}.zip"
  ZIPPATH="${TRAIN_DIR}/${ZIPNAME}"
  URL="${BASE_URL}/${ZIPNAME}"

  echo "Downloading ${PART}..."
  curl -L --fail --retry 3 -o "${ZIPPATH}" "${URL}"

  echo "Extracting ${ZIPNAME}..."
  unzip -q "${ZIPPATH}" -d "${TRAIN_DIR}"

  echo "Removing ${ZIPNAME}..."
  rm -f "${ZIPPATH}"
done

echo "Done downloading training set. You can now add validation manually."
