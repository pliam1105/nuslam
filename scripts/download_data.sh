#!/usr/bin/env bash
# Download and extract the nuScenes mini split (v1.0-mini, ~4.2 GB compressed).
#
#   ./scripts/download_nuscenes.sh [DATAROOT]
#
# DATAROOT defaults to $NUSCENES_DATAROOT, then ./data/nuscenes.
# The mini split is served without authentication. The full trainval splits are
# not: log in at https://www.nuscenes.org/nuscenes#download, grab the URLs, and
# extract them into the same DATAROOT.
set -euo pipefail

DATAROOT="${1:-${NUSCENES_DATAROOT:-./data/nuscenes}}"
URL="https://www.nuscenes.org/data/v1.0-mini.tgz"
ARCHIVE="${DATAROOT}/v1.0-mini.tgz"

mkdir -p "${DATAROOT}"

# ~4.2 GB download + ~4.4 GB extracted; refuse to start if that clearly won't fit.
avail_kb=$(df -Pk "${DATAROOT}" | awk 'NR==2 {print $4}')
if [ "${avail_kb}" -lt 9000000 ]; then
  echo "ERROR: need ~9 GB free at ${DATAROOT}, have $((avail_kb / 1024)) MB." >&2
  echo "Free space or pass a DATAROOT on a larger volume." >&2
  exit 1
fi

echo "Downloading v1.0-mini -> ${ARCHIVE} (resumable)"
curl -L -C - -o "${ARCHIVE}" "${URL}"

echo "Extracting into ${DATAROOT}"
tar -xzf "${ARCHIVE}" -C "${DATAROOT}"

echo "Done. Expect ${DATAROOT}/{v1.0-mini,samples,sweeps,maps}"
echo "Delete ${ARCHIVE} to reclaim ~4.2 GB once the extract is verified."
