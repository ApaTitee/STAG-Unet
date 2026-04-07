#!/bin/bash
set -euo pipefail

# ============================================================
# BEETLE Dataset Downloader
# Downloads individual files (NOT the on-the-fly archive)
# to enable reliable resume (wget --continue) and MD5 verification.
#
# Prerequisites: wget, unzip, md5sum (coreutils)
# Total size: ~151 GB
# ============================================================

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
cd "$SCRIPT_DIR"

DATA_DIR="./data"
RECORD_ID="16812932"
BASE_URL="https://zenodo.org/api/records/${RECORD_ID}/files"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# File definitions: filename|expected_md5
# MD5 values from the official Zenodo API record
FILES=(
    "data_overview.csv|fa3093bec3a80e7f3464dcc7cbc36fad"
    "annotations.zip|20c3ed8ae74a392eb2b4ba2baf75494a"
    "model.zip|c1c82ed123484cd8760b6343feeee14f"
    "images.zip|7bfd8524615fc4914b9998b8bfc80f9e"
)

MAX_RETRIES=5
RETRY_WAIT=10   # seconds between retries

download_with_retry() {
    local filename="$1"
    local url="$2"
    local retry=0

    while [ $retry -lt $MAX_RETRIES ]; do
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Downloading: $filename (attempt $((retry + 1))/$MAX_RETRIES)"
        if wget --continue --progress=bar:force:noscroll \
                --timeout=60 --tries=3 \
                -O "$filename" "$url"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Successfully downloaded: $filename"
            return 0
        else
            retry=$((retry + 1))
            if [ $retry -lt $MAX_RETRIES ]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Download failed, retrying in ${RETRY_WAIT}s..."
                sleep "$RETRY_WAIT"
            fi
        fi
    done

    echo "ERROR: Failed to download $filename after $MAX_RETRIES attempts."
    return 1
}

verify_md5() {
    local filename="$1"
    local expected_md5="$2"

    echo "Verifying MD5 for $filename ..."
    local actual_md5
    actual_md5=$(md5sum "$filename" | awk '{print $1}')

    if [ "$actual_md5" = "$expected_md5" ]; then
        echo "  ✓ MD5 OK: $actual_md5"
        return 0
    else
        echo "  ✗ MD5 MISMATCH!"
        echo "    Expected: $expected_md5"
        echo "    Actual:   $actual_md5"
        return 1
    fi
}

# ---- Main download loop ----
echo "=============================================="
echo " BEETLE Dataset Downloader"
echo " Record: https://zenodo.org/records/${RECORD_ID}"
echo "=============================================="

for entry in "${FILES[@]}"; do
    IFS='|' read -r filename expected_md5 <<< "$entry"
    download_url="${BASE_URL}/${filename}/content"

    # If file already exists and MD5 matches, skip
    if [ -f "$filename" ]; then
        if verify_md5 "$filename" "$expected_md5"; then
            echo "Skipping $filename (already downloaded and verified)."
            continue
        else
            echo "Existing $filename has wrong checksum, re-downloading..."
            rm -f "$filename"
        fi
    fi

    download_with_retry "$filename" "$download_url"
    verify_md5 "$filename" "$expected_md5" || {
        echo "ERROR: MD5 verification failed for $filename. Please re-run the script to retry."
        exit 1
    }
    echo ""
done

# ---- Extract sub-archives ----
echo "=============================================="
echo " Extracting archives..."
echo "=============================================="

for archive in annotations.zip model.zip; do
    if [ -f "$archive" ]; then
        echo "Extracting $archive ..."
        unzip -o "$archive"
        rm "$archive"
        echo ""
    else
        echo "Warning: $archive not found, skipping..."
    fi
done

# images.zip is very large; optionally remove after extraction
if [ -f "images.zip" ]; then
    read -rp "images.zip extracted. Remove the archive to free ~147 GB of disk space? [y/N] " choice
    case "$choice" in
        [yY]|[yY][eE][sS])
            rm -f "images.zip"
            echo "Removed images.zip."
            ;;
        *)
            echo "Kept images.zip."
            ;;
    esac
fi

echo "=============================================="
echo " All done!"
echo "=============================================="
