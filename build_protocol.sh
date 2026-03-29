#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FBS_FILE="$SCRIPT_DIR/protocol/voxeldust.fbs"
OUT_DIR="$SCRIPT_DIR/core/src"

if ! command -v flatc &> /dev/null; then
    echo "Error: flatc not found. Install FlatBuffers compiler:"
    echo "  brew install flatbuffers"
    exit 1
fi

echo "Generating Rust code from $FBS_FILE..."
flatc --rust -o "$OUT_DIR" "$FBS_FILE"

# flatc generates voxeldust.protocol/voxeldust_generated.rs or similar.
# Move it to the expected location.
GENERATED="$OUT_DIR/voxeldust_generated.rs"
if [ -f "$GENERATED" ]; then
    echo "Generated: $GENERATED"
else
    # flatc may create a subdirectory based on namespace
    FOUND=$(find "$OUT_DIR" -name "*_generated.rs" -newer "$FBS_FILE" 2>/dev/null | head -1)
    if [ -n "$FOUND" ]; then
        mv "$FOUND" "$GENERATED"
        # Clean up empty namespace dirs
        find "$OUT_DIR" -type d -empty -delete 2>/dev/null || true
        echo "Generated: $GENERATED"
    else
        echo "Warning: No generated file found. Check flatc output."
        exit 1
    fi
fi

echo "Done."
