#!/usr/bin/env bash
# Batch FBX → GLB import for one character class.
#
# Reads `manifest.<class>.txt` from this directory, runs `convert.sh`
# once per entry. Validates every input file is present BEFORE starting
# any conversion — partial outputs would leave the asset registry in a
# broken half-loaded state on next client launch.
#
# Usage:
#   tools/fbx_to_glb/import_class.sh <class_name> <mixamo_download_dir>
#
# Example:
#   tools/fbx_to_glb/import_class.sh human_default ~/Downloads/mixamo

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ---------------------------------------------------------------------------
# Args.
# ---------------------------------------------------------------------------
if [ "$#" -ne 2 ]; then
    cat >&2 <<EOF
Usage: $0 <class_name> <mixamo_download_dir>

  <class_name>            Matches manifest.<class_name>.txt in this dir.
  <mixamo_download_dir>   Folder containing the Mixamo .fbx downloads
                          named per the manifest.

Example:
  $0 human_default ~/Downloads/mixamo
EOF
    exit 2
fi
CLASS="$1"
MIXAMO_DIR="$2"
MANIFEST="$SCRIPT_DIR/manifest.${CLASS}.txt"

if [ ! -f "$MANIFEST" ]; then
    echo "ERROR: no manifest for class '$CLASS' (looked for $MANIFEST)" >&2
    exit 3
fi
if [ ! -d "$MIXAMO_DIR" ]; then
    echo "ERROR: mixamo download directory does not exist: $MIXAMO_DIR" >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Pre-flight: read manifest + verify every input file exists.
# We collect (input_path, output_path) pairs; bail out early on missing
# files so a partial run never produces a half-state asset tree.
# ---------------------------------------------------------------------------
PAIRS_INPUT=()
PAIRS_OUTPUT=()
MISSING=()
# TAB-separated parse — Mixamo download filenames contain spaces
# ("Falling Idle.fbx", "Left Turn 90.fbx"), so the manifest uses a
# literal tab between the two columns. Anything but tab is part of
# the filename or output path.
while IFS=$'\t' read -r INPUT_NAME OUTPUT_REL || [ -n "$INPUT_NAME$OUTPUT_REL" ]; do
    # Strip everything after the first '#' on the input column (so a
    # commented line like `# foo` is skipped), then trim.
    INPUT_NAME="${INPUT_NAME%%#*}"
    INPUT_NAME="${INPUT_NAME## }"
    INPUT_NAME="${INPUT_NAME%% }"
    OUTPUT_REL="${OUTPUT_REL%%#*}"
    OUTPUT_REL="${OUTPUT_REL## }"
    OUTPUT_REL="${OUTPUT_REL%% }"
    # Skip blank or pure-comment lines.
    [ -z "$INPUT_NAME" ] && [ -z "$OUTPUT_REL" ] && continue
    # Half-populated row = malformed.
    if [ -z "$INPUT_NAME" ] || [ -z "$OUTPUT_REL" ]; then
        echo "ERROR: malformed manifest line (need TAB-separated input.fbx<TAB>output.glb):" >&2
        echo "  input='$INPUT_NAME' output='$OUTPUT_REL'" >&2
        exit 4
    fi
    INPUT_PATH="$MIXAMO_DIR/$INPUT_NAME"
    OUTPUT_PATH="$REPO_ROOT/$OUTPUT_REL"
    if [ ! -f "$INPUT_PATH" ]; then
        MISSING+=("$INPUT_NAME")
    fi
    PAIRS_INPUT+=("$INPUT_PATH")
    PAIRS_OUTPUT+=("$OUTPUT_PATH")
done < "$MANIFEST"

if [ "${#MISSING[@]}" -ne 0 ]; then
    echo "ERROR: the following files are missing from $MIXAMO_DIR:" >&2
    for f in "${MISSING[@]}"; do
        echo "  - $f" >&2
    done
    cat >&2 <<EOF

Each Mixamo download produces a single .fbx; rename them to the names
above before re-running. The manifest is at:
  $MANIFEST
EOF
    exit 5
fi

# ---------------------------------------------------------------------------
# Convert each pair sequentially. Blender startup is ~1 s per call, so
# 9 conversions take ~10–15 s in total — acceptable for a one-time
# asset bootstrap. (Future: switch to a single Blender invocation that
# loops all pairs in Python if this becomes a hot path.)
# ---------------------------------------------------------------------------
echo "Converting ${#PAIRS_INPUT[@]} files for class '$CLASS'..."
for i in "${!PAIRS_INPUT[@]}"; do
    "$SCRIPT_DIR/convert.sh" "${PAIRS_INPUT[$i]}" "${PAIRS_OUTPUT[$i]}"
done
echo "Done. Class '$CLASS' assets installed under client/assets/characters/$CLASS/."
