#!/usr/bin/env bash
# Single-file FBX → GLB conversion.
#
# Usage:
#   tools/fbx_to_glb/convert.sh INPUT.fbx OUTPUT.glb
#
# For batch conversion of an entire character class's clip set, use
# `import_class.sh <class_name> <mixamo_dir>` instead.
#
# Backend selection (in priority order):
#   1. FBX2glTF binary at tools/fbx_to_glb/bin/FBX2glTF
#       — Facebook's open-source converter, handles every FBX version
#         (Mixamo character downloads in 2026 are still FBX 6.1 binary;
#         Blender 5.x dropped 6.x support). Preferred when present.
#   2. Blender (PATH or /Applications/Blender.app or Flatpak)
#       — handles FBX 7.1+. Used as a fallback when FBX2glTF is absent.
#
# Either backend produces a glTF Binary suitable for `bevy_gltf` — the
# rest of the pipeline (loader.rs, render systems) is backend-agnostic.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERT_PY="$SCRIPT_DIR/convert.py"
FBX2GLTF_BIN="$SCRIPT_DIR/bin/FBX2glTF"

# -----------------------------------------------------------------------------
# Args.
# -----------------------------------------------------------------------------
if [ "$#" -ne 2 ]; then
    cat >&2 <<EOF
Usage: $0 INPUT.fbx OUTPUT.glb

Example:
  $0 ~/Downloads/Y_Bot.fbx \\
     client/assets/characters/human_default/y_bot.glb
EOF
    exit 2
fi
INPUT_FBX="$1"
OUTPUT_GLB="$2"

if [ ! -f "$INPUT_FBX" ]; then
    echo "ERROR: input file does not exist: $INPUT_FBX" >&2
    exit 3
fi

# Make sure the output directory exists; downstream tools expect the
# whole client/assets/characters/<class>/ tree to materialize as a side
# effect of the converter.
mkdir -p "$(dirname "$OUTPUT_GLB")"

# -----------------------------------------------------------------------------
# Backend 1: FBX2glTF (preferred — handles FBX 6.x).
# -----------------------------------------------------------------------------
if [ -x "$FBX2GLTF_BIN" ]; then
    echo ">>> [FBX2glTF] $(basename "$INPUT_FBX") -> $OUTPUT_GLB"
    # Flags chosen for parity with the Blender backend:
    #   --binary          → emit GLB (single file with embedded buffers)
    #   --keep-attribute  → preserve UV + normals + colors + skin weights
    #   --pbr-metallic-roughness → standard PBR material model bevy reads
    #   --no-flip-v       → Mixamo UVs are already in glTF convention,
    #                       skip the V flip that breaks them
    #   --anim-framerate bake30 → bake to 30fps (matches Mixamo export)
    #
    # `-o OUTPUT_NO_EXT` per FBX2glTF convention — the binary appends
    # `.glb` on its own.
    OUTPUT_NO_EXT="${OUTPUT_GLB%.glb}"
    "$FBX2GLTF_BIN" \
        --input "$INPUT_FBX" \
        --output "$OUTPUT_NO_EXT" \
        --binary \
        --keep-attribute auto \
        --pbr-metallic-roughness \
        --no-flip-v \
        --anim-framerate bake30
    exit 0
fi

# -----------------------------------------------------------------------------
# Backend 2: Blender headless. Auto-detect on PATH then canonical paths.
# -----------------------------------------------------------------------------
find_blender() {
    if command -v blender > /dev/null 2>&1; then
        command -v blender
        return 0
    fi
    for cand in \
        /Applications/Blender.app/Contents/MacOS/Blender \
        /Applications/Blender/Blender.app/Contents/MacOS/Blender ; do
        if [ -x "$cand" ]; then
            echo "$cand"
            return 0
        fi
    done
    if command -v flatpak > /dev/null 2>&1 && \
       flatpak info org.blender.Blender > /dev/null 2>&1; then
        echo "flatpak run org.blender.Blender"
        return 0
    fi
    return 1
}

BLENDER="$(find_blender || true)"
if [ -z "${BLENDER:-}" ]; then
    cat >&2 <<'EOF'
ERROR: no FBX → GLB converter available.
  Either:
    - place the FBX2glTF binary at tools/fbx_to_glb/bin/FBX2glTF
      (https://github.com/facebookincubator/FBX2glTF/releases — handles
      every FBX version), OR
    - install Blender 4.x / 5.x and put it on PATH or in /Applications/.
EOF
    exit 1
fi

echo ">>> [Blender] $(basename "$INPUT_FBX") -> $OUTPUT_GLB"
# shellcheck disable=SC2086  # Word-splitting intentional for the
# `flatpak run org.blender.Blender` two-word case.
$BLENDER --background --python "$CONVERT_PY" -- "$INPUT_FBX" "$OUTPUT_GLB"
