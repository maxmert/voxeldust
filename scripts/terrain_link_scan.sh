#!/bin/zsh
# THE FLOAT FENCE, layer 3 (the voxel foundation, slice 5; SL10 clause 4): the generator's and the
# leaf's OWN object code must name no platform math symbol. `nm -u` lists the symbols an object needs
# from outside itself; a transcendental among them means a sine, a power or an exponent reached the
# seed-shaped path from THESE TWO CRATES — a direct call the lint missed, or a generic or inlined body
# of a dependency. A non-generic function of a dependency compiles into THAT crate's object and is not
# seen here; the crate-isolation gate (`tests/tests/crate_isolation.rs`) is what keeps such a
# dependency out of the two crates. Scans the rlibs (the crates' own objects), never the staticlib,
# whose bundled std would name the libc math it links for itself.
#
# `--control` is the observed-failing control: it scans the motion crate's rlib, which calls the
# platform's sine and cosine on purpose, and expects to FIND them. A scan that finds nothing there
# proves nothing here.
set -u
cd "$(dirname "$0")/.."
pattern=' _?(sin|cos|tan|asin|acos|atan|atan2|exp|exp2|expm1|log|log2|log10|log1p|pow|cbrt|hypot|sinh|cosh|tanh|fma|sincos|sinf|cosf|tanf|asinf|acosf|atanf|atan2f|expf|exp2f|logf|log2f|log10f|powf|cbrtf|hypotf|fmaf|sincosf|__sincos_stret|__sincosf_stret)$'
if [ "${1:-}" = "--control" ]; then
  found=0
  for rlib in target/release/deps/libvd_physics-*.rlib; do
    if [ ! -f "$rlib" ]; then
      echo "terrain-link-scan --control: missing $rlib (build --release -p vd-physics first)"
      exit 2
    fi
    hits=$(nm -u "$rlib" 2>/dev/null | grep -E "$pattern" || true)
    if [ -n "$hits" ]; then
      found=1
      echo "terrain-link-scan --control: $rlib names platform math, as it must:"
      echo "$hits" | head -5
    fi
  done
  if [ "$found" -ne 1 ]; then
    echo "terrain-link-scan --control: FAIL — the scan found no platform math in the motion crate, so it can find nothing anywhere"
    exit 1
  fi
  echo "terrain-link-scan --control: PASS (the scan can go red)"
  exit 0
fi
found=0
for rlib in target/release/deps/libvd_terrain-*.rlib target/release/deps/libvd_seed-*.rlib; do
  if [ ! -f "$rlib" ]; then
    echo "terrain-link-scan: missing $rlib (build --release first)"
    exit 2
  fi
  hits=$(nm -u "$rlib" 2>/dev/null | grep -E "$pattern" || true)
  if [ -n "$hits" ]; then
    echo "terrain-link-scan: FAIL — $rlib names platform math:"
    echo "$hits"
    found=1
  else
    echo "terrain-link-scan: $rlib names no platform math symbol"
  fi
done
if [ "$found" -ne 0 ]; then
  exit 1
fi
echo "terrain-link-scan: PASS"
