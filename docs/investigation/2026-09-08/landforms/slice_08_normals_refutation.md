# Slice 8, the packed normal — the refutation and its answers (2026-09-10)

The refuter (read-only, Opus) attacked the packed normal under ruling V18: the octahedral code,
the per-rung form, the shaders, the ball, the gate. Ten findings; every one answered below, and
the gate chain runs on the answered tree.

| # | Finding | Verdict | Answer |
|---|---|---|---|
| N-1 | `sine_error` guarded a far-side candidate (`dot < 0`) that no candidate can reach — 302 744 modelled inputs never took it — a red coverage gate in a Tier-A crate. | CONFIRMED | The guard is gone; the doc names why no candidate can be far. |
| N-2 | The encoder ran on the main thread in the harvest loop (four candidates a vertex, each a decode and a cross product), against the P-8 rule that moved the bounds to the worker; and no moving-eye flight measured it. | CONFIRMED (placement); PLAUSIBLE (cost) | The worker packs: `ChunkGeometry::packed_normals`, built beside the normals and copied onto the skirts; `mesh_of` inserts the worker's bytes and packs nothing (the flat dev look alone packs on the main thread, by its nature). The moving eye flies in the final chain and its harvest cost is read. |
| N-3 | The zero-normal test's comment claimed 48 black-speck pixels; §18.2 in the same change says those pixels were the overlay's tick readout. One statement was false. | CONFIRMED | The comment now says what was measured (the readout) and what was not (no zero normal ever). |
| N-4 | A zero normal draws dark on the exact path and lit under the packed one; the guard test sampled four rungs. | CONFIRMED | The test walks every rung of the home planet at four columns and asserts the packed vector's length too. The encoder's fold of a zero to the square's centre stays, documented as a vertex no triangle draws. |
| N-5 | `assert!(widest <= widest_plain)` could not fail: the plain pair is one of the four candidates. | CONFIRMED | Strict now: the precise search wins on the chunk (4.2e-5 against 6.3e-5 rad). |
| N-6 | `oct_decode` divides by 32 767 with no clamp; the GPU reads −32 768 as −1. The encoder never emits it, but the public function did not hold its doc's contract. | CONFIRMED asymmetry | The decode clamps at −1, the GPU's own rule. |
| N-7 | `EXACT_NORMAL_RUNG` was a bare constant, against the one-config-struct convention. | PLAUSIBLE | `TerrainConfig::exact_normal_rung`, default `EXACT_NORMAL_RUNG`, with its measurement. |
| N-8 | Two silent fallbacks (`Vec::new()`, `vec![[0, 0]; count]`) hid a missing attribute. | CONFIRMED | Both panic with the reason. |
| N-9 | `mesh_of` cloned, inserted, read back, packed and removed the exact normals for every packed chunk. | CONFIRMED | Gone with N-2: the worker's packed bytes are inserted directly. |
| N-10 | The gate fell back to the previous run's picture when a stand had no frozen exact reference, so two lossy steps could ratchet; a size mismatch skipped the compare. | CONFIRMED | A missing reference or a size mismatch is a red gate with a message; every stand has its frozen exact picture. |

**Sound, per the refuter:** the fold and its inverse at the boundary, the axes, the diagonals
and the poles; the WGSL `select` order and the −0.0 case; the four-candidate box (0 of 20 000
normals where a wider box wins); the Bevy 0.18 names; extra attributes ignored by the standard
material; the shadow pass compiling with the define; the probe mask untouched by the normal;
`mesh_bytes` right for both forms; attribute alignment; the flat path's duplication; the exact
references untouched.
