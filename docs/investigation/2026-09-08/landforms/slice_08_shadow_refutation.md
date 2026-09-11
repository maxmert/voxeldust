# Slice 8, the shadow ladder — the refutation and its answers (2026-09-11)

The refuter (read-only, Opus) attacked the shadow ladder (option A) and the instruments around it.
Nine findings; every one answered below, and the gate chain runs on the answered tree.

| # | Finding | Verdict | Answer |
|---|---|---|---|
| 1 | A coarse caster and a drawn chunk shared ONE residency in the lane: a key built as a caster reads resident, so when the ladder came to want it (a crossfade band) it was never requested — the coarse rung drew nothing under the band and the instruments (which read residency) saw no gap. | CONFIRMED | The invariant is stated and enforced (`want_caster`/`unwant_caster`, `convert`): a key the ladder draws casts itself while a finer chunk wants it and is a non-caster otherwise; a key held as a caster that the ladder comes to want is dropped, released and rebuilt as a drawn chunk; the picture gate's stands cannot see a band, so the moving eye is the measurement. |
| 2 | The caster's last unwant released the key from the lane while its DRAWN entity stayed; the rebuild replaced the `Drawn` record and leaked the old entity, its twin, its bytes and its counts for ever. | CONFIRMED | `unwant_caster` never releases a drawn key (it restores `NotShadowCaster` on it); a `Drawn` replaced in the map now despawns the old entity and twin and returns its counts, bytes and caster's want. |
| 3 | The want leaked: a count with no caster and no job; the design's fallback ("a key both want is drawn and casts itself") was not in the code — a drawn chunk got `NotShadowCaster` whenever it had asked for a caster, even when it was wanted as one. | CONFIRMED | A drawn chunk casts itself iff its key is wanted (`casts_itself`); the no-body path forgets the key's wants; the request fires whenever a want begins and no caster stands. |
| 4 | The caster's culling box did not cover its own sink at any step above one. | CONFIRMED | The box grows by the caster's sink (`caster_sink_m`, shared with the material). |
| 5 | The freeze and the report-only modes were not exclusive: a look-grid picture could be frozen as the gate's reference. | CONFIRMED | A freeze asserts the gate's own grid (neither the report-only nor the census-only switch set). |
| 6 | A caster outranked every margin chunk of its rung (class 2, index 0). | CONFIRMED, low | The caster's priority carries the widest index: behind every margin chunk of its rung. |
| 7 | Casters counted in the stamp's nearest and farthest chunk. | CONFIRMED | A `ShadowCaster` marker; the placement counts drawn chunks only. |
| 8 | DEFERRED 18 and §19.10 described the first form (from rung 1, two rungs, 44.8) while the code ships from rung 0, one rung (56.2, 221 MB). | CONFIRMED | Both say what ships and what it costs. |
| 9 | `parent_keys` lost its doc comment to `coarse_key`. | CONFIRMED | Back in place. |

**Sound, per the refuter:** `coarse_key`'s halving in every axis and its bounds, its test's reach
over every arm; the pipeline key through `ExtendedMaterial` (the define fires in the shadow pass);
the shader's define combinations; the sink's sign and units; the sun's layers and the cameras'
blindness to layer 2; the caster's own bookkeeping edges; the boarding loop's geometry (the
berths lie 150° off the hull's flight path, 160 m clear of each other).

**What the pictures cannot see, per the refuter and agreed:** with the ring at the eye's feet cast
by a 2 m mesh, a metre-wide rock, a step or a doorway's edge loses its shadow there; the four
stands hold no such feature. A stand with one is owed to the picture gate.
