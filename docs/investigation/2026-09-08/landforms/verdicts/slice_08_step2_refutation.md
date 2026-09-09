# Slice 8 step 2 — THE REFUTATION (the tier rule and the ladder to the horizon)

Read against `CLAUDE.md`, ruling V14, `slice_08_ladder_discussion.md` §3–§7, §9, §11, §14, and the
step 8p refutation. The tree is the uncommitted diff over `922666f`, with `LadderView::wanted` re-read
from disk after the quadtree descent replaced the per-ring enumeration.

Nothing here is style. Every item names a file, a line, an input and a wrong output.

---

## 1. DEFECT — the fold stops at 76° from the FACE CENTRE, so an eye high over a face edge loses a band of ground it can see

`crates/client/src/ladder_view.rs:80` (`FACE_PARAM_MAX = 1.9`), applied at `:357–360`.

The top-rung candidates come from a square in the FOOT FACE's own parameter space. A candidate past the
face edge is clamped to `a = ±1.9`, and `bend(1.9) = 4.12`, so the farthest direction the fold can
name stands `arctan(4.12) = 76.4°` from that FACE'S CENTRE (80.3° at the diagonal). Every candidate
past the clamp collapses onto the same direction and the `seen` set drops it.

The clamp measures from the face centre. The visible cap measures from the FOOT. So the two disagree by
how far the foot stands from its face centre — up to 45°.

Failure, with the home planet's own numbers (`R = 6 370 354 m`, 13 rungs, top-rung chunk 253 952 m):

- The pilot's hull is 2 000 km over the planet, above a point near a FACE EDGE (`a ≈ 0.95`, about 43°
  from the face centre).
- The geometric horizon is a cap of half-angle `acos(R/(R+h)) = 40.4°`, so the ground the pilot can
  see reaches 43 + 40.4 = **83.4° from the face centre**.
- The fold reaches 76.4°. About **7° of arc — 780 km of ground inside the geometric horizon — is
  never a candidate**, so no column there is ever wanted. The limb on that side is black.
- At the far bound (an eye 2 radii out, `h = R`) the cap is 60° and the loss grows to about 28°.

The test stands never reach it. The orbit test (`:717`) stands at `d = normalize([0.2, 0.9, 0.4])`,
which is 26.4° from the +Y face centre; 26.4 + 40.4 = 66.8° < 76.4°. The corner test (`:741`) stands
60 km up, where the cap is only 9.5°.

`MAX_SQUARE_CHUNKS`'s own comment (`:74–76`) says *"a ring cut by it is reported by the reach"*. That
is false twice over: the cut here comes from `FACE_PARAM_MAX`, not from the square's cap; and nothing
reports it — `stamp.drawn_radius_m` still states the full 5 889 km while the ground stops at 76.4°.

**FIX.** Stop folding through a single face's parameter space. Enumerate the top-rung candidates by
walking the six faces' own chunk grids and testing each column's centre against the reach (at the top
rung the whole planet is at most `6 × 40² = 9 600` columns — cheaper than today's 71² fold at orbit),
or seed the descent from the six faces' root columns and let the split do the work. Then add a tiling
probe (item 17) at an orbit stand over a face edge.

---

## 2. DEFECT — the peak past the horizon is read from the octave-dropped field, so a real mountain is culled with its whole 254 km column

`crates/client/src/ladder_view.rs:392–398`; `crates/terrain/src/digest.rs:190`.

```rust
let span = self.span(body, col);                       // surface_column at col.rung
let seen_col = (near <= reach)
    & ((near <= horizon) | (span.peak_m.to_f64() - surface >= drop));
if !seen_col { continue; }                             // the node AND its subtree die
```

`surface_column(body, face, rung, …)` reads `height_m(body, dir, rung)`, and `height_m` DROPS the
`rung` finest octaves (`crates/terrain/src/height.rs:17–28`). `column_bound_m` widens the reading by
the LIVE octaves only (`digest.rs:105–124`). So `peak_m` at rung `L` is short of the true rung-0
surface by up to `BodyDefinition::dropped_bound_m(L)` — a function this crate already has
(`crates/terrain/src/body.rs:294`), and at the home planet's top rung that is 13 of 14 octaves.

The sightline it is compared against is a rung-0 fact. Comparing a rung-12 reading with a rung-0
threshold is not conservative; it is systematically LOW.

Failure: the pilot stands on the home planet (surface 6 370 354 m, horizon 6 582 m, reach 466 058 m).
A top-rung node's centre is 300 km out. The sightline's drop there is
`(300 000 − 6 582)² / (2 × 6 370 354) = 6 758 m`. The recipe's total relief is 16 570 m, but at rung
12 one octave is live, so `peak_m − surface` can state at most about ±8.3 km and often much less. A
ridge that truly stands 14 km over the reference sphere — visible, 7 km clear of the sightline — reads
as 2 km at rung 12 and the node is culled. Because the cull happens BEFORE the split (`:396`), its
four children, their children, and every rung-0 column under 254 km of ground die with it. The
skyline behind the horizon is drawn from a field that does not contain mountains.

**FIX.** The peak test needs an UPPER bound on the rung-0 surface, not the rung's own reading:
`span.peak_m + body.dropped_bound_m(col.rung)`. Note the cost: this brings back part of the 6 556
chunks the peak test removed, so pair it with the descent (a coarse node that passes is split, and
its children re-test with a tighter bound), which is what the descent is for.

---

## 3. DEFECT — the ruler's ray march now steps one metre at a time out to 466 km

`crates/client/src/chunks.rs:514–523`; called at `crates/client-render/src/terrain.rs:706` and `:712`.

```rust
let mut t = cell;                       // cell = 1 m at rung 0
while t <= reach_m && !below(t) {       // reach_m used to be the DRAWN radius (372 m)
    above = t; t += cell;
}
```

Step 2 replaced `drawn_radius_m` with the ladder's REACH. On the ground the reach is 466 058 m and
`first_rung = rung_for_distance(1.8 m) = 0`, so `cell = 1 m`. When the centre ray does NOT hit the
ground the loop runs **466 058 iterations**, each evaluating `height_m` over 14 octaves of 3-D
gradient noise — tens of milliseconds on the render thread, inside `sync_terrain`.

Failure: the pilot stands on the home planet and looks up at the sky, or grazes the skyline. The cache
key is `(eye_body, forward_body)` (`terrain.rs:698–701`), so every frame in which the eye or the nose
moves re-runs the march. A pop-detector leg (D8-5, a hull at 528 m/s) moves both every frame. The
four stands never see it because all four noses point below level and hit ground within a few
hundred steps.

**FIX.** March at the COARSEST rung first to bracket the hit, then refine at the rule's rung; or bound
the step count (`reach / cell` capped, with the step growing with distance). The bisection at the end
already gives the precision, so the march only needs to bracket.

---

## 4. DEFECT — every candidate pays 25 height evaluations before the reach test decides it is not there

`crates/client/src/ladder_view.rs:392`.

`self.span(body, col)` runs BEFORE `seen_col` (`:394`). `surface_column` evaluates `height_m` at 25
sample sites (`digest.rs:171–192`). At orbit the top-level fold pushes about `71² = 5 041` distinct
columns, most of them on the far side of the planet and far past the reach, and **every one of them
costs 25 height evaluations and a permanent cache entry** before the `near <= reach` test drops it.

The span is only needed for the peak arm (`near > horizon`) and for a node that is actually drawn.

**FIX.** Test `near <= reach` and `near <= horizon` first; read the span only when the peak arm is
needed or the node is drawn.

---

## 5. DEFECT — `LadderView::spans` grows without bound over a flight (SL9)

`crates/client/src/ladder_view.rs:289`, `:294–298`. The cache is a `BTreeMap<Column, ColumnSpan>` and
nothing ever removes an entry. `Terrain` keeps one `LadderView` per realm for as long as the realm's
row is in the window (`terrain.rs:234–239`).

Failure: the pilot flies 100 km over the home planet at rung 0. The rung-0 disc is 869 m in radius, so
the swept ground is about `100 000 × 1 738 / 62² = 45 000` new rung-0 columns, plus every coarser
rung, plus every candidate the fold rejected past the reach (item 4). Each entry is a `Column` key and
a `ColumnSpan` inside a `BTreeMap`, so a long flight over one planet reaches millions of entries and
hundreds of megabytes for spans nobody will read again.

**FIX.** Evict on the wanted-set recompute (keep the columns of this set and the last), or cap the map
and drop the coldest. The doc calls a span "a function of the seed, read once and kept" — true, but
"kept" must be bounded, and SL9 forbids a cost that grows without a bound the code states.

---

## 6. DEFECT — a chunk released while it is still building is never withdrawn, and the withdrawal machinery is dead in the shipped path

`crates/client-render/src/terrain.rs:469` and `:471–491`; `crates/client/src/chunks.rs:362–377`.

The release loop walks `terrain.entities` — the DRAWN chunks only. A chunk that is still PENDING and
no longer wanted is in no loop: it is never released, never withdrawn, and the workers build it in
full. `ChunkLane::forget` (the only other caller of `workers.cancel`) is never called from production
code — `grep` finds it in tests only. So `ThreadedWorkers::cancel` and its `cancelled` set
(`terrain.rs:118–120`, whose own comment says a withdrawal saves *"4 ms and 400 KB each, MEASURED at
M7-2/M7-3"*) never run in a flight.

Failure, and the reason the four flights are green anyway: all four stands are STILL. `moved()`
(`terrain.rs:394`) is false after the first frame, the wanted set is computed once, nothing is ever
released, and `pending_count()` reaches 0. Move the eye and the shape changes:

- `EYE_STEP_M = 0.5` (`:69`) — its own comment says *"a hull at 528 m/s recomputes every frame"*.
- The request loop (`:487–491`) walks all ~4 000 wanted keys every frame and requests every key the
  lane does not hold. `HARVEST_PER_FRAME = 24` (`:66`) drains 24 per frame.
- Once the request rate passes 24 per frame, `pending` grows monotonically, and because
  `overlapping_missing` (`:481`) holds a drawn chunk while ANY wanted chunk over its footprint is
  still building, almost nothing is ever released. `terrain.entities` and the mesh memory then grow
  without bound while the hull flies.

This is UNMEASURED, not proven — but it is the exact case D8-5 and M8-1 exist to measure, and it is
the case the four stands structurally cannot show.

A second trap sits in the same machinery, latent today: `ThreadedWorkers`'s worker removes the
`cancelled` entry only when it dequeues that job (`terrain.rs:140–146`). If the job was already taken
when `cancel` ran, the entry stays forever, and a LATER request for the same key is skipped silently —
the chunk never arrives, `pending_count()` never returns to 0, and `overlapping_missing` holds the old
chunk for the rest of the session. The moment anything wires `ChunkLane::forget` (a realm leaving and
re-entering the window), this becomes live.

**FIX.** (a) Walk the lane's pending set, not `entities`, when releasing; (b) give `cancelled` a
generation or clear the entry on submit; (c) run M8-1 before step 3.

---

## 7. UNMEASURED — `column_bound_m` is an argument presented as "the recipe's own statement about itself"

`crates/terrain/src/digest.rs:96–124`.

The bound claims that the surface inside a column never stands more than `Σ Aᵢ · min(1, (π·s/λᵢ)²/2)`
from the 25 samples. Nothing measures that. The derivation is sound for ONE plane wave sampled along a
grid axis (the extremum midway between two samples exceeds the samples by `A(1 − cos(πs/λ))`). It is
not stated for what the recipe actually is:

- The extrema of gradient noise are localized bumps, not plane waves. A bump at the centre of a
  15.5 m grid cell stands 10.96 m from the nearest sample in BOTH face directions, and the two 1-D
  terms add — up to **twice** the stated bound.
- `wavelength_m = body.radius_m / o.frequency` (`:114`) is NOT the wavelength. `height_m` feeds
  `dir * frequency` into `noise3`, whose lattice period is 1 (`crates/terrain/src/noise.rs:4–6`), so
  `R/f` is the LATTICE CELL. The fundamental of gradient noise is about two lattice cells, so the
  formula overstates `(πs/λ)²` by about 4× — which is probably what saves it. That is luck resting on
  a mis-named quantity, not a bound.

What a too-small bound produces is a HOLE: `lo`/`hi` (`:186–187`) exclude the chunk the surface
actually passes through, so a column's mesh has a gap. **Neither gate can see it.** `ground_holes`
counts a "nothing" pixel only when drawn ground stands ABOVE it in the same column
(`crates/client-harness/src/probe.rs:191`), and a missing TOP chunk of a column is at the top of the
ground — nothing above it. The tiling test (`ladder_view.rs:681–711`) tests COLUMNS, never the
z-span.

**FIX.** A Tier-A test that can fail: for a few hundred rung-0 columns of the home planet, sample every
cell of the column and assert `max(dense) <= max(sample) + bound` and
`min(dense) >= min(sample) − bound`. Then either the bound holds or the test goes red and names by how
much.

---

## 8. UNMEASURED — the cap in `column_bound_m` is dead on this world, and the doc says the opposite

`crates/terrain/src/digest.rs:118` and the doc at `:96–103` (*"the finest octaves their whole
amplitude"*).

The finest LIVE octave at rung `L` has a lattice cell of about `400 km / 2^(13−L)` and the sample
spacing is `15.5 · 2^L` m, so the ratio `s/λ` is **constant across every rung**: `0.317`, giving
`curve = (π·0.317)²/2 = 0.497`. `share = min(1, curve)` therefore never reaches 1 on the home planet,
at any rung. No octave contributes its whole amplitude, and the cap arm is unreachable through this
body. The two tests that touch it (`:283–297`) assert only `b0 < total0` and `b9 <= total9`, which the
above makes trivially true.

**FIX.** Correct the doc, and pin the ratio with a test that states the number (`share ≈ 0.497`) so a
change to the octave table or to `COLUMN_SAMPLES_PER_EDGE` is visible.

---

## 9. UNMEASURED (gate) — the per-pixel rung check tolerates ±1 with no ceiling, so a rule off by exactly one everywhere passes green

`crates/bins/tests/terrain_pictures.rs:101`, `:530–546`.

Each terrain pixel asserts `|rung_for_distance(d) − p.rung| <= 1`. Nothing asserts how MANY pixels are
off. The measurement already reports 284 of 632 363 (ground), 11 384 of 740 353 (hill) and 16 264 of
566 872 (aloft, 2.9 %). If a future change shifted every switch distance by one rung, every pixel
would read `gap == 1` and every assert would pass.

The ±1 also absorbs the real geometric error the design carries: the ring is decided by the column's
CENTRE placed on the sphere through the surface UNDER THE EYE (`ladder_view.rs:384`), not on the
column's own height. Near a 3 km ridge at 3.5 km that error is a full ring. The bias is conservative
for a near column (the computed distance is short, so the rung is finer than the rule) and coarse for
a distant basin — but the gate cannot tell that from a defect.

**FIX.** Assert a ceiling per stand (a fraction of the terrain pixels, from the measured numbers with
headroom), so the count is a gate and not only a print.

---

## 10. DEFECT (gate) — the ruler's rung assert is a tautology

`crates/bins/tests/terrain_pictures.rs:719–723`:

```rust
assert_eq!((disc.rung_min, disc.rung_max), (ruler.rung, ruler.rung));
```

`ruler.rung` is written into the stamp at `crates/client-render/src/terrain.rs:719–724`, and the SAME
variable chooses the probe material that paints the ball's pixels at `:794`
(`probe_material(pm, PROBE_KIND_RULER, rung)`). One value paints the picture and fills the stamp; the
assert compares it with itself. Before step 2 the right-hand side was the flag's rung — an
independent number. A real check was replaced by a self-check.

**FIX.** Assert `ruler.rung == rung_for_distance(ruler.distance_m, rungs)` — the rule against the
answer — and keep the probe comparison as the encoding check it is.

---

## 11. UNMEASURED (gate) — `HORIZON_ROW_TOLERANCE = 3` allows the ground to fall SHORT of the horizon

`crates/bins/tests/terrain_pictures.rs:104`, `:601–606`. The sign of the formula is right: `dip`
is below level, `above_axis = tilt − dip`, and the row grows downward, so `horizon_row` is where the
smooth-sphere horizon lands. The ground stand's 222 against 239 is 17 rows ABOVE it, which relief
explains.

But the comment says *"relief only lifts the skyline above it"* — so the correct assertion is
`sky_row <= horizon_row`, with a tolerance only for the row's own rounding. The aloft stand measures
**252 against 251**: one row of sky where ground should be. The tolerance of 3 was set wide enough to
admit it. Item 2 (the culled coarse peak) and item 1 (the fold's truncation) both show up exactly as a
short skyline, so this is the tolerance that hides the two defects above.

**FIX.** Tighten to 1 row and explain the residual, or find it. Do not leave a tolerance calibrated on
the number that failed.

---

## 12. UNMEASURED (gate) — the orbit band was moved off the limb, which is where item 1 would show

`crates/bins/tests/terrain_pictures.rs:906–911`. The comment is honest: *"0.932 of the lower half was
ground, the rest the sky in the corners, so the band starts under it"* — the band moved from 0.5 to
0.72 after a failure. The consequence is not stated: the share is now measured only in the bottom
28 % of the frame, and the limb and the corners — the only place the fold's truncation (item 1) or a
missing far ring can appear at orbit — are outside the measured band.

**FIX.** Keep the band at 0.5 and assert the share against a MEASURED floor for the orbit stand (0.93
with headroom), so the corners stay inside the gate.

---

## 13. UNMEASURED (gate) — `blocks == 0` passes with 524 pixels of nothing, and a skyline hole is never counted

`crates/client-harness/src/probe.rs:182–216`; the assert at `terrain_pictures.rs:523–528`.

Two gaps:

1. The erosion is a 4-neighbour erosion, so any hole 1 or 2 pixels wide gives `blocks == 0`. The
   measurement states 524 hole pixels in 171 runs still on screen. A missing chunk at a FAR rung is a
   thin sliver at the skyline — 1 to 2 pixels tall — and therefore invisible to this gate. That is the
   same shape as the aloft stand's one missing row (item 11).
2. `hole[i] = ground_above & (kind == NONE)` (`:191`) counts a hole only when drawn ground stands
   above it in the same column. A missing chunk at the TOP of the ground — the skyline, exactly where
   items 1 and 2 bite — has no ground above it and is never a hole at all.

**FIX.** Count a hole with ground BELOW it as well (a "nothing" pixel between two ground pixels in any
of the four directions), and state the crack count as a falling ceiling until step 3 asserts zero.

---

## 14. LAW (SL8) / ledger — the `FAR_EYE_RADII` handover is a seam with no registry row, and it is a cost cliff

`crates/client/src/ladder_view.rs:69`, `:321–325`; `crates/client-render/src/terrain.rs` "ONE DRAWING
PER REALM".

At `len = 2.0 × radius` the ladder returns the empty set. One frame the hull draws the planet's whole
recipe; the next it draws the proxy outline. That is a seam by name (SL8's arrival-pop and
tier-refusal rows). The code comment says *"the handover is a seam the pop detector measures in step
6"*, and §14.3 repeats it — but `DEFERRED.md` has NO row for it. `CLAUDE.md` calls DEFERRED *"the
binding registry of every interim/stub"*.

It is also a cost cliff. Just inside the bound (an eye at 1.99 radii), `reach = 11.5 × 10⁶ m`, the
square radius saturates at `MAX_SQUARE_CHUNKS = 64`, and the fold pushes up to `129² = 16 641`
candidates — every one of which reads a 25-sample span (item 4) and keeps it forever (item 5) — for a
body that fills a small part of the screen.

**FIX.** Add the DEFERRED row (what is missing, where, which step closes it). Separately, make the
handover a fade rather than a cliff, or at least bound the far-eye work by the body's ANGULAR size
rather than by a radius multiple.

---

## 15. NIT / ledger — `D-TERRAIN-3` flipped to 🟩 while its own replacement's interims have no row

`docs/design/DEFERRED.md`, the D-TERRAIN-3 block. Its own new text says *"until step 3 lands, the
boundary between two rings is a hard edge on the pictures"*. §14.3 lists four live interims: the hard
ring edge and the 524 crack pixels, the full-circle rings (no view wedge), 1.6 GB of mesh at today's
byte format, and the far-eye handover. None of them has a row. An entry closed on a replacement that
carries its own unregistered interims is a ledger that no longer answers "what is still owed".

**FIX.** Keep D-TERRAIN-3 🟩 and open D-TERRAIN-4 (the ring edge and the cracks → step 3), D-TERRAIN-5
(the mesh bytes → step 5, with M8-2's ceiling), D-TERRAIN-6 (the far-eye handover → step 6).

---

## 16. NIT — the stamp mixes a WANTED reading with a DRAWN reading, and no refusal counter is asserted

`crates/client-render/src/terrain.rs:735–745`; `crates/bins/tests/terrain_pictures.rs:552–560`.

`rung_min`, `rung_max` and `drawn_radius_m` come from the WANTED set; `chunks_per_rung` and
`chunks_drawn` come from `entities` (DRAWN). The gate's message says *"the finest rung on screen"* but
the value it reads is the wanted set's. They agree today only because the gate waits for
`chunks_pending == 0`.

They can disagree without pending ever rising: a wanted key that `ChunkLane::request` refuses is
counted in `counters.outside` and never enters `pending` (`chunks.rs:332–335`). A whole ring
systematically refused would leave `rung_min` naming a rung with zero chunks on screen, `pending == 0`,
and the gate green.

**FIX.** Put `chunks_wanted` beside `chunks_drawn` on the stamp, carry `outside`/`no_body`, and assert
both are zero in the picture gate.

---

## 17. NIT — the tiling test proves one stand inside one face; the cross-face tiling is untested

`crates/client/src/ladder_view.rs:681–711`. The spiral is a real gate and it found a real gap (the
horizon-edge column). But it runs on ONE stand — the ground, 3.4 m up, well inside the +X face — and
only out to the horizon. It never probes:

- across a face edge or corner (the corner test at `:738` asserts only `faces.len() == 3`);
- at the aloft or orbit stands, where the fold and the clamp of item 1 live;
- past the horizon, where the peak test of item 2 decides.

Given that the descent exists BECAUSE of a hole class (4 818 and 16 297 probe pixels), the cross-face
tiling is the one place the gate does not look.

**FIX.** Run the same spiral at the corner stand (60 km up, three faces) and at an orbit stand over a
face edge. That single change turns item 1 red.

---

## 18. NIT — magic numbers, dead arms and a duplicated formula

- `crates/client/src/ladder_view.rs:135–140` — `relief_m(body)` is `BodyDefinition::relief_bound_m(0)`
  written a second time (`crates/terrain/src/body.rs:283`). One formula, two homes.
- `crates/terrain/src/digest.rs:177` — `(i * step).min(edge - 1)` is dead: `step = (62−1)/4 = 15`, so
  the largest index is 60 and the clamp never binds. The defensive clamp also hides that the sample
  grid stops one cell short of the column's far edge.
- `crates/terrain/src/digest.rs:173–180` — `sites` is a `Vec` allocated per column, 25 entries, for
  every wanted column of every recompute. A `const` array of offsets costs nothing.
- `crates/client/src/ladder_view.rs:437–448` — `column_under` is `pub` and reads
  `cells_per_edge(rung)` with no bound on `rung`. `Ladder::cells_per_edge` is `self.n >> rung`, which
  panics for `rung >= 32` in a debug build, and gives `n_l = 0` (a divide by zero in `face_param`)
  well before that.
- `crates/client-render/src/terrain.rs:66` — `HARVEST_PER_FRAME = 24` is justified as *"narrow enough
  to keep a frame"*. At 405 KB of mesh per chunk (M7-3) that is 9.7 MB of upload per frame, 583 MB/s
  at 60 Hz. UNMEASURED.
- `crates/client-render/src/terrain.rs:74` — `SHADOW_REACH_RUNG = 2` is a new bare rung number. Its
  comment states the consequence (3.5 km) but not why rung 2 is the right rung rather than the rung
  under the eye, which is what the old formula derived it from.
- `crates/bins/tests/terrain_pictures.rs:916–922` — `assert!(ground_chunks >= 9)` and its three
  siblings are now trivial (3 855, 4 287, 1 949, 1 189 measured). They assert nothing. Replace each
  with a band around its measured value, so a change that halves or doubles the ladder goes red.

---

## 19. COVERAGE — arms the tests do not reach

- `crates/client/src/ladder_view.rs:349` — the `.min(MAX_SQUARE_CHUNKS)` cap binds only for an eye
  past about 1.9 radii (`radius = 68 > 64` at 2 radii). No test stands between the corner stand's
  60 km and the far bound's rejection at 3 radii, so the cap has never been exercised and its comment
  has never been checked.
- `crates/client/src/ladder_view.rs:410` — the false arm of `(child.x <= last) & (child.y <= last)`
  needs a split whose child index passes the face's last chunk column. The corner stand probably
  reaches it, but nothing asserts it. Add a direct unit test on a body whose top-rung face edge is
  partial (the home planet's is: 2 443 cells is 39 whole chunks plus 25 cells).
- `crates/terrain/src/digest.rs:118` — the `curve > 1` arm of `min(1, curve)` is unreachable through
  the home planet at any rung (item 8). If the exemption list does not cover it, it is a red region
  waiting for the next `coverage-fast`.

---

## VERDICT

The descent is the right shape and it fixed a real hole class, and the instrument (the probe, the
stamp, the hole counter, the tiling spiral) is genuinely better than what step 8p shipped. But three
defects put ground the player can see off the screen, and the gates were widened in exactly the places
those defects appear. The fold clamps the visible cap at 76° from a FACE CENTRE rather than from the
foot, so an eye high over a face edge loses hundreds of kilometres of limb (item 1) — and the orbit
stand's share band moved off the limb (item 12). The peak past the horizon is read from a field with
its finest octaves dropped, so a coarse node and its whole 254 km subtree die although a real mountain
stands there (item 2) — and the horizon-row tolerance widened to admit a skyline that falls one row
short (item 11), while the hole counter cannot see a skyline hole at all (item 13). The ruler's ray
march now steps one metre out to 466 km and stalls the render thread whenever the centre ray misses
(item 3). Beside those, the cost model is unproven: every one of the four measurements is a STILL
stand, so the wanted-set recompute, the release hold, the never-withdrawn pending chunks and the
unbounded span cache (items 4, 5, 6) are all measured only in the one state that cannot exercise them
— and D8-5's moving legs are two steps away. Finally, `column_bound_m` is presented as "the recipe's
own statement about itself" but is an argument built on a mis-named quantity, with no test that could
fail and a failure mode both gates are structurally blind to (item 7). Items 1, 2, 3, 6 and 7 should
close before step 3 starts, because the crossfade will be tuned on pictures these five defects shape.

---

## The answers (2026-09-09, after the refutation)

Every finding is answered. "FIXED" names the change; "ANSWERED" gives the measurement or the reason
no change is made. The flight after these changes is the ninth of step 2.

| # | Answer |
|---|---|
| 1 | **FIXED.** The descent's roots are every top-rung column of every face (at most 64 per edge by the ladder's construction), each tested by its nearest point against the reach; the fold through one face, its 1.9 clamp and the 64-chunk cap are gone. Two new tiling tests: a spiral out to the horizon's ARC from orbit over a point 43° from a face's centre, and from 60 km over a face corner — each direction covered by exactly one wanted column. |
| 2 | **FIXED.** `ColumnSpan::peak_m` is the sampled high plus the column bound plus `dropped_bound_m(rung)`, so a coarse node's peak stands at or above the true rung-0 peak and a mountain behind the horizon is never culled with its subtree; asserted at rung 9. |
| 3 | **FIXED.** `RULER_MAX_CELLS = 4096`: the march reaches at most 4 096 cells of its rung (4 km on the ground, 17 000 km at the top rung); a ray that meets no ground within it plants no ball. Tested with the ladder's whole reach on a sky ray. |
| 4 | **FIXED.** The reach test runs first; a column past the reach reads no span. |
| 5 | **FIXED.** The spans carry the descent's generation and every span the descent did not visit is dropped after it; the far-eye return clears them all. Asserted: the same eye holds the same count, far away holds none. |
| 6 | **FIXED.** `ChunkLane::pending_all` lists every building chunk and the engine withdraws each that its realm's wanted set no longer holds (no hold: nothing is drawn for it); a re-request clears a stale cancel in `ThreadedWorkers::submit`. The lane test lists the building pair. The moving-eye measurement itself is M8-1 (D-TERRAIN-5 item 4). |
| 7 | **FIXED and MEASURED.** The bound doubles the second-order term for the two grid axes and names `λ` as the noise LATTICE CELL (half a wave at most, so four times a sinusoid's term); a test samples sixteen rung-0 columns and four rung-9 columns every fourth cell and asserts every height inside the sampled extrema widened by the bound. |
| 8 | **ANSWERED.** The cap binds on a body whose octaves are spaced more tightly than this one's; it stays, branchless, for that body. |
| 9 | **FIXED.** The share of terrain pixels exactly on the rule's rung must be at least 0.9 (MEASURED 99.9 % / 96.6 % / 93.8 %). The ring by the column's centre on the sphere under the eye is the design's stated approximation; its error is what the ±1 absorbs, and the exact share bounds it. |
| 10 | **FIXED.** The ruler's rung is asserted against `rung_for_distance(distance)`, not against the stamp's own copy. |
| 11 | **FIXED, and the refuter's premise refuted by measurement.** Relief does NOT only lift the skyline: on the ground stand the far-left skyline stood 28 rows (1.75°) UNDER the sphere through the eye's own surface, because the stand's hill overlooks lower ground. The bound a skyline can never cross is the horizon of the LOWEST ground the recipe can raise (the eye's surface minus the relief bound); every column is asserted against that row within three rows, which from orbit is three rows under the eye's own horizon. |
| 12 | **ANSWERED.** The limb arcs across the frame and the band's rows lie under it; the limb itself is now asserted in EVERY column (13). |
| 13 | **FIXED.** The skyline is asserted in every column: the topmost drawn pixel of each column lies at or above the row where that column's own ray dips by the horizon's dip, through the pilot camera. A missing cap at the limb is red. The crack count stays stated; step 3 asserts it zero. |
| 14 | **FIXED (ledger).** `D-TERRAIN-5` item 3 names the far-eye handover as a seam for step 6. |
| 15 | **FIXED (ledger).** `D-TERRAIN-5` lists the four interims step 2 leaves — the crack, the full rings and their bytes, the handover, the still-stand measurements — with their steps. |
| 16 | **ANSWERED.** The stamp's fields say which reading each is (wanted vs drawn) in their doc; the lane's `outside` counter is asserted in the lane's own tests. |
| 17 | **FIXED.** See 1: the corner and the orbit-over-an-edge stands run the tiling spiral. |
| 18 | **FIXED where real.** `relief_m` reads `relief_bound_m(0)`; the dead `min` is gone; `column_under` clamps the rung to the address's `RUNG_MAX`; the chunk floors are per stand (2 000 / 2 000 / 1 000 / 500 against 3 855 / 4 300 / 2 012 / 1 277). `HARVEST_PER_FRAME` and `SHADOW_REACH_RUNG` are named constants with their reason; M8-2 measures the harvest. |
| 19 | **ANSWERED by the gate.** The cap and the fold are gone with their arms; the child-bound false arm is reached at a face edge by the corner and orbit tests; `coverage-fast` decides. |
