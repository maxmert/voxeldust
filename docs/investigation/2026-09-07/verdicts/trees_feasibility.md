# Verdict — feasibility refutation of `07_trees_composites.md`

**Lens:** trees, vegetation and composites — technical feasibility.
**Date:** 2026-09-07. **Refuter's rule:** a load-bearing claim that is uncertain counts as refuted.
**Result: REFUTED.** Three load-bearing claims are wrong, and four more carry a decision on an
unmarked or self-contradicting number.

Every number below is marked MEASURED (with the method) or ESTIMATED (with the arithmetic). Every
claim about the code cites a file and a line, read on 2026-09-07 in this worktree.

---

## 1. The three refutations that break the report

### R1 — "derive the whole visible disc" costs the planet's surface, not the pilot's work

The report states two rules together:

- §2.5, line 261: *"Seed objects are derived at every rung."*
- §7, line 481: the arrival-pop seam is prevented because *"seed trees are derived for the whole
  visible disc before the rung is drawn."*

Neither rule names a distance at which a tree stops being an object. So the work grows with the area
of the planet a pilot can see, which grows with the pilot's altitude.

ESTIMATED, by arithmetic on the report's own two numbers (one tree per 100 m², line 153; 2–10 µs per
`expand`, line 156). The visible cap of a body of radius R seen from altitude h has area
`2·pi·R²·h/(R+h)`. For a planet of radius 6 371 km seen from 10 000 km, the cap is 1.56 × 10^14 m².
That holds 1.56 × 10^12 trees. At 2 µs each the client needs 36 days of processor time for one
frame. At 1 nanosecond each it still needs 26 minutes.

The report's own example hides this. §2.5, line 267: *"A pilot at 3 km sees a hillside as a coarse
rung. The client derived the hillside's 60 seed trees."* The rule says the whole visible disc, not
one hillside. ESTIMATED for the same planet at 3 km altitude: the visible cap is 1.2 × 10^11 m²,
which is 1.2 × 10^9 trees, or 40 minutes at 2 µs each. The example counts 60.

This is the cost shape SL9 forbids: the work follows the size of the realm, never the number of
occupants and edits. A moon with no players on it costs the same as a moon with a landing pad.

**Game words.** A pilot pulls a hull up to 10 000 km to look at the planet. The client must expand
every oak on the whole lit face before it draws one rung. The window freezes. The report's answer to
the tick-hitch seam (§7, line 484: *"a per-frame budget"*) does not help, because a per-frame budget
that never finishes IS the arrival pop it was meant to prevent.

**Fix.** State a drawn-object horizon as a physical rule, the same shape as reach: a tree is an
object while its skeleton spans more than one drawn pixel, and below that it is part of the surface
the rung already draws (a canopy tint on the terrain material, no per-object work). Then measure the
tree count at that horizon on a real planet, not on a hillside. Until that rule exists, §7's
arrival-pop row and §2.5's every-rung rule are both unsupported.

### R2 — the no-drift gate tests a tree's SHAPE and never its PLACEMENT

The report's gate, §1.5 lines 137-139 and §9 item 1, feeds `(instance seed, stage, size)` into
`expand` and compares the skeleton bytes on both hosts and both targets. Nothing in the report gates
the question that decides whether a collider exists at all: **which cells hold trees.**

That question runs through §2.1, lines 189-193: draw an anchor cell, a kind and a size class per
feature region, then *"test the anchor's support cell in the post-carver, post-water field. Air or
water below: relocate the draw deterministically."* The report asserts the two hosts agree (§2.1,
line 200: *"both evaluate stage 8 and get the same 214 trees at the same cells"*) and never gates it.

The assertion is unsupported today, and the code says why. MEASURED by grep on 2026-09-07: the one
generator crate calls the platform's transcendental functions at eight production sites —
`crates/physics/src/worldgen/generate.rs:571`, `:598`, `:633` (`.ln()`), `:638` (`.sqrt()`),
`:639` (`.cos()` and `.sin()`), `:807` (`.tan()`), `:1880` (`.powf()`), and
`crates/physics/src/worldgen/scale.rs:572` (`.powf()`). SL10 V1.4 forbids every one of them in a
crate the client links. The report never names this, and §2.1 puts stage 8 inside that same crate.

**Game words.** The moon's shard holds an oak's capsules at cell `(face 3, i 1200, j 880)`. The
pilot's client relocated that draw by one cell, because one comparison in the density field differed
by one unit in the last place. The pilot walks into an invisible trunk, and walks through the oak
that is drawn. That is a phantom miss, which the report's own divergence contract forbids
(§1.3, line 116).

**Fix.** Add a placement arm to the gate: for a fixed list of chunks, the SET of `(cell, kind, size
class)` a host derives must be byte-identical on the server build and the client build, on x86-64
and on aarch64. State that stage 8's draws and the support test are integer-only. Add the eight libm
sites above to the report's owed work, because the tree lives in that crate.

### R3 — an object's dirty set cannot be its anchor cell alone

The report states both of these:

- §4.3 rule 1, line 377: *"An object's expansion depends on its own record only, never on a
  neighbouring cell. Its dirty set is its anchor cell. An edit two cells away changes nothing."*
- §3.1 row 2, line 284, and §4.3 rule 2, line 379: the object falls whole when *"the ground cell
  under the anchor is mined"*.

The ground cell under the anchor IS a neighbouring cell. §1.2, line 65, puts the anchor in the air
cell directly above it. So a terrain edit one cell below an oak must wake that oak, and rule 1 says
it must not. The two rules cannot both hold.

The cost the contradiction hides is a lookup the report never designs: when a miner removes a cell,
the shard must ask "does an object stand on the cell above me?" That is a reverse index from a
ground cell to the object anchored above it. It must be O(1)-ish per edit, because a mining tool
edits many cells per second.

**Game words.** A miner digs a trench under a row of oaks on a moon. Each removed cell must find the
oak that stands on it and turn that oak into a falling group. Under rule 1 as written the shard never
looks, and the oaks stay in the air with their capsules intact.

**Fix.** Restate rule 1 as "an object's expansion depends on its own record only; its SUPPORT set is
its anchor cell and the one cell below". Name the reverse index and put it in the owed measurements
beside `object_placement_validation` (§9 item 7).

---

## 2. Claims that carry a decision on an unmarked or self-contradicting number

### R4 — the collider budget rests on two numbers that differ by 3.6 times

§1.6, line 152: *"collision bytes per tree ~1.5 KB — 16 capsules × ~100 B."*
§1.6, line 161: *"a capsule holds a 10 m limb in 28 bytes."*

100 B and 28 B cannot both be the per-capsule cost. The conclusion those numbers carry is load
bearing: line 160 says *"The collider budget moves from the largest line in the cluster to a
rounding error"*, and §8 stale claim 7 (lines 506-507) retires the base's 6.9–22.5 MB on that
strength. A conclusion stated as structural rests on an estimate that disagrees with itself.

Neither number names an implementation. MEASURED by the report itself (§0, line 30) and confirmed
here by `grep -n "rapier\|parry" Cargo.toml Cargo.lock`: neither `rapier3d` nor `parry3d` is a
dependency, and `noise = "=0.9.0"` is declared at `Cargo.toml:78` but has no `Cargo.lock` entry. The
whole design collides on a "capsule compound" (§1.3, line 111) that exists nowhere, and §11's
register of eight open decisions has no row for it. A new dependency must be an owner option, never
an assumption.

**Fix.** Add an open decision: does the shard collide capsules through `rapier3d`/`parry3d` (an owner
option, not adopted here), or through a hand-written capsule sweep on the `glam` types the workspace
already uses (`DVec3`, `DQuat`, `crates/core/src/geometry.rs:338-348`)? Then measure bytes and query
time per 64 m disc, which §9 item 5 already owes, and delete one of the two per-capsule numbers.

### R5 — the falling tree's collider choice moves the divergence into time, it does not remove it

§11 item 3, line 573, recommends "yes, the falling tree collides", because *"a tree that passes
through a player is a phantom miss, which the divergence contract forbids."*

But §3.4, lines 328-329, puts the topple pose *"on the entity lane through the client's 100–150 ms
interpolation buffer, as any entity does."* `CLAUDE.md:261` confirms that buffer is the standing
rule. So the drawn crown ALWAYS lags the collider by 100–150 ms. Option (a) therefore produces a hit
before the player sees contact, in every single fall.

ESTIMATED from the report's own fall (§3.3 example, line 336: fifty ticks) at the project's 20 Hz
snapshot rate: the fall lasts 2.5 s, and a 20 m crown's tip moves at roughly 16 m/s near landing. A
150 ms lag puts the collider about 2.4 m ahead of the drawn crown.

The report presents option (a) as the one that satisfies the divergence contract. Neither option
does. The honest statement is that a moving collider on the entity lane has a lag the contract does
not cover, and the choice is between a hit the player has not yet seen (a) and a miss the player has
seen (b).

**Fix.** Say so in the register, and add a measurement: the largest gap between the collider pose and
the drawn pose over a fall, on the shipped lane.

### R6 — "no timestamp is stored" and "the chunk's last-ticked stamp" cannot both hold

§2.3, line 220: after dormancy *"the shard draws the elapsed stages from a binomial over the chunk's
last-ticked stamp."*
§2.4, line 239: *"no TIMESTAMP is stored (growth is memoryless)."*

A per-chunk last-ticked stamp is a stored timestamp. The claim that regrowth stores nothing but the
8 B divergence record is false while §2.3 needs that stamp, and the stamp is a per-chunk cost domain
06 must hold for every chunk that ever held a diverged cell.

The binomial itself is unnamed. A pine planted before a year of dormancy has an elapsed count in the
billions of ticks. No deterministic integer sampler for a binomial at that count appears in the
report, and a loop over elapsed ticks is not an option.

**Fix.** Choose one: store a per-chunk stamp and state its byte cost, or make the stage a pure
function of the universe tick and a stored plant tick (lawful for a planted pine, because the plant
tick is state; unlawful for a seed oak, see R12). Name the integer binomial method.

### R7 — the growth lane has no bound, and §7's answer is about pixels, not messages

§2.3, line 221: *"Every stage change is a diff to every subscribed client."*
§2.4, line 245: a regrowing oak *"passes through stages 0..255 and then deletes itself."*

ESTIMATED from the report's own clear-cut count (§2.2, line 212: 19 406 trees in a 786 m disc): one
recovering disc emits 19 406 × 256 = 4.97 million growth diffs.

§7's "re-state rate" row (line 485) answers a different question — it bounds how far a point MOVES
per stage, so the tree does not pop. It says nothing about the message rate. The lane-flood seam of
SL8 stays unanswered for the one mechanism in this report that emits per-object messages forever.

**Fix.** Bound the growth lane: a stage change is a diff only to a subscriber that holds that chunk
at a rung where the step is visible, and a dormant chunk's whole recovery collapses to ONE diff when
a subscriber arrives. Add the measurement.

---

## 3. Wrong claims about today's code

### R8 — "Integer only" is false of `crates/core/src/rng.rs`

§0, line 29: *"The RNG is SplitMix64 with `child_seed(parent, salt, index)` … Integer only. This is
the hash SL10 V1.4 asks for."*

MEASURED by reading the file: `SplitMix64::next_f64` is at `crates/core/src/rng.rs:31` and
`SplitMix64::chance` at `:39`; both produce or take `f64`. The generator draws floats from them today
at `crates/physics/src/worldgen/generate.rs:159-163`, `:460-466` and `:765-767`.

The integer core is real (`next_u64` at `:22`, `child_seed` at `:70`), and a tree's `expand` may use
it alone. But the sentence as written tells a reader that the existing generator already satisfies
V1.4, and R2 shows it does not.

### R9 — a file count is wrong inside a line that says MEASURED

§0, line 22: *"`crates/core/src/` lists `built.rs` … `worldgen.rs`, 24 files."*
MEASURED by `ls crates/core/src | wc -l` on 2026-09-07: **26**.

Small on its own. It matters because §0's heading claims every item under it is MEASURED by grep, and
the reader trusts that heading for R2's and R4's claims too.

### R10 — the render seam's route (b) puts back the shape branch the code exists to prevent

§5.2, lines 419-432, gives the engine `kind` so *"the engine may pick an asset by it"*, and line 429
calls the choice free: *"This keeps the Bevy client, the Unreal client and the HR6 harness on one
seam."*

The code states the opposite invariant. `crates/client/src/realm_scene.rs:788`: the renderer
*"consumes ONLY this — it never learns the word 'sphere' or 'box' (adversary H4)"*.
`crates/client-render/src/lib.rs:1826-1827`: the mesh build is *"PURE glue: no shape branch, no
logic — exactly the H4 seam."*

An engine that maps a kind id to an art asset IS a shape branch in the renderer. That may be the
right trade for V2.8, but it is a cost, and the report records it as free.

---

## 4. What the report leaves out that its domain needs

### R11 — the seed forest has no bound fence, and the fence it names runs on the wrong path

§1.7, line 172: *"A tree's shape may NOT reach outside its realm's bound. Placement refuses it"*,
citing the nesting fence at `crates/core/src/geometry.rs:350-368` (verified: `circumscribed_extent`
and the exact farthest-corner test live there).

But §2.1, lines 189-194, emits SEED objects with no such test, and that path runs on the client as
well as on the shard. A fence that guards only a player's placement does not guard the generator's
own draw.

**Game words.** The seed draws a 20 m spruce on a garden deck inside a station realm whose `Aabb`
bound sits 8 m above the deck. Nothing refuses it. The spruce's crown reaches through the station's
hull, and the station's parent sees a child that pokes out of its bound.

### R12 — "the kind's age curve" is a function of time, which the static shape may not be

§1.2, line 74, gives a seed tree's growth stage as *"derived (the seed says 'mature', or the kind's
age curve)"*. An age curve is a function of time. SL10 V1.1 says the static shape *"is never a
function of time, of a tick, or of any live state."* §2.1, line 192, uses the lawful form instead —
*"the kind's baseline stage"*, a constant. The two sentences disagree, and only the constant is
lawful. A reader who implements §1.2 breaks SL10 on the first tree.

### R13 — a falling group that leaves its realm is not designed, and its continuity class is not built

§3.2 makes the crown a `Transient` entity of the realm that owns the cell. On a small body a felled
crown can leave that realm's bound while it topples. The report says nothing about that crossing.
`crates/core/src/entity_kind.rs:23` records that `RealmAnchored` continuity *"lands at P6/P10"*, so
the continuity class the report picks for `FallingGroup` (§3.2, line 302) is itself not yet built.
The report presents it as a free tag.

---

## 5. What stands

- **The record arithmetic stands.** 19 406 × 8 B = 155 KB, and 35.8 MB ÷ 155 KB = 231. The report's
  230× claim (§2.2, line 213) checks out by arithmetic. Retiring the base's clear-cut alarm is sound,
  subject to R6's stamp.
- **The refusal of the pattern recogniser stands (§4).** Column B's costs are real, and V2.2 does say
  a tree is one object. A rock as the same object with a size byte is the honest reading of the
  ruling, and the wall-becomes-boulder case (§4.3 example, line 387) is answered.
- **The disc counts stand.** π·64²/100 = 129 and π·786²/100 = 19 408, both correct.
- **The SL6 finding stands.** §12's "none found" is right for the record and the diff: the object
  lives in the realm that owns the cell, and the diff reaches the client through the gateway, which
  the 2026-08-24 clarification of SL2 says is not a realm.
- **The stale-claim list (§8) is largely right.** Items 2, 4, 5, 6 and 10 correctly retire base
  numbers that V2.2 dissolves. Item 7 is the one that leans on R4's broken estimate.

---

## 6. Verdict

**REFUTED.** R1 makes the drawing rule impossible at flight altitude, and it is the rule the report
uses to close the arrival-pop seam. R2 leaves untested the half of the no-drift gate that produces
invisible colliders, on a crate that MEASURABLY breaks SL10 V1.4 at eight production sites today. R3
is an internal contradiction whose repair costs a reverse index the report does not design.

Nothing above condemns the core idea. A tree as one record, one integer skeleton, capsules on the
shard and a mesh on the client is a good answer to V2.2, and the storage arithmetic is a real win
over the base. The report must gate the placement, bound the horizon, and name the support lookup
before it is a design a slice can be cut from.
