# THE GALAXY, THE SKY AND THE RULER — IMPLEMENTATION PLAN OF RECORD

**Date** 2026-08-24 · **Tree** `worktree-warp` @ `c88ef0d` + the uncommitted SL9 work · **Status** plan, awaiting build order
**Read-only survey.** No build, test, benchmark or binary was run. Every number below is read from a named
file and line in this tree, or is arithmetic whose working is shown. Anything I could not check is labelled
**UNMEASURED**.

**Binding above this file:** `docs/design/owner_decisions_2026-08-24.md` (the rulings) and `CLAUDE.md`
(HR1–HR6, SL1–SL9). Where the design of record
(`scratchpad/galaxy_sky_design.md`) and the rulings disagree, the rulings win, and this plan follows the
rulings.

---

## 0. WHICH SPINE, AND WHAT WAS GRAFTED

**The spine chosen: *nothing moves until the refusal that would catch it moving is landed, measured red,
and turned green.*** (Plan 1's spine, corrected.)

It was chosen on one measured structural fact that the other two spines get wrong, and everything else
follows from it:

> **The coordinate step is INERT until the galaxy is a realm that can own things.** `frame_for_realm`
> (`crates/core/src/pose.rs:147-163`, read in full — six arms) has **no arm that produces `GalaxySpace`**;
> every region's frame comes from it (`crates/physics/src/worldgen/body.rs:187`); `FrameRef::tier` maps every
> live frame to `Tier::Fine`. So **zero coarse-tier positions exist anywhere in the product today**, and a
> plan that re-values the coarse step in one slice and makes the galaxy realm-bearing several slices later
> has, in between, a tier nothing can select and a set of gates that pass vacuously.

The tier change, the realm arms, the frame arms, the radii and the directory-key migration are therefore
**one slice** (S9), and this plan says so.

**Grafted in, named:**

| Taken | From | Why |
|---|---|---|
| Decouple the heaviest star from the galaxy's size, and print the cap at each candidate step **before** the step ships | Plan 2 slice 7 | Measured: `solve_mass_cap` reads `let budget_m = REAL_GALAXY_R_M;` (`crates/physics/src/worldgen/scale.rs:250`) and the cap **is the star draw's own upper bound**. The golden's own header records the last time this fired: *"the cap is the DRAW's own bound, so seed 0's stars were RE-DRAWN as well as re-placed … 99 of 99 Planet rows moved."* Without this graft, Q8 condition 1 (enlarge the world, touch nothing discovered) is unobtainable and "in-system bytes unchanged" is false. |
| Swept membership as its own slice, with a differential control | Plan 2 slice 3 | The only route that decouples enterability from a child's size, which the movement law (A4) will need. Its equivalence control expires when the step moves. |
| The deferral rule, verbatim, then honoured: *"the forces themselves are a phase, not a slice, and nothing here may be planned to depend on them"* | Plan 2 | Plan 1 scheduled P8 as a slice; Plan 2 wrote the rule and then broke it. This plan writes it and keeps it. |
| Honest edge labels — *"depends on nothing structurally"* where that is true | Plan 2 | A plan whose every edge claims to be forced cannot be audited. |
| **The sky lane BEFORE the census raise** | Plan 3 (and the dependency judge) | Measured in the opposite direction from Plans 1 and 2: `window.rs:265-282` ships the level as one un-split `rows: realms.to_vec()` on `MsgClass::RealmSnapshot` / `Durability::Ephemeral`. Raising the census first produces an **empty sky with nothing in any log**. |
| The **arithmetic** reason bands precede the step (not the obedience reason) | Plan 3 | The shipped 3 m band (`body.rs:21,23`) is **1.5 cells** at a 2 m step, so `band_edge_cells`' floor-and-ceil (`geometry.rs:1195-1203`) would silently become the doorway. |
| Read the constant, not the comment | Plan 3 | Compression is `24.567816882382665` (the frozen constant, confirmed in the golden header: *"chi 16.378 -> 24.568"*). A comment near it still says 16.378. Every figure quoted here cites the literal or the assertion. |
| The gate must assert on a **typed refusal** and on the file being byte-unchanged, never on a field being present | Plan 1 slice 1 | The tree already holds one write-only stamp: `TRANSFER_SCHEMA_VERSION` (`crates/wire/src/intershard.rs:114`), assigned at five sites and compared at none. |
| The child-index degradation hazard, gated inside the band slice | Plan 1 slice 2 | Measured: `radius_m = shape.circumscribed_extent() + r.band.outset()` (`crates/sim/src/stub/regions.rs:236`) feeds `grid_edge_m = next_power_of_two_m(2.0 * widest)` (`crates/core/src/child_index.rs:155-157`). |

**Found in the judging and carried here as my own, because no plan contained them:**

1. **The headline "red today" gate in all three plans is GREEN today.** "Two entities 100 m apart at the
   placement radius must draw exactly 100 m apart" cannot fail. Working shown in §S4.
2. **The governed-ceiling walk is a bounded-radius query, not an unsolvable lower-envelope query.** Working
   shown in §S10. This dissolves the one hazard Plan 1 admitted it might have to revisit.
3. **A store refusal at boot becomes an unbounded respawn loop**, because a child that forks and then exits
   is counted as a *successful* launch: `Ok(node) => { … self.launches.fail_streak.remove(&path); }`
   (`crates/node/src/rlm_runtime.rs:325-331`, read).
4. **The swept primitives are `f64` while the shipped verdict is integer**, so "wire them in" is a category
   error. `segment_shell_crossing(p0: DVec3, p1: DVec3, r: f64)` (`geometry.rs:173`) versus
   `shell_member_cells` on `i128` cell squares (`geometry.rs:1209`). Sized accordingly in S5.

**Where the plans disagreed on a fact, and which I believe:**

| Disagreement | I believe | Why |
|---|---|---|
| Is the coarse tier selectable once `Tier` gains a variant? | **No, not until the galaxy is realm-bearing** | Read `frame_for_realm` in full; no `GalaxySpace` arm exists. |
| Can in-system golden rows be held byte-identical across a radius change? | **No, not until the mass cap is decoupled** | The golden's own header: 99 of 99 Planet rows moved when the cap last changed. |
| Is `guard_root_representable`'s refusal factor 1,024×? | **No — 2^25 = 33,554,432× against the universe root** | `K_SPAN·R/edge/(CELL_DOMAIN_MAX+1)` = `2·2^76·2^10/2^62` = `2^25`. The 1,024× figure is for a galaxy *shell* at 1 m, not the root the fence reads. Same conclusion, different number. |
| Is the upward tier-conversion residual 2^-63 m? | **No — 2^-51 m = 4.4409e-16 m per axis** | The coarse residual is below one 2 m cell; `ulp(2)` = 2^(1-52). The design is wrong by 4,096×; its conclusion (far below a proton) survives. |
| Does the 150,000-system population fill 95 % of the galaxy shell? | **No — 47.5 % of the RADIUS, 10.7 % of the volume, at the ruled 2 m step** | 2.19123e18 / 2^62 = 0.4751; cubed = 0.1072. The 95 % figure is the design's *one-metre* solve, which the owner overruled. |
| Is the sky lane blocked on having 150,000 stars? | **No, it is the reverse** | The emit path must be partitioned *before* the census, or the failure is silence. |

---

## 1. OWNER SUMMARY

We are going to make the world big enough to be real, and we are going to do it in an order where every
change that could quietly break something lands **after** the thing that would catch it breaking.

**First, two locks.** Nothing we save to disk today says which world wrote it or what unit its positions are
counted in, so changing the ruler would make an old save read as a player a thousand times further out —
silently. So the first job is a short label on every saved file and a refusal that reads it, out loud, naming
what differed. The second job is the doorways: every boundary in the world today has the same three-metre-wide
crossing zone, which is no doorway at all when a ship covers seventy-eight thousand kilometres in a single
tick. That is not a theory — one test flight today crosses the same boundary eighteen times in eighteen
seconds, and the test is parked on that measurement. We give every doorway its own thickness, matched to how
fast anything can arrive at it.

**Then four things nobody can see, that make the big change safe.** Programs learn to refuse to talk to each
other when they disagree about the ruler. The drawing maths starts subtracting before it rounds, which fixes a
defect that is live today: two ships flying side by side are each rounded independently to a quarter of a
metre. The world learns to ask "what did this ship pass *through* this tick", not just "where did it end up".
And the heaviest star the world may contain stops being whatever the galaxy happens to afford, and becomes a
stated physical number the galaxy must be big enough for — which is what makes enlarging the world later a
free decision instead of a launch-freezing one.

**Then the visible one.** The ruler changes: a star system keeps counting in millimetres, a galaxy starts
counting in two-metre steps, the space between galaxies in about thirty-two-kilometre steps. This moves every
distance in the world at once, which is why it is one job and not five. The moment it lands you can fly the
same three stars **at their true separation** — today they are about twenty-five times too close, which makes
them about twenty-five times too big and their parallax twenty-five times too strong. That error becomes
exactly one. That is the first thing you will be able to fly that looks different.

**Then the sky.** Each star's position gets sent once instead of fifty times a second, in its own small
message the client keeps on disk. Doing this *after* raising the star count would produce an empty sky with
nothing in any log, so it goes first. Then a hundred and fifty thousand stars, placed by *where they are*
rather than by counting outward, with the gap between any two set by their own sizes. Then the level above:
about sixty-one galaxies, each different from the others, and the ability to create realms while the game is
running — which is what player-built ships need, and what the corridor between galaxies will need when its own
design is done.

**Then we fly the whole journey and measure what nobody has measured.**

**What you can fly, and when.** Nothing new until the doorways are sized — at that point the boundary thrash
stops and the parked flight test comes back green. Nothing visible until the ruler moves — at that point the
sky is physically correct for the first time. Then a sky that costs nothing, then a hundred and fifty thousand
stars in it, then flying out of the galaxy, then the full warp leg end to end. Five of the fourteen steps show
you nothing new. That is the price of the order, and I am stating it rather than hiding it.

---

## 2. THE ORDER AT A GLANCE

| # | Slice | Delivers | The gate that could fail | Unlocks |
|---|---|---|---|---|
| **S1** | The saved-data stamp, and the refusals that read it | A stamp on every durable file, verified before any scan; a refusal that names the field and both values with units; the spawn-side refusal so it cannot respawn-loop; the deploy rule the refusal forces | Write under one generation, open under another → typed refusal **and the file byte-unchanged**; a non-empty unstamped store refused, never adopted | S3 (same digest on the wire), S8 (the step refuses old data by itself), S11 (the client's cache) |
| **S2** | Each boundary owns its band; the world says what each band must be | Band moves inside the per-region map at **today's values**; the speed↔band solve as one named pair; a measuring sweep over THE world's forest | Bit-identity of the whole booted forest before/after; the sweep asserts non-vacuity (every parented region yields all five numbers) on the shipped seed and the swept range | S5 (its numbers), S6 (the sizing), S7 (the radius chain) |
| **S3** | A refusal on the wire, before the unit moves; positions that say what they are | The coordinate digest in the client handshake and in the inter-node tag; twelve one-line diagnostics that name frame and unit | A peer with a different edge table is refused, naming both units; a node with a different generation cannot complete a handshake | S8 (the change this catches) |
| **S4** | Subtract before you flatten | The eye subtraction moves into the integer lattice on the draw path; the child index keys on cells, not metres | Drawn separation error over a swept set of separations at the placement radius: **non-zero before, exactly zero after** | S6 (index quality), S9 (drawn positions at galaxy magnitudes), S10 |
| **S5** | Membership tests the whole tick's motion | Integer swept segment-vs-shell and segment-vs-box; `region_verdict` consumes the previous position | A subject travelling one tick further than a child's whole diameter still acquires it; **and** swept ≡ point, bit for bit, for every sub-band subject | S9 (per-tick travel ×2,051), and A4 at P8 |
| **S6** | Bands sized from real closing speed | Every band a function of the ceiling in force; `guard_quantum_band` written; the outset term added to the sibling and nesting fences | **Un-ignore the parked thrash test** with its bound and floor untouched; every band ≥ N cells at its own tier; index candidates-per-query stays small | S7, S9, and A4's precondition discharged |
| **S7** | The heaviest star stops depending on the galaxy's size | `solve_mass_cap`'s budget becomes an argument; the cap becomes a stated physical bound the galaxy must afford; a harness prints cap, largest shell and per-pair gap at each candidate step | Reproduce the shipped answer at the shipped budget **bit-for-bit** before trusting it at a new one; the affordability fence refuses a world too small for its own cap | S9 (the step chosen on a number), S12 (Q8 condition 1 becomes achievable) |
| **S8** | The ladder, built but not climbed | A third and fourth tier; the coarse step re-valued to 2 m; integer tier conversion with a loud refusal; the storage fence reads the frame's own step | Down-conversion bit-identical over the bounded domain and **refusing** above it; up-conversion residual ≤ 2^-51 m; the fence at exact equality at all three tiers | S9 |
| **S9** | The world at its true distances | Realm-bearing galaxy and universe; radii at the fence equality; directory-key migration; goldens split **by mover** | Galaxy→system→planet→back returns the identical position bit-for-bit at every hop; every non-moving golden row byte-identical; the movers named and non-empty | Everything after it |
| **S10** | A parent's cost stops growing with its children | Interest fold inverted to observers-outer; the governed-ceiling walk becomes a bounded-radius query; relay fingerprint incremental; emit path partitioned | Allocation count and emitted bytes **equal** (not "within noise") at 3 children and at the census, as a function | S11, S12 |
| **S11** | The sky stated once | Compact catalogue message; send-on-change on a re-driven lane; keep-alive compares a counter; liveness digest; login level chunked; one instanced point cloud | Per-tick sky bytes zero after the first tick; the **received** star count equals the census; encoded catalogue ≡ folded catalogue, byte for byte; not re-sent on a scene-epoch bump | S12 (the census can now be raised) |
| **S12** | A galaxy generated where you look | Position-addressed lazy placement; per-pair minimum gap with a one-light-year floor; binaries as one realm; census 150,000; the all-pairs fence replaced, never deleted | **Grow the census N→N+1 and every existing system is bit-identical**; minimum separation meets the per-pair rule; boot never enumerates | S13, S14 |
| **S13** | The level above, and room left for a door | Universe as a running realm; galaxies generated lazily one level up and differing by kind; the galaxy shell crossable outward; runtime realm creation | Outward galaxy-shell handover shows the same zero drawn difference the system shell does; galaxy 61 differs from galaxy 1 on a statistic that would be equal under identical rules | The tunnel's own design pass (out of scope) |
| **S14** | Fly it, and the honest capacity plan | The full warp leg on THE world; the four-part not-a-loading-screen instrument; the shard-count, reconnect-burst and traveller-count measurements | No black frame; zero drawn delta across the handover; a thousand reconnects inside a stated budget **under the real session cap**; galaxy tick inside budget at a stated traveller count | — |

---

## 3. WHAT MUST BE TRUE BEFORE SLICE 1

The owner named two preconditions. Both are slices in this plan, not chores, and both are forced by
arithmetic as well as by instruction. One thing is a **question**, not work, and it must be answered first.

### P0 — ONE QUESTION FOR THE OWNER, BEFORE S1 IS CUT

**Does any deployed volume currently hold a world someone has played in?**

**UNMEASURED from this tree, and it decides S1's shape.** What I can measure: dev-cluster stores live under
`$TMPDIR` with `VD_STORE_EPHEMERAL_OK` set and are wiped by `dev-cluster down`
(`crates/bins/src/bin/vd-devcluster.rs:265-271`) — so a local store is never a migration subject. But
`deploy/k3d/30-orch.yaml:90` declares a `PersistentVolumeClaim` that survives a redeploy.

- **If no:** "refuse, do not migrate" is honest and free, and per-family record migration stays deferred.
- **If yes:** S1 grows a reader for the pre-stamp shape, roughly doubling that slice, and changes nothing
  else in the order.

Serves: owner Q1 condition 1; D-48; the standing rule that a claim must be a measurement.

### P1 — THE SAVED-DATA FORMAT STAMP (owner Q1 condition 1) → **slice S1**

Stated as work with a gate in §S1. The owner's condition is literal: *"the saved-data format stamp lands
FIRST."*

**Why it is also forced, not merely instructed:** everything from S2 onward changes something the bytes on
disk depend on — the bands change which realm holds an entity, the ladder changes what a whole number means,
the realm arms change the directory key. A generation **folded over those constants** rather than typed by
hand is written once and never revisited. Typing an integer someone must remember to bump guarantees it is
forgotten on exactly the change it exists for.

### P2 — SPEED-SIZED BOUNDARY BANDS (owner A4) → **slices S2 and S6**

The owner's sentence is *"this law cannot land before the bands are speed-sized"*. Delivered across two
slices because the measurement must precede the sizing, and because the sizing moves the world's own radii
(see S7).

**Why it is also forced by arithmetic, independent of A4** (grafted from Plan 3, re-derived here):
`band_edge_cells` (`crates/core/src/geometry.rs:1195-1203`) converts each band edge to **whole cells** —
floor for the acquire edge, ceil for the release edge. The shipped band is 1.0 m inset + 2.0 m outset
(`crates/physics/src/worldgen/body.rs:21,23`) = 3.0 m total. At the ruled 2 m galaxy step that is **1.5
cells**; at a 2^15 m universe step it is **9.155e-5 cells**. The configured metres would stop meaning
anything and the rounding rule would become the doorway. So the bands must be sized before the step moves
whatever the owner's instruction had been.

### P3 — ONE WRITTEN RULING, TAKEN BEFORE S9 IS CUT (one paragraph, no code)

**Does `FrameRef::realm()` become total?** Making the galaxy frame realm-bearing removes the only cover for
two branches whose source documents them as *"otherwise uncoverable, no live shard uses `GalaxySpace`"*
(`crates/sim/src/stub/containment.rs:485-490`; the same shape at `crates/sim/src/stub/regions.rs:275`,
whose `root_frame` falls back to `FrameRef::GalaxySpace`). Under HR5 the 100 % gate then goes red on
branches nothing can reach. This is a decision, not a discovery: take it in writing, with a grep of the
`FrameRef::realm()` call sites, **before** S9 starts, so S9's size is the mechanical work only.

---

## 4. THE SLICES

Every slice states its size honestly, **including the 100 % region+branch cost**. Tier-A crates
(`vd-core`, `vd-node`, …) are held at 100 %; `vd-io-prod` is Tier-B at a ratcheted 94 floor and is *excluded*
from the 100 % gate — so a decision that must be gated goes in `vd-core`, and `vd-io-prod` keeps only
plumbing. `MemStore` has no `open()`, so anything enforced at `RedbStore::open` is invisible to every Tier-A
harness test; each such slice names its process-tier gate explicitly.

---

### S1 — THE SAVED-DATA STAMP, AND THE REFUSALS THAT READ IT

**Serves:** owner Q1 condition 1; D-48 item 1; D-47 item 1(b)/(c).

**Delivers.** A stamp record and a **pure verifier in `vd-core`** (so the 100 % gate holds the decision while
`vd-io-prod` stays plumbing at its 94 floor). `RedbStore::open` (`crates/io-prod/src/store.rs:555` — the one
function all three production redb files funnel through) verifies it **before any scan**.

Contents, and nothing that is not actually compared (a field with a lenient check is inert code and fails
HR5): a layout version; a **coordinate-unit generation folded over a new `Tier::ALL` edge table** rather than
typed by hand (`Tier` has no `ALL` today — `crates/core/src/pose.rs:207-222`); a world-law generation folded
over the named constants that shape the forest (the band law and the radii are both in it); the **world
seed**, which is env-only today (`VD_UNIVERSE_SEED`) and recorded nowhere durable; the universe epoch; and a
file-role tag so an outbox can never be opened as a saga log.

Four separated outcomes: absent on a provably-empty file → write and proceed; absent on a **non-empty** file →
refuse (pre-stamp bytes are of unknown shape); present and equal → proceed; present and different → refuse,
**naming the field and printing both values with units** (Q1 condition 3). One new `StoreError` arm — both
orchestrator opens already propagate with `?`, so the process already exits non-zero with the message.
`VD_STORE_ALLOW_GENESIS` as the explicit wipe opt-in (D-47 names it; no such variable exists today).

**Three things no plan contained, all measured, all in this slice because they are what make the refusal
safe rather than merely loud:**

1. **The refusal must not become an unbounded respawn loop.** `Ok(node) => { … fail_streak.remove(&path); }`
   (`crates/node/src/rlm_runtime.rs:325-331`) treats a successful fork/exec as a successful launch. A
   demand-spawned shard that forks fine and then exits at store-open therefore never engages the exponential
   backoff, and the reconciler respawns it at the base cooldown forever. Fix: refuse at the **spawn decision**
   too, and count a child that dies within N ticks of a successful launch as a launch failure.
2. **The orchestrator must not orphan its own children when it refuses.** Children are exec'd into their own
   process group (`crates/bins/src/lib.rs:1927`), `Child` does not kill on drop (said in as many words at
   `lib.rs:3045`), and the reclamation state is the **launch ledger**, opened through the same funnel the
   stamp refuses (`crates/bins/src/bin/orchestrator.rs:226`). So the launch ledger's refusal policy must
   differ from the saga store's: it stays readable-for-reaping, or the orchestrator enumerates and kills its
   recorded process group before exiting.
3. **The deploy procedure the refusal forces.** All three StatefulSets declare
   `podManagementPolicy: Parallel` and **no `updateStrategy`** (`deploy/k3d/30-orch.yaml:28`,
   `40-gateway.yaml:28`, `50-shard.yaml:29`), so the default RollingUpdate applies and every release
   produces a mixed-version fleet by design. With the refusal armed, that is a cluster partition. Set
   `updateStrategy: OnDelete` (or a documented scale-to-zero) and refuse a rolling update of a unit-changed
   image in the deploy tooling, in this slice, as a standing operational rule.

Plus the cure for the one format version the tree already has, which fails silent on exactly its own event:
`scan_all` drops version-mismatched rows with no count and no log
(`crates/io-prod/src/outbox.rs:316-327`) and `replay_outbox` then reports *"genesis / already-drained"*
(`:552`). Twenty lines. Landing a new stamp beside a working example of the failure mode is how the new one
ends up copying it.

**The gate, and why it is red today.** Nothing in this tree compares a store-level version at all:
`rehydrate` (`crates/node/src/saga_runtime/store.rs:196-303`) scans five families and compares none, and
decides genesis-vs-recover by the **clock family being absent** (`:210`).

- Write under generation N, open under N+1 **and the inverse**: assert the typed refusal **and that the file
  is byte-unchanged afterwards**. Production holds exactly one caller of the current value; the test injects.
- All four open outcomes driven separately.
- A checked-in golden holding the postcard bytes of one canonical instance of each of the **seven** durable
  value shapes, so a field added or reordered moves the bytes and the only way back to green is a
  regeneration that required the version to move. (postcard is positional and non-self-describing: two
  different structs decode from the same bytes with no error, which is why the stamp is checked *before* any
  scan and why the byte golden exists.)
- **The unit gate:** change a tier's metres-per-cell in a fixture, assert the derived generation moves, then
  assert an old-generation store is refused.
- The refusal reason is reachable **without** a successful connection: served on the health endpoint before
  exit, or written into the mounted volume. A refusal an operator cannot act on becomes an unofficial
  `rm -rf`, which is the loss the refusal exists to prevent.

**Consumes:** nothing. This is the only slice here that needs no other.
**Consumed by:** S3 (the same digest on the wire — a store and a peer must never disagree about which world
they are in); S8 (the generation moves by itself, because it folds over the table S8 extends); S11 (the
client's on-disk catalogue cache carries the same stamp — which is *why* the type goes in `vd-core` and not
`vd-io-prod`).

**Size — MEDIUM.** ~200 production lines, ~350 gate lines, plus the supervisor and orchestrator fixes.
Measured basis, not guessed: `BootCounter::read` + `write_durable` is **80 lines**
(`crates/io-prod/src/boot.rs:561-640`) for the same job — magic, checksum, atomic write, typed refusal; the
version-ledger self-check that reads its own source is **122 lines** (`crates/wire/src/version.rs:486-608`);
`StoreError`'s three arms are **12 lines** (`store.rs:136-147`). Coverage: the verifier is pure and
monomorphic, unit-tested in `vd-core` at 100 %; the end-to-end refusal is **process-tier** and the Tier-B
floor for `vd-io-prod` must be **re-measured**, not assumed, since a new refusal branch adds regions to a
floored crate.

**Risks.**
- *A stamp written and not read is worse than none.* The tree already holds one (`TRANSFER_SCHEMA_VERSION`).
  Cured only by gating on an observed refusal.
- Writing the stamp at open flips four assertions that a fresh store is empty (`store.rs:887, 942, 1001,
  1013`). Rewrite them as "empty apart from the stamp"; **never delete them** — they are the durability
  contract's own control.
- `RedbStore::is_empty()` is production-dead (its only callers are io-prod's own tests). Writing a stamp row
  at open is safe today and would silently become a bug if anyone later "simplifies" `rehydrate` to use it.
  Pin that with a test in this slice.
- **Deliberately not shaped now:** cluster identity and orchestrator identity. Neither exists as a value
  anywhere in the tree; inventing one is a design decision, not a line of code.

---

### S2 — EACH BOUNDARY OWNS ITS BAND, AND THE WORLD SAYS WHAT EACH BAND MUST BE

**Serves:** owner A4 (the precondition, measurement half); D-WORLD-4b; SL4 (worldgen is the lawful producer).

**Delivers.** The single shared `ContainmentBand` built once and copied onto every region in the universe
(`crates/physics/src/worldgen/body.rs:167-171`, used at `:192`) moves **inside** the per-region map, computed
by a named function that returns today's 1 m / 2 m for every input — **byte-identical**. The speed↔band solve
becomes one named pair in `vd-core`: `band_for_speed(v, dt, N) = v·dt·N` and its inverse
`speed_for_band(W, dt, N) = W/(dt·N)` — the owner's *"band width and top speed are two readings of one
solve"* made into code.

Plus a sweep over THE world's booted forest that, for every parented region, computes and prints: the
parent's ceiling; the ungoverned in-band tick count; the band the child needs at the parent's ceiling; the
band it can afford against its own extent; and the band it can afford against its nearest sibling's clear
gap.

**One decision that must be written down in this slice, not discovered in S7:** two different tick counts
exist for one idea and neither knows about the other. `ContainmentBand::width_safe_for` uses
`K_SAFETY = 2.0` (`crates/core/src/geometry.rs:792`); the world's own geometry solve uses
`BAND_TICKS_N = max(K_SAFETY, n_entry) = 3.0` and doubles it again with `BAND_TAU_HEADROOM = 2.0`
(`crates/physics/src/worldgen/scale.rs:99-104`). Picking one silently changes either the safety margin **or
every world radius** — because `REAL_GALAXY_R_M = R_uni − v_cap(R_uni)·0.02·3·2` is a live expression
(`scale.rs:84-88`), not a literal. That is the precise trigger for the band → radius → mass-cap → every-star
chain, and it is the measured reason S7 exists.

**The gate, and why it is red today.**
- **(a) Bit-identity of the whole booted forest** before and after. The pattern already exists
  (`w.band == d.band`, `crates/physics/src/worldgen/tests.rs:823`). It goes red if any band value moved,
  which is what makes "byte-identical" a measurement rather than a claim.
- **(b) Non-vacuity, two statements that fail for different reasons**, copied from the shape that already
  works (`crates/bins/tests/flight_table.rs:378-395`): every parented region produces all five numbers, on
  the seed the cluster boots **and** across the existing swept-seed range; and the row count equals the
  derived parented count **and** a pinned census. A print cannot fail; this is what makes the sweep evidence
  rather than documentation.

**Consumes:** S1 only in **schedule**, not structure — and I say so rather than dressing it up. The band
lands on `RealmRegion`, which is not written through the stamped funnel; the world-law generation folding the
band constants is a design choice that *makes* the edge true, not a fact that forced it. The owner's
instruction, not a dependency, is what fixes S1 before S2.
**Consumed by:** S5 (its ungoverned-in-band-tick column is the evidence that a point-sample test is unsound
at speed); S6 (the solve and the affordable-band columns size every band); S7 (the tick-count decision moves
every radius).

**Size — SMALL-MEDIUM in code, MEDIUM-LARGE in blast radius.** ~30 lines in the generator, ~40 in `vd-core`
for the solve pair (branchless, monomorphic — HR5's generic trap does not apply), ~120 for the sweep. The
cost is the reach: `RealmRegion` is `Serialize`d into the client's region file, and roughly **twenty gate
sites** read `cfg.band.outset_m` as one number for the whole world (`crates/bins/src/flight.rs:118,196`;
`crates/bins/src/lib.rs:2700,2779`; eight process-tier test files).

**Risks.**
- The two tick counts above. Name which, in writing, and pin it.
- `BandConfig::build` passes `dt = 1.0` as a literal while the cluster ticks at 0.02 s — a dormant 50×
  units mismatch hiding behind the zero velocity. The geometry solve deliberately uses a **fixed**
  `GEOMETRY_TICK_DT_S = 0.02` so two clusters at different tick rates boot the identical world
  (`scale.rs:96-98`). The containment band must make the same choice **explicitly**, or the world forks by
  tick rate.

---

### S3 — A REFUSAL ON THE WIRE, BEFORE THE UNIT MOVES

**Serves:** owner Q1 conditions 2 and 3.

**Delivers.** S1's unit generation folded into the client handshake, with `PROTO_MINOR` and its floor raised
together and the mandatory ledger paragraph the tree's own completeness check will demand
(`crates/wire/src/version.rs:239, 259, 486-608`). The same generation folded into the inter-node ALPN tag
(`crates/io-prod/src/trust.rs:18`, today the bare `vd-intershard/1`), so a mismatched node is refused at the
transport handshake with **no new message and no new wire arm** — which keeps the fleet half out of SL6's ask
entirely, and costs one constant plus two config builders.

**A correction to both plans, which both claimed a diagnosis the ALPN route cannot deliver:** a rustls ALPN
mismatch produces a `no_application_protocol` alert — a connection error with no field, no value, no unit. Q1
condition 3 exists precisely to remove that ambiguity. So the ALPN route buys the **refusal**, and the
**diagnosis** must be reachable another way: each node states its generation on its admin/health endpoint and
in its startup log, and the dialling side logs the exact tag it offered on failure. Gate on those artefacts,
not on a wire message that route can never carry.

Plus Q1 condition 3 finished: twelve one-line edits so every position diagnostic names its frame and its unit
(`crates/sim/src/stub/containment.rs:689,691,698`; `crates/sim/src/stub/conversion.rs:86,89,299,386,422,426,429`
all print bare flattened metres and name neither). Cheap now; painful during an incident.

**The gate, and why it is red today.** Negotiation exists at **exactly one site** in the product
(`crates/connection-plane/src/gateway/client.rs:79`), client-to-gateway only; between nodes nothing is
compared beyond mutual TLS and that one tag. And — the reason this slice exists at all — **a step change moves
zero bytes**: `Tier` derives only `Clone, Copy, Debug, PartialEq, Eq` (`crates/core/src/pose.rs:207`) with no
`Serialize`, and the edges are compile-time constants. No golden reddens, no decode fails, nothing goes red by
itself. Every safeguard here must be built deliberately.

The contract is a **value with exactly one production caller and an injectable test constructor**, so the
gate drives a real `negotiate()` refusal against an injected mismatched contract, rather than unit-testing the
digest and calling it a handshake test. Both arms driven — the refusal is the entire product.

**Consumes:** S1's digest, already derived from the tier edge table rather than hand-typed, so it moves on
exactly the change it exists for.
**Consumed by:** S8 and S9 — the changes that make this refusal fire.

**Size — SMALL-MEDIUM.** ~120 lines plus the ledger paragraph, one constant on the fleet route, twelve
one-line log edits. **This is a flag day**: adding a field to the hello message is *not* postcard-additive
(the wire module's own header says so, `crates/wire/src/version.rs:9-14`). This plan pays **two** flag days —
this one and S9's new discriminants — and counts them openly. Landing them together would pay one but would
leave no window in which the refusal provably exists before the unit moves; landing them apart is the price
of the spine, and it is free while no fleet is live.

**Risk.** A reader will conclude the handshake already covers this, because
`crates/client/src/interp.rs:22-39` says the unit is *"a value that was SHIPPED, not one the renderer
inferred from context"*. It is not: the frame **name** ships, and the unit is a compile-time table on the
receiver (`interp.rs:40-42`). Fix that comment in this slice or it points the next reader the wrong way.

---

### S4 — SUBTRACT BEFORE YOU FLATTEN

**Serves:** no ruling — this is a **defect that is live on the world as it stands**, and it is the only gate
in the plan that is red today without any new world.

**Delivers.** The eye subtraction moves into the integer lattice on the client draw path. Today
`DeliveredView::world_pos` (`crates/client/src/view.rs:400`) and `RealmBox::draw_center`
(`crates/client/src/realm_scene.rs:129`) each flatten from the realm origin with
`delta_m(LatticePos::default(), tier)`, and `eye_relative` then subtracts two already-flattened `f64` values
(`crates/client-harness/src/camera.rs:202`). `RenderEye` becomes a lattice position plus a tier, and
`eye_relative` takes lattice halves.

**The measured production touch list, closed** (both plans' lists came from the design of record, which is
half wrong — five of the sites it names are inside `#[cfg(test)]` modules and are test helpers, and it misses
four production sites): `client/src/view.rs:400`; `client/src/realm_scene.rs:129`;
`client-harness/src/camera.rs:202, 455, 756`; `client-harness/src/verdict.rs:55, 70`;
`client-render/src/lib.rs:404, 764, 847`; plus `FollowCamera::eye` (`camera.rs:58`) and
`pilot_capture_camera` (`camera.rs:328`), which build the eye from an already-flattened value.

Server side, the same cure but for a **different defect** — say which is being fixed where or the gate will
be written against the wrong failure. The child index landed today keys its grid on flattened metres
(`crates/core/src/child_index.rs:91, 109, 190`) and so does its one caller
(`crates/sim/src/stub/containment.rs:632-638`). Here the division is already exact — `grid_edge_m` is a power
of two (`child_index.rs:155-163`) — so the error enters **only** through `as_dvec3`, and the hazard is a
**key flip near a cell boundary**, not cancellation. At galaxy magnitudes a 256 m flatten error is over a
hundred times the whole containment band, which would key a genuine near-boundary candidate into a cell its
child was never registered in — and the index's contract means the caller then **skips** it as a positive
"not here".

**THE GATE — AND A CORRECTION THAT MATTERS.** All three plans lead this slice with: *"two entities 100 m
apart, both at today's placement radius, must draw exactly 100.000 m apart — RED BEFORE."*
**It is green today. The gate as stated cannot fail.** Working, shown:

```
placement radius        R = 1,498,979,587,153,876 m      (the frozen value, golden header)
in fine cells           R / 2^-10 = 1.534955097245569e18 ,  log2 = 60.413  → binade [2^60, 2^61)
f64 ulp there           2^(60-52) = 256 cells = 256 × 2^-10 m = 0.25 m
100 m in cells          100 × 1024 = 102,400 ;  102,400 mod 256 = 0
```

Both endpoints therefore share a residue modulo the rounding quantum, both flattens shift by the **same**
amount, and the drawn difference is exactly 100.000 m. Worse, this generalises: every whole-metre separation
is `n × 1024` cells and `1024 = 4 × 256`, so **any whole-metre gate at this radius is exact today**. The
sub-cell offset cannot rescue it either — a residual below 2^-10 m added to a 1.5e15 m magnitude whose ulp is
0.25 m is absorbed entirely.

**The gate that can fail:** sweep separations that are **not** multiples of the rounding quantum (e.g.
`cellB = cellA + 102,501`, i.e. 100.0986 m — `102,501 mod 256 = 101`) and assert the **maximum drawn error is
non-zero before the fix and exactly zero after**, in the same test body. The pre-fix bound is **one ulp of the
absolute coordinate, 0.25 m per axis** — not the 0.5 m the source's own doc claims
(`crates/core/src/pose.rs:469`, a stale bound someone will quote).

Two ships flying in convoy at today's star placement radius each round independently to a quarter of a metre
**today**. This is not a future problem being paid for early.

**Consumes:** nothing structurally. Scheduled here because it must precede any galaxy-scale draw, and because
the child index was written **today** — re-keying it is 25 lines and one caller now, and grows with every
future caller.
**Consumed by:** S6 (the index-quality gate measures the real thing only once the keys are integers); S9 (at
the populated rim one f64 step is 256 m and at the full galaxy shell 1,024 m — a visible warp cannot land on
a drawn position that rounds to a kilometre); S10; S14 (its zero-drawn-delta arm is unachievable without it).

**Size — SMALL-MEDIUM.** Thirteen measured production call sites plus three lines in the index and its one
caller. Coverage: `client-render`/`client-harness` are Tier-B; the `vd-core` index change is Tier-A and
monomorphic.

**Risk.** Working from the design of record's citation list will send an implementer to five test helpers and
miss four production sites. Use the measured list above.

---

### S5 — MEMBERSHIP TESTS THE WHOLE TICK'S MOTION

**Serves:** A4 at P8 (the only route that decouples enterability from a child's size); and it makes the
module's own header true.

**Delivers.** `region_verdict` (`crates/core/src/geometry.rs:1166`) consumes the previous tick's position and
tests the **segment**. The module header at `geometry.rs:8-11` already states in as many words that
*"membership evaluation tests the tick's whole motion segment against the shell, so a 5.5 km/tick body cannot
tunnel undetected."* It does not: the verdict samples the pose only. This slice makes that sentence true.

**A CORRECTION TO BOTH PLANS THAT CHANGES THE SIZE.** Plan 2 sized this at "~150-250 lines" on the premise
that the swept primitives are *"built, covered, zero consumers"* and merely need wiring. They are built and
have zero callers — confirmed. But their signatures are **f64 metres**:
`segment_shell_crossing(p0: DVec3, p1: DVec3, r: f64)` (`geometry.rs:173`),
`segment_aabb_crossing(p0: DVec3, p1: DVec3, half: DVec3)` (`geometry.rs:426`), `Boundary::swept`
(`geometry.rs:289`). The shipped verdict decides on **i128 cell squares** through `shell_member_cells`
(`:1209`) and `aabb_member_cells` (`:1237`) — specifically so the answer is exact at every magnitude,
bit-identical across hosts, with the overflow hole cured by a Chebyshev pre-test rather than by luck. Wiring
the f64 pair in would abandon integer exactness on the deciding path. So this slice is **new integer swept
math**, with the existing f64 pair demoted to a differential oracle.

**A second correction.** Plan 2 said *"the caller already holds `dot.pose` before and after integrate."* Two
of the three production callers do not. `region_verdict` has exactly three: the containment scan
(`crates/sim/src/stub/containment.rs:658`, which does hold a prior) and the two crossing-abort flush guards
(`crates/sim/src/stub/conversion.rs:272` and `:369`), which ask a **point** question about a pose re-read
after a freeze drain. State in the slice whether those keep a point form — which forks "the one band
question" into two functions and needs an HR3 justification, since both sites' comments insist they ask the
same rule the scan asks — or receive a degenerate segment. Then drive both arms.

**The gate, and why it is red today.**
- A subject travelling one tick further than a child's whole **diameter** still acquires that child. Today it
  does not — acquisition needs a sample landing at least `inset` **inside** the surface
  (`ContainmentBand::member`, `geometry.rs:815`), so no band widening can ever buy acquisition.
- ~~**The differential arm:** for every subject whose per-tick travel is below the band, the swept verdict
  must **equal** the point verdict, bit for bit.~~ **★ THIS GATE IS FALSE AS WRITTEN, AND WAS REPLACED WHEN
  S5 LANDED — see [[D-S5]].** Two cells of travel (1.95 mm, three orders below the shipped 3 m band)
  genuinely clips a region at every size: for `a = (E,−1,0) → b = (E,+1,0)` both endpoints are outside and
  the midpoint sits exactly on the surface. Confirmed firing at seven real radii, from a 5 m station to the
  galaxy bound. Written as an equality this gate would be **RED for a correct implementation**, and the
  cheapest way to make it green would be to weaken the code. What shipped instead — four arms, each able to
  fail, and the first two still bit-equalities: stationary identity; **one-cell equality** (proved, then
  checked exhaustively, with the bound shown attained so it is tight); an exactly-tight integer divergence
  envelope; and a float-free lattice oracle. The intent the plan had here survives: the arm that protects
  walking players — landing and colliding — is arm 2, and it is permanent.
- A separate fast-regime arm with an **independent oracle** — an analytically computed segment-versus-sphere
  crossing time — so the fast path is not self-certifying once the sub-band control no longer covers it.

**Consumes:** S2's ungoverned-in-band-tick column, which turns "the point test might be unsound" into a
number.
**Consumed by:** S9. **The honest edge:** this slice's only *hard* requirement is that it precede S9, where
one tick of galaxy travel becomes 100.7 home-system diameters
(`v_cap = 2·2^62/180 = 5.1241e16 m/s × 0.02 s = 1.0248e15 m` against a 1.0178e13 m home-system diameter). It
is **not** required by S6 — widening the outset changes the release edge, never the acquire edge. I place it
here, and label the edge as scheduling, because it shares the containment surface with S6 and doing the
containment answer twice in two slices is the expensive way; and because **its equivalence control expires**:
the sub-band differential covers every subject today and covers no fast subject after S9.

**Size — MEDIUM-LARGE, larger than either plan estimated.** New integer segment-vs-sphere and
segment-vs-box math with its own overflow pre-test, plus a full HR5 pass over three shapes × five
`ShellCrossing` arms. All branching stays monomorphic (the `crates/core/src/tlv.rs` shape); nothing here is
generic, so the per-monomorphisation trap does not apply. `vd-core` is Tier-A: every arm must be driven.

**Risk.** This is the largest correctness change in the plan and it touches **the** containment answer —
whose last breakage cost a week and produced a turn-back tag. It is mitigated entirely by landing it while it
is provably equivalent on today's world. If the differential cannot be made to hold, the slice has found a
real disagreement, and finding it here is far cheaper than after the step moves.

---

### S6 — BANDS SIZED FROM REAL CLOSING SPEED

**Serves:** owner A4's stated precondition; D-WORLD-4b; D-REAL-4; SL4 (worldgen names motion, the sim only
reads a number); SL9 (the index must survive it).

**Delivers.** Every band becomes a function of the ceiling in force at that boundary, computed in worldgen —
the lawful site under SL4, and the pattern already exists once: the area-of-interest band feeds the child's
own closing speed through `InterestConfig::build` and freezes a plain number onto `RealmRegion` that the sim
only reads (`crates/physics/src/worldgen/config.rs:246-258`).

`guard_quantum_band` is **written**. It is named at `crates/core/src/geometry.rs:1156` — *"guard_quantum_band-
class fences keep every band ≥ 3 orders above"* the cell quantum — and a grep of the whole tree returns that
one doc comment and **no definition**. The claim is true by 3.49 orders at the millimetre tier and would be
false by 4.04 orders at a 2^15 m step.

The outset term is added to the sibling-separation fence (`crates/physics/src/worldgen/guards.rs:267`
compares **bounds only** today) and to the nesting fence. About twenty gate sites are re-pointed off the one
global outset number.

**The gate, and why it is red today.**
- **Un-ignore `a_durable_player_flies_the_chain_node_per_realm_without_freezing_or_fence_thrash`**
  (`crates/bins/tests/node_per_realm_walk.rs:241`) with `MAX_ENTITY_FENCE = 12` and `CROSSINGS = 5`
  **untouched**. It is parked red carrying a measurement, not a weakening — its own text records 21 crossing
  sagas of which **eighteen** crossed one boundary in eighteen seconds, nine in and nine out, strictly
  alternating, every one at `attempt = 0`, i.e. eighteen fresh decisions rather than one re-driven. The
  deferred row names un-ignoring it as the proof the band landed. It goes green when that boundary's outset
  reaches about `v·k_dwell·dt = 3.9167e9 × 5 × 0.02 = 3.917e8 m` — **0.11 % of that system's 3.525e11 m
  shell**, easily affordable.
- `guard_quantum_band` asserts every shipped band is at least N cells wide at its own tier — true at the
  millimetre tier (3 m = 3,072 cells) and red the instant a coarse tier exists.
- A fence asserting no realm's ceiling lets one tick cross its own thinnest band — A4's safety derivation
  made structural before A4's control change lands.
- **The index-quality gate, inside this slice, not five slices later:** max and mean candidates-per-query
  over THE world's booted forest stay at or below a stated small number after the bands are sized.

**Consumes:** S2 (the per-region band and the named solve); S4 (integer keys, so the index-quality gate
measures the real thing); S5 (acquisition survives at speed — the honest form: S5 is not *required* for S6 to
be safe, but shipping both without S5 would leave the enterability arithmetic unanswered while the bands are
being justified by it).
**Consumed by:** S7 (the band solve is already an input to the world's radii, so the mass cap cannot be
trusted until the band story is settled); S9; and A4 at P8, where the solve is re-read as a function of the
new ceiling law rather than rewritten.

**Size — MEDIUM in code, LARGE in blast radius.** ~30 lines to move the band inside the map, ~60 for the two
fences, ~20 gate sites re-pointed. `vd-core` fences are Tier-A at 100 %; both arms of each fence driven.

**Risks.**
- **Speed-sized outsets inflate the child index and coarsen its grid for every child**, because the grid edge
  follows the widest radius. That degrades the O(1) lookup that landed today back toward the scan it just
  replaced — an SL9 defect introduced by an SL4-lawful change. Hence the gate inside this slice. If it fails,
  the index needs nested grids per magnitude, and that is a design change to schedule **here**.
- Adding an outset term to the sibling-disjointness fence **may refuse THE world**. Record that refusal as the
  measurement it is rather than relaxing the fence — this tree's own history warns about disarming a fence
  whose refusal is the world's negative control. Speed-sized outsets can also make two siblings' membership
  regions overlap while their bounds stay disjoint, and the containment code already documents the
  consequence: the tie breaks on the **lower realm id**, not the nearer realm.
- **This slice does NOT retire the approach governor.** See §6 and §7.

---

### S7 — THE HEAVIEST STAR STOPS DEPENDING ON THE GALAXY'S SIZE

**Serves:** owner Q9 rulings 1 and 2 (*"re-solve the real cap with OUR solver, on THE world, at our step,
before building"*); owner Q8 condition 1 (*"worth more than the choice itself"*); Q3 condition 2.

**This is the graft from Plan 2 that the dependency judge identified as the missing half of every other
plan.** Without it, "in-system bytes unchanged" is false and Q8 condition 1 is unobtainable no matter how
position-addressed the placement becomes.

**The chain, measured end to end:**

```
guard_root_representable passes at EXACT equality       → R_uni is pinned to the step   (guards.rs:69-75)
REAL_GALAXY_R_M = R_uni − v_cap(R_uni)·dt·N·headroom     an EXPRESSION, not a literal    (scale.rs:84-88)
solve_mass_cap:  let budget_m = REAL_GALAXY_R_M;                                          (scale.rs:250)
StellarConfig.mass_hi_msun = imf_mass_hi_msun()          the IMF DRAW's upper bound       (config.rs:324)
```

The golden's own header records the last time this fired: *"cap 120 → 16.360034882257757 M_sun … The cap is
the DRAW's own bound, so seed 0's stars were RE-DRAWN as well as re-placed … 99 of 99 Planet rows moved; 6 of
12 System rows moved; 0 of 9 Star rows moved."* **The movers are in-system rows.**

**Delivers.** `solve_mass_cap`'s budget becomes an **argument** rather than a module read. Then the design
change the owner's Q9 ruling already implies: the heaviest star stops being *whatever the galaxy can afford*
and becomes a **stated physical bound** (the initial mass function's own top) that the galaxy must be big
enough for — with the affordability check becoming a **fence** rather than a solve. Q9 ruling 1 says the
binding limit should become the coordinate one and that in practice every stellar type is kept; at the ruled
step the coordinate limit very likely **stops binding entirely**, and this is the shape that lets it.

Plus a harness that prints, on THE world, at each candidate step: the derived cap, the largest system shell,
the demand at the cap, and the resulting per-pair minimum gap.

**It also settles a structural finding:** the shipped galaxy radius is derived as `R_uni` minus a band. That
is a **single-lattice artefact** — once the galaxy has its own lattice, its radius is its own fence equality
(2^61 cells × its step) and that derivation dies.

**The gate, and why it can fail.** The harness must **reproduce the shipped answer before it is trusted at a
new step**: the printed cap at today's budget equals the pinned arm exactly
(`target_system_bound_max_m() == system_shell_r_m` at the cap, `crates/physics/src/worldgen/tests.rs:1432-1435`),
and the *"one part per million above the cap is unaffordable"* assertion holds at every candidate budget.
**Red if turning the constant into an argument moved any shipped answer by one bit** — which is the whole
point of running it against the old budget first. Then: the affordability fence refuses a world too small for
its own stated cap, with the refusal arm driven.

**An arithmetic estimate, labelled as such.** At the pinned numbers the demand at the cap is essentially
three system shells (clearance 749,817,826,779,791.8 m against a shell of 749,489,793,576,937.9 m;
clearance + 2×shell = 2.2488e15 = `R_gal` exactly), so the rule is `shell ≤ R_gal/3`. At the ruled 2 m step
`R_gal` becomes 2^62 = 4.611686e18 m, a budget **2,050.73×** today's. Propagating the design's own two-point
fit (`shell ~ M^1.833`) gives a cap around **1,048 solar masses** — far above the physical top of the initial
mass function, which is what "the coordinate limit stops binding" means. **This is arithmetic through a
FITTED exponent, not a measurement.** The authority is the bisection, and this slice runs it. Do not build
against 1,048 any more than against the design's 30.

**Consumes:** S6. The band solve is already an input to `REAL_GALAXY_R_M`, and reconciling S2's two tick
counts moves that radius and therefore the cap. State the edge in that precise form or it will be dropped as
soft.
**Consumed by:** S9 (the step is chosen on the solver's own number, and moving the radius no longer re-draws
every star); S12 (the per-pair gap is sized from the resulting shells, and Q8 condition 1 becomes achievable).

**Size — SMALL-MEDIUM.** Mostly turning one module constant into a parameter, plus a printing-and-asserting
gate. `vd-physics` coverage tier: **UNMEASURED here** — check the justfile's Tier-A package list before
sizing the test volume.

**Risk.** This is the slice that could **invalidate the ruled step**, which is precisely why it exists before
the one-way door. If the solver disagrees with the estimate, the owner re-rules while it is still free. Its
own hazard is smaller: changing `solve_mass_cap`'s input touches the derived cap, which every star draw
reads, so the reproduce-the-old-answer arm is not optional.

---

### S8 — THE LADDER, BUILT BUT NOT YET CLIMBED

**Serves:** owner Q1.

**Delivers.** A third and fourth tier; the coarse constant re-valued from a light-year to **2 m** and the
light-year moved to its own name (still needed as Q3's gap floor, but never as a unit). The `i128` for the
tier ratio dies: at a 2 m step the galaxy ratio is exactly **2,048** and the universe-to-galaxy ratio exactly
**16,384** — both powers of two, so the conversions are bit shifts. The cross-tier conversion
(`crates/core/src/pose.rs:365-372`, today a fold through f64 metres) becomes integer arithmetic with a loud
refusal above its bound. `guard_root_representable` (`crates/physics/src/worldgen/guards.rs:69`) takes the
frame's own step and runs per level. The seven production lines that hard-code the fine step re-read it from
a frame.

**The gate, and why it is red today.**
- **(a) Downward bit-identical.** For a separation bounded by the destination realm's own extent, coarse →
  fine → coarse is the identity on the integer half, bit for bit, over a proptest across the bounded domain
  — **and the out-of-bound case refuses**, with that arm driven. An **absolute** coarse cell scaled down
  overflows: `(CELL_DOMAIN_MAX + 1) × 2048 = 2^73 = 9.4447e21` against `i64::MAX = 9.2234e18`. The current
  code is safe only **incidentally**, because its one caller happens to pass a destination-relative value
  (`crates/core/src/frame.rs:224`). That safety is not typed, and a second caller would reintroduce it
  silently.
- **(b) Upward within a stated bound.** Exact in the integer half; residual **at or below 2^-51 m =
  4.4409e-16 m per axis**. The design says 2^-63, which is wrong by 4,096×. Assert the measured bound.
- **(c) The fence at each tier.** Passes at exact equality and refuses one octave up, extending the existing
  one-tier pin (`crates/physics/src/worldgen/tests.rs:1448-1451`, occupancy == 0.5 and headroom == 2.0) to
  three. **Red before:** `guards.rs:69` reads the millimetre constant unconditionally and, applied to a
  2^76 m universe root, refuses by `2^25 = 33,554,432×` — not the 1,024× the design and the ruling both
  quote, which is the figure for a one-metre galaxy *shell*, not the root the fence reads.
- **(d)** The **root radius becomes an expression of its tier's step**, not a literal
  (`REAL_UNIVERSE_R_M: f64 = 2_251_799_813_685_248.0`, `scale.rs:63`, whose tie to the step lives only in a
  doc comment). Otherwise the step and the radius do not move together by themselves and only one pin at one
  tier catches the drift.

**Consumes:** S3 (the installed refusal — this slice re-values a coordinate unit, which is exactly what S3
exists to catch across a mixed fleet); S1 (the generation moves by itself, because it folds over the table
this slice extends).
**Consumed by:** S9, which turns the ladder on.

**Size — MEDIUM.** ~40 production lines in the coordinate module (the tier variants, the edge table, the
frame-to-tier map, plus the two other exhaustive frame matches in the same file), ~30 for the integer
conversion and its refusal arm, ~15 in the fence with three production callers to re-check
(`crates/bins/src/bin/gateway.rs:129`, `shard.rs:359`, `vd-seedsearch.rs:234`), plus seven test files that
name the coarse tier by hand. `vd-core` is Tier-A: the refusal arm must be driven, not merely written.

**Risks.**
- **Today's coarse edge is a light-year and is NOT a power of two** (`9,460,730,472,580,800 = 2^6 ×
  147,823,913,634,075`), so the normalising constructor is not exactly idempotent at that tier — the very
  property the file says the power-of-two edge exists to guarantee. It is inert **only** because no coarse
  position is ever produced. **Do not light the tier up at its current value even to test the plumbing.**
- `pose.rs:178`'s own doc boasts that *"the edge is a compile-time constant, never serialized, so this moves
  zero bytes."* That sentence is what the ruling overturns; it must change here or it becomes a stale doc
  pointing the next reader the wrong way. Same for `Separation::metres`' doc bound at `:469`.

---

### S9 — THE WORLD AT ITS TRUE DISTANCES

**Serves:** owner Q1; Q7 (*"the largest single re-measurement in the plan, to be scheduled as one"*); Q5
(the universe level becomes load-bearing).

**Delivers.** The galaxy and the universe become realms that can own things: new realm-id arms, a seed on the
galaxy frame, an appended universe frame, the forward map arms, and the lineage stand-ins retired
(`crates/core/src/realm_path.rs:25-27, 85-95`). **The directory-key migration rides with it and cannot be
split from it**, because `RealmCoord::lowered()` maps Universe → `System(0)` and Galaxy → `System(1)`
(`crates/core/src/realm_coord.rs:49-56`) and `lowered()` **is** the directory key, so giving them real arms
strands their owner records.

The radii move to the fence's equality at each tier — the galaxy at 2^62 m = **487.46 light years**, of which
150,000 systems at real density occupy a populated radius of 2.19123e18 m = 231.62 ly, i.e. **47.5 % of the
radius and 10.7 % of the volume** (the ruled two-metre step buys eight times the volume of the design's one
metre, which is the owner's stated reason for taking it); and the universe at 2^76 m = **7.99 million light
years**, which is where the sixty-one galaxies come from.

**The golden is split BY MOVER, not by anchor level, as this slice's first task** — and this is a correction
to all three plans. Both proposed an anchor-level split; the golden's own header proves it is the wrong axis
(the movers when the cap last changed were the **in-system planet rows**). With S7 landed, the cap no longer
moves when the radius does, so the assertable statement becomes: *these rows move because the placement radius
moved, and nothing else does.* Every flight budget and distance-bearing gate is re-derived.

**The gate, and why it is red today.**
- An occupant crosses galaxy → system → planet and back, and the round trip returns the identical position
  **bit for bit on the integer half at every hop**. Red before because **no arm can produce a galaxy frame at
  all**, so no occupant can be owned by a galaxy and the round trip cannot be attempted.
- The storage fence passes at exact equality at all three tiers.
- Every non-moving golden row byte-identical, and the movers **named and non-empty** — never "the declared
  movers may have moved", which is satisfiable by nothing moving.
- The existing visibility-climb fence still refuses a world in which another galaxy's stars could reach you.

**Consumes:** S8 (the ladder), S7 (a cap that no longer follows the radius), S6 (bands that are not one and a
half cells wide at the new step), S5 (per-tick travel becomes 100.7 system diameters), S4 (drawn positions
that round against the answer, not against the world), S3 (the refusal, which now fires), S1 (the generation
moves by itself).
**Consumed by:** everything after it.

**Size — LARGE, and deliberately so.** Twelve exhaustive `RealmId` matches, not the eight both plans counted
(`crates/core/src/pose.rs:52, 92, 153`; `crates/core/src/worldgen.rs:63`;
`crates/core/src/realm_path.rs:93`; `crates/bins/src/lib.rs:1440, 1460, 1536, 2830, 2875`;
`crates/physics/src/worldgen/plant.rs:158`; `crates/client/src/realm_scene.rs:576`). **Two of the misses are
`match &str` realm-name parsers** (`bins/src/lib.rs:1455-1466` and `2870-2880`) — the compiler will **not**
force a new arm there, so a galaxy or universe realm would silently fail to parse from a CLI or devctl string
and the build would stay green. Add a name → id → name round-trip test over every arm, so the omission is a
red that does not depend on exhaustiveness checking. Plus three exhaustive frame matches, the directory-key
migration, the golden split and re-baseline, the radii, and every flight budget in eight process-tier test
files. This is the second flag day.

**Risks.**
- **The HR5 trap, which must be decided in P3 above and not here:** making the galaxy frame realm-bearing
  removes the only cover for two documented-uncoverable fallbacks.
- **Rotation.** At a two-metre step the exact-rotation reach is `cell_edge / f64::EPSILON` = 9.0072e15 m =
  **0.952 light years** (`crates/core/src/pose.rs:540`), **below** the 3.89-light-year mean star gap, so any
  non-identity orientation on an ordinary interstellar conversion is refused loud. It is masked today only
  because every generated placement is identity-oriented (the identity quaternion short-circuits bit-exactly).
  **Decide the policy here; do not discover it.** Galactic rotation is refused at every tier by this bound —
  proper motion, if ever wanted, belongs in the catalogue as a per-star velocity, never as a rotating frame.
- **Homes.** `StoredHome::in_realm` is **derived** at gateway config time from the world regions, not
  persisted — so the stamp cannot protect it, and the distance change would **silently relocate every
  account's home** with no migration and no refusal. (Both plans got this backwards: Plan 2's ground claims
  `StoredHome` persists a pose; it does not, and the real consequence is the opposite of the one stated.) Add
  the gate: an account's home realm and its offset inside that realm are unchanged across the step change, or
  the plan states deliberately that homes move and says where to.
- **Abandonment.** This is the one slice that is not flyable at its midpoint. Split it into a **prepare-half**
  (golden split by mover; directory dual-read) and a **commit-half**, so an abandonment leaves a readable
  directory. See §8.

---

### S10 — A PARENT'S COST STOPS GROWING WITH ITS CHILDREN

> ⚠️ **SUPERSEDED IN PART — READ `owner_decisions_2026-08-26_movement.md` AND `owner_decisions_2026-08-24.md`
> BEFORE BUILDING OR RELAYING THIS SECTION.**
>
> The governed-ceiling passage below plans to KEEP the approach governor and make its per-child walk cheap.
> **The owner's movement law removes that governor** — its own words: *"this law removes that guarantee"*
> about not being able to fly through a moon. A parent never sets a speed; a child hands up acceleration and
> torque and the parent integrates.
>
> ⚠️ **BUT DO NOT DELETE IT IN THIS SLICE EITHER.** The governor is load-bearing TODAY: the realm ceiling is
> not a cap above a speed, it IS the speed (D-MOVE-1), so removing it now pins every occupant at foot pace,
> and S6 sized every band in the world from its guarantee. **In S10 the governor is simply NOT TOUCHED** —
> optimising it is wasted work on something that is going away, and deleting it belongs to the force phase
> (P5), together with the band re-solve. S10 therefore has FIVE mechanisms, not six.
>
> ⚠️ **AND S6's BAND NUMBERS ARE NOT SAFE ACROSS THAT CHANGE.** Slice 6 sized every band in the world FROM
> the governor's guarantee, and measured the alternative as *"bands thousands of times larger than the
> bodies they wrap"*. Removing the governor re-opens that solve.
>
> The other five mechanisms in this slice — the interest fold, the relay fingerprint, the keep-alive, the
> level message, the client's per-delta clone — are unaffected and stand as written.
>
> *(This banner exists because the section was relayed to the owner as current on 2026-08-26, after the
> ruling that overtook it. A plan agreeing with itself proves nothing.)*

**Serves:** SL9, whose own text says this *"must be measured on a realm with many, not argued"*.

**Delivers.** The interest fold inverted so it walks observers, which are bounded, rather than children,
which are not — today `for (region, pose) in &placements` is the outer loop with `observers.iter().any(...)`
inside and an allocating `path.clone()` in the inner body (`crates/sim/src/stub/aoi.rs:528-560`). The relay
fingerprint's whole-state re-encode replaced by an incremental digest. The keep-alive's baseline clear
replaced by a counter compare. The shard-to-gateway level message partitioned. The client's per-delta clone
of the whole box map removed.

**And the governed-ceiling walk, which both plans treated as either an SL9 defect they had to keep or a
hazard they might have to revisit.** Today `governed_ceiling_for_frame` allocates a `Vec` over **every**
direct child, per subject, per tick (`crates/sim/src/stub/dot.rs:355` into
`crates/sim/src/stub/regions.rs:454`). Plan 1 admitted this might be unsolvable — *"the ceiling is a
lower-envelope query, not a containment query"* — and named its escape as deleting the governor, which cannot
happen before forces exist.

**It is solvable exactly, and the reason is a reading of the formula, not an argument.**
`approach_ceiling_mps(child_cap, dist, τ) = child_cap + max(0, dist)/τ`
(`crates/core/src/flight.rs:104`) is **monotone increasing in distance**, and `child_cap` is floored at the
foot speed by `realm_speed_cap_mps` (`flight.rs:93`). So a child at boundary distance `d` contributes at
least `v_foot + d/τ`, and it can only bind the running minimum `v` if `d < (v − v_foot)·τ`. **The
lower-envelope query is therefore a bounded-radius query**, answerable by the grid the child index already
is, by walking cells outward until the shell distance exceeds the bound.

The bound, measured on the world S9 creates: galaxy lawful ceiling `2 × 2^62 / 180 = 5.1241e16 m/s`, τ =
1.10 s, so the search radius is `5.6365e16 m = 5.96 light years`. Against a mean nearest-neighbour separation
of 3.89 ly, and at the placed density, a ball of that radius holds roughly **three** systems, not 150,000.
**The governor stays, and stops walking.**

**The gate, and why it is red today.** Measure wall time, **allocation count** and **emitted bytes** as a
**function** of the child count — at three children and at the census, with one traveller — and assert
**equality** (not "within noise") on the two deterministic arms. Wall time is **advisory only**, printed under
a stated quiet-machine protocol with verified artefacts: this project's own standing rule records three false
results in one session from a raced run, a stale binary and a load-induced red, and a gate that goes red for
the wrong reason gets weakened, which is how negative controls die.

Plus a correctness arm for the bounded-radius query: over a swept set of subject positions on THE world, the
bounded query returns **exactly** the ceiling the full walk returns, bit for bit.

**Consumes:** S9 (final units and frames, so the inverted fold and the query are written once against the
units they will run in — part scheduling, part real: the frame set is exhaustively matched); S4 (an index
keyed on integers).
**Consumed by:** S11 (its byte budget rests on the fold being observer-bounded); S12 (the world neither boots
nor ticks at the census without this, and SL5 forbids a reduced variant to prove it on).

**Size — LARGE.** Six mechanisms in five crates, each with its own cost curve. **The census raise itself is
S12's**, and the cost gate here is taken on the **pure functions** with a child list of any length passed as
an argument — which is a function's input, not a second world, and therefore inside SL5. That split is what
breaks the 7↔8 cycle both this spine's ancestor and Plan 2 contained.

**Risk.** `crates/sim/src/stub/tests.rs` is **15,839 lines** and every sim-side change here adds to it under
HR5. Split it into per-subject modules as this slice's first task, gated only on test-count parity and
unchanged coverage — a mechanical move with a measurable green, done before the file grows again.

---

### S11 — THE SKY STATED ONCE

**Serves:** owner Q2 (all four conditions); Q4 (the reversal, and all four of its conditions).

**Ordered BEFORE the census raise**, against both Plans 1 and 2 and with Plan 3 and the dependency judge.
Their stated edge — *"there must be 150,000 stars to catalogue"* — runs backwards. The gate that matters
("per-tick sky bytes are zero after the first tick") is writable at three children, and the real edge is that
`window.rs:265-282` ships the level as one **un-split** `rows: realms.to_vec()` on `MsgClass::RealmSnapshot`
with `Durability::Ephemeral` — a drop lane. Raising the census first ships an **empty sky with nothing in any
log**.

**Delivers.** The compact catalogue message, granted by the owner's Q4 reversal, with all four conditions:
the client is its only recipient and nothing is sent to realms (a ship holds the same generator and asks it);
a test asserting the **encoded** catalogue and the one **folded from the seed** are identical, byte for byte
(one truth, two producers, which drifts at the first patch and nobody notices); the generation derived from
the content, never hand-incremented; and the client caching it on disk, with the catalogue a **drawing aid
only** — the server validates every warp destination itself.

Placements sent on change, on a re-driven acknowledged lane, with the **receiver stating its generation and
the sender answering from that** rather than from its own memory of what it sent. The keep-alive comparing a
counter instead of clearing its send-on-change memory. A **liveness digest**, so silence is provably "nothing
changed" rather than "nobody is working". The login level chunked under the frame cap, encoded once per
gateway. The client drawing one instanced point cloud rather than an entity per star.

**The gate, and a correction both plans need.** Both gate on *"per-tick sky bytes are ZERO after the first
tick"* — **which is also exactly what a dropped oversize datagram produces.** The gate as written passes on
the failure it exists to prevent. So:

- Gate on the **received** catalogue: the client's rendered star count equals the census on a fresh session.
- A deliberately oversized catalogue is **refused loud at the sender** with a named counter, never dropped.
- The catalogue is **not** re-sent when a crossing bumps the scene epoch — a warp leg is two crossings, so
  getting this wrong re-transmits the whole sky twice per journey.
- The encoded and the seed-folded catalogues byte-identical.
- The **liveness digest is the subject of an assertion**, not a listed deliverable. **LANDED
  2026-08-27** as `ShardToGateway::StarSkyAlive` / `ServerControlMsg::SkyAlive` — a new wire arm, so it
  went through the SL6 gate and the owner approved it. Carries one number, the generation the server
  believes is current, restated on the AoI cadence **whether or not it changed**: it is the one
  statement on this lane that must not be suppressed when it repeats, because the repetition is the
  message. Three readings replace one ambiguous silence — generation matches ⇒ *"nothing changed"*,
  differs ⇒ *"I hold the wrong sky"*, no beat ⇒ *"nobody is working"*. The third arm is an ABSENCE and
  so is left to a caller holding a clock (the client counts confirmations); the watchdog policy is owed
  with the renderer. See D-S11-SKY.
- **LANDED 2026-08-27 — the cache and the generation exchange.** The client holds the galaxy on its own
  disk under `StoreRole::ClientCatalogue`'s stamp plus the catalogue generation, and states what it
  holds; the gateway asks the shard only while some client behind it needs one; the shard answers and
  remembers nothing. `SkyStatedTo` — the memory that could not be made correct — is deleted. MEASURED:
  every byte flip in a cache's row payload reaches the fold, and none is caught by an earlier test, so
  the digest is doing the work rather than the fixture. Owed: the file I/O (no client persistence seam
  exists yet), the quiet-lane watchdog policy, and the instanced point cloud.
- The client's cache carries S1's stamp **and a content digest**; a mismatch **discards and re-requests**
  rather than drawing, and the decode is bounded (row-count cap, magnitude cap) with a counter. A
  byte-flipped cache produces a counted refusal and a re-request, never a drawn frame. (postcard is
  positional: an edited cache decodes into garbage rather than failing, and a point cloud built from garbage
  magnitudes is an unbounded draw.)

**Consumes:** S10 (an emit path already partitioned and a fingerprint already incremental); S1 (the cache's
stamp).
**Consumed by:** S12 (the census can now be raised without the sky going silent); S13 (the same lane one level
up); S14 (the reconnect number).

**Size — LARGE.** The window emit, the keep-alive, the relay fingerprint, the gateway's retained ring, a new
wire arm and the renderer. The scale arms of the gate (client frame time flat at 150,000 points; one
catalogue per gateway at a thousand sessions) are taken in S12 and S14, where their subjects exist.

**Risks.**
- **The failure mode of doing this half-way is silence**, so every part is load-bearing and none is an
  optimisation.
- Send-on-change must be a **value comparison** and never a "does this child move?" test, which would be a
  specific on a lookup path and is forbidden by name (SL4). Enforce with a module dependency rule, not care.
- The owner's law that only full state rides a drop lane is preserved by moving unchanged placements to the
  reliable lane and keeping full state on the drop lane for what changed. **This is the subtlest point in the
  whole plan and the one that most needs a careful reviewer.**
- The catalogue must **never** be client-generated from the seed however tempting — that hands every player
  the ability to enumerate the galaxy offline, and that door does not close again.
- **HR4 needs this lane to pass an identical fixture on two shard kinds**, and the natural second kind — a
  planet with many player-built areas — does not exist today. **UNMEASURED** how that is satisfied; name it
  in the slice.

---

### S12 — A GALAXY GENERATED WHERE YOU LOOK

**Serves:** owner Q8 condition 1; Q9 rulings 1, 3 and 4; Q3 condition 1; Q7 consequence 1 (lazy generation
kept).

### ⚠ THIS SECTION WAS WRITTEN AGAINST AN OLDER GENERATOR — RE-MEASURED 2026-08-28

Four of its claims are stale. Measured against the code in the tree:

| The section says | Measured 2026-08-28 |
|---|---|
| placement is `f(index)` | the DIRECTION is `f(seed, system)`, drawn per system |
| the collinear ring must go | already deleted (owner ruling Q-B, 2026-08-18) |
| the radius follows the count | `real_placement_r_m()` never reads the count |
| the count is a stated target | the DRAW is seeded (`galaxy_system_count`); its bounds are pinned EQUAL |
| the reversibility gate is "red by construction" | **GREEN** — `growing_the_system_count_does_not_move_the_systems_already_placed`, 16 seeds, nothing moves |

Reversibility holds because each system streams from its OWN lineage, so its position is
`f(universe_seed, itself)` and a neighbour appearing cannot reach it.

★ **BUT THE PLACEMENT IS STILL WRONG, AND MORE WRONG THAN "ONE SHARED RADIUS".** Both angles are drawn
UNIFORM ON THE SPHERE, and a uniform sphere is a BALL, not a galaxy: as many stars above the disc as in
it, and no arms, by construction. So all THREE coordinates must come from the shape —

| coordinate | today | needed |
|---|---|---|
| polar | uniform on the sphere | concentrated near the disc plane, because a disc is thin |
| azimuth | uniform | concentrated into arms, because that is what an arm IS |
| radius | ONE constant for every system | a density profile with a bulge |

What IS reusable is the plumbing: the per-system lineage stream, which is what makes G4 hold.

**THE GATE THAT IS ACTUALLY RED** is the one nobody wrote: the shape. See
`owner_decisions_2026-08-27_galaxy_shape.md` G2, G8–G13.

---

**Q8 CONDITION 1 IS ANSWERED HERE, AND THE ANSWER TODAY IS NO — MEASURED.** The owner marked it *"unverified;
worth more than the choice itself."* `system_center_at` (`crates/physics/src/worldgen/generate.rs:215-234`,
read in full) takes the index `n` and returns `direction × (config.stellar.system_ring_r_m × [n != 0])` —
**every non-home system sits at exactly one radius**, with only its direction seeded, and its identity is
`f(galaxy, index)` through `SYSTEM_SALT` (`generate.rs:44-47`). That one radius is derived from the galaxy
radius. **So today, changing the count, the cap or the galaxy's size moves every star that exists**, and the
galaxy count and the universe size are launch-freezing decisions — exactly what the condition exists to
prevent.

**Delivers.** Placement becomes **position-addressed**: a system's identity and placement are drawn from the
cell of space it occupies, and generation is lazy and per-cell. The **per-pair minimum gap**: each pair's own
radii plus a margin, with one light year as the floor for ordinary pairs (Q9 ruling 1) — a rare giant simply
pushes its neighbours away, and every stellar type is kept. **A binary is written down as one system realm
containing two stars** (Q3 condition 1), before the gap rule silently deletes binaries. The all-pairs boot
fence replaced by a construction proof plus a sampler plus an **exhaustive check over the neighbourhood the
game actually touches**. The census raised to 150,000 from today's three (`WORLD_SYSTEM_COUNT: u32 = 3`,
`generate.rs:51`).

**The gate, and why it is red today.**
- **The reversibility arm, and take the achievable form:** grow the census from N to N+1 and assert every
  previously generated system's identity **and** position are bit-identical. Red today by construction. (The
  owner's own phrasing — enlarge the *universe* radius — becomes achievable too once S7 has decoupled the
  cap; assert both, and note that only S7 makes the second one gateable at all.)
- Minimum separation over a large random sample meets each pair's own rule and the one-light-year floor, and
  the construction proof's bound is asserted **tight** against that sample, so a proof that drifts from the
  generator is caught rather than trusted.
- **Boot never enumerates** — a counter that must stay at zero. Red today: the all-pairs fence
  (`crates/physics/src/worldgen/guards.rs:262-277`) would be **11,249,925,000** comparisons per boot at
  150,000, in every world-deriving process, and there is a 284-seed sweep over it.
- The scale arms deferred from S11: client frame time flat at the census, one catalogue per gateway.

**Consumes:** S11 (a lane that survives the census); S10 (per-tick cost independent of the child count); S9
(the coordinate room — the population needs a 2.19e18 m radius and today's galaxy shell is 2.2488e15 m,
**974× too small**); S7 (the cap that sizes the per-pair rule); S6 (bands the sibling fence can now see).
**Consumed by:** S13 (the same position-addressed placement, one level up); S14.

**Size — LARGE.** It replaces the placement law, the census, the laziness and two boot fences at once, and it
cannot be split further without leaving the world in a state where the fence and the placement disagree.

**Risks.**
- **The quadratic fence is the world's negative control**, and its replacement is strictly weaker. Replace it,
  never delete it; keep the exhaustive form over the touched neighbourhood as a hard gate; make the
  construction proof carry the guarantee.
- **The star field must never look evenly spaced** (owner, explicit), and the neighbouring worktree's
  jittered-lattice answer was measured **5.75× too even**. The draw-and-retry placement's actual regularity
  must be **sampled and looked at** before it ships. **Nobody has yet seen this sky.**
- The mean nearest-neighbour separation of the placed field is **UNMEASURED** — generate it and look.

---

### S13 — THE LEVEL ABOVE, AND ROOM LEFT FOR A DOOR

**Serves:** owner Q8 (61 galaxies, condition 2); Q5 (the universe level becomes load-bearing; runtime realm
creation).

**Delivers.** The universe as a running realm with children; galaxies generated lazily by S12's
position-addressed placement one level up; the galaxy shell crossable **outward**, which has never been done;
**galaxy kinds that differ measurably** (Q8 condition 2, or galaxy 61 is 150,000 systems statistically
identical to galaxy 1 and the whole feature is a long corridor to more of the same); and the mutator for a
shard's region set, which does not exist today — `RealmRegions` is boot-only — and which the owner named as
owed by both player-built ships and player-built tunnels.

**The gate, and why it is red today.**
- Fly out of a galaxy into the space above, and assert the handover at the galaxy shell shows the **same zero
  drawn difference** the system shell does. **Not** "fly to another galaxy": Q5 rules physical inter-galaxy
  flight dead under A4, and the corridor is the answer. Reaching galaxy 61 is a **generation** assertion here,
  not a flight. (Both plans wrote a gate that quietly required the out-of-scope tunnel; this is the re-cut.)
- Galaxy 61 differs from galaxy 1 on a statistic that would be **equal** if both were drawn from identical
  rules — the gate for a condition that is otherwise unfalsifiable.
- The existing visibility-climb fence still refuses a world in which another galaxy's stars could reach you.
- Red before: nothing has ever been drawn from the universe frame, the galaxy shell has never been crossed
  outward, and the region set is fixed at start-up with no way to add to it.

**Consumes:** S12 (the placement law reused one level up rather than re-invented — the whole reason the
placement rewrite comes first); S9 (the universe tier and the realm-bearing frames).
**Consumed by:** the tunnel realm's own design pass, which is out of scope.

**Size — MEDIUM-LARGE.**

**Risks.**
- **Leave room without designing it.** The corridor is expected to be a **sibling of the galaxies under the
  universe**, so this slice must not assume the universe's children are all galaxies; the per-tier fence must
  not assume a fixed number of tiers; and the coordinate unit must stay a property **carried by a frame**,
  never a small enum matched exhaustively outside its own module.
- From the owner's own recorded list: **a player who can create a realm can make the cluster start
  processes**, so creation must cost enough and be capped **where realms are started**, not where they are
  built.

---

### S14 — FLY IT, AND THE HONEST CAPACITY PLAN

**Serves:** SL8 (seamlessness); owner Q2's owed measurement; Q7 consequence 2; the standing measure-don't-
argue rule.

**Delivers.** The full warp leg flown on THE world: leave a planet, watch the star shrink, cross into the
galaxy, accelerate, see real parallax passing stars, slow down, arrive with a star growing from a dot. Plus
the four-part **not-a-loading-screen** instrument kept as a standing tool for policing any future high-speed
sequence: the world keeps running, the player keeps control, the frame never stops updating or goes flat, and
the frame-to-frame pixel difference stays inside the band of ordinary high-speed flight.

And the numbers nobody has: the shard port band stated as a **capacity plan** rather than an accident (~500
shards from a 1,000-port band, `crates/bins/src/lib.rs:200-202`, binding around 550 concurrent players); the
galaxy shard's tick against traveller count (Q7 consequence 2, *"nobody has measured a galaxy shard's tick
against traveller count"*); and the reconnect burst.

**The gate.**
- No black frame; no brightness step at the handover; drawn centre, drawn radius and drawn colour differ by
  **exactly zero** across the handover frame; the live realm count never exceeds a handful; frame-to-frame
  pixel difference across the crossing tick inside the band of an ordinary tick at the same speed.
- **The reconnect gate asserts the refusal behaviour under the production cap**, not only egress. The cap
  already refuses: `sessions.by_session.len() >= config.tuning.max_sessions` increments
  `sessions_refused_capacity` and pushes a close (`crates/connection-plane/src/gateway/client.rs:107-114`). A
  synchronised thousand-player reconnect hits that refusal **before** any byte budget, and a refusal with no
  retry discipline is the classic thundering herd. Assert a refused session carries a retry-after the client
  honours with jitter, and that the burst **converges** rather than oscillating. **Do not raise the cap for
  the test** — measuring above it measures a system that will never exist.
- The galaxy shard's tick inside a stated budget at a stated traveller count, written so it **can** refuse.

**Consumes:** everything.
**Size — MEDIUM**, and mostly gate rather than production code — which is what it should be if every slice
before it did its job.

**Risk.** These measurements need a quiet machine and verified artefacts, or they are worse than no
measurement at all.

---

## 5. THE MEASUREMENTS THIS PLAN OWES

Every figure the owner's decision record marked ⚠ (the design's, not a measurement), and which slice turns it
into a number. Plus figures I corrected in the reading.

| # | The design's figure | Where it comes from | Turned into a number by |
|---|---|---|---|
| 1 | The universe step and the resulting galaxy count | Q1 | **S8** (the step, at the fence equality) and **S13** (the count that falls out) |
| 2 | ~21.75 MB per player per tick; a login message ~27× over the frame cap | Q2 | **S11** (measured on the real emit path, at the real census in S12) |
| 3 | **A thousand players reconnecting at once** — owner-owed before it ships | Q2 | **S14**, against the real session cap |
| 4 | ~1/80 of space forbidden at one light year; ~1/10 at two | Q3 | **S12** (sampled on the placed field) |
| 5 | Compression ~24.6× today | Q7 | **ALREADY MEASURED** — `24.567816882382665`, the frozen constant, confirmed in the golden header (*"chi 16.378 → 24.568"*). ⚠ A comment near the assertion still says 16.378; fix it in the first slice that touches that file. |
| 6 | **A galaxy shard's tick against traveller count** — explicitly named unanswered | Q7 consequence 2 | **S14** |
| 7 | ~9 million systems; 61 galaxies; Local-Group spacing | Q8 | **S13** |
| 8 | **Does the generator place by position or by counting outward?** — *"unverified; worth more than the choice itself"* | Q8 condition 1 | **ANSWERED HERE: it counts.** `system_center_at` is `f(index)` on one shared shell radius (`generate.rs:215-234`). **S12** makes it position-addressed; **S7** removes the second half of the irreversibility (the mass cap). |
| 9 | The heaviest star: ~6 suns at a global gap; ~30 at the coordinate limit; ~9 at one-eighth density | Q9 rulings 1 and 2 | **S7**, by the code's own bisection, at each candidate step, with the reproduce-the-shipped-answer arm |
| 10 | Mean nearest neighbour ~3.9 ly; galaxy 232 ly → 215 ly at a lower count | Q9 | **S12** (measured on the placed field, not the formula) |
| 11 | One-eighth density to double the gap | Q9 | **S7** — likely moot: if the coordinate limit stops binding, density is not the knob |
| 12 | The jittered lattice measured 5.75× too even | Q3 | **S12** — sample the regularity and **look at the sky** before shipping |
| 13 | `guard_root_representable` would refuse the new world by 1,024× | Q1, and the design | **MEASURED HERE AND CORRECTED: 2^25 = 33,554,432×** against the universe root; 2,048× against the galaxy shell. The 1,024× figure is a one-metre galaxy shell, not the root the fence reads. |
| 14 | Upward tier-conversion residual 2^-63 m | the design §4 | **CORRECTED HERE: 2^-51 m = 4.4409e-16 m per axis.** Conclusion (far below a proton) survives. Asserted in **S8**. |
| 15 | 150,000 systems fill 95 % of the galaxy shell | the design §4.3 (its **one-metre** solve) | **CORRECTED HERE for the ruled two-metre step: 47.5 % of the radius, 10.7 % of the volume.** |
| 16 | "Two entities 100 m apart must draw 100 m apart — red today" | all three plans | **CORRECTED HERE: the gate is GREEN today and cannot fail.** Working in §S4. The defect is real; the gate was not. |

**Still UNMEASURED after this plan, and named as such:** whether a played-in world exists on a persistent
volume (§P0, a question for the owner); the mean nearest-neighbour separation of the *placed* field, as
opposed to the formula (S12 measures it); how HR4's two-shard-kind requirement is satisfied for the sky lane
(S11 names it); and `vd-physics`' coverage tier, which I did not check.

---

## 6. WHAT WE ARE DELIBERATELY NOT DOING

**Owner-deferred:**

- **The tunnel realm's internals.** Out of scope by ruling until just before the voxel block design. This plan
  leaves **room, not design**: S13 must not assume the universe's children are all galaxies; the per-tier
  fence must not assume a fixed tier count; the coordinate unit stays a property carried by a frame; and
  runtime realm creation is capped where realms are **started**, not where they are built.

**Because the machinery they need does not exist — and the rule this plan keeps, from Plan 2, verbatim:
*the forces themselves are a phase, not a slice, and nothing here may be planned to depend on them*:**

- **Ships, engines, thrusters, mass and forces** — the producers of A4's *"a ship computes its own forces"*.
  There is **no force anywhere in this tree**: the integrator sets velocity from a commanded speed
  (`crates/sim/src/stub/dot.rs:429`), every transient advance passes zero acceleration explicitly, and
  `crates/sim/src/coupling.rs` is **60 lines** of sealed `EffectFree` marker whose ports are deferred by name
  to P8.
- **Retiring the approach governor**, and therefore the *control* half of the movement law. This plan lands
  A4's **precondition** (speed-sized bands, S6), its **safety machinery** (swept membership, S5) and the
  **clamp shape** the transient path already has — and stops. Retiring the governor is not a schedulable
  choice today: it is the **only** thing that produces speed, because the room-scaled ceiling *is* the stick's
  multiplier (`v_allowed` → `throttle_axes_scale`, `dot.rs:405-419`). Delete it with no engines and every
  occupant is pinned at the foot speed, and a galaxy crossing becomes `2 × 2.2488e15 / 500 = 9.0e12 s`. That
  is why the flight-budget re-baseline in this plan happens **once**, at S9, and not twice.
- **A basic flight computer in every hull**, and player-written autopilots — owed with A4, carrying A5's three
  recorded risks: a bounded loop delay the autopilot can read; **a realm stating its own physical character**,
  which the ruling already flags as probably a fresh SL6 ask; and a hard per-ship time budget so an overrun
  degrades the ship, never the tick.
- **Trajectory as durable checkpoint-carried state.** Today velocity is an output, so nothing accumulates and
  nothing needs carrying. The moment forces integrate, it becomes state that must ride the transfer envelope
  and the checkpoint.

**Because they are orthogonal to this arc and would enlarge every slice:**

- **Per-family, field-by-field migration of stored records** (D-48 item 2). Refusal, not conversion, is the
  honest answer while no played-in world exists — a migration needs a reader for the version *before* the
  stamp, and that version has no stamp, so a pre-stamp store is indistinguishable from a corrupt one. It
  becomes mandatory at the first rolling redeploy over a live store. **Contingent on §P0's answer.**
- **The in-flight saga phase-evolution decision** (D-48 item 3). Take the written drain-before-deploy policy
  now — one paragraph, enforced in the deploy procedure alongside S1's `OnDelete` rule — rather than the large
  ordinate scheme.
- **Cluster and orchestrator identity in the stamp** (D-47 item 1(b)). Neither exists as a value anywhere in
  the tree; the field is shaped in S1 and filled when the identity exists, because a field with a lenient
  check is inert code and fails HR5.
- **The boot-counter sidecar's own format version.** A real gap — a fourth durable artefact with magic but no
  version — but on the incarnation path, not the coordinate path.
- **Backup, restore and replication of the durable store** (D-47 items 2-3); the Tombstone half of D-6.
- **The three fences the rewritten SL1 depends on (D-SL1-2).** Nothing in this plan consumes a realm being
  *told* its own placement — the sky's consumer is the client, and **the client is not a realm**. The plan must
  therefore not accidentally build the *reading*, and the fences stay owed against the day something does.
- **Per-entity interest filtering on snapshots (D-9) and the read-subscription lane for remote avatars
  (D-RLM-18)**, with the range and line-of-sight bound on streaming a hull's interior to outsiders. The
  clarified SL2 makes both lawful and urgent for any crowded scene; nothing in this plan needs them.
- **Galactic rotation.** Refused at every tier by the exact-rotation reach (0.952 ly at a two-metre step
  against a 3.89 ly star gap). Proper motion, if ever wanted, belongs in the catalogue as a per-star velocity.
- **The brightness control's demotion.** With compression at exactly one the sky is physically correct, so the
  knob survives as a **picture** control and not a physics one. That is a record to write, not code to build.
- **Splitting a galaxy's authority across hosts without creating new realms** — the reserve the owner named if
  S14's traveller measurement says one galaxy per process does not survive. Measured in S14, unbuilt here.

---

## 7. WHERE THIS PLAN CAN BREAK

| # | Finding | Resolution, or stated acceptance |
|---|---|---|
| 1 | **§P0 is unanswered:** whether a deployed volume holds a played-in world. UNMEASURED from this tree. | **ASK before S1 is cut.** If yes, S1 grows a pre-stamp reader and roughly doubles; nothing else in the order changes. |
| 2 | **A boot refusal becomes an unbounded respawn loop**, because a fork-then-exit is counted as a successful launch (`rlm_runtime.rs:325-331`). | **Resolved inside S1**: refuse at the spawn decision too, and count an early death as a launch failure. Gate on a **widening retry interval**, not on an eventual give-up. |
| 3 | **An orchestrator refusal orphans its own children** and destroys the path to reclaiming them (own process group; `Child` does not kill on drop; the launch ledger is behind the same refusal). | **Resolved inside S1**: the launch ledger stays readable-for-reaping, or the orchestrator kills its recorded process group before exiting. Gated: every recorded child dead and every port free before exit. |
| 4 | **The ALPN refusal cannot carry a diagnosis** (a `no_application_protocol` alert has no field, no value, no unit), yet Q1 condition 3 demands one. | **Resolved inside S3**: keep ALPN as the hard refusal (free, outside SL6); put the unit readout on the health endpoint, the startup log and the dialling side's failure log, and gate on **those**. |
| 5 | **Speed-sized outsets coarsen the child index for every child** and can re-break today's O(1) lookup. | **Resolved inside S6** with a candidates-per-query gate on THE world's forest. If it fails, nested grids per magnitude are a design change scheduled **there**. |
| 6 | **Two flag days on the wire** (S3's shape, S9's discriminants). A spine that folded them together would pay one. | **ACCEPTED, and stated.** The price buys a window-free ordering: the refusal provably exists before the unit moves. Free while no fleet is live; two forced client updates if one were. S1's `OnDelete` rule attaches to each. |
| 7 | **S9 moves every distance in the world at once.** Fixed test values, recorded pictures and every flight budget shift together. | **ACCEPTED — it is the owner's own instruction** (*"the largest single re-measurement in the plan, to be scheduled as one"*). Splitting it would re-baseline the same gates twice. Mitigated by S7 landing first, so the *mass cap* does not move with it. |
| 8 | **S9 is the one slice not flyable at its midpoint.** | **Resolved** by splitting it into a prepare-half (golden split by mover; directory dual-read) and a commit-half. See §8. |
| 9 | **Making the galaxy frame realm-bearing breaks HR5 in two documented places.** | **Resolved by §P3**: take the "does `FrameRef::realm()` become total?" ruling in writing before S9 starts, not when the gate goes red. |
| 10 | **Rotation is refused for ordinary interstellar conversions** at a two-metre step (0.952 ly reach vs 3.89 ly gap), masked today only because every placement is identity-oriented. | **Decide the policy inside S9.** Either rotations are applied only in the local frame (a structural fence) or the refusal is accepted with a named consumer. |
| 11 | **Homes are derived, not persisted**, so the distance change silently relocates every account's home and the stamp cannot protect it. | **Resolved inside S9** with a gate: home realm and in-realm offset unchanged across the step change, or a deliberate statement that homes move and where to. |
| 12 | **The universe radius is a literal**, its tie to the step a doc comment. | **Resolved inside S8**: make it an expression of its tier's step, and extend the exact-equality pin to all three tiers with the refusal arm driven. |
| 13 | **The all-pairs separation fence is the world's negative control** and its replacement is strictly weaker. | **ACCEPTED with mitigation** in S12: replace, never delete; keep the exhaustive form over the touched neighbourhood; assert the construction proof's bound **tight** against a large sample. |
| 14 | **The sky gate passes on the failure it exists to prevent** (zero sent bytes == a dropped datagram). | **Resolved inside S11**: gate on the **received** star count and on a loud refusal of an oversize catalogue. |
| 15 | **The client's cache is untrusted input on the client's own decode path.** | **Resolved inside S11**: stamp + content digest; mismatch discards and re-requests; bounded decode with a counter. |
| 16 | **A synchronised reconnect hits the session cap before any byte budget**, and a refusal with no retry discipline amplifies the herd. | **Resolved inside S14**: assert the refusal behaviour under the **production** cap, with retry-after and jitter, and show convergence. |
| 17 | **Wall time is not a falsifiable bound on this machine** (three false results in one session, on record). | **Resolved**: allocation count and emitted bytes are the gate arms and are asserted **equal**; wall time is advisory under a quiet-machine protocol. |
| 18 | **The star field may still look like a lattice.** Nobody has seen this sky. | **ACCEPTED with an instrument**: S12 samples the regularity and the sky is **looked at** before it ships. |
| 19 | **S7 could invalidate the ruled step.** | **This is the point of S7.** It runs before the one-way door, so the owner re-rules while it is free. Its own overhead if the step turns out right is the printing harness — paid deliberately. |
| 20 | **Five slices show the owner nothing new to fly.** | **ACCEPTED, and stated in the owner summary.** A spine that front-loaded the sky would show something sooner and pay for it by re-baselining the sky when the distances moved under it. |
| 21 | **`crates/sim/src/stub/tests.rs` is 15,839 lines** and three slices add to it under HR5. | **Resolved**: a mechanical per-subject split as the first task of the first sim-side slice that follows S9 (S10), gated on test parity and unchanged coverage. |
| 22 | **HR4's two-shard-kind requirement for the sky lane** has no natural second kind today. | **UNMEASURED.** Named in S11; must be answered when that slice is cut. |

---

## 8. IF WE STOP HALF WAY

Every slice boundary below is stated as **flyable** (the game runs and is no worse) or **not flyable**.

| After | Flyable? | What the world is |
|---|---|---|
| **S1** | ✅ | Unchanged to play. Saved files now carry a label and refuse a mismatch out loud; a refusing shard backs off instead of respawning forever; the deploy procedure is written down. |
| **S2** | ✅ | Byte-identical, measured. Every boundary owns its band; we now know, per realm, what each band must be. |
| **S3** | ✅ | A flag day is paid. Mismatched builds refuse each other; every position log names its frame and unit. |
| **S4** | ✅ **and better** | Convoy flight is exact. The quarter-metre drawing grid at star distances is gone. |
| **S5** | ✅ | Containment answers the same thing it answered before (proved bit-for-bit) and can now answer it for a fast mover. |
| **S6** | ✅ **and visibly better** | Boundary thrash stops; the parked flight test comes back green with its bound untouched. A4's precondition is discharged. |
| **S7** | ✅ | One golden regeneration if the cap value moves — with a stated reason. The heaviest star no longer follows the galaxy's size. |
| **S8** | ✅ | The ladder exists and nothing selects it. **This is the last safe stopping point before the one-way door.** |
| **S9 prepare-half** | ✅ | Goldens split by mover; the directory reads both key shapes. Nothing has moved. |
| **S9 commit-half** | ⚠️ **NOT flyable mid-way** | Distances, directory key, radii and flight budgets move together. Abandoning **inside** it leaves owner records keyed one way and goldens pinning another. **This is the only such slice in the plan, and the split above is what bounds it.** |
| **S9 complete** | ✅ **and the first visible change** | Three stars at their **true** separation. Compression 24.568 → 1. Sizes, brightness and parallax physically correct. |
| **S10** | ✅ | Same world, cost no longer growing with the child count. |
| **S11** | ✅ | The sky costs zero bytes per tick and survives a reconnect. |
| **S12** | ✅ | A hundred and fifty thousand stars, placed by position, lazily. |
| **S13** | ✅ | The universe is a place; galaxies differ; realms can be created while the game runs. |
| **S14** | ✅ | The full leg flown, and the capacity numbers are numbers. |

**THE ANSWER, PLAINLY: the last safe stopping point is the end of S8.** Everything up to there leaves a
working game that is strictly better than today, and no coordinate one-way door has been walked. From S9
onward the world is at its true distances and there is no going back — which is why S7 and S3 sit in front of
it, and why S9 is split into a prepare-half and a commit-half.

**The second natural stopping point is the end of S12**, which is a complete, coherent, playable world: one
galaxy, a hundred and fifty thousand real stars, a sky that costs nothing, all at true scale. S13 and S14 are
the level above and the proof.
