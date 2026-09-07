# Verdict — the law refutation of report 08, the render seam and seamlessness

**Report under test:** `docs/investigation/2026-09-07/08_render_seam_seamless.md`
**Lens:** the engine-agnostic render seam and seamlessness (V2.8, V2.9, SL8).
**Date:** 2026-09-07.

**Result: REFUTED.** The report holds three load-bearing claims that break a law, two numbers that
are wrong, and four gaps the owner cannot decide around. The report's ground-truth section is
otherwise accurate: I re-ran every code citation and the great majority are true.

Read this file with the report open. Each finding names the report's own section, the law it meets,
the file and line in the code, and the fix.

---

## A. What I checked, and what stands

I re-ran every `file:line` in section 0 of the report. These claims are TRUE:

- The client library is renderer-free, and its own package description says "NO client-side
  prediction" (`crates/client/Cargo.toml:6`, `crates/client/src/lib.rs:1-15`).
- The engine consumes one type, `MeshPrim`, and never learns the word "sphere"
  (`crates/client/src/realm_scene.rs:769-796`). The rotation field is there
  (`realm_scene.rs:784`) and the renderer applies it (`crates/client-render/src/lib.rs:1569`).
- The Bevy renderer builds a mesh from a vertex array and nothing else
  (`crates/client-render/src/lib.rs:1828-1846`), and its own comment says the terrain quads slot in
  there unchanged (`lib.rs:1824-1827`).
- A drawn realm is one `SceneRow` — a realm id, a parent id, a stamped pose and a skip-unknown bag
  (`crates/wire/src/channels.rs:378-391`). The bag carries three tags
  (`crates/core/src/look.rs:30-48`). A realm states its own look
  (`crates/sim/src/stub/window.rs:485-496`).
- `BulkKind::ChunkSnapshot` and `BulkKind::ChunkDelta` are reserved names with no producer
  (`crates/wire/src/channels.rs:292-294`).
- The eye sits at the render origin and reduces on the lattice before it flattens
  (`crates/client-render/src/lib.rs:534-566`, `1264-1275`).
- The camera's up is the origin frame's `+Y` (`lib.rs:1136-1145`). There is no local up anywhere.
- The renderer never names `Msaa`. Two comments in the whole workspace mention it, neither in the
  renderer (`crates/bins/tests/render_boxes_smoke.rs:107`,
  `crates/client-harness/src/verdict.rs:149`).
- `noise = "=0.9.0"` is declared and no crate uses it (`Cargo.toml:78`). Rapier is not in the
  manifest.
- The world-generation tag exists and names no generator version and no arithmetic profile
  (`crates/core/src/store_stamp.rs:118-147`, `crates/wire/src/admin.rs:373-378`).

The report's central law argument in section 2.2 also STANDS, and I tried hard to break it. Under
SL10 a moon's look for its ground IS one statement: its seed, its address, and the diffs its miners
made. Every rung is a pure function of that one statement. So the pilot's engine choosing a rung is
the observer sampling what the moon already said, exactly as one star cloud is one statement drawn
at any distance. The moon still draws itself. No pose crosses for a rung. SL3 is not broken.

---

## B. The findings

### B1. The diff never reaches a looker who is not standing on the moon (MISSING)

**The report says** (§2.3 item 1, §6 item 8, §8 item 3): the realm's shard derives the diff interest
from the poses it authors, and the client states no interest.

**The code says** a realm's shard does not know where a looker is unless the looker is its own
occupant. A window carries a SCOPE and no eye: `WindowScope` is `Occupants` or `Child(RealmId)`
(`crates/wire/src/session_flow.rs:582-590`), and the window registry is keyed by the opener, never by
an observer (`crates/sim/src/stub/window.rs:38-46`).

**The law says** no occupant pose crosses a realm boundary (SL2). The pilot's pose may not enter the
moon.

**So the report's own example fails.** In §2.3 a miner digs a tunnel on a moon. A pilot ten
kilometres up, inside her hull, is told to receive the moon's coarse pyramid entry for that tunnel.
The pilot is an occupant of the HULL, not of the moon. The moon holds nobody. The moon therefore has
nothing to derive an interest from, and the report has refused the only other source by name. No
carrier exists. The tunnel is invisible to every looker who is not on the ground — which is seam
kind 6, arrival pop, produced by the design itself.

**Fix.** Name the carrier and make the SL6 ask out loud. The lawful shape is the connection plane,
because SL2's own clarification says the gateway is not a realm: the gateway holds both sides, so
the gateway can state one scalar interest radius per open window, and the moon ships pyramid entries
inside it. State the data (one radius per window), the direction (gateway to realm, not realm to
realm), why the moon cannot compute it, and the cost of doing without. Do not leave §8 with two asks
and a refusal.

---

### B2. The predicted eye breaks the no-prediction fence (BREAKS_LAW)

**The report says** (§3.2): "the client library evaluates each chunk's tier at the predicted eye
`p + v·t_warm` … The velocity is the delivered row's velocity; the client derives nothing."

**Both halves fail.**

1. The eye stands at the pilot's own avatar, which is an OCCUPANT, not a realm row. The client's
   render sample for an occupant is `RenderPose`, and `RenderPose` has no velocity field at all —
   frame, cell, pos, orient, tier, and nothing else (`crates/client/src/interp.rs:84-101`). There is
   no "delivered row's velocity" for the eye to read.
2. The client crate states a STRUCTURAL fence, twice: "the interpolation sample is a `RenderPose`
   with NO velocity field, and `StampedPose::advanced_ballistic`/`.vel` are never read on the render
   path" (`crates/client/src/lib.rs:11-15`, repeated at `crates/client/src/interp.rs:10-13`). The
   binding mandate is in CLAUDE.md: NO client-side prediction. SL10 §7 repeats it in the owner's
   newest words: the client never derives a pose or a velocity.

Computing `p + v·t_warm` derives a future pose of the pilot's own avatar on the client. The report
lists it as a law conflict nowhere in §8.

**Fix.** Warm ahead without a derived pose, and put both lawful shapes to the owner. Either the
moon's shard states the warm radius it already grows by closing speed (the reach ruling R8 grows a
reach that way, and `crates/sim/src/stub/aoi.rs:369` already reads a predictive horizon), or the
residency rule reads the SPREAD of the delivered track — the distance the avatar actually moved
between the last two delivered stamps — which measures the past and predicts nothing. Whichever the
owner picks, keep the fence: name the module rule that holds the residency reader out of the draw
path, with a control that fails when somebody deletes it.

---

### B3. The engine instances a tree, and the tree has collision (BREAKS_LAW)

**The report says** two things that cannot both be true.

- §1.1, the cell lane: the engine "may read the cell field … It uses them for textures, decoration
  density, **instancing of trees (V2.2)**, HUD widgets on faces, and ambient occlusion."
- §8 item 5: "A seed-placed tree is a static shape function of `(seed, address)` … The server
  computes its collision from the same record."

**The owner's words** (V2.2) are: a tree is one object on the surface, the client renders it as a
tree, "but then the collisions are calculated on the server, so server somehow should know the
shape".

A tree therefore has a collision shape. If Unreal instances the tree with its own foliage tools, the
tree's drawn trunk is C++ output while the tree's collision trunk is Rust output. That is two
implementations of one shape. SL10 §2 forbids it ("a port is FORBIDDEN") and SL10 §5 forbids its
result ("what the player sees is what the player stands on"). The report refuses option (a) in its
own table for this exact reason — "an engine that MESHES writes a second surface extractor in C++.
Refused." — and then permits the same thing through the cell lane.

**Fix.** Split the cell lane by one test, and write that test into the seam contract: **anything the
server answers a collision question about comes out of the shape lane; the cell lane carries only
what nobody can walk into.** A tree's trunk and its branches are shape. Its leaf cards, its bark
texture, its wind sway and the grass under it are cell lane. Say so in §1.1, and put the trunk
through the shared mesher with the ground.

---

### B4. The 870 metre interest radius is a person's number, not a cell's (WRONG)

**The report says** (§2.3 item 1): the cells whose fine diffs matter are those "within the distance
at which one 1 m cell is drawable, `1 m / 1.15 × 10⁻³ rad ≈ 870 m`".

**The code says** 870 m is the reach of a PERSON. `OCCUPANT_FIGURE_EXTENT_M = 0.5` — the
circumscribed radius of a figure one metre across — and its own example reads: "at this extent and
the drawable angle … a lone occupant is shipped to an observer inside about 870 m"
(`crates/core/src/look.rs:61-70`). The formula is `extent × cot(θ/2)`
(`crates/core/src/geometry.rs:1148-1154`), not size divided by the angle.

A cell one metre on a side has a circumscribed radius of `√3/2 = 0.866 m`. Its reach is
`0.866 × cot(θ/2)`, about **1,505 m** (ESTIMATED; arithmetic on `geometry.rs:1152` at the angle
`drawable_theta_min_rad()` returns, `geometry.rs:1171-1174`, asserted 1.1506 × 10⁻³ rad at
`geometry.rs:3000-3006`). The report's radius is too small by 1.73 times.

**Why this is load-bearing.** The number sets how far the moon ships its fine diffs. If the moon
ships them to 870 m while the pilot's client draws fine cells to 1,505 m, then the tunnel a miner dug
is a hole in the ground inside 870 m and solid rock in the ring beyond it. The player walks toward
the tunnel and it appears. That is seam kind 6 again, manufactured by an arithmetic slip.

**Fix.** Take the radius from the code's own function with the CELL's circumscribed extent, and say
which extent it is. Then re-derive every other range in §3.1 and §3.2 from that same call.

---

### B5. The sun is not an ancestor (WRONG)

**The report says** (§1.2): `sun_set(&snapshot)` comes "from the luminous rows in the ancestor chain
(kind-blind)".

**The world says** otherwise. The chain is universe, galaxy, star system, planet. The STAR is a child
of the star system, so for a pilot on a planet the star is the planet's SIBLING, never its ancestor.
A star is its own realm kind (`RealmId::Star`, matched at
`crates/client/src/realm_scene.rs:719-731`). Walking the ancestor chain finds no sun, and the pilot
lands in the dark.

The report contradicts itself thirty lines later: §4.4 says "the sun direction from `star row −
observer` in the composed frame", which is right.

**Fix.** Write "the luminous rows IN THE WINDOW", not "in the ancestor chain". The window already
ships the children in range (the reach ruling), and the star's row is one of them with its
photometric tag (`crates/core/src/look.rs:32-34`). Keep it kind-blind: a row is a sun because it
carries light, never because it is a `Star`.

---

### B6. The seam contract states no vertex quantum, so it shuts the door on V2.4 (MISSING)

**The report says** (§9, first door): the `ChunkGeometry` seam contract — "cell-space integers,
packed attributes, indices, the header with the tier and the skirt" — must shut BEFORE the Unreal
plugin starts and before the Bevy encoder is written. It is a one-way door.

**The report never says what one integer step is worth.** V2.4 is the owner's requirement for blocks
smaller than one metre, so a player can build little details that fit inside the one-metre cell. If
the vertex integers count whole metres, a half-metre handle on a hull door cannot be expressed, and
the door §9 says must shut first shuts on V2.4. The report's §5 item 3.1 says only that the
collision cell stays one metre, which is a different statement about a different thing.

**Fix.** Put the vertex quantum in the `ChunkGeometry` header as a named field beside the tier, and
state its value as a decision for the owner (a sixteenth of a metre and a thirty-second are the two
candidates the sub-metre report should price). Then one seam carries a hull's half-metre handle and a
moon's ground through the same integers.

---

### B7. No structural fence keeps the client's generator to shape only (MISSING)

**The report asks** (§8 item 1) for the world seed and each body's shape record — "per body its
address, radius, grid family and tier depth" — to reach the client once.

**The law is stricter than the ask.** SL1 clause 5 says the placement machinery may not NAME a
realm's own position, and it says how: "Enforced by a crate/module dependency rule with an
observed-failing control … never by care." SL4 demands the same shape for motion. SL10 §7 says the
client never derives a pose. The galaxy-shape ruling says every number the placement reads is
seed-derived — so a client that holds the seed and a body's address holds the means to fold that
body's absolute from the root, which SL1 clause 4 forbids in the strongest words in the file.

The report proposes no fence, and §9 lists no such door.

**Fix.** Add the door. The generator crate the client links must expose SHAPE AT AN ADDRESS and
nothing else: no placement function, no orbit, no absolute. Split the crate if the placement solve
lives in it today, and add a control that fails when the client-linked surface grows a placement
symbol — the same shape the crossing path already uses against motion. This is a one-way door,
because the Unreal plugin links whatever the split leaves behind.

---

### B8. "The shipped tick is 50 Hz" is the dev cluster's tick (UNMEASURED_AS_FACT)

**The report says** (§0.5 and §10 item 7): "The shipped tick is 50 Hz (`crates/bins/src/lib.rs:414`)."

**The code says** that constant is named `DEV`, and its doc reads "The standard **dev-cluster**
parameters (fast ticks for snappy bring-up)" (`crates/bins/src/lib.rs:412-414`). The rate is a
per-shard knob passed in, and the code says so where it is read: "`tick_hz` is a per-shard knob
passed in" (`crates/core/src/kinematics.rs:148-153`). The wire carries the rate as a message the
client learns (`crates/wire/src/channels.rs:136`).

The substance the report wanted — the base's "20 Hz everywhere" is stale — holds. The fact it stated
does not.

**Fix.** Write: the tick is a per-shard knob; the dev cluster runs it at 50 Hz; the client learns the
rate from the wire. Then no residency or bandwidth number is quoted at a rate nobody ships.

---

### B9. The multisampling quarter is reasoning wearing a measurement's clothes (UNMEASURED_AS_FACT)

**The report says** (§0.3, and it leans on this for the anti-aliasing door in §5 item 3.8):
"**Multisampling at 4× dims a one-pixel star to a quarter.** MEASURED 2026-09-07 with multisampling
off and on."

**The ledger says** less. `docs/design/DEFERRED.md:1334-1339` records: "with multisampling OFF the
12 px misses vanished in one run while the underflow loss did not — multisampling at 4× dims a
one-pixel star to a quarter, which the eye's own law may or may not want; not decided here." What ran
was one run, and what it counted was twelve missing pixels. The quarter is coverage arithmetic, not a
painted luminance. The report's own §7 item 5 still OWES that measurement — the report admits the
number is unmeasured in one section and calls it MEASURED in another.

The standing rule is exact here: say UNMEASURED rather than dress reasoning as a result.

**Fix.** Mark §0.3 as ESTIMATED from coverage, cite the one-run observation as what was seen, and
keep §7 item 5 as the bench that settles it. The anti-aliasing recommendation may still stand on its
other two reasons (the dither crossfade needs a temporal resolve; alpha-to-coverage foliage is
foreclosed anyway), and it should stand on those alone until a star's painted luminance is measured.

---

### B10. "Zero hits" is not what the grep returns (WRONG, narrow)

**The report says** (§0.1): "`FrameSpace`, `SphericalSpace` and `reanchor` do not exist in any crate
(grep, zero hits)."

**The grep returns six hits:** `crates/core/src/fence.rs:18`, `crates/sim/src/capability.rs:11`,
`:18`, `:35`, `crates/sim/src/stub/transient.rs:239`, and a test function named
`rehome_event_for_reanchors_…` at `crates/node/src/saga_runtime/tests.rs:3513`.

The substance holds — no such TYPE exists, and the code says so in its own words at
`crates/sim/src/capability.rs:18` ("`FrameSpace` does not exist yet"). But a stated measurement that
fails when a reader re-runs it costs the report credit it has earned everywhere else.

**Fix.** Write: the names appear only in comments and one test name; no type and no module exists,
and `capability.rs:18` says the seam is still owed at P5.

---

### B11. A tick-hitch tolerance must be physical (BREAKS_LAW, SL8)

**The report says** (§3.3 item 7): the tolerance is "the 1 % low frame time stays within two times
the median".

**SL8 says** tolerances are PHYSICAL. Twice the median is a ratio between two engineering numbers. On
the dev cluster's 50 Hz it permits a 40 ms frame beside a 20 ms one — a visible stutter as the hull
descends, which is the seam the item exists to forbid.

**Fix.** State the tolerance the way the report states its others: in what the PLAYER sees. A frame
is a hitch when the drawn scene jumps by more than the drawable angle between two frames — the same
one-pixel rule the reach already uses (`crates/core/src/geometry.rs:1171-1174`). Then the number
follows from the flight speed and the field of view, not from the machine.

---

### B12. A realm's colour still comes from what KIND it is (MISSING)

**The report says** (§3.3 item 4): the client's kind-keyed helpers "are cosmetic colour only and must
never reach the tier or the material path".

**The code** matches on realm kind to pick a colour: `role_hsv` maps `Planet`, `System`, `Ship`,
`Station`, `Area` and `Star` each to its own hue (`crates/client/src/realm_scene.rs:719-731`). SL3
says the REALM authors how it looks. A colour chosen on the client by what kind a realm is, is the
client authoring part of a realm's look — the root of seam kind 4, detail-by-box, in its mildest
form. The code's own note already says the drawn photometric colour rides the look bag's `TAG_LUMA`,
"never this fallback family".

The report accepts the table and does not ledger its removal.

**Fix.** Add one line to §10 or to DEFERRED: `role_hsv` is a fallback for a realm that states no
colour, and it retires when a realm's material comes from its own seed record. Then a moon is grey
because its ground is, and not because it is a moon.

---

### B13. One small mis-reading of the wire (WRONG, minor)

**The report says** (§10 item 2): "`coarsen_level` is a pose-precision hop counter
(`crates/wire/src/intershard.rs:1185`)."

**The code says** it is "The deleted lane's legs-travelled diagnostic", and it sits inside a TOMBSTONE
kept only so a reserved discriminant keeps a decodable shape. The neighbouring field in the same
struct is labelled "the SL2 breach that condemned it" (`crates/wire/src/intershard.rs:1180-1189`).

The conclusion the report draws — no terrain-rung field exists on the wire — is TRUE and stands. The
description of the field is not.

**Fix.** Write: the only `coarsen_level` in the workspace is a diagnostic on a dead lane's tombstone,
so the wire carries no rung field at all.

---

## C. Laws I tried to break and could not

- **SL1.** No realm in the report derives or passes on its own placement. A chunk address is
  realm-local, and a chunk rides its realm's row, which the parent authored (§4.3; verified at
  `crates/client/src/realm_scene.rs:829-851` and `crates/client-render/src/lib.rs:1562-1570`). The
  one open risk is the unbuilt fence, which is B7.
- **SL2.** Nothing in the seam ships an occupant's pose into another realm. The one hole is the
  missing carrier, which is B1.
- **SL3.** A parent draws no child. The report keeps the coarsest rung as the realm's OWN look
  (§10 item 6), which is what `crates/sim/src/stub/window.rs:485-496` already does. The one residue
  is B12.
- **SL4.** The crossing path in the report names no motion. The origin swap re-expresses a picture;
  it never asks how a hull moves.
- **SL5.** No reduced world, no preset, no second generator. The report's whole shape is one
  generator in two hosts.
- **SL7 and SL9.** No per-child per-tick cost and no scan appears. Chunks are keyed by address inside
  one realm; the realms in the window stay the reach ruling's business (§2.2).
- **HR3 and HR4.** Nothing branches on a shard kind. The report names a G-IDENTICAL fixture that
  meshes one chunk on a planet profile and on a hull profile (§2.2).
- **The movement contract.** No velocity crosses upward, and no parent sets a speed. The one velocity
  defect is downward and on the client, which is B2.
- **The seed ruling.** The report keeps a valuable substance off the seed: §2.3 item 2 and §8 item 2
  put every live-state deposit on the diff lane.
- **Reach and visibility.** No tree walk. A dormant realm is never drawn: §3.2 makes the shape
  parameters `None` until the realm runs and states its look, which is the same presence gate the
  window already uses (`crates/sim/src/stub/window.rs:485-496`).
- **The final-backend law.** No stand-in, no placeholder mesh, no config-selected second
  implementation — except the tree, which is B3.

---

## D. The owner's requirements

V2.1, V2.3, V2.5, V2.6, V2.7, V2.8 and V2.9 all keep their doors open in this report. **V2.2 is half
shut by B3**: an engine-instanced tree drifts from the collision the server computes. **V2.4 is shut
by B6**: a seam of whole-metre integers cannot carry a half-metre handle on a hull door. Fix both
before the `ChunkGeometry` door shuts, because §9 says that door shuts first.

---

## E. The order the fixes should take

1. B3 and B6, because both change the `ChunkGeometry` seam contract, and §9 shuts that door first.
2. B1 and B2, because both change what crosses and both need the owner: one is an SL6 ask, one is a
   choice between two warm-ahead shapes.
3. B7, because the fence must exist before the Unreal plugin links anything.
4. B4, B5, B8, B9, B10, B13 — corrections to the report's text, no design change.
5. B11 and B12 — one tolerance to restate, one deletion to ledger.
