# Verdict — the engine-agnostic render seam and seamlessness

**Lens:** feasibility refutation of `docs/investigation/2026-09-07/08_render_seam_seamless.md`.
**Date:** 2026-09-07. **Method:** I read the binding law first. I read the report second. I then
grepped the code, the owner rulings and the vendored `bevy_pbr 0.18.1` source. Every claim below
cites `file:line`. Every number says MEASURED or ESTIMATED.

**Verdict: REFUTED.** The report is careful and most of its code claims are true. Four load-bearing
claims fail: it never reads SL10 §4, the rule that makes its central recommendation possible; it
refuses option (a) for a reason its own cell lane then commits for trees; it marks an anti-aliasing
number MEASURED that nobody measured; and it reads the realm's reach off the wrong angle. Six more
findings are gaps the domain needs.

---

## 1. What survives

I confirmed these, and the owner can rely on them.

- The render seam is one type. `MeshPrim { vertices, color_rgba, transform }` with
  `PrimTransform { translation, scale, rotation }` (`crates/client/src/realm_scene.rs:769-795`). The
  rotation field is real and the renderer applies it
  (`crates/client-render/src/lib.rs:1569`). **Example:** a hull that turns shows its nose today.
- The Bevy renderer builds a mesh from the vertex array and nothing else
  (`crates/client-render/src/lib.rs:1834-1843`). It is a non-indexed triangle list, so the report's
  `ChunkGeometry` index buffer is genuinely new, not a fit to the old type.
- The author hand-over despawns and respawns in one frame
  (`crates/client-render/src/lib.rs:1572-1598`).
- The eye sits on the lattice with its unit (`crates/client-render/src/lib.rs:541-566`).
- The drawable angle is 1.1506 × 10⁻³ rad and a test asserts it
  (`crates/core/src/geometry.rs:1164-1173`, `geometry.rs:3000-3005`). MEASURED by that test.
- The camera's up is the origin frame's `+Y` (`crates/client-render/src/lib.rs:1144`). There is no
  planet-centre up in the renderer.
- The harness command set is exactly as listed (`crates/devproto/src/dispatch.rs:15-76`).
- `noise = "=0.9.0"` is declared and unused (`Cargo.toml:78`); no crate names rapier
  (only a comment, `Cargo.toml:96`).
- `bevy_pbr 0.18.1` does ship a Hillaire sky, and its scattering medium is a shared asset handle
  (`Handle<ScatteringMedium>`, `bevy_pbr-0.18.1/src/atmosphere/mod.rs:229`). The report's plan to
  share a small set of media by handle is better supported than the report knew.
- Bevy's upload throttle exists and meshes take part
  (`bevy_render-0.18.1/src/render_asset.rs:469-486`; `bevy_render-0.18.1/src/mesh/mod.rs:145`).

---

## 2. The findings that refute the report

### F1 — SL10 §4 never appears, so no-drift is argued as measurable, never as achievable

**The claim.** The §1.1 table gives option (b) the verdict "One surface. The drift gate compares
bytes. **Holds.**" and "`ChunkGeometry` bytes compare server-to-client. **Measurable.**" That cell
carries the whole recommendation.

**The evidence.** The report cites SL10 §1, §2, §3, §5 and §6 (report lines 119, 120, 121, 244, 596,
659). It never cites SL10 §4. SL10 §4 is the ruling that says HOW the bytes come out the same:
integer hashing for every random draw, a fixed evaluation order, no fast-math flag, no fused
multiply-add contraction, **no call into the platform's transcendental functions** (`sin`, `exp`,
`pow` from libm), and only add, subtract, multiply, divide and square root, which are IEEE-exact on
every target (`docs/design/owner_decisions_2026-09-07_voxels.md:36-41`).

**Why it matters.** A gate that compares bytes proves nothing about whether the bytes CAN agree. A
mesher that packs a normal with `atan2` is red on an Apple aarch64 client and green on a Linux
x86-64 shard, and no amount of gating cures it. The report's own §1.2 mixes the two kinds of
function in one list and calls them all "engine-free and Tier-A": `chunk_poll` must obey §4, while
`sun_set`, `exposure_ev100` and `atmosphere_params` compute `2·atan(R/d)` and a colour temperature
and must NOT be held to §4, because their output is never compared byte for byte. An implementer
reading §1.2 has no rule that tells the two apart.

**Fix.** Split the §1.2 function set in two. Mark the shape lane §4-bound: the generator, the mesher,
the normal packing and every attribute inside `ChunkGeometry`. Mark the lighting lane §4-free and say
why. Then state that byte identity across x86-64 and aarch64 is **UNMEASURED** until the §7 item 4
gate runs, and never write "Holds".

**Example.** A pilot on an Apple laptop and a pilot on a Linux desktop fly to the same hill on the
same moon. Under §4 the two clients cut the same slope and the moon's shard agrees with both. Without
§4 one pilot's boots sink a centimetre into a hill the other pilot walks over.

### F2 — The report refuses option (a) for a reason its own cell lane commits for trees

**The claim.** §1.1 refuses option (a): "The server's collision surface comes from the Rust mesher.
The engine's display surface comes from its own code. Two surfaces. **Drift by construction.**"
Then §1.1 item 2 authorises the engine's cell lane to do "instancing of trees (V2.2)".

**The evidence.** V2.2 is explicit that the server must know the tree's shape: *"the collisions are
calculated on the server, so server somehow should know the shape"*
(`docs/design/owner_decisions_2026-09-07_voxels.md:74-78`). If Unreal's instancer builds the trunk
from the cell record and the moon's shard builds a collider from the shared crate, that is exactly
the two surfaces the report refused two paragraphs earlier. §8 item 5 says "The server computes its
collision from the same record" but never says the DRAWN trunk must come from the shared crate.

**Fix.** Split a tree at the collision line, in the seam itself. The trunk and the main limbs — what
a walker can hit — ride the SHAPE lane out of the shared mesher. Only the non-colliding decoration
— leaf cards, wind sway, bark detail — may be engine-side on the cell lane. State that line as a
one-way door beside the `ChunkGeometry` contract.

**Example.** A walker on a moon runs at a tree. Under the split the trunk she hits is the same
surface the moon's shard swept her against. Under the report as written, Unreal chose the trunk's
radius and she stops half a metre before the bark, or walks through it.

### F3 — "Multisampling at 4× dims a one-pixel star to a quarter" is marked MEASURED and is not

**The claim.** §0.3 and §3.3 item 10 mark it MEASURED, citing `DEFERRED.md:1335-1337`. §5 item 3.8
uses it as one of three reasons to choose the temporal anti-aliasing family, and §9 lists that family
as a one-way door whose cost if wrong is "every capture baseline moves twice".

**The evidence.** `docs/design/DEFERRED.md:1337-1340` records something narrower: *"with
multisampling OFF the 12 px misses vanished in one run while the underflow loss did not —
multisampling at 4× dims a one-pixel star to a quarter, which the eye's own law may or may not want;
not decided here."* What ran is one run, and its result is twelve pixels that came back. The quarter
is coverage arithmetic, not a photometric reading. The report's own §7 item 5 owes exactly that
reading ("a one-pixel star's painted luminance against the law's amplitude, with the star probe"),
which proves the number is not in hand.

**Fix.** Mark it ESTIMATED. Make §7 item 5 a precondition of the anti-aliasing door, not a follow-up.
A one-way door may not be opened on an unmeasured number, by the standing rule.

### F4 — The realm's reach is read off the wrong angle

**The claim.** §0.5: "The drawable angle: one pixel at the reference view … = 1.1506 × 10⁻³ rad …
A realm's reach is `extent × cot(θ/2)` from that angle (`geometry.rs:1152`)."

**The evidence.** The reach uses the DOT angle, not the drawable angle. `crates/core/src/geometry.rs:1140-1146`
says so in its own words: *"a realm states its own reach by size as `extent · cot(θ/2)` with THIS θ"*,
and THIS θ is `VISIBILITY_THETA_MIN_RAD = 0.026_180` (`geometry.rs:1146`). The drawable angle is
defined 20 lines later and its doc separates the two jobs: *"the dot angle … is the bar for WAKING a
realm (expensive)"* against the drawable angle as *"the floor for SHIPPING an occupant to an
observer (cheap)"* (`geometry.rs:1163-1169`). The two differ by 22.8×.

**Why it matters.** §3.2 argues the client can generate the coarse rungs early "because the pilot was
inside its reach the whole time". That argument is about the wake radius and must be made with the
dot angle. Written with the drawable angle it claims a reach 22.8 times too small, which would make
the warm-ahead budget look far tighter than it is.

**Fix.** Name the two angles apart everywhere: the dot angle wakes a realm, the drawable angle ships
an occupant and sizes a cell. Re-check §3.2's warm-ahead budget against the dot angle.

---

## 3. The gaps the domain needs

### F5 — A rotation joint and a turret have no seam

V2.5 names a rotation joint that turns what is built on it, a manipulator and a remote-controlled
turret (`docs/design/owner_decisions_2026-09-07_voxels.md:88-92`). §3.3 item 8 rules that a chunk
mesh "rides the row's placement. It has NO pose lane of its own" and that terrain gets "zero new
per-tick data". A turret is part of a hull's grid, so it is not a realm with a row of its own; and it
turns every tick, so it cannot ride the hull's row. The report never mentions a joint or a turret.
A HUD widget on a turret's face inherits the same hole. **Fix:** decide, in the seam, whether a
joint is a nested realm row or a new per-part transform in `ChunkGeometry`'s header, and state the
per-tick byte cost either way. **Example:** a gunner turns a turret on a hull's spine while the pilot
flies; the seam must move that turret's chunks and its HUD widget, and today it cannot.

### F6 — The Bevy sky traps are relayed from an older Bevy

§4.4 lists "three traps in Bevy's shipped sky, all data fixes … verified in the vendored source by
that document" and names the Mie absorption and scattering as transposed. In `bevy_pbr 0.18.1` the
`Atmosphere` component has no Mie fields at all: it holds `bottom_radius`, `top_radius`,
`ground_albedo` and a `Handle<ScatteringMedium>` (`bevy_pbr-0.18.1/src/atmosphere/mod.rs:210-231`).
Mie survives only as comments inside the medium
(`bevy_pbr-0.18.1/src/medium.rs:130`, `medium.rs:143`). The Earth radius pair is likewise not
hardcoded: it lives in a named constructor, `Atmosphere::earthlike`
(`mod.rs:234-243`), and both radii are public fields. Only the 32 km aerial-view range is as
described (`mod.rs:359`, `aerial_view_lut_max_distance: 3.2e4`, a public field). **Fix:** re-verify
the traps against 0.18.1 and drop the transposition.

### F7 — The `+Y` up claim is right, and the cure the report states is too strong

The assumption is real, and the shader says so itself: `get_view_position` adds
`vec3(0.0, atmosphere.bottom_radius, 0.0)` to the world position
(`bevy_pbr-0.18.1/src/atmosphere/functions.wgsl:303-306`), and `get_local_up` carries the comment
*"We assume the `up` vector at the view position is the y axis, since the world is locally
flat/level. … NOTE: this means that if your world is actually spherical, this will be wrong."*
(`functions.wgsl:308-313`). But §4.4's cure — "on Bevy the planet-tangent frame is the only
configuration in which the shipped sky is correct" — fixes the EYE's up only. The flat approximation
along the ray at `functions.wgsl:311-312` stays, so the sky is still approximate toward the limb and
at altitude. **Fix:** say that the tangent frame is necessary and not sufficient, and owe a
measurement of the residual error at the horizon and from orbit. **Example:** a walker on a moon
looks at the limb from a ridge; the sky above her head is right and the sky at the limb is the flat
approximation, and nobody has measured how far off it is.

### F8 — The chunk worker pool names no mechanism and no dependency

§1.2 and §3.3 item 7 put generation and meshing on "the client library's own worker pool". `vd-client`
is Tier-A at 100 % region and branch under HR5, and it is fenced structurally: no clock, no sockets,
no sleep, no default-hasher maps (`crates/client/clippy.toml:6-18`). No thread-pool crate is a
dependency of it (`crates/client/Cargo.toml:8-19`), and rayon is nowhere in the workspace. The report
correctly lists `cbindgen` as an owner option (§1.3) and then never lists the pool. **Fix:** name the
mechanism — `std::thread` plus `crossbeam-channel`, which is already a workspace dependency
(`Cargo.toml:66`) — or list rayon as an owner option, and say how a worker pool reaches 100 % branch
coverage.

### F9 — The new look-bag tag is not costed against the lane that already broke twice

SL6 ask 1 puts the world seed and each body's shape record on the existing look bag and calls that
cheap: "a new TAG on the existing skip-unknown look bag … not a new wire arm". The report itself
records what that lane costs: the datagram budget is 1,200 bytes and a 1,420-byte hull window frame
was dropped forever (§0.5; `docs/design/owner_decisions_2026-09-02_reach.md:200-202`, MEASURED), and
the galaxy's keep-alive reached 22 MB (§3.3 item 11; `owner_decisions_2026-09-02_reach.md:237-238`,
MEASURED). The bag rides a per-row lane whose row count is unbounded under SL9. The report never
states the tag's bytes, and never states that it is stated once per realm rather than on every
keep-alive. **Fix:** state the tag's size, declare it Retained and stated-on-change like the reach
datum (`crates/sim/src/stub/reach.rs:52-64` is the shape to copy), and gate it against the
1,200-byte budget. **Example:** a pilot in a busy star system gets one window frame per tick; if each
body's shape record rides every frame, the frame passes 1,200 bytes and the transport drops the lot,
which is the defect the reach ruling already paid for once.

### F10 — Who owns the tier rule: the engine, or the library?

§2.1 says "The engine picks the tier for each resident chunk" and §1.2's `chunk_request` takes the
tier as an argument. §3.3 item 4 then gives a tolerance that asserts a formula,
`L = ceil(log2(d × cell_angle_max))`, "with one `cell_angle_max` field", and a fixture for a moon, a
hull and a station. Both cannot hold. If Unreal picks freely, the detail-by-box tolerance cannot be
gated on an Unreal client, and V2.8's engine-agnostic promise loses its only seam test. **Fix:** put
the rule in the client library as `chunk_tier_for(distance)`; let an engine override only for a debug
flight, and keep the override count in `DevState` beside the refusal counter.

### F11 — A sub-metre block on a smooth slope has two collision surfaces

§5 item 3.1 answers "Sub-metre blocks (V2.4) live INSIDE the 1 m cell and the collision cell stays
1 m." §5 item 3.2 rules a smooth iso-surface for natural material and says "The smooth lane's surface
IS the collision surface (SL10 §5)". An iso-surface cuts a cell at sub-cell precision. A 1 m
collision cell and a sub-cell iso-surface cannot both be what the boots meet. **Fix:** state which
surface answers a sweep where a sub-metre block sits on a smooth slope, and say what holds the block
up. **Example:** a builder puts a quarter-metre step on a hillside; the hill is a smooth iso-surface
and the step is a sub-metre block; a walker's boot must meet one definite surface at the join.

### F12 — Two seam cases are not covered: a mined cell under a placed block, and an edit during a crossing

§3.3 item 1 covers the crossing and item 2 covers a re-mesh, and neither covers their overlap: a diff
arrives for a realm while the origin swaps to it. Nor does the report say what happens when a miner
removes the cell UNDER a placed block — whether the block falls, floats, or is refused — although
that decides whether one edit can invalidate another chunk's pyramid entry and therefore whether the
diff lane's latest-wins-per-key rule is enough. **Fix:** add both to §3.3 with a tolerance.

### F13 — Small corrections

- "233,220 stars are drawn from a berth inside a hull (MEASURED …)". The ruling attributes 233,220 to
  the WATCHER at the spawn, not to the berth (`owner_decisions_2026-09-02_reach.md:192-193`), and the
  same ruling records that the drawn-versus-held instrument was still owed at step 1
  (`reach.md:203-205`). The MEASURED drawn-in-view figure is 41,345 (`DEFERRED.md:1334`).
- "`RenderAssetBytesPerFrame` 8 MiB in Bevy". The type exists but its default is `max_bytes: None`,
  which means no throttle at all (`bevy_render-0.18.1/src/render_asset.rs:468-470`). The client must
  set it. The mechanism does work, because meshes report their size
  (`bevy_render-0.18.1/src/mesh/mod.rs:145`).
- The report breaks its own reference-view rule. §7 item 12 says every rung range must be re-derived
  from ONE reference view before any residency number is quoted; §3.2 then quotes rung 14 from
  1.27 × 10⁻³ rad while §2.3 quotes 870 m from 1.1506 × 10⁻³ rad.
- Two numbers in examples carry no MEASURED or ESTIMATED mark: "the 505 tier-0 columns" (§1.2) and
  "1,900 quads" (§1.1). The report's own method paragraph promises every number is marked.
- §3.3 item 2 cites `lib.rs:1572-1600` as the mechanism for "a chunk mesh swap". That code swaps a
  whole realm's body on an author hand-over; it does not serve chunks, which do not exist yet. Say
  the pattern is BORROWED, not that the mechanism is in place.

---

## 4. Scale (SL9), and the frame budget

- **SL9 holds in the report's shape.** Generation cost grows with the tier depth, which grows with
  the logarithm of a body's radius, and with the occupants a shard holds — never with a parent's
  child count. Diff interest is derived per occupant (§2.3 item 1). I found no per-child walk.
- **One SL9 risk stays open,** and it is F9: the shape-record tag rides a per-row lane whose row count
  is unbounded.
- **The frame budget is not answered, and the report says so honestly.** Chunk generation runs on the
  library's worker pool and the render thread only uploads (§3.3 item 7), but F8 shows the pool has
  no mechanism yet. The three benches of §7 — the per-tier mesher cost, the sustained columns per
  second at flight speed through the upload, and the resident chunks of a 100 km view while moving —
  are correctly named as owed and correctly marked as the gate on any rung range.

## 5. One-way doors

Five of the six doors in §9 are real and their deadlines are right. Two need changes:

- **The anti-aliasing door** must not shut until the star-probe luminance bench (§7 item 5) runs. See
  F3.
- **The `ChunkGeometry` door** must also carry the tree's collision line (F2) and the SL10 §4 boundary
  (F1), because both change the contract's content, and both are cheap now and expensive after two
  engine plugins consume it.

One door is missing: **who owns the tier rule** (F10). It must shut before the Unreal plugin starts,
for the same reason the geometry contract must.

## 6. What the owner should read differently

The recommendation of option (c)-narrowed survives, and I did not find a better option. What does not
survive is the confidence. The report should say: option (c) is lawful IF the shape lane obeys SL10
§4, IF a tree's colliding geometry rides the shape lane, and IF the byte-identity gate goes green on
both targets — and today all three are UNMEASURED.
