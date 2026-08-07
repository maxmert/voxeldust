# P1.5 Slice 3 — Renderer + HR6 capture (plan, for review)

The final P1.5 slice: open a real window over the renderer-FREE `vd-client` core, draw
delivered entities, drive them with WASD+mouse, and stand up the HR6 capture pipeline
(`vdctl screenshot/record`, the `runs/` manifest, and the permanent `G-RENDER-SMOKE`
gate). Investigated via workflow `wf_af1d4184` (ground + 5 lenses); de-risked by three
spikes (below). This doc is the spec — **review before implementation**.

---

## 0. Status — de-risking DONE, decisions locked

**Spikes (all complete):**
- **glam 0.29→0.30 bump** ✅ — workspace unified on glam 0.30.10 (matches Bevy); 38
  suites green, Tier-A 100%, clippy clean. No render-boundary conversion shim needed.
- **Pure G-RENDER-SMOKE assertions** ✅ — `magenta` / `content_present` / `luminance` /
  `region_nonempty` / `differing_pixel_count` / `nan_count_f32` / `render_smoke` proven
  **100% region+branch on synthetic buffers, zero GPU** (`/tmp/vd-visual-spike`). Drops
  into `vd-client-harness`.
- **Apple-Silicon Bevy readback** ✅ **GO** — all 4 HR6 properties pass byte-exact &
  stable on this M-series machine (offscreen→CPU + 256-byte padding strip; headless
  no-window; **post-egui composite**; wgpu#6827 clean). Working spike + PNG proof at
  `docs/design/spikes/bevy-readback/`.

**User-locked decisions:** full Bevy 0.18 (now unconditional); temporary billboard dot
marker; glam unified at 0.30; plan-doc first.

**Locked dep set** (resolved on stable 1.94.1): `bevy 0.18.1`, `bevy_egui 0.39.1`,
`wgpu 27.0.1`, `glam 0.30.10`, `winit 0.30.13`. Cold compile ≈2.5 min / 374 deps /
≈8.8 GB target; **incremental rebuild sub-2 s**. Dev profile: `opt-level=1` our code,
`3` deps.

---

## 1. Scope (from PLAN.md P1.5 + coverage_e2e.md)

**In:** a window drawing delivered entities as dots; **WASD+mouse** through the SAME
`InputState` setters dev-control uses (byte-identical); an egui debug overlay (sub_ids,
authoritative sub per player, frame_ids, interp depth); HR6 capture — `vdctl screenshot
[--at-tick T]` via in-engine wgpu readback, `walk-to`/`look-at`/`wait-until` closed
loop, `runs/<UTC>__<scenario>/` manifests, `vdctl record`; the permanent
**G-RENDER-SMOKE** gate.

**Out** (later phases, must not creep in): voxel meshes (P4), physics/collision (P5),
ships/nested frames (P8), detail levels (P4 — see below), client-side prediction (never).

> **SUPERSEDED 2026-08-03 — "LOD (never)" is RETRACTED by owner ruling.** Detail levels are now
> REQUIRED ("we will need LODs everywhere, but in the smartest way"). They remain out of *this* slice,
> which still ships one representation only; the ladder lands with terrain at P4. Design:
> `docs/investigation/block_system_design_addendum_2.md`. The wire already anticipated this — `intershard.rs`
> reserves the WHAT lane "at the LOD the observed realm controls" plus a `coarsen_level` ladder.

**Definition of Done:** a human opens the client, sees their own dot (distinct color)
and other dots moving on one stub shard, drives their dot with WASD+mouse, smooth at the
100–150 ms buffer with zero prediction; the debug overlay shows exactly one authoritative
sub for the local player; `G-RENDER-SMOKE` is green; the agent can `dev-cluster up` →
`client --dev-control` → `vdctl walk-to … && vdctl screenshot` → read a PNG + manifest.

---

## 2. Crate layout (the structural decision — needs sign-off)

The dependency rule (`bins → node → sim → wire → core`) and the renderer-FREE mandate on
`vd-client` (clippy-enforced) mean the wgpu/winit/egui glue CANNOT live in `vd-client`.
Two new crates split the pure logic (Tier-A, must stay 100%) from the glue (Tier-B):

| Crate | Tier | Contents |
|---|---|---|
| `vd-client` | A (unchanged) | net/decode/interp/view/input/render_clock — renderer-free. Gains the `pump_inbound`/`assemble_input` split (§3). |
| `vd-devproto` | A (extended) | + `Screenshot`/`Record`/`WalkTo`/`LookAt` request variants + their pure handlers (§6). |
| **`vd-client-harness`** (new) | **A — 100%** | The pure Slice-3 logic: the G-RENDER-SMOKE visual assertions (already prototyped), the `NavController` (walk-to/look-at closed-loop math), the `runs/` manifest serde, the `--at-tick` alignment, the camera math, and the keyboard→`InputAction` mapping (pure). |
| **`vd-client-render`** (new) | **B (`/bin`-adjacent, coverage-exempt)** | The Bevy app: window + offscreen modes, the billboard/egui draw, the `ImageCopyDriver` readback node, the winit input → mailbox glue. Depends on `vd-client` + `vd-client-harness` + `vd-devproto` + bevy/bevy_egui. |
| `client` bin | B | Thin: a feature-gated `render` mode wires `vd-client-render`; the Slice-2 headless mode stays default so the parity tests build lean. |

`vd-client-harness` is added to the justfile `tier_a` set. `vd-client-render` is excluded
by the existing `--ignore-filename-regex '(/bin/|/tests/)'` — confirm it also matches the
render-crate path (likely add `vd-client-render` to the regex, OR keep its glue in the
`client` bin and the *only* render-crate code be uncoverable-by-nature; decision T0).

---

## 3. The core ↔ render bridge (threading model)

**Principle:** decouple rendering (display refresh, 60–120 Hz) from input/step (20 Hz),
and keep the Slice-2 robustness (the dev-control listener never hostage to a window
stall). The chosen model unifies windowed + headless and makes byte-identical input
**structural**:

```
┌─ tokio runtime ─────────────┐   ┌─ CORE step thread (20 Hz, Slice-2 loop) ───────────┐
│ MeshTransport reader/writer │   │ 1. drain the bounded InputAction mailbox →         │
│ dev-control listener (cfg)  │   │    apply_input_action (dev-control AND keyboard)   │
│   reads published RenderFrame│──▶│ 2. pump_inbound(now_s)  (drain net, anchor cursor) │
│   enqueues InputAction/cmds  │   │ 3. assemble_input()     (the 20 Hz InputDatagram)  │
└─────────────────────────────┘   │ 4. publish ArcSwap<RenderFrame{view,clock,own,…}>  │
                                   │    + ArcSwap<DevState>; bump step_seq               │
┌─ render thread (Bevy) ──────┐   └────────────────────────────────────────────────────┘
│ winit input → enqueue       │──────────────▲  (same mailbox as dev-control)
│   InputAction to the mailbox│
│ load RenderFrame → sample    │◀─────────────┘  (lock-free read)
│   view.render(cursor(now_s)) │   draw billboards + egui overlay; capture node reads back
└──────────────────────────────┘
```

- **`ClientState::step` splits** into `pump_inbound(transport, now_s)` (drain+ingest, every
  frame) + `assemble_input(transport)` (tick++ and the one 20 Hz `InputDatagram`/Hello/Bye).
  `step()` stays as `pump_inbound(); assemble_input()` so the **headless `client` bin +
  the parity/load tests are byte-identical** (the existing `net.rs` byte-identical test
  must still pass). Both are branchless Tier-A shims; the branching stays in `ingest`/
  `send_outbound`.
- **All input rides the mailbox** — keyboard/mouse on the render thread enqueue
  `InputAction`s exactly as the dev-control listener does, drained+applied by the step
  thread via `apply_input_action`. Byte-identical-to-dev-control becomes structural
  (one apply path). `set_movement` latest-wins + `add_look` accumulate make per-frame
  enqueue → 20 Hz drain semantically correct.
- **No prediction:** the render thread only SAMPLES `view.render(cursor)` from the
  published `RenderFrame` at its own continuously-advancing cursor. `RenderPose` has no
  velocity; nothing extrapolates. The published `RenderFrame` is a cheap clone of
  `DeliveredView` + the `RenderClock` anchor each 20 Hz step; the render thread
  interpolates between updates at display refresh (smooth, still no prediction).
- **Headless gate** uses the SAME core driver; only Bevy's run mode differs
  (`ScheduleRunnerPlugin` + `RenderTarget::Image`, no winit).

*Alternative considered (rejected as primary): core ON the render thread (pump_inbound
in a Bevy system). Simpler, but a window stall pauses the core → the dev-control listener
sees stale state and queued commands. The step-thread model keeps the agent loop robust.*

**Bridge sub-decision — LOCKED:** the published `RenderFrame` clones the whole
`DeliveredView`+`RenderClock` each step (simple, fine at P1.5 entity counts; a
`cache-when-counts-grow` note matches the deferred per-frame-alloc item).

---

## 4. Input + camera + the dot

- **Key/axis mapping (pure, Tier-A):** held WASD → `set_movement([fwd,strafe,vert])`
  (axis order from `channels.rs`), mouse-motion delta → `add_look([yaw,pitch])` scaled by
  ONE named sensitivity const, jump/interact → `set_action_bit(1<<index, pressed)` on
  press/release edges. Diagonal normalization deferred (keep the per-axis clamp) until the
  server movement model needs it.
- **Camera (pure math, Tier-A `camera.rs`):** free-fly / follow-own-dot first-person,
  parameterized by an **up-vector** (world-Y now; planet-radial + ship-local later — the
  one seam that must not need reshaping), consuming `world_pos(pose)`. **Mouse-look is a
  LOCAL view-only response:** the camera yaw/pitch turns immediately at render refresh
  from the raw delta; the SAME delta also feeds `add_look` for the wire. The rendered
  ENTITY orientation stays server-delivered (this is camera view, NOT entity prediction).
  **Default LOCKED: follow-own-dot first-person** (the own-dot from `own_entity()` is always
  framed — makes the overlay's "one authoritative sub" easy to verify); free-fly stays a
  thin policy over the same core for later.
- **Dot marker (sign-off taken):** a small camera-facing billboard/point marker per
  delivered entity, own-dot highlighted via `own_entity()`, **documented in code as
  temporary P1.5 scaffolding REPLACED (not extended) when real meshes land (P4+)** —
  explicitly distinct from the banned voxel-proxy. Size/color = named consts.

---

## 5. HR6 capture pipeline (the spike recipe, locked)

- **Render + capture (Bevy, from the spike — use the render-graph path, NOT the
  `Screenshot` component):** render egui INTO the camera's image target via
  `EguiMultipassSchedule` on an offscreen `RenderTarget::Image` camera
  (`auto_create_primary_context=false`); read back with the canonical `headless_renderer`
  `ImageCopyDriver` node (`copy_texture_to_buffer`, `align_copy_bytes_per_row`,
  `map_async` + `device.poll(wait)` + crossbeam to the main world); strip the 256-byte row
  padding. (The `Screenshot` component hits the open Bevy #16689 and drops the egui
  overlay — do not use it for the gate.) `RenderTarget` is a separate component in 0.18;
  use full Bevy default features (trimming panics `bevy_text`'s `Assets<Font>`).
- **`screenshot [--at-tick T]` determinism contract (Tier-A alignment):** `--at-tick T`
  = "block until delivered/applied tick ≥ T (reuse the Slice-2 `wait-until` path), then
  capture the next rendered frame, and record the ACTUAL `(cursor, freshest_tick,
  snapshots_applied)` in `manifest.json`". The captured STATE dump is deterministic (wire
  truth); pixels are structurally-reproducible only (documented honestly).
- **`record --fps N --secs D`:** reduced-rate frame sequence; **ffmpeg-optional** (not on
  PATH here) → PNG/APNG fallback. The PNG/APNG encode + frame-cadence math are Tier-A.
- **`runs/<UTC>__<scenario>/`** = `{manifest.json, frames/, shots/, state/, client.log}`;
  the manifest aligns every capture to its tick AND its `vdctl state` dump (manifest serde
  is Tier-A; timestamp passed in, never read from a clock inside the lib).
- **Visual assertions (Tier-A, prototyped at 100%):** `render_smoke` = no-magenta **and**
  content-present; plus luminance-band, HUD-anchor-non-empty, frame-stability, no-NaN —
  all pure over `(&[u8], w, h)` / `&[f32]`.

---

## 6. vd-devproto protocol additions (Tier-A, 100%)

New `DevRequest` variants (postcard/JSON additive — appended enum variants are safe):
- `Screenshot { at_tick: Option<u64>, label: Option<String> }` — non-input render command
  → a SECOND bounded mailbox to the render thread (the capture path); reply
  `Captured { path, tick }`.
- `Record { fps: u32, secs: f64, label: Option<String> }` — non-input → render mailbox;
  reply `Recorded { path, frames }`.
- `WalkTo { target: [f64;3], arrive_epsilon: f64 }` / `LookAt { target: [f64;3] }` —
  BLOCKING closed-loop commands (handled like `wait-until`): the step thread runs a pure
  `NavController` each tick (read own pose from `DeliveredView` → axes via `set_movement` /
  look via `add_look` → arrival check), the listener awaits arrival/timeout and replies
  `State`/`Timeout`. `as_input_action` returns `None` for these (they are loops, not single
  actions); the `NavController` math is Tier-A.

`is_mutating()` extended: `Screenshot`/`Record`/`WalkTo`/`LookAt` are agent-driving
commands — gate the privileged ones consistently with `--allow-dev-control`.

---

## 7. G-RENDER-SMOKE gate

A new `just render-smoke` recipe (none exists today): bring up a dev cluster, launch the
`client` in **headless offscreen** mode under `--dev-control`, `vdctl walk-to` the own-dot,
`vdctl screenshot`, then run `render_smoke` (no-magenta + content-present) over the
readback — green or fail. Runs unattended (no window/display server — proven by the
spike). Permanent gate; a paired visual scenario lands beside each future transition
phase (the multi-client cross-boundary screenshot arrives with P2 compositing).

---

## 8. Implementation sub-ordering (tasks, each landing green)

- **T0** — crate scaffolding (`vd-client-harness` Tier-A + `vd-client-render` Tier-B +
  `client` `render` feature), workspace members, the coverage tier_a/regex update, dep
  pins + dev-profile tuning. Gate: builds; tier split holds.
- **T1** — `vd-client-harness` PURE logic, **Tier-A 100%**: the visual assertions
  (port the proven spike), `NavController` (walk-to/look-at), manifest serde, `--at-tick`
  alignment, camera math, the keyboard→`InputAction` mapping. Test-first; zero GPU.
- **T2** — `vd-client` `step()` split (`pump_inbound`/`assemble_input`); headless bin +
  parity/load tests stay byte-identical (the byte-identical test is the gate).
- **T3** — `vd-devproto` variants + handlers (Tier-A 100%); the render-command + nav
  blocking dispatch shapes.
- **T4** — `vd-client-render`: the Bevy window + the §3 bridge + WASD/mouse → mailbox +
  billboard dots + egui overlay. **Human DoD: walk a dot on screen, smooth, zero
  prediction.** (Run live; you confirm in-window per "don't commit broken".)
- **T5** — the capture pipeline: offscreen readback node (spike recipe) +
  `screenshot`/`record` + the `runs/` manifest. `vdctl screenshot` produces a PNG +
  manifest aligned to a tick.
- **T6** — the `G-RENDER-SMOKE` `just` target + the paired headless visual scenario;
  wire it into the gate family.
- **T7** — adversarial review (workflow) → fix criticals → re-audit, as for Slices 0–2.

---

## 9. Decisions — ALL RESOLVED (ready for T0)

1. **Crate layout** (§2) — ✅ SPLIT: `vd-client-harness` (Tier-A) + `vd-client-render`
   (Tier-B) + a feature-gated render mode in the `client` bin. `vd-client` stays
   renderer-free.
2. **Camera default** (§4) — ✅ follow-own-dot first-person (free-fly kept as a policy).
3. **Bridge `RenderFrame`** (§3) — ✅ full `DeliveredView`+`RenderClock` clone per step.
4. **Coverage exemption** for `vd-client-render` — ✅ via the `--ignore-filename-regex`
   path exclusion (extend it to the render-crate path), same mechanism as `/bin/`.

Plus the spike-settled + earlier-locked decisions: Bevy 0.18 (committed), glam 0.30,
temporary billboard dot marker, all-input-via-mailbox. **Nothing left open — T0 can start
on go-ahead.**

---

## 10. Signal-vision forward-compatibility (architecture review `wf_321511c8`)

A 5-lens review checked Slice 3 + the foundation against the full signal-heavy end-game
(Functional-Panel-emits-named-signals → Thruster receivers, station control-handoff,
cross-owner signal-read approvals, ship-part→debris kind conversion, functional blocks +
subsystems, signal-driven HUDs / video screens). **Verdict: Slice 3 conflicts with NOTHING
— every lens is "supported-with-additive-work" or clean.** The foundation was designed for
this (Signal pub/sub + `SignalScope` + `AccessPolicy`/grants in `sealed_shards.md §3`; HR2
generic transfer + the kind registry already model `Debris`/`NamedConstruction`; HR3/HR4
shard-agnostic blocks). The capture's offscreen-camera→`Image` recipe is, additionally, the
canonical mechanism for **future video-screen blocks** (a camera feed → a block-face texture)
— a genuine Bevy strength.

**Slice-3 keep-right rules (binding for T1/T4 — these CONFIRM the plan, change no decision):**
- **Input stays raw + opaque.** The keyboard→`InputAction` mapping is PHYSICAL-key→bit-INDEX
  only (`space → bit 0`), NEVER `space → jump`. `action_bits:u32` is an OPAQUE physical-action
  channel set; the bit→named-signal binding is SERVER-SIDE (the future Functional Panel). Add
  the comment at `input.rs::set_action_bit` + the harness mapping. Movement/look stay on the
  datagram; configurable/discrete actions beyond them ride a FUTURE reliable C→G message family
  (never a wider `InputDatagram` — appended struct fields aren't postcard-safe).
- **Debug overlay ≠ game HUD.** Render the egui debug overlay in its OWN context/pass; no
  gameplay code reads overlay state. The configurable signal-driven HUD is a separate P9 system.
- **`RenderFrame` stays extensible.** The full-`DeliveredView` clone is the right shape *because*
  future per-block signal/HUD display values fold in via a new additive `BulkMsg` variant
  (mirror the `on_snapshot` arm) — not pose-only-baked.
- **The dot marker is gated for REPLACEMENT** (not extension) so a real block/entity mesh path
  slots in at P4+.

**Scheduled FUTURE design deliverables surfaced by the review (NOT Slice 3 work — recorded so
they're not lost):** the in-realm **SignalGraph execution model** (eval order, cycle handling,
per-tick propagation, programming-block mlua, engine-combining aggregation, and the runtime
signal-SOURCE swap for control-handoff) is the largest undesigned gameplay piece → a first-class
**P9** charter. **Moving/mechanical blocks** (pistons/rails/rotors that physically move/connect
sub-grids) have no seam — the geometry model assumes a STATIC grid → a named **P8** design
question. The ship-detach **structural-split** algorithm (connectivity excision, mass/COM
recompute, fragment-kind selection) → **P8**. Invariant to lock early: a cross-shard **kind
change** is always *despawn-old-EntityId + mint-new-EntityId-with-dest-kind at the SOURCE, then
a plain same-kind Transfer*. The station-takeover needs a **per-publisher pre-emption/exclusive-
control** axis on the signal channel (override ≠ mere allow-list) at P9. None blocks P1.5.
