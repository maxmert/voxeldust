# Dual-shard process cluster design + adversary (wf_18a0fc4a)

## DESIGN

I have everything. The `WalkTo` dev-control request already exists as a server-side closed loop (`dev_control_nav.rs`), so the capture test just issues `DevRequest::WalkTo`. Here is the ordered, implementation-ready plan.

---

# DUAL-SHARD PROCESS CLUSTER (Track R / 1d.2) + render_crossing_smoke — ordered plan

**Node roster (fixed):** ORCH=1, GATEWAY=2, SHARD=3 (realm System(7), login/source), **DEST=4 (realm System(8), new)**. Matches the harness reference (`tests/src/lib.rs:38-42`). Everywhere below "inert" = a single-shard `up` (no `SHARD_B`/no `VD_KNOWN_SHARDS`) is **byte-identical** to today.

**One HR3 rule for the whole plan:** nowhere does any bin/lib `match` on a realm or shard *kind*. The DEST is a `NodeId` chosen from a config SET / directory head; realm identity is `RealmId::System(seed)` data. Adding shard #4 is a roster/set extension, never a code branch.

---

## Slice 0 — `vd-bins` roster consts + `DevClusterParams` twin fields (Tier-A lib)

**`crates/bins/src/lib.rs:119`** — add the 2nd shard const under the roster block:
```rust
pub const SHARD: NodeId = NodeId(3);
pub const SHARD_B: NodeId = NodeId(4); // Track R / 1d.2 DEST shard (realm B). Absent from a single-shard `up`.
```
**`crates/bins/src/lib.rs:182`** (`DevClusterParams`) — add twin realm knob (do NOT reuse `realm_seed`):
```rust
pub realm_seed: u64,
pub realm_seed_b: u64, // DEST realm seed (Track R). System(realm_seed_b); MUST differ from realm_seed.
```
**`crates/bins/src/lib.rs:204`** (`DEV`) — `realm_seed_b: 8,` (matches harness `dest_stub_config` `System(8)`).

Add a compile-time guard next to the existing asserts (`lib.rs:216`):
```rust
const _: () = assert!(DEV.realm_seed != DEV.realm_seed_b, "DEST realm must differ from source realm");
```
**Coverage:** consts + `const _` assert = free (compile-time). No new runtime branch. **Proves:** the roster/param surface exists, distinctness enforced at build.

---

## Slice 1 — `ClusterAddrs` + port scheme grow (Tier-A: devproto lib + vd-bins struct)

The DEST shard needs its own bind addr + probe. Two ports added per slot.

**`crates/devproto/src/lib.rs`** — grow `RESERVED_NODE_PORTS` 7→9 (`:51`), add offsets (`:60`):
```rust
const SHARD_B_OFFSET: u16 = 7;
const PROBE_SHARD_B_OFFSET: u16 = 8;
```
`SlotPorts` (`:113`) add `pub shard_b: u16, pub probe_shard_b: u16,`. `slot_ports` (`:166`) set them `base + SHARD_B_OFFSET` / `base + PROBE_SHARD_B_OFFSET`. The `dev_control_base`/`client_quic_base` already key off `RESERVED_NODE_PORTS`, so they shift automatically — **update the unit tests** in the same file: `slot_zero_resolves_...` (`:225-240`) expected ports shift by +2 (dev_control(0) 7007→7009, etc.), and `block_size` default 32 still holds `9 + 2·4 = 17`. Keep the `validate` invariant `block_size >= RESERVED_NODE_PORTS + 2·K` — now 9+8=17 ≤ 32, ✓.

**`crates/bins/src/lib.rs:349`** (`ClusterAddrs`) — add `pub shard_b: SocketAddr, pub shard_b_probe: SocketAddr,`.

**Coverage (HR5):** the offsets are covered by the amended devproto unit tests (they read every port). The two new `ClusterAddrs` fields are struct data (no branch). **Proves:** deterministic, collision-free ports for a 2nd shard, still slot-bounded (the `WORKTREE_SLOT_CEILING` math already leaves headroom: block stays 32).

---

## Slice 2 — gateway bin: parse `VD_KNOWN_SHARDS` (Tier-B bin; the parser is Tier-A, already exists)

The library is already multi-shard (`known_shards: BTreeSet`, per-shard `SubId`s, reverse-index fan-out). Only the bin hardcodes the singleton.

**`crates/bins/src/bin/gateway.rs:74-76`** — replace the singleton with the set (reuse the EXISTING `EnvConfig::node_list`, `io-prod/src/runtime.rs:167` — no new parser owed, the map's "owed helper" is already there):
```rust
// Track R / 1d.2: the STABLE routable-shard roster — EVERY shard so a (render-ready) DEST's
// frames are node-class dispatchable (is_known_shard). The login shard is ALWAYS a member.
// Absent VD_KNOWN_SHARDS ⇒ just {shard} (single-shard stays byte-identical).
known_shards: {
    let mut set = std::collections::BTreeSet::from([shard]);
    set.extend(env.node_list("VD_KNOWN_SHARDS").unwrap_or_default());
    set
},
```
Match to harness reference: `known_shards = every shard`, `shard = SHARD` login default (`tests/src/lib.rs:193-197`).

**Coverage:** this is a `/bin` (Tier-B, coverage-exempt). `node_list` (Tier-A) is already covered by `peer_books_and_node_lists_parse_and_reject` (`runtime.rs:320`). `unwrap_or_default()` on absence keeps the single-shard path inert. **Proves:** a DEST→gateway frame is classed `is_known_shard` and reaches `on_shard_frame` (the map's finding: without this DEST frames drop as unknown-peer).

---

## Slice 3 — orchestrator bin: roster from a shard SET (Tier-B bin)

`select_rehome_target` (`saga_runtime.rs:1813`) needs the DEST `NodeId` in the roster with a capable `ShardProfile`, or every cross-shard re-home parks. For P3 bare-point subjects `CapRequest::default()` ⇒ the empty/stub profile satisfies (`capability.rs:407`).

**`crates/bins/src/bin/orchestrator.rs:213-216`** — replace the empty map:
```rust
// Track R / 1d.2: the D-37 re-home roster is the routable-shard SET, each an empty/stub
// ShardProfile (P3 bare-point subjects: CapRequest::default() ⇒ empty profile satisfies).
// This node ALWAYS hosts a shard? No — the roster is the DEST candidates. Absent ⇒ empty (inert).
roster: {
    let mut roster = std::collections::BTreeMap::new();
    for node in env.node_list("VD_ROSTER").unwrap_or_default() {
        roster.insert(
            node,
            vd_sim::capability::ShardProfile::build(vd_sim::capability::CapRequest::default())
                .expect("empty profile is coherent"),
        );
    }
    roster
},
```
Mirrors the harness `stub_roster` (empty profile per stub, `tests/src/lib.rs:129-132`).

**`crates/bins/src/bin/orchestrator.rs:209`** — clock_peers already reads `VD_CLOCK_PEERS` via `node_list`; no bin edit (the env value grows in Slice 5). **Proves:** `head(Realm(DEST))` has a landing target — the crossing's DEST resolution (`handle_crossing_request` reads `dest_rec.authority.node()`, `saga_runtime.rs:1415-1459`; NOT a hardcoded dest — HR3-clean). Coverage: `/bin`, exempt; `ShardProfile::build`/`node_list` Tier-A already covered.

---

## Slice 4 — `vd-bins` env builders: `shard_b_env` + multi-shard peer books (Tier-A lib)

All additive. A single-shard `up` calls the unchanged 3 builders → byte-identical.

**`crates/bins/src/lib.rs:724`** — new `shard_b_env` (twin of `shard_env`, DEST identity + source-realm boundaries do NOT belong here — DEST hosts realm B). Its `VD_PEERS` books ORCH, GATEWAY, **and SHARD** (mesh dials by book; cross-shard transfer needs each shard to book the other):
```rust
pub fn shard_b_env(a: &ClusterAddrs, p: &DevClusterParams) -> Vec<(&'static str, String)> {
    vec![
        str_pair("VD_NODE_ID", SHARD_B.0),
        str_pair("VD_BIND", a.shard_b),
        ("VD_PEERS", book(&[(ORCH, a.orchestrator), (GATEWAY, a.gateway), (SHARD, a.shard)])),
        str_pair("VD_REALM_SEED", p.realm_seed_b),   // System(8) — DEST realm identity
        str_pair("VD_SPEED", p.move_speed),
        str_pair("VD_TICK_DT", p.tick_dt),
        str_pair("VD_ORCH", ORCH.0),
        str_pair("VD_MINT_SEED", p.mint_seed.wrapping_add(6)), // distinct mint (harness DEST=17 vs 11)
        str_pair("VD_INPUT_LOG_CAP", p.input_log_cap),
        str_pair("VD_REALM_RECHECK", p.realm_recheck),
        str_pair("VD_SNAPSHOT_BUDGET", p.snapshot_budget),
        str_pair("VD_PROBE_ADDR", a.shard_b_probe),
        // NO VD_REALM_BOUNDARIES here: the SOURCE shard hosts the crossing trigger into realm B.
        // (Nesting/self-heal boundaries on DEST are a later additive knob — Slice 8 optional.)
    ]
}
```

**`crates/bins/src/lib.rs`** — add a `dual: bool` (or an `Option<&ClusterAddrs>` shard-b) parameter to the three existing builders so their peer books gain the DEST entry ONLY in dual mode. Concretely:
- `orchestrator_env` (`:684`): peers `+ (SHARD_B, a.shard_b)` when dual (orchestrator INITIATES realm-grants/ReHome to DEST).
- `orchestrator_env` clock_peers (`:687`): `format!("{},{},{}", GATEWAY.0, SHARD.0, SHARD_B.0)` when dual (DEST's follower clock never advances otherwise — harness rule `tests/src/lib.rs:99-101`).
- `gateway_env` (`:706`): peers `+ (SHARD_B, a.shard_b)` + a `("VD_KNOWN_SHARDS", format!("{}", SHARD_B.0))` line when dual (Slice 2 consumes it). Route inputs/cut-drains to a client transferred onto DEST.
- `shard_env` (`:730`): source shard's peers `+ (SHARD_B, a.shard_b)` when dual (each shard books the other).

Keep single-shard callers passing `dual=false` → **identical** output (verify against `process_parity.rs`).

**Coverage (HR5, Tier-A):** add unit tests in `lib.rs` `#[cfg(test)]`: (a) `shard_b_env` emits `VD_NODE_ID=4`, `VD_REALM_SEED=8`, books SHARD; (b) dual `gateway_env` emits `VD_KNOWN_SHARDS=4` and books DEST while `dual=false` omits both (the two arms of the `dual` branch, each `assert_eq!` — split, per HR5). **Proves:** every process bin gets the exact peer-book / clock-peer / known-shards wiring the in-process harness wires, and single-shard stays inert.

---

## Slice 5 — `vd-devcluster` bin: spawn DEST + `VD_REALM_BOUNDARIES` on SOURCE (Tier-B bin)

The launcher decides dual vs single from a flag; a plain `up` stays 3 nodes.

**`crates/bins/src/bin/vd-devcluster.rs:81`** — parse an optional `--dual` (or `--dest-boundaries <path>`) after the slot; default single. Gate everything below on `dual`.

**`crates/bins/src/bin/vd-devcluster.rs:192-200`** (`ClusterAddrs`) — add `shard_b: loopback(ports.shard_b), shard_b_probe: loopback(ports.probe_shard_b),`.

**`crates/bins/src/bin/vd-devcluster.rs:219-229`** (the spawn array) — when dual, append a 4th entry and set the source-boundaries env. The SOURCE shard (realm 7) gets `VD_REALM_BOUNDARIES` pointing at a `boxes.json` whose exterior `realm=System(7)`, `to_realm=System(8)` (the guard `guard_boundaries_in_realm`, `lib.rs:1110`, rejects a mismatched file LOUD):
```rust
let mut nodes = vec![
    ("vd-orchestrator", orchestrator_env(&addrs, &DEV, &store_str, dual)),
    ("vd-gateway",      gateway_env(&addrs, &clients, &auth_pubkey, &DEV, dual)),
    ("vd-shard",        shard_env(&addrs, &DEV, dual)),
];
if dual {
    // SOURCE hosts the crossing trigger into realm B; DEST hosts realm B (no boundary).
    let boundaries_path = write_source_boundaries(work, &DEV)?; // System(7)→System(8) box, single-sourced
    let mut src = nodes.pop().unwrap(); // ("vd-shard", env)
    src.1.push(("VD_REALM_BOUNDARIES", boundaries_path));
    nodes.push(src);
    nodes.push(("vd-shard-b", shard_b_env(&addrs, &DEV)));
}
for (name, node_env) in nodes { /* spawn + record_pid, unchanged */ }
```
`write_source_boundaries` writes the SAME `Vec<RealmBoundary>` the render_crossing_smoke test projects (single-sourced — a helper in `vd-bins` returns the Vec so both the launcher and the test call it; mirror the `render_boxes_smoke` `one_box_boundaries` pattern but with `to_realm: System(8)` and a source-realm exterior).

**`await_ready`** (`vd-devcluster.rs:270`): dual readiness must confirm BOTH realms granted. `admin_bootstrapped` (`:288`) uses `AdminSnapshot::cluster_bootstrapped()` — extend that Tier-A predicate (in `vd-wire`/admin) to accept an expected-realm-count, OR add a dual check that the directory holds `Realm(System(8))` too (the crossing M2 anti-vacuity precondition, harness `crossing_e2e.rs:164-172`). Keep single-shard's count=1 path identical.

**Coverage:** `/bin` (exempt) except `write_source_boundaries`/the `cluster_bootstrapped` predicate change (Tier-A) — cover those in unit tests (a boundaries Vec projects + serializes; the predicate is true iff N realms present). **Proves:** a real 4-process cluster boots, DEST holds realm 8 in the ONE orchestrator directory, SOURCE arms the geometric trigger into realm 8. Matches harness invariants 1–5.

---

## Slice 6 — surface `subs_holding` on `DevState` for the out-of-process H1 (Tier-A: devproto + client)

`crossing_was_real` (H1) needs the source+dest holder SET for the own entity across the process boundary. `DevState` today only carries the single composited `authoritative_sub` — the two-holder fact is exactly what compositing hides. Additive (the map's option 1).

**`crates/devproto/src/state.rs:31`** (`DevEntityRow`) — add:
```rust
pub authoritative_sub: u32,
/// Track R H1: EVERY sub currently holding a live track for this entity (source+dest during a
/// crossing overlap). One-element in steady state; two during the SubId(0)+SubId(1) overlap —
/// the anti-vacuity probe `crossing_was_real` reads this. Purely additive diagnosis (never pixels).
pub subs_holding: Vec<u32>,
```
**`crates/client/src/net.rs:417-422`** — populate from the existing `view.subs_holding(entity)` (`view.rs:171`):
```rust
.map(|(entity, sub, pose)| DevEntityRow {
    entity: entity.to_string(),
    pos: sanitize_vec3(pose.pos),
    orient: sanitize_quat(pose.orient),
    authoritative_sub: sub.0,
    subs_holding: self.view.subs_holding(entity).into_iter().map(|s| s.0).collect(),
})
```

**Coverage (HR5, Tier-A):** extend the existing `devstate_when_live_reports_the_decoded_delivered_view` test (`net.rs:1115`) — assert `entities[0].subs_holding == vec![0]` in steady state; add a two-held-sub case (`held_subs = {SubId(0), SubId(1)}`, both tracks live) asserting `subs_holding` sorted-equals `[0, 1]`. Use `assert_eq!` on the sorted Vec (HR5: equality, not `matches!`). **Proves:** the overlap set crosses the wire; the process gate can assert H1 non-vacuously.

---

## Slice 7 — `render_crossing_smoke.rs` (Tier-B `/tests`) + justfile recipe + gate

New sibling to `render_boxes_smoke.rs`, forked from it. `#![cfg(all(feature = "dev-control", feature = "render"))]`.

**`crates/bins/tests/render_crossing_smoke.rs`** — flow:

1. **Slot:** `CROSSING_SLOT = WORKTREE_SLOT_CEILING + 20` (distinct from render-boxes' +19). `DevClusterDown` reaper; `devcluster(launcher, "up", ... "--dual")` — stands up the **4-process** cluster (Slice 5) with the SOURCE boundary planted (System(7)→System(8)).
2. **Fixture single-source:** call the shared `vd_bins` boundaries helper (Slice 5) → the SAME `Vec<RealmBoundary>` the launcher planted. `RealmScene::from_boundaries` → project. This carries BOTH boxes (source realm 7 box exterior + dest realm 8 box) so `fit_camera_to_scene` frames both for BEFORE/AFTER.
3. **Launch capture client** (unchanged spawn pattern, `--capture --realm-boxes <both-boxes.json>`). `await_listener`. Subscribe: login opens `SubId(0)` on SHARD automatically; the DEST sub opens via the transfer saga (no client config — driven by the crossing).
4. **BEFORE:** `WaitUntil UniverseTick>=1`; read state; assert own row is in box A world-space via `expected_box(&scene, own_world) == Some(System(7))`. `Screenshot{label:"before"}` → decode → clear-calibrate → `magenta==0` + `dot_pixels_within_box_region(before_rgba, dot_aabb(own_before), boxA_aabb)` (H2 for box A). Assert the dot projects (`projected_point_aabb(...).is_some()`).
5. **Walk across (non-vacuous, trigger-fired):** issue `DevRequest::WalkTo { target: dest_box_center, arrive_epsilon }` — the EXISTING server-side closed loop (`dev_control_nav.rs:168`), `arrive_epsilon > move_speed·tick_dt` (the `walk_to` oscillation contract). The dot physically crosses the planted boundary; the SOURCE dwell-detector fires the `CrossingRequest` → orchestrator saga → DEST sub opens. **Never hand-fed** (no `trigger_transfer` — the crossing FIRES from geometry, like `crossing_e2e.rs:6-12`).
6. **H1 overlap assert:** during/after the walk, poll state; assert the own row's `subs_holding` (Slice 6) contains BOTH the source and dest subs at the overlap frame — the `crossing_was_real` predicate reconstructed over the wire (`holders.contains(&source) && holders.contains(&dest)`). This is the anti-vacuity gate: without it "landed in box B" passes vacuously.
7. **AFTER:** poll until `expected_box(&scene, own_world) == Some(System(8))` (the flip). `Screenshot{label:"after"}` → decode → `magenta==0` + `dot_pixels_within_box_region(after_rgba, dot_aabb(own_after), boxB_aabb)` (H2 for box B).
8. **HR6 manifest:** assert both `before`/`after` screenshots recorded (like `render_boxes_smoke.rs:305-317`).

Own-world pos each shot from the state row's `pos` (the composited `world_pos`, already `sanitize_vec3`'d). `dot_aabb`/`box[A|B]_aabb` via `projected_point_aabb` + `fit_camera_to_scene` (the ONE legit screen-space step; membership stays world-space).

**`justfile:167`** — add after `render-boxes-smoke`:
```make
render-crossing-smoke:
    cargo test -p vd-bins --features dev-control,render --test render_crossing_smoke -- --nocapture
```
**`justfile:172`** (`gate`) — append `render-crossing-smoke`. Also add `render_crossing_smoke` to the lint-combos targets (`justfile:72-74`) so the dual-shard test's `dev-control,render` surface is clippy-gated like the others.

**Coverage:** `/tests` (Tier-B, exempt). The verdicts it calls (`expected_box`, `crossing_was_real`, `dot_pixels_within_box_region`, `projected_point_aabb`, `walk_to`) are Tier-A, already 100%-covered by their own unit tests. **Proves (the end goal V4):** a SEPARATE-PROCESS capture client watches a REAL cross-shard crossing — the dot is pixel-visible in box A, physically walks across a geometric boundary that FIRES the transfer, both source+dest subs hold it during the overlap (H1 real, over the wire), and it lands pixel-visible in box B (H2), zero magenta, non-vacuously.

---

## What each process bin must MATCH in the in-process harness reference

| Invariant | Harness (`tests/src/lib.rs`) | Process bin (this plan) |
|---|---|---|
| Roster + build order ORCH→GW→shards | `build_cluster:156-242` | spawn array Slice 5 (orch, gw, source-shard, shard-b) |
| clock_peers = {GW, SHARD, DEST} | `p2_cluster:265` | `VD_CLOCK_PEERS="2,3,4"` Slice 4 |
| ONE directory, BOTH realms granted | `orch_config:183` + M2 gate | Slice 5 dual `await_ready` (both realms) |
| `known_shards = {SHARD, DEST}`, `shard=SHARD` login | `:193-197` | `VD_KNOWN_SHARDS` Slice 2+4 |
| roster {SHARD, DEST}→empty ShardProfile | `stub_roster:129-132` | `VD_ROSTER` Slice 3 |
| DEST head resolved dynamically, not hardcoded | `handle_crossing_request:1415-1459` | unchanged (HR3-clean; bin only supplies the SET) |
| DEST realm = System(8), distinct mint | `dest_stub_config:88` | `shard_b_env` Slice 4 |
| each shard books the other (mesh) | `fabric.register` mesh | `VD_PEERS` cross-entries Slice 4 |

## HR / inertness ledger
- **HR1:** no new cross-shard bytes — reuses the proven saga/`InterShardFlow`; `subs_holding` is a read-only local diagnosis surface, never a shard-to-shard payload.
- **HR2:** ONE transfer machinery; DEST is a generic `TransferableKind` sink via the same batched machinery.
- **HR3:** no bin/lib branches on a realm/shard KIND. DEST is a `NodeId` in a SET / a directory head; realm is `System(seed)` data. Adding shard #4 = roster extension.
- **HR4:** the crossing fixture is the same geometric-boundary machinery the in-process `crossing_e2e` proves — features-once, now run on the process tier.
- **HR5:** every new Tier-A line (Slice 0/1/4/6 lib+devproto+client) has a unit test asserting equality; the two new bins/tests (Slice 2/3/5/7) are `/bin`+`/tests`, coverage-exempt.
- **Inert:** a single-shard `up` (no `--dual`) never emits `VD_KNOWN_SHARDS`/`VD_ROSTER`/`VD_REALM_BOUNDARIES`, spawns 3 nodes, books the same peers, uses ports shifted only by the +2 reserved-port grow (verify via `process_parity.rs`, which must be updated for the new port offsets in Slice 1).

## Files touched (all absolute)
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/bins/src/lib.rs` — consts (`:119`), `DevClusterParams`/`DEV` (`:182`,`:204`), `ClusterAddrs` (`:349`), `orchestrator_env`/`gateway_env`/`shard_env` dual param (`:676`,`:700`,`:724`), new `shard_b_env` + `write_source_boundaries` helper, new unit tests.
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/devproto/src/lib.rs` — `RESERVED_NODE_PORTS`/offsets (`:51-60`), `SlotPorts` (`:113`), `slot_ports` (`:166`), amend port unit tests (`:225-258`).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/devproto/src/state.rs` — `DevEntityRow.subs_holding` (`:31`).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/client/src/net.rs` — populate `subs_holding` (`:417`), extend `devstate` tests (`:1115`).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/bins/src/bin/gateway.rs` — `VD_KNOWN_SHARDS` (`:74-76`).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/bins/src/bin/orchestrator.rs` — `VD_ROSTER` (`:213-216`).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/bins/src/bin/vd-devcluster.rs` — `--dual`, `ClusterAddrs` shard_b (`:192-200`), 4th spawn + source-boundaries (`:219-229`), dual `await_ready` (`:270`).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/bins/tests/render_crossing_smoke.rs` — NEW.
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/crates/bins/tests/process_parity.rs` — update expected ports/env for the +2 reserved-port grow (single-shard must stay parity-clean).
- `/Users/maxim/Projects/my/voxeldust/.claude/worktrees/new-system/justfile` — `render-crossing-smoke` recipe (`:167`) + append to `gate` (`:172`) + lint-combos (`:72-74`).

**Reuse (do NOT reinvent):** the proven transfer saga (`handle_crossing_request`), `EnvConfig::node_list` (parser already exists — the maps' "owed parser" is not owed), `DeliveredView::subs_holding` (exists), `crossing_was_real`/`expected_box`/`dot_pixels_within_box_region`/`projected_point_aabb` (Tier-A, 100%-covered), `DevRequest::WalkTo` server-side closed loop (`dev_control_nav.rs`), the `render_boxes_smoke` cluster-up/spawn/capture/decode scaffolding.

**Order:** 0→1→2→3→4→5→6→7. Slices 0-1 are prerequisite data/ports; 2-5 are the cluster wiring (each independently testable, single-shard inert); 6 unblocks the H1 wire-surfacing; 7 is the capstone that exercises the whole stack and is the V4 visual proof + the k3d (#123) multi-shard base.

---

## ADVERSARY

I now have complete ground truth. `expected_box` is a pure GEOMETRIC membership test on world position vs box AABBs — it is realm-frame agnostic. So the AFTER assertion works IF the dot's world position is inside box B's AABB, regardless of the seed-7 frame the DEST renders. The key issue is whether the two boxes overlap in world space (both AABBs in the same world coordinate system). This is a real fixture-geometry concern the design glosses. I have everything needed for a rigorous verdict.

---

# HOSTILE REVIEW — dual-shard process cluster + `render_crossing_smoke`

I read the actual code behind every load-bearing claim. The design is largely sound and the reuse story is real (`known_shards` is genuinely a `BTreeSet`, `subs_holding` exists, `handle_crossing_request` resolves the DEST head dynamically, the harness reference is faithfully described). But there are **five real defects that make the capture either vacuous or hung**, one **factual error in the plan's file list**, and several under-specified spots. Details below, each with the exact code evidence.

---

## CRITICAL findings (block GO)

### C1 — `await_ready` returns before DEST realm 8 is in the directory → the crossing silently `crossing_unresolved`, H1 unsatisfiable
**Severity: CRITICAL (vacuous/hung capture).**
`admin_bootstrapped` (vd-devcluster.rs:288) judges readiness via `AdminSnapshot::cluster_bootstrapped()`, whose body (admin.rs:115-119) is:
```rust
self.directory.iter().any(|entry| entry.authority.starts_with("shard:"))
```
It returns `true` as soon as **any one** shard grants **any** realm. In dual mode the SOURCE (realm 7) will typically grant first, `await_ready` returns, and the test proceeds to plant the boundary + issue `WalkTo` **before DEST has granted realm 8**. Then `handle_crossing_request` (saga_runtime.rs:1434) hits the `(Some(_subj), _, _) => runtime.crossing_unresolved += 1` arm — the exact silent-count path the harness M2 gate (crossing_e2e.rs:162-172) exists to prevent. No saga starts, the DEST sub never opens, `subs_holding` stays `[0]`, and H1 (`crossing_was_real`) is **permanently false** → the test hangs to its tick budget then fails, OR (if the author "fixes" it by dropping H1) the capture passes vacuously.

The design *names* this ("dual readiness must confirm BOTH realms granted... OR add a dual check that the directory holds `Realm(System(8))`") but marks it "extend that Tier-A predicate ... OR". **This is not optional — it is the single load-bearing precondition.** It must be a hard, explicit both-realms gate.
**Fix (required):** add a Tier-A predicate `AdminSnapshot::realms_present(&self, &[RealmId]) -> bool` (or `bootstrapped_realm_count`) that checks the directory holds a `shard:`-authority record for **each** expected realm, unit-tested with 0/1/2 realms present. The launcher's dual `await_ready` calls it with `[System(7), System(8)]`. The single-shard path calls the existing `cluster_bootstrapped()` unchanged (inert). Do NOT ship the "OR" — pick the explicit-realm-set check.

### C2 — the AFTER H2 assertion depends on the two box AABBs being disjoint in WORLD space, but the fixture geometry is unspecified and the harness shell places BOTH boxes at the origin
**Severity: CRITICAL (H2 vacuous or unsatisfiable).**
`expected_box` (verdict.rs:65-80) and `dot_pixels_within_box_region` (verdict.rs:106-138) are pure **world-space geometric** tests — they do NOT care about the seed-7 vs seed-8 `FrameRef`. So the BEFORE/AFTER flip requires the source box and the dest box to occupy **distinct, non-overlapping world-space AABBs**, and the walk must carry the dot from inside A to inside B. But:
- The harness `shell()` (lib.rs:1451, and `plant_one_crossing_shell` in tests/src/lib.rs:493) plants the boundary at `LatticePos::local(DVec3::ZERO)` — **the origin** — with `to_realm` being a pure authority relabel. The harness crossing never moves the box; it flips authority in place. There is no second box at a different world location.
- The design says "`write_source_boundaries` ... mirror `render_boxes_smoke`'s `one_box_boundaries` pattern but with `to_realm: System(8)`" — a `shell(System(7), System(8))` at one location. That produces **exactly one box** in world space (the source box). `RealmScene::from_boundaries` on a single shell gives the scene ONE box (`System(7)`). There is no `System(8)` box for `expected_box` to ever return, so the AFTER assert `expected_box(&scene, own_world) == Some(System(8))` is **structurally unreachable** — after the dot walks out of box A it reads `None`, never `Some(System(8))`.

The design's step 2 says the scene "carries BOTH boxes (source realm 7 box exterior + dest realm 8 box)" — but the plan for `write_source_boundaries` only emits the source shell. The two facts contradict. You cannot both (a) plant a single source-realm boundary the shard's guard accepts (`guard_boundaries_in_realm` at lib.rs:1110 rejects any boundary whose `realm != System(7)`) AND (b) have a `System(8)` box in the SAME file for the client scene.
**Fix (required):** separate the two fixtures explicitly:
1. The **shard-planted** boundary file = one `System(7)`-exterior shell with `to_realm=System(8)`, placed at a source-box world location (passes the guard).
2. The **client** `--realm-boxes` scene = a `Vec<RealmBoundary>` with **two disjoint boxes**: a `System(7)` box at location A and a `System(8)` box at location B, positioned so a straight `WalkTo` from A-center to B-center crosses the planted boundary. This scene is client-only (never handed to a shard, so the guard never sees the `System(8)` box).
3. The `WalkTo` target = box-B center; the planted boundary geometry must lie on the A→B path so the dwell detector fires mid-walk.
Single-source the SHARED geometry (box centers/half-extents) in one `vd-bins` module so the planted boundary and the two client boxes are derived from the same constants, but they are **two different projections** of it. The design's "single Vec both planted and projected" is wrong here — the shard file and the client scene are necessarily different sets.

### C3 — after the crossing the dot's world pose keeps the SOURCE-realm frame (seed 7); the AFTER capture may render the dot in box A's world region, not box B
**Severity: CRITICAL (H2 pins the dot to the wrong box).**
crossing_e2e.rs:303-307 proves the DEST renders `FrameRef::SystemSpace { system_seed: 7 }` — the **crossed source-realm frame**, NOT its own seed-8 frame — and `world_pos` for a `SystemSpace` frame is the identity (view.rs:227, verdict.rs unit tests). So the dot's rendered world position after the crossing is `source_frame_pos`, i.e. it advances **within the seed-7 coordinate system**. Whether that lands inside the client's `System(8)` box AABB depends **entirely** on where the two boxes sit and how far the dot walked — the crossing itself does NOT teleport the dot into box B's region; only the accumulated `Move` integration does. If box B is placed far from box A (like render_boxes' `(500,0,0)` offset) but the dot only walks a few units past the boundary, the dot's world pose is still near box A's edge → `dot_pixels_within_box_region(after, dot_aabb, boxB_aabb)` returns `false` (dot pixels fall outside box B) → H2 fails; OR if the boxes are adjacent/overlapping, H2 passes but `expected_box` can't cleanly flip.
**Fix (required):** the fixture must guarantee the `WalkTo` target (box-B center) is reachable within `max_ticks` at `move_speed·tick_dt`, AND the dot actually arrives inside box B (assert `arrived`, not just `max_ticks` elapsed), AND box B's half-extent is large enough that the arrived dot's projected pixels fall inside box B's projected rectangle. Add an explicit `expected_box(&scene, own_after) == Some(System(8))` gate BEFORE the AFTER screenshot (the design has this in step 7) — but it is only reachable if C2 is fixed (a real `System(8)` box exists in the scene) and the walk genuinely arrives. Make the walk-arrival a hard assertion (`WalkTo` reply must be `arrived`, not `Timeout`).

### C4 — H1 (`subs_holding`) is populated inside the `rendered()` compositing loop, which dedupes to ONE sub per entity — the two-holder fact is exactly what that loop hides
**Severity: CRITICAL (H1 self-defeating if implemented as written).**
The design's Slice 6 snippet adds `subs_holding` **inside** the `.map(|(entity, sub, pose)| ...)` over `self.view.rendered(cursor)` (net.rs:417). But `rendered` → `chosen_subs` (view.rs:144-157) **collapses each entity to a single row** ("Suppresses cross-sub duplicates (each entity once)"). The per-row `subs_holding` field the design adds IS a separate call to `self.view.subs_holding(entity)` (which reads `self.tracks.keys()` directly, view.rs:171 — correctly un-deduped), so the *value* is right. But the concern is subtler and real: during the overlap window, `chosen_subs` picks the authoritative sub (SubId(1) once `AuthorityChanged` re-points, else the lowest). The row exists once; its `subs_holding` correctly reports `[0,1]`. **This actually works** — but ONLY because `subs_holding(entity)` bypasses `chosen_subs`. The design must be explicit that `subs_holding` is read from the raw track keys (not derived from the composited `sub`), or an implementer "simplifying" it to `vec![sub]` (the composited sub already in scope) would silently make H1 vacuous. **Fix:** call `self.view.subs_holding(entity)` explicitly (as the snippet does) and add the two-holder unit test the design specifies (`held_subs={SubId(0),SubId(1)}` → `subs_holding` sorted-eq `[0,1]`). Credit: the design already prescribes this test and the correct call. Downgrade to HIGH only because the snippet is correct as written — but flag the "don't reuse the composited sub" trap loudly.

### C5 — the timing of the H1 overlap read is a race: the overlap window is one-tick, and an out-of-process poll over dev-control can miss it entirely
**Severity: CRITICAL (H1 flaky/vacuous).**
The gateway holds BOTH subs only from `SubscriptionReady` (dest `SubId(1)` opens) until `ReleaseSubscribe` closes the source `SubId(0)` — the map-gateway report says the source close happens "with a one-tick drain grace" (gateway.rs:5730). The client's `DeliveredView` holds both tracks only while both subs deliver frames. Over dev-control the test polls `DevState` JSON at whatever cadence `WaitUntil`/round-trips allow — it does **not** observe every tick. The design's step 6 ("during/after the walk, poll state; assert ... contains BOTH") can trivially poll on a tick where only `[0]` or only `[1]` is live, and the two-holder window may have already closed by the time the AFTER-flip condition is met. The in-process harness sees every tick via `topo.step()`; the process test cannot.
**Fix (required):** do NOT rely on catching the transient two-holder frame by polling. Instead surface a **latching** diagnosis: add a `crossing_overlap_observed: bool` (or a max-holders-seen counter) to `DevState`, set once in the client's `on_snapshot`/view-update path the instant `subs_holding(own).len() >= 2` is ever true, and never cleared for the run. The gate then asserts the latch is set — proving the overlap was real without a tick-perfect poll. This is a Tier-A client + devproto change (additive, inert: the latch stays false in single-shard). Without it H1 is a race that will flake in CI.

---

## HIGH findings

### H-1 — factual error: `process_parity.rs` does NOT use the port scheme, so Slice 1's "update process_parity for the +2 port grow" is wrong and masks the real inert-parity risk
**Severity: HIGH (misdirected work + a real gap left unguarded).**
process_parity.rs:150-155 builds addresses from `reserve_udp_addr()`/`reserve_tcp_addr()` — ephemeral OS-assigned ports, NOT `DevPortScheme::slot_ports`. Growing `RESERVED_NODE_PORTS` 7→9 has **zero** effect on process_parity's addresses. The design's claim (Slice 1 + the files-touched list: "process_parity.rs — update expected ports/env for the +2 reserved-port grow") is factually false; there are no expected ports to update there. The genuine parity risk is different: `process_parity` calls `gateway_env(&addrs, &clients, &auth_pubkey_hex, &DEV)` and `orchestrator_env`/`shard_env` with their **current signatures**. Slice 4 adds a `dual: bool` param to all three builders — that **changes their signatures** and breaks the `process_parity` call sites (compile error). The design must update process_parity's `*_env(...)` calls to pass `dual=false` (or `None`), and assert the emitted env is byte-identical (it should be, since `dual=false` is the inert arm). **Fix:** correct the files-touched note; the real parity work is signature-compat + a `dual=false`-emits-identical-env unit test, not port numbers.

### H-2 — the devproto port-scheme doc block and layout comment must move in lockstep with the offsets, or the `vd-slot` bash surface silently misaligns
**Severity: HIGH (operator foot-gun).**
Slice 1 grows `RESERVED_NODE_PORTS` 7→9 and adds `SHARD_B_OFFSET=7`, `PROBE_SHARD_B_OFFSET=8`. But `dev_control_base = base + RESERVED_NODE_PORTS` (devproto/lib.rs:167) and `client_quic_base = base + RESERVED_NODE_PORTS + k` shift by +2 automatically — the design got this right and correctly notes the unit tests at :225-268 shift by +2. HOWEVER the module doc block (lib.rs:22-28) hard-codes the layout `7 .. 7+K dev-control`, `7+K .. 7+2K client-quic` — that comment becomes wrong (now 9..9+K). And `vd-slot.rs` (the bash-facing helper) surfaces these ports; if it hard-codes any offset it silently misaligns every `client.sh`. **Fix:** update the doc block, and verify `vd-slot` reads ports only through `SlotPorts` accessors (never inlined). Add a shard_b/probe_shard_b accessor. This is Tier-A (`vd-slot` logic if any is a lib) — confirm it stays covered.

### H-3 — the roster `BTreeMap` + `known_shards` `BTreeSet` are deterministic, but `select_rehome_target` picks the LOWEST live node — with roster `{SHARD(3), SHARD_B(4)}` a crossing FROM realm 7 could re-home a D-37 orphan onto SHARD itself, not DEST
**Severity: HIGH (wrong-target risk, but only on the D-37 re-home path, not the crossing path).**
`select_rehome_target` (saga_runtime.rs:1813) returns the **lowest live** roster node whose profile satisfies — with empty profiles, that's always `SHARD(3)`. This is fine for the **crossing** path (which uses `dir.head(Realm(to_realm)).authority.node()`, NOT the roster — HR3-clean, correctly noted). But the design puts BOTH shards in `VD_ROSTER` "for D-37 re-home." If a D-37 forward-re-home ever fires in this cluster, it'd pick SHARD(3) as the re-home target regardless of which realm the orphan belongs to — because `select_rehome_target` is realm-blind (it keys on NodeId + profile only). That's a latent correctness issue for D-37 in a multi-realm cluster, but it does NOT affect the crossing capture (task #83/D-37 is separately in-progress). **Fix:** scope `VD_ROSTER` to what the crossing needs (the DEST candidates) and add a note that realm-aware re-home target selection is a D-37 concern; do not let this plan imply the roster fixes multi-realm re-home. For the crossing gate, the roster's only job is that `head(Realm(8))` resolves — which comes from DEST's realm grant, NOT the roster at all. The design slightly over-claims the roster's role.

---

## MEDIUM findings

### M-1 — HR3 audit: the plan is clean, but `write_source_boundaries` and the `--dual` flag are the two spots where a KIND-match could sneak in
**Severity: MEDIUM (guard against regression).**
No bin/lib matches on a realm/shard kind in the design as written — DEST is `NodeId(4)`, realm is `System(seed)` data, `known_shards`/`roster` are set/map extensions. Correct, credit given. The two risk spots: (a) `write_source_boundaries` must derive `to_realm` from a config field (`realm_seed_b`), not a literal `System(8)` inline — the design mostly does this but the snippet shows `to_realm: System(8)` hard-coded; wire it from `DevClusterParams::realm_seed_b`. (b) the `--dual` flag gates 3-node vs 4-node — ensure it's a **roster-size** decision (spawn N shards from a list), not a `if dual { spawn shard_b }` special-case, so extending to N shards (cloud, #123) is adding list entries. As written, `--dual` is a hard 2-shard boolean → see M-2.

### M-2 — cloud (#123) cornering: `--dual` hardcodes exactly 2 shards; the plan does not extend to N shards + k3d
**Severity: MEDIUM (paint-into-corner risk for the stated dual-purpose goal).**
The goal states this cluster "ALSO serves the cloud/k3d deploy (#123)" and the end-goal is "multisharding meshes ... hundreds." But `--dual` is a boolean, `SHARD_B = NodeId(4)` is a single const, `shard_b_env` is a single twin builder, `ClusterAddrs.shard_b` is a single field, and `RESERVED_NODE_PORTS` grows by exactly 2 (one shard's 2 ports). This is a **2-shard hardcode**, not an N-shard roster. For a local visual playground that's acceptable; for the k3d deploy it is not — k8s scales shards as replicas/StatefulSet ordinals, each with its own realm and its own peer entries, and the gateway's `known_shards` + orchestrator's `roster`/`clock_peers` must grow to N. The design's HR3 ledger claims "adding shard #4 = roster extension, never a code branch" — but the *bin wiring* it proposes IS a per-shard code addition (new const, new field, new builder, +2 ports). **Fix / decision needed:** either (a) explicitly scope this plan to the **local 2-process playground only** and open a separate slice for N-shard k3d roster generalization (honest, and matches "do it LATER" from the visual-testable-endstate memory), OR (b) generalize now: `ClusterAddrs.shards: Vec<SocketAddr>`, `shard_env(idx, realm_seed)`, `VD_KNOWN_SHARDS`/`VD_ROSTER`/`VD_CLOCK_PEERS` built from a shard-count. Option (a) is the lower-risk pick given the memory's "P3 main plan first," but the plan must **say so** — as written it silently bakes in 2 and would need rework for #123. This contradicts the plan's own claim of serving #123.

### M-3 — the `orchestrator_env`/`gateway_env`/`shard_env` cross-peer entries are needed, but the design doesn't verify the mesh actually dials symmetric peers without a deadlock/ordering issue
**Severity: MEDIUM (verify, likely fine).**
The maps correctly establish that each shard must book the other (`VD_PEERS` cross-entries) and the gateway/orchestrator must book DEST. The in-process harness uses `fabric.register` (a shared fault fabric) — every node reaches every other by construction. The process mesh dials by static address book; the design adds the cross-entries. This should work (the mesh is connectionless-ish QUIC with reconnect), but the harness reference does NOT prove the **process** mesh handles a 4th late-booting peer that other nodes already have in their book (they'll get `NodeUnreachable` until DEST binds — process_parity.rs:90 shows this is tolerated with retry). **Fix:** confirm the spawn order (DEST last) + the readiness retry loop tolerate DEST-not-yet-up NodeUnreachable frames, and that C1's both-realms gate waits long enough for DEST's realm grant to land after its late boot. Low risk, but it's an untested process-tier interaction the in-process reference cannot vouch for.

### M-4 — HR5: the compile-time `assert!(DEV.realm_seed != DEV.realm_seed_b)` is good, but `shard_b_env` and the dual-arm env builders are Tier-A and need the split-equality tests the design promises
**Severity: MEDIUM (coverage discipline).**
Slice 4's builders are Tier-A lib (`crates/bins/src/lib.rs`), so per HR5 every new branch (the `dual` arm) needs both arms exercised with `assert_eq!` equality (not `matches!`), split per the CLAUDE.md gotcha. The design prescribes this (Slice 4 coverage note: "the two arms of the `dual` branch, each `assert_eq!` — split"). Good. **But**: the `dual: bool` param introduces an `if dual { ... }` inside each of three builders — that's three branches × two arms = six covered regions, each needing a test that inspects the emitted `Vec<(&str,String)>` for presence/absence of the DEST peer entry, `VD_KNOWN_SHARDS`, and the extended `VD_CLOCK_PEERS`. The design lists two of these; ensure all three builders get both-arm coverage, and that `common_env`/`sh_quote`/`book` are unaffected. Also `write_source_boundaries` (Tier-A, returns a `Vec<RealmBoundary>` + writes a file) needs a serialize-roundtrip + guard-passes test.

---

## LOW / credit

- **Credit — HR1 clean:** `subs_holding` is a read-only local view of `self.tracks`, never a cross-shard payload; no new `InterShardFlow` arm; the crossing reuses the proven saga. Correct.
- **Credit — HR3 head resolution:** the design correctly relies on `handle_crossing_request` reading `dir.head(Realm(to_realm)).authority.node()` (saga_runtime.rs:1436) — the DEST is never hardcoded in the bin. Correct and load-bearing.
- **Credit — single-shard inertness (mostly):** absent `--dual`, no `VD_KNOWN_SHARDS`/`VD_ROSTER`/`VD_REALM_BOUNDARIES` is emitted, `known_shards` stays `{shard}` via `unwrap_or_default()`, roster stays empty, 3 nodes spawn. The `node_list` parser + `unwrap_or_default()` on absence is the right inert mechanism (runtime.rs:167). The +2 port grow shifts dev-control/client-quic bases but keeps `block_size=32` valid (9+2·4=17 ≤ 32) — verified against `validate` (devproto/lib.rs:132). One caveat: the shifted ports mean a running single-shard cluster on the OLD binary and a NEW binary use **different** dev-control ports for the same slot — a developer with a live cluster across the upgrade sees a port move. Benign (they re-`up`), but worth a note.
- **Credit — anti-vacuity awareness:** the design correctly identifies H1 (`crossing_was_real` two-holder) and H2 (`dot_pixels_within_box_region` non-empty) as the anti-vacuity gates and that both must be real. The failure is in the *mechanism* of observing H1 out-of-process (C5) and the fixture geometry for H2 (C2/C3), not the intent.
- **LOW — `dev_control_nav.rs` mischaracterized:** the design calls WalkTo a "server-side closed loop (`dev_control_nav.rs`)". It is actually a **client-side** loop driven tick-by-tick by the dev-control listener (dispatch.rs:82 "driven tick-by-tick by the harness"), and `dev_control_nav.rs` is a **test file** (`crates/bins/tests/dev_control_nav.rs`), not a server module. The mechanism is real and adequate; only the description is wrong. Fix the wording so an implementer doesn't hunt for a nonexistent server module.

---

## VERDICT: **GO-WITH-FIXES**

The architecture is right, the reuse is real, the library genuinely already supports multi-shard, and single-shard inertness is achievable as designed. But **the capture as specified is vacuous or hung** on three independent counts (C1 readiness race, C2/C3 fixture geometry, C5 overlap-observation race), so it cannot merge as written.

### Required before GO
1. **C1 — explicit both-realms readiness gate.** New Tier-A `AdminSnapshot::realms_present(&[RealmId])` (or bootstrapped-realm-count), unit-tested 0/1/2 realms; dual `await_ready` waits for `[System(7), System(8)]`. No "OR". Without this the crossing counts `crossing_unresolved` and never fires.
2. **C2 — two-fixture split.** The shard-planted boundary file (one `System(7)` shell, `to_realm=System(8)`, guard-passing) and the client `--realm-boxes` scene (two disjoint world-space boxes `System(7)`@A and `System(8)`@B) are DIFFERENT sets single-sourced from shared geometry — not one Vec. The `System(8)` box must exist in the client scene or the AFTER `expected_box` flip is unreachable.
3. **C3 — walk must genuinely arrive inside box B.** Assert the `WalkTo` reply is `arrived` (not `Timeout`); `max_ticks·move_speed·tick_dt` must exceed the A→B world distance; box B's half-extent must be large enough that the arrived dot's projected pixels land inside box B's projected rectangle. Gate `expected_box(scene, own_after)==Some(System(8))` before the AFTER shot.
4. **C5 — latch the overlap, don't poll for it.** Add an additive, never-cleared `crossing_overlap_observed` (or max-holders-seen) to `DevState`, set in the client the instant `subs_holding(own).len()>=2`; the gate asserts the latch. Tier-A, inert in single-shard. Eliminates the tick-perfect-poll race that would flake H1.
5. **H-1 — correct the process_parity note.** It uses `reserve_*_addr`, not the port scheme; the real work is updating the three `*_env` call sites for the new `dual` param and asserting `dual=false` emits byte-identical env. Fix the files-touched list.
6. **M-2 decision — declare the scope.** Either explicitly scope this plan to the 2-process local playground (open a follow-up for N-shard k3d roster generalization), or generalize the bin wiring to a shard-count now. As written it hardcodes 2 and does NOT extend to #123 despite claiming to. Pick one and state it.

### Strongly recommended (not strictly blocking)
- H-2 (fix the devproto layout doc block + verify `vd-slot` uses accessors), H-3 (scope `VD_ROSTER` and stop implying it fixes multi-realm re-home), M-4 (all-three-builders both-arm coverage + `write_source_boundaries` test), M-1 (`to_realm` from `realm_seed_b`, not inline), the LOW wording fix for the WalkTo/`dev_control_nav.rs` mischaracterization.

**Load-bearing evidence files:** `crates/wire/src/admin.rs:115-119` (C1), `crates/node/src/saga_runtime.rs:1415-1459` (C1/H-3), `crates/client-harness/src/verdict.rs:65-138` (C2/C3), `tests/tests/crossing_e2e.rs:162-172,287-317` (C1/C3), `tests/src/lib.rs:493` + `crates/bins/src/lib.rs:1451` (C2), `crates/client/src/view.rs:131-177` (C4/C5), `crates/connection-plane/src/gateway.rs:5730` (C5), `crates/bins/tests/process_parity.rs:150-208` (H-1), `crates/devproto/src/lib.rs:22-28,132,167` (H-2/inertness), `crates/devproto/src/dispatch.rs:82` (LOW).