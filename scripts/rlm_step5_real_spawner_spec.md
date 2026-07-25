# RLM Step 5 — The Real-Process Realm Spawner (`ProcSpawner`) + Retire the Static Forest

**Status:** VETTED spec, buildable. Synthesis of three adversarially-critiqued designs
(reuse-substrate spine + clean-launcher's exact kernel wiring + k8s-forward's seam framing).
Steps 1–4 committed. Step 5 = the design's Slice 5 (`scripts/realm_lifecycle_design.md:327-333`).

**One-line thesis:** supply the real process behind `Box<dyn RealmSpawner>`
(`orchestrator.rs:239`), shape the trait contract by what the eventual k8s launcher (#123)
demands so it drops in as a *second* backend with zero rework, and retire the static forest —
**resolving every CRITICAL/HIGH the critics found: the crash-recovery double-spawn (C-1), F2
id-reuse across crash (H-1), unreapable rehydrated pods (H-3), the `live_nodes()` probe storm
(H-3′), the stranded directory head (H-4), the HR5 Tier-A mislabel (H-4′), the false
`SpawnError` value (H-2), and the shard-coord/boot-ticks plumbing (H-3″).**

---

## §0 — What is FROZEN and stays untouched (the backend-swap invariant)

Step 5 is a **backend swap + a bounded set of additive edits**. These are byte-for-byte unchanged:

- The three `RealmSpawner` signatures (`crates/sim/src/io/mod.rs:447-463`).
- The `MemSpawner` twin (`crates/sim/src/io/mem.rs:440-513`) — still drives ALL deterministic tests.
- The reconcile kernel `vd_sim::rlm::reconcile` (`rlm.rs`) — the pure DECIDE logic.
- The EXECUTE glue `exec_spinup`/`exec_kill`/`exec_force_reap`/`reconcile_launches`
  (`crates/node/src/rlm_runtime.rs:177-266`) — written against the trait, backend-swap-transparent.
- The frozen `InterShardFlow::RealmDemand` arm (`crates/wire/src/intershard.rs:257,517-545`) — **no new wire**.
- Directory CAS / grant / revoke / head (`crates/sim/src/directory.rs`), the re-home saga, redb.

**Additive edits Step 5 DOES make (each justified below, each with an inertness/coverage story):**
1. `SpawnError::LaunchFailed { reason }` — one additive frozen-taxonomy arm (§9 OQ-1; **user decision**).
2. `LaunchLedger::rehydrate` + a durable **launch-intent domain** + a durable **monotone id/port
   high-water** in the orchestrator boot path (`vd-node`, Tier-A) — the C-1/H-1 fix.
3. Directory rehydration wired before the first RLM sweep — the C-1 fix (second half).
4. `NodeKind::StubShard → NodeKind::Shard(profile)` at `shard.rs:44` + `VD_OWN_COORD` /
   `VD_BOOT_TICKS_P99` env carriers into `StubConfig` — the profile-select + demand-loop fix.
5. A reconcile **orphan-head sweep** (paths with a directory head but no ledger cell) — the H-4 fix.
6. `select_rehome_target` made realm-aware (resolve via `head(Realm(to_realm))`) — the roster-dynamic fix.

Everything else is new code in `ProcSpawner` (split across a Tier-A decision half in `vd-node`
and a thin syscall shim in `vd-bins`, §1.6 — the HR5 fix).

---

## §1 — The generic `RealmSpawner` process backend

### 1.1 The HR5-correct split (resolves H-4′: `vd-bins` is NOT Tier-A)

`Justfile:12` `tier_a` = `vd-core vd-devproto vd-wire vd-sim vd-node vd-connection-plane
vd-harness vd-client vd-client-harness` — **`vd-bins` is absent**. A `ProcSpawner` living wholly in
`vd-bins` would leave its crash-prone branching (rehydrate reconstruction, id/port allocator,
kill-taxonomy, TTL, orphan-head logic) with ZERO region+branch enforcement. That is a corner the
"no corners" mandate forbids. Resolution — **split the backend by coverage tier**:

- **`vd_node::rlm_spawn::SpawnCore` (Tier-A, `vd-node`)** — ALL decision/branch logic as a
  monomorphic struct over an injected `trait LaunchBackend` (object-safe, no generics-in-monomorph
  gotcha): the F2 monotone allocator, the deterministic port cursor, the `live`/`killed` maps, the
  `kill_realm` two-arm taxonomy, `live_nodes()` set-assembly + cache policy (§1.5), the rehydrate
  reconstruction, and the durable-intent read/write. Covered 100% region+branch by unit tests
  against a **fake `LaunchBackend`** (a `Fn`-hook that never forks a real process), using
  `assert_eq!`/`expect_err` (HR5(d)), split `&&` (HR5(d)). This is where every critic-found crash
  branch lives, so it is where the gate must bite.
- **`vd_bins::ProcLaunchBackend` (Tier-B, `vd-bins`)** — the thin, branch-poor OS shim implementing
  `LaunchBackend`: `spawn_node` (`lib.rs:1677`), `try_wait`, `Child`/pid/pgid signal teardown, the
  durable pidfile fsync, the `admin_get_body` probe (`lib.rs:1694`). Exercised by the process-tier
  gate (§8), NOT coverage-gated; the un-mockable syscall lines carry an explicit
  `#[cfg_attr(coverage_nightly, coverage(off))]` / `coverage-exemptions.toml` entry.

`trait LaunchBackend` is object-safe (`&dyn`), carries no generics — no monomorphization region debt
(HR5). The public `RealmSpawner` impl is `ProcSpawner = SpawnCore<ProcLaunchBackend>` wired at
`orchestrator.rs:239`; the fake-backend tests use `SpawnCore<FakeLaunchBackend>` in `vd-node` unit
tests. **The frozen trait, `MemSpawner`, and the reconciler stay 100% Tier-A with no new regions.**

### 1.2 State

```
// vd_node::rlm_spawn
pub struct SpawnCore<B: LaunchBackend> {
    inner: Arc<Mutex<SpawnInner<B>>>,   // &self receivers → interior mutability (mirrors MemSpawner)
}
struct SpawnInner<B> {
    next_node:  u64,                          // F2 monotone id allocator — durable high-water (§4.2)
    next_port:  u16,                          // deterministic port cursor — durable high-water (§4.2)
    live:       BTreeMap<NodeId, LiveSlot>,   // minted, not reaped — deterministic order
    killed:     BTreeSet<NodeId>,             // F2: retired ids, never resurfaced (RAM + reconstructed, §4.2)
    intent:     Arc<dyn Store>,               // the DURABLE launch-intent domain (disjoint key set, §4.1)
    backend:    B,                            // the injected OS/fake launcher
    orch_control: Arc<MeshControl>,           // book new peers into the ORCH mesh (CA-1 S2)
    tuning:     ProcSpawnTuning,              // port range, workdir, drain grace, probe cadence — ONE config
    live_cache: LiveCache,                    // §1.5 — amortizes live_nodes() to O(1)/sweep
}
struct LiveSlot {
    handle:  SlotHandle,       // Owned(B::Child) | Orphan{pid,pgid} — Orphan after rehydrate (§4)
    node:    NodeId,
    coord:   RealmCoord,       // full lineage coord (NOT root_coord) — carries profile + AoI authority
    bind:    SocketAddr, probe: SocketAddr,
    cookie:  IncarnationCookie,// per-spawn nonce echoed by the shard on probe — guards pid reuse (§4.3)
    at_tick: UniverseTick,
}
```

> **IMPLEMENTATION NOTE (5b, LAYERING CORRECTION).** The `orch_control: Arc<MeshControl>` field above is
> IMPOSSIBLE as drawn: `MeshControl` lives in `vd-io-prod`, and the dependency rule is `bins → node → sim`
> — **vd-node cannot depend on vd-io-prod**. So peer-booking is delegated THROUGH the backend seam:
> `LaunchBackend::book_peer(node, addr)` (the real `ProcLaunchBackend` in vd-bins calls
> `MeshControl::update_peer_addr`; the fake records it). `SpawnInner` therefore holds NO mesh handle — the
> kernel stays in vd-node, the mesh handle stays in vd-bins. Likewise the SHIPPED 5b `SpawnInner` is
> flatter than drawn: `store: Box<dyn Store + Send + Sync>` (not `Arc<dyn Store>`) inside the mutex for
> `&mut`-through-lock; `next_port` is a `u32` cursor (so port 65535 is usable and exhaustion is loud, never
> wrapped); `LiveSlot` carries `{coord, addr, at_tick}` only — `SlotHandle`/`IncarnationCookie`/`pgid` are
> deferred to 5c (no dead 5b state), and `live_cache` folds into `live_nodes`'s prune-and-cache path.
> `SpawnCore<B>` holds `backend: B` beside `inner: Arc<Mutex<SpawnInner>>` (backend outside the mutex, so a
> launch/probe never holds the state lock across an OS call).

`Arc<Mutex<..>>` gives `Send + Sync` with `&self` — satisfies `Box<dyn RealmSpawner + Send + Sync>`
(`rlm_runtime.rs:65`), exactly the `MemSpawner` pattern. All maps `BTreeMap`/`BTreeSet`
(deterministic iteration — matters for `live_nodes()` ordering and replay).

### 1.3 `spawn_realm(&self, coord, at_tick) -> Result<NodeId, SpawnError>`

1. **Idempotency-by-coord guard (resolves C-1's guard bug).** Scan `live` for any slot (Owned OR
   Orphan) serving `coord.path()` **whose probe says alive** → return its existing `NodeId` (no-op
   spawn). The guard keys on **`coord.path()` alone**, NOT `(coord.path, incarnation)` — the
   incarnation is used only as a staleness disambiguator, never as the adoption key. This is what
   makes cross-crash adoption work: a rehydrated orphan for `coord` short-circuits a re-issued
   `SpinUp` so no second process races it for the head.
2. **Reserve the id + port DURABLY, then mint (F2, resolves H-1).** Bump the durable monotone
   high-water counters (`next_node`, `next_port`) and **fsync the intent record BEFORE the process
   is launched** (reserve-before-launch), so a crash in the launch window can never re-mint an id
   whose process is already running with `VD_NODE_ID=id`. `next_node`/`next_port` are seeded from
   the durable high-water on boot, never from `max(survivors)` (§4.2). Id is decoupled from
   pid/port/ordinal/coord — a respawned realm always gets a strictly-newer id (F2).
3. **Allocate bind + probe addrs** from `tuning.port_range` on `127.0.0.1` (single-host, §6/M-2
   scale note). Deterministic cursor — no OS-ephemeral-port nondeterminism. The spawner CHOOSES the
   addr (the process-tier analog of "k8s assigns a pod IP"); the shard constructs its own
   `MeshTransport` at `VD_BIND` inside `boot_mesh_and_replay` (HR1 — transport lives inside the
   sealed shard).
4. **Build the env from the FULL lineage coord** (reuse `realm_shard_env`'s vocabulary
   `lib.rs:1089-1137`, generalized to one coord):
   - `VD_NODE_ID=id`, `VD_BIND=bind`, `VD_PROBE_ADDR=probe`
   - `VD_REALM_KIND=realm_kind_token(coord.lowered())`, `VD_REALM_SEED=<coord seed>`
   - **`VD_OWN_COORD=<serialized RealmPath of coord>`** (resolves H-3″ half 1): the shard's real
     lineage, so its decentralized `evaluate_realm_aoi` can `coord.child(level)` its true children,
     not just root-level ones. `StubConfig.own_coord` is fed from this (replacing
     `StubConfig::root_coord(own_realm)` at `shard.rs:121`).
   - **`VD_BOOT_TICKS_P99=<measured process boot p99>`** (resolves H-3″ half 2): a nonzero
     predictive-AoI horizon (`stub.rs:138-141`), else F7 pre-warm is inert.
   - `VD_INCARNATION_COOKIE=<cookie>` (§4.3), `VD_PROCESS_INCARNATION` (F6, `lib.rs:837`).
   - `VD_PEERS` = the **ancestor closure + the two anchors (ORCH, GATEWAY)** — NOT every live
     sibling (resolves M-6/M-7: all-live booking is O(N²) and non-deterministic). Siblings and
     re-home crossing peers resolve lazily via CA-1 reply-on-connection (§7).
   - `VD_UNIVERSE_SEED/SCALE`, `VD_TICK_*`, `VD_MINT_SEED` (derived per-id so no two shards alias
     entity ids), `VD_ORCH`, `VD_TRUST_DIR`, D-3 vars — from `tuning`. **No `VD_PROFILE` and no
     `VD_HELD_REALMS`** (§3, §5): the shard derives its own profile from `VD_OWN_COORD`; single-realm
     always (anti-thrash, resolves L-3).
5. **Launch** via `backend.launch(&node_env)` (→ `spawn_node`, `lib.rs:1677`; `process_group(0)` on
   unix so teardown signals the subtree). On `Err(io)` → **`Err(SpawnError::LaunchFailed{reason})`**
   (§9 OQ-1) — NOT the semantically-false `UnknownNode` the reuse-substrate/k8s-forward drafts
   proposed. The `Result` MUST stay fallible: a real OS spawn fails on EMFILE/ENOMEM/port-exhaust,
   and k8s WILL fail on quota — `exec_spinup`'s exponential backoff (`BACKOFF_CAP=10`,
   `rlm_runtime.rs:179-192`) depends on it. On failure, the reserved id/port are NOT reclaimed
   (F2 monotonicity beats port thrift at dev-tier scale; ports reclaimed only from confirmed-dead
   slots, §4.2).
6. **Book into the orchestrator's own mesh** (control surface): `orch_control.update_peer_addr(id,
   bind)` (`mesh.rs:953`, CA-1 S2) so the orchestrator can immediately dial for grant CAS / revoke /
   reap. **Also book the dest→gateway edge at spawn** when the coord is a re-home destination
   (resolves M-1 first-contact window): the orchestrator owns the addr, so it pushes it into the
   gateway's mesh too, closing the input-drop window before the dest's first snapshot dial.
7. **Persist the full intent record** `{node, pid, pgid, bind, probe, coord.path(), cookie,
   at_tick}` durably (§4.1) and **write the `LaunchLedger.minted` entry** — see §4.1: the intent
   domain and `LaunchLedger` are repopulated together on recover so the kernel's `launch_present`
   suppression works after a crash.
8. **Record & return** `live.insert(id, slot); Ok(id)`.

### 1.4 `kill_realm(&self, node) -> Result<(), SpawnError>`

Mirror `MemSpawner::kill_realm` (`mem.rs:497-509`), then OS teardown:
- `node ∈ killed` → `Err(AlreadyKilled)`; `node ∉ live` → `Err(UnknownNode)`; else **drain-then-kill**:
  SIGTERM the process group (Owned via `Child`; Orphan via `kill(-pgid, …)` using the persisted
  `pgid` — resolves H-3/L-1 unreapable-orphan) → bounded `tuning.drain_grace` → SIGKILL + reap →
  drop the mesh book → delete the durable intent record → `live.remove; killed.insert`.
- **PID-reuse guard (resolves M-3/M-5):** before signalling an Orphan, re-confirm identity via the
  `IncarnationCookie` probe (§4.3), never bare `kill(pid,0)` — a recycled pid is not our process.
- Idempotent by contract: `exec_kill`/`exec_force_reap` discard the `Result` (`let _ =`,
  `rlm_runtime.rs:222,241`), so `AlreadyKilled`/`UnknownNode` converge silently.
- **The drain is the Cat-C flush-on-down seam** (§3): today `drain_grace` is a bounded no-op; at
  P7 it becomes "flush Store B, then kill." Planted, not built — a P7 arm, not a rewrite. L-1: a
  test asserts SIGTERM is delivered and the grace honored now, so P7 adds an arm not a bug.

### 1.5 `live_nodes(&self) -> BTreeSet<NodeId>` — external truth, O(1)-amortized (resolves H-3′)

**Authoritative external-truth poll, NOT an internal echo** (else silent-death detection is a
no-op). `reconcile_launches` calls it EVERY drive (`rlm_runtime.rs:253`), so it MUST be
O(1)-amortized, not O(live) syscalls/probes per sweep:
- **Owned slots:** `try_wait()` is a cheap non-blocking `waitpid`; poll every sweep.
- **Orphan slots** (post-rehydrate, no `Child` handle): probing `VD_PROBE_ADDR` via HTTP is
  expensive → **cache with a slow refresh cadence** (`tuning.orphan_probe_interval`), and **promote
  an orphan to Owned-equivalent "confirmed" on first successful contact** so it leaves the probe
  path (it self-grants a head shortly after, at which point the reconciler tracks it by head, not
  by `live_nodes`). A reaped/dead slot is monotone — it never resurfaces (`killed` set), so no
  flapping (resolves the anti-thrash `live_nodes` monotonicity requirement, §5).
- Returns `live.keys()` filtered to still-alive. The trait doc gains a note: **`live_nodes()` is
  called per-sweep; backends MUST make it cheap** (the k8s label-list is O(1) apiserver-cached).

### 1.6 File layout
`crates/node/src/rlm_spawn.rs` (`SpawnCore`, `LaunchBackend`, fake backend tests — Tier-A);
`crates/bins/src/proc_launch.rs` (`ProcLaunchBackend` OS shim — Tier-B);
`crates/bins/src/bin/orchestrator.rs:239` (the `Box::new(ProcSpawner::new(...))` swap).

---

## §2 — Retire the static forest (with a green-gate transition)

### 2.1 What is deleted (the compile-time realm enumeration)

- **The roster `Vec`:** `ClusterShape::Forest` + `extra_realm_shards`'s 5-element literal
  (`lib.rs:170-176,208-252`).
- **The fixed roster NodeIds as a compile-time set:** `PLANET_A_SHARD..AREA_A_SHARD`, `GALAXY`
  (`lib.rs:128-145`) and the fixed realm-seed constants (`lib.rs:149,153,344-345`) — retained ONLY
  as bootstrap-anchor ids (ORCH, GATEWAY) + mem-twin scenario ids.
- **The launcher pre-computation:** the `--forest` spawn loop (`vd-devcluster.rs:351-369`) and its
  hand-enumerated readiness gate `expected_realms(Forest)` (`:443-460`).
- **The pre-baked `VD_PEERS`/`VD_ROSTER`/`VD_KNOWN_SHARDS` fan-out** over that `Vec`.

### 2.2 What replaces it — demand-launch WITH a real bootstrap seed (resolves M-8)

The clean-launcher critic correctly noted: demand spin-up only fires on a `RealmDemand::SpinUp` from
`evaluate_realm_aoi` *inside a live realm shard* — so at cold boot, before any realm shard exists,
**nothing emits the first demand** unless a seed injector exists. **Resolution — Step 5 builds the
minimal bootstrap seed-demand (do NOT defer the whole bootstrap to Step 6):**

- `vd-devcluster --demand` boots ONLY the root chain: orchestrator + gateway + the root `System`
  shard. The orchestrator, at boot, **resolves the deepest realm containing the durable player
  spawn position from the generator** (the `project_dynamic_realm_lifecycle` bootstrap: "generator
  finds deepest containing realm from stored pos") and **seeds the initial demand** — either by
  injecting the bootstrap occupant's stored pose into the AoI evaluator (so `evaluate_realm_aoi`
  organically emits `SpinUp` for the ancestor closure) or, minimally, by staging the first
  `SpinUp{bootstrap_coord}` directly into the reconciler's demand ledger. From there the AoI loop
  grows the rest with NO hand-enumeration. The realm SET becomes the live directory heads + the
  demand ledger — never a `Vec`.
- `select_rehome_target` becomes realm-aware: resolve the target from `head(Realm(to_realm))`
  (HR3-clean, roster-independent) instead of "lowest live capable NodeId" (`orchestrator.rs`
  realm-BLIND today). Fixes D-37 target-pick agreement with the dynamic roster.

**Step 6 remains** the full warp E2E + richer multi-hop bootstrap; Step 5 owns the minimal
seed-demand that makes the demand loop self-starting. This means Step 5 genuinely retires the static
forest (the end state has no `Vec`), not "add-alongside forever."

### 2.3 Inertness / byte-identity during transition (keeps every gate green)

Three inertness layers, each proven by a gate:
1. **`RlmTuning::default()` is inert** (`orchestrator.rs:233` → `reconcile_interval_ticks == 0` →
   never sweeps → `ProcSpawner` never invoked). Every non-demand run keeps this default ⇒ swapping
   `MemSpawner → ProcSpawner` at the wiring site is byte-identical (the spawner is dead code until a
   scenario arms `RlmTuning`). So Step 5 lands `ProcSpawner` BEFORE flipping any gate.
2. **`--static-forest` compat is a TEMPORARY migration scaffold** (design line 328): the old
   `--forest` bring-up is renamed `--static-forest`, keeps the literal `Vec` + `MemSpawner`-inert
   orchestrator, and every legacy gate (`node_per_realm_walk.rs`, parity/E2E) runs on it UNCHANGED
   until individually migrated to `--demand`. **The final slice (5g) deletes `--static-forest` and
   the `Vec` entirely** once the last gate migrates — so there is NO permanent static-forest
   residue (satisfies the auto-REWORK constraint: the end state is forest-free).
3. **`VD_OWN_COORD`/profile parity:** `--static-forest` is upgraded to emit the SAME full lineage
   `VD_OWN_COORD` + `Shard(profile)` as `--demand` (resolves M-1/H-2 contradiction: a static shard
   and a demand shard must be byte-identical for `process_parity` to assert equality). The
   capability-inertness proof (§3) shows this changes no observable P3 behavior.

---

## §3 — Profile-select + state-load-on-spin-up

### 3.1 Profile select `coord.profile_kind() → ShardProfile` (HR3, the C2 fix)

The path exists and is Tier-A-covered (`realm_coord.rs:63,97-106` `profile_kind`/`profile_kind_of`,
total over 6 tags, `Universe/Galaxy → Galaxy`; `capability.rs:250-266` `profile_for`, infallible for
every kind `profile_kind_of` produces, pinned by `capability.rs:491`). Step 5 wires it into boot:

- **`shard.rs:44`:** `kind: NodeKind::StubShard` becomes
  `kind: NodeKind::Shard(profile_for(coord.profile_kind())?)` where `coord` is built from
  `VD_OWN_COORD` (§1.3). The `?` fires only for a `ProfileKind` no realm yields (dead in practice —
  handled as a loud boot refusal for coverage totality). **The ONE kind→profile match stays exactly
  in `profile_kind_of`** — a capability selector, never a feature/reconciler branch (HR3;
  `reconcile`/`drive` never match on kind).
- **Why `VD_OWN_COORD` and not re-derive from `own_realm`:** a Galaxy collapses to `System(1)` under
  `RealmId::System`, so re-deriving from `VD_REALM_KIND`+`VD_REALM_SEED` would drop `signal_relay`.
  The un-collapsed `RealmCoord` (via `VD_OWN_COORD`) preserves `Galaxy → ProfileKind::Galaxy →
  profiles::galaxy()` with `signal_relay` — the capability that later uncorners Signals (P9). This
  is the verified C2 point.

### 3.2 The capability-inertness proof (resolves H-2: NOT byte-identical, wider blast radius)

The critics are right that `StubShard` (zero caps) → `Shard(profile)` (real caps) is a *behavior
change*, and that `register_stub_shard` + the author-and-ship schedule are gated on
`NodeKind::StubShard` at multiple sites (`stub.rs`, `capability.rs:422`, `runtime.rs:187`,
`app.rs:332`). **Resolution — prove CAPABILITY-INERTNESS at P1–P3, do not assert byte-identity:**

- At P1–P3 **no capability-gated system exists yet** — no feature consults `ShardProfile` caps to
  decide behavior. So the extra caps a `Shard(profile)` carries are DORMANT: the author-and-ship
  schedule (`evaluate_realm_boundaries`/`emit_realm_frames`/`evaluate_realm_aoi`) runs identically,
  gated only by `has_synced`, over the same occupant set. Therefore the *observable output*
  (authored frames, grants, snapshots) is byte-identical even though the carried profile differs.
- **This must be PROVEN, not asserted (Slice 5a gate):** a dedicated test asserts
  `Shard(profile_for(coord.profile_kind()))` produces byte-identical authored frames + grants to
  `StubShard` at the same synced tick, for every `ProfileKind` arm `build_app` can receive. The
  StubShard-gating sites are re-gated onto `Shard(profile)` (or `StubShard` is retained as the
  `ProfileKind::Stub` test-only path) and the `all_kinds` coverage test (`capability.rs:419-423`) is
  updated. This is more than "one Tier-A edit" — it is scoped as its own slice (5a) with the
  inertness proof as the acceptance gate.

### 3.3 State-load-on-spin-up: Cat-A now, Cat-C planted

**Step 5 timeframe = closed-form Category-A recompute, NO stored state load.**
- **Cat-A (`f(seed, universe_tick)`):** the shard computes geometry via `boot_regions_and_movers`
  (`shard.rs:93`, closed-form from `VD_UNIVERSE_SEED`) and child poses via `child_placements(tick)`.
  It authors NOTHING before its first `ClockSync → FollowerClock` (`has_synced` run-condition,
  `stub.rs:1229/1236/1249`) and self-grants via **pull-boot** `request_pending_grants`
  (`stub.rs:1256`) — no orchestrator push. **Byte-identical by construction (DET-1):** a spun-up
  shard == an always-on shard, same build. This is exactly why an occupant-less ancestor needs no
  special mode — it recomputes its frame from seed and integrates zero occupant signals (the
  zero-signal degenerate case, §0-law).
- **Bootstrap occupant pose (resolves M-3):** ProcSpawner spawns geometry (Cat-A); the bootstrap
  player's stored pose enters via the **existing admit / `ReHomeState::PoseOnly` path**, NOT a realm
  checkpoint. Stated explicitly so bootstrap is not read as "needs no pose source."
- **Cat-C (checkpoint-carried redb load-on-up) — DEFERRED past Step 5 (D-RLM-2).** Occupant/rapier/
  blocks (P4+) load from per-realm Store B with an `(EpochId, universe_tick)` header failing LOUD on
  `CheckpointEpochMismatch`. Step 4 landed only 4a/4b; the durable occupant snapshot (4c) is
  deferred/soak-gated. The seam is live: `ReHomeState::Snapshot` (`stub.rs:2361-2362`), the
  `spawn_realm` boot (load-on-up) + `kill_realm` drain (flush-on-down) hooks — both no-ops today.

---

## §4 — Crash re-discovery + monotone NodeId + determinism

### 4.1 The C-1 fix: rehydrate BOTH the LaunchLedger AND the DirectoryCore (not just the spawner)

**The verified hole (both critics, confirmed against `rlm.rs:612-614`):** the kernel's spin-up
suppression is `absent = head.is_none() & !launch_present`, where `launch_present` reads
`launch_live = LaunchLedger.minted` (`rlm_runtime.rs:153`), and `live_nodes()` is read ONLY by
`reconcile_launches` to DRAIN entries already in `minted`. On recover, `RlmReconcilerRes::new`
starts with `LaunchLedger::default()` (empty) and `DirectoryCore` is empty. So a pre-crash child
whose head was not yet committed → `head.is_none() & !launch_present` = true → **double-spawn +
headless zombie**. Rehydrating `ProcSpawner.live` alone does NOT close this — `live_nodes()` never
feeds the spin decision when `minted` is empty.

**Resolution — three durable rehydrations before the first sweep:**
1. **Durable launch-intent domain.** `spawn_realm` persists `{coord.path → node, at_tick, addr,
   pid, pgid, cookie}` in the orchestrator's redb `Store` under a **disjoint key domain**
   (`RealmPath`-keyed, structurally orthogonal to the `NodeId`-keyed durable demand ledger — same
   `Store` seam, no `reconcile` signature change). Persisted in the **same group-commit barrier as
   the directory head** so intent and head can never diverge.
2. **`LaunchLedger::rehydrate(store)`** (new, `vd-node` Tier-A) repopulates `minted` from the intent
   domain on RECOVER, **before the first sweep**. Now `launch_present` is true for a pre-crash
   child, the duplicate `SpinUp` is suppressed, and `reconcile_launches` (with `live_nodes()` seeded
   from the rehydrated spawner) keeps the child until it grabs the head.
3. **Directory rehydration** (reuse-substrate C-1): the orchestrator's `DirectoryCore` heads+fences
   are rehydrated from redb before the first reconcile, so a re-adopted running shard's head
   survives the restart and a duplicate incarnation's CAS FAILS (head still held) instead of
   succeeding. Both halves are needed: (2) suppresses the spin, (3) makes the CAS lose even if a spin
   slips through.

### 4.2 The H-1 fix: durable monotone id/port high-water (never resurrect a retired id)

`next_node`/`next_port` are seeded from a **durable monotone high-water counter** persisted in the
intent domain (reuse the R-6 durable incarnation/boot-counter pattern, `saga_runtime.rs`), bumped +
fsync'd **before** `spawn_node` (reserve-before-launch, §1.3 step 2). **Never `max(survivors)`** —
that ignores retired ids higher than the surviving max (the verified H-1 scenario: mint 1000/1001/
1002, kill 1002, crash, `max(survivors)=1001` → re-mints 1002). The durable high-water is monotone
across crashes independent of pidfile discovery, so a killed-then-crashed id is never re-minted (F2).
Ports reclaimed only from confirmed-dead slots (a fresh spawn never reuses a surviving port → no
bind collision / no probe-aliasing-the-wrong-process, resolves M-5).

### 4.3 Orphan re-discovery + PID-reuse guard (resolves H-3/M-3)

After orchestrator kill-9, children reparent to init; the rebuilt orchestrator no longer holds their
`Child` handles. `ProcSpawner::rehydrate(store, backend)` reads the durable intent records,
reconstructs `live` with `handle = Orphan{pid, pgid}`, and probes each via `VD_PROBE_ADDR` +
`IncarnationCookie` echo. **The cookie is the identity token beyond bare pid** — a recycled pid
belongs to an unrelated process and will not echo the cookie, so re-adoption verifies identity, not
mere pid-liveness (`kill(pid,0)` alone is unsafe). Survivors seed `live` + re-book their addrs
(`update_peer_addr`); stale records (dead/cookie-mismatch) are dropped and a fresh `SpinUp`
re-fires with a NEW monotone id. `next_node` resumes from the durable high-water (§4.2), never from
the discovered max.

### 4.4 The H-4 fix: orphan-head sweep (stranded directory head with no ledger cell)

`reconcile` iterates `ledger.iter()` only (`rlm.rs`), so a pre-crash head whose realm is never
re-demanded (parent also down, or intent record lost) lingers forever — invisible to reconcile,
holding a port + lease. **Resolution — after the freeze window, `reconcile` sweeps directory
realm-heads with no ledger cell AND no launch record and either synthesizes an adoption cell (AoI/
re-home will confirm it, keeping it) or `ForceReap`s it.** Paired with the durable id allocator
(§4.2) so every live process is always discoverable, and gated behind the crash-freeze so it never
fires before demands re-accrue.

### 4.5 The crash-freeze (already landed, kept as tuning)

`arm_quiesce`/`rlm_quiesced_until` (`rlm_runtime.rs:126`) blocks **teardown** until
`now >= rlm_quiesced_until` so a rebuilt orchestrator never mass-reaps before demands re-accrue.
Note (verified): the freeze gates `Kill`/`teardown_ready` (`rlm.rs:470`), NOT `ForceReap` or
spin-up — so it is belt-and-suspenders for teardown, and the REAL spin-side protection is §4.1's
ledger+directory rehydration. Step 4a armed the freeze; Step 5 keeps it as a config knob sized for
the slower backend (§6 inv-5) and adds the missing spin-side rehydration.

### 4.6 Determinism boundary

The mem-twin harness path stays byte-identical (`BTreeMap`/`BTreeSet`, no clock/RNG) — the pure
`reconcile` kernel drives both harness and cluster. `ProcSpawner` is a real OS backend
(non-deterministic pids/timing) and appears ONLY in the real orchestrator bin + the process-tier
gate — never in the deterministic harness. The deterministic port cursor + monotone id keep the
same demand sequence assigning the same ids/ports per run where a test needs it; the process gate
asserts **behavioral parity** (on-demand shard byte-identical in OUTPUT to a static one), not
byte-identical spawner internals.

---

## §5 — Pre-warm + anti-thrash

- **Predictive AoI is already built (Step 2), Step 5 makes the horizon honest.**
  `boot_ticks_p99 → horizon_s = boot_ticks_p99·tick_dt_s → aoi_min_dist projects occ_pos +
  occ_vel·horizon_s` (`stub.rs:138-141,4336,4412-4415`). Step 5's obligations: (a) plumb
  `VD_BOOT_TICKS_P99` into `StubConfig` so the shard's decentralized AoI has a nonzero horizon
  (§1.3, resolves H-3″); (b) MEASURE the real process boot p99 (spawn + mesh dial + first ClockSync
  + pull-boot self-grant, Cat-A only) in the process gate and feed it back. The predictive horizon
  must cover **dial completion** (M-4), not just process start, so a re-home source→dest push over a
  learned lane is reachable before the crossing fires.
- **Tuning invariant (resolves M-2 tuning):** `min_dwell_ticks ≥ boot_ticks_p99 + settle` — add the
  cross-field check to `RlmTuning::validate` so a realm demanded-then-undemanded during a slow boot
  can't churn.
- **Node-per-realm 1:1 is the anti-thrash base.** `ProcSpawner` spawns exactly ONE process per
  realm coord → every re-home is cross-node (entity LEAVES the node, containment detector stops
  seeing it) → the co-hosting `source==dest` stale-pose limit cycle that froze the player
  (`project_node_per_realm_rehome`) is structurally impossible. **`ProcSpawner` NEVER sets
  `VD_HELD_REALMS` (single-realm always) — a test asserts it unset** (resolves L-3), so a future
  edit can't silently re-introduce co-hosting. The trait carries NO 1:1 assumption (D-44 inverse
  stays openable, §6 inv-6); Step 5 just never builds co-hosting.
- **Reconciler hysteresis (Step 2/3) unchanged:** min-dwell (`spawn_watermark`), per-coord
  exponential backoff (`BACKOFF_CAP=10`), `launch_ttl_ticks` grace, the crash-freeze. Step 5 only
  makes `spawn_realm` fallible-and-retryable so the backoff has something real to back off from, and
  keeps `live_nodes()` monotone (§1.5) so a flap never causes a spurious re-spawn.
- **Warm-spare-pool DEFERRED** (design line 354, load-bearing for warp latency F7): behind the same
  `RealmSpawner` trait with zero caller change (pre-forked idle `vd-shard`s handed a coord via a
  control message instead of `Command::spawn`). Flagged for the user (§9 OQ-2) because it changes the
  boot handshake. Step 5 does not foreclose it (the trait carries no cold-spawn-only assumption).

---

## §6 — The clean k8s seam (#123) WITHOUT building it

Step 5 is process-tier (real `vd-shard` OS process + real `MeshTransport` + real child-liveness
poll), explicitly the **dev/CI single-host tier** (M-2: dozens–hundreds of realms; ephemeral port
space ~64K, PID/FD/RAM ceilings; 100K genuinely needs k8s — say so, do not present the process poll
as scale-representative). Every k8s concern is a packaging/launcher-impl concern in `deploy/k3d` +
a future `K8sSpawner`, never the trait. Invariants Step 5 preserves so k8s drops in with zero rework:

1. **`spawn_realm` FALLIBLE + `exec_spinup` backoff** — k8s fails on quota/scheduling. (The new
   `LaunchFailed{reason}` variant, §9 OQ-1, gives k8s a truthful discriminant to log/telemeter.)
2. **`NodeId` monotone + decoupled from ordinal/coord/pid** — StatefulSet ordinal-0 recreated mints
   a NEW id; the durable high-water (§4.2) is the incarnation contract; a pod stamps its minted
   `NodeId` as a label so `live_nodes()` reconstructs after a rebuild.
3. **`live_nodes()` = authoritative external-truth poll, made cheap** (§1.5): process = cached
   `try_wait`/probe; k8s = O(1) apiserver label-list. Trait doc states it is per-sweep.
4. **Transport registration + addressing OUT of the trait** — `spawn_realm` returns only a `NodeId`;
   the CA-1 plane (`update_peer_addr` + auto-pusher DNS re-resolve + reply-on-connection) is
   separate. This is exactly why DNS-vs-process addressing differs with zero trait change: a
   rescheduled pod's new IP is picked up by the auto-pusher, not by re-spawning.
5. **`launch_ttl_ticks` / `rlm_quiesced_until` / `boot_ticks_p99` / `min_dwell` as config** — sized
   per backend at deploy (a process spawn is near-instant; a pod is not).
6. **Node-per-realm 1:1 as the base, NOT baked into the trait** — the D-44 co-host inverse stays
   openable; Step 5 need not build co-hosting.

k8s-specifics deferred to `deploy/k3d` + `K8sSpawner`: StatefulSet ordinals, per-pod
`volumeClaimTemplates` PVC (the M3 boot-counter), headless Service `publishNotReadyAddresses`, label
selectors (the `live_nodes()` source), NetworkPolicy, deterministic-DNS-from-`RealmPath` roster
growth feeding `VD_PEER_HOSTS`. Ledgered **D-RLM-1**. Any k8s apiserver client crate is flagged for
the user (§9 OQ-3), never adopted in Step 5.

---

## §7 — Peer addressing (CA-1, no DNS) — the resolved convergence rule

The mesh dials by `SocketAddr`, never by name. Step 5 grows the roster dynamically with zero new
wire and zero DNS, reusing three CA-1 planes — **but booking only the ancestor closure, not all-live
(resolves M-6/M-7 O(N²)):**

1. **Orch → shard (control): explicit booking.** `update_peer_addr(id, bind)` at spawn (§1.3 step 6)
   — trusted control surface; an untrusted dial-in only ever populates `LearnedPeers`. Plus the
   dest→gateway booking at spawn (§1.3, resolves M-1 first-contact window).
2. **Shard → parent/gateway/sibling (reply direction): reply-on-connection (`LearnedPeers`).** A
   realm shard is the initiator of first contact in every flow that matters (streams snapshots TO
   the gateway, dials orch to self-grant, initiates the saga's `TransientGo` TOWARD its sibling).
   Each dial-in is recorded in the peer's `LearnedPeers` (accept-path only, capped, shed-loud), so
   the peer replies with zero addressing config (the codified D-18 lesson: one dispatcher, identical
   DATA+ACK loop on every connection). This closes the re-home direction: `SpinUp B` → B boots,
   dials its ancestor closure incl. source A → A learns B by reply-on-connection → A pushes the
   crossing over the learned lane. **First-contact ordering invariant** (a node never needs to
   initiate to a shard before that shard has dialed it — gateway→shard routing begins only after a
   subject re-homes onto the shard, strictly after the shard completed a saga in which it dialed
   orch+gateway) — a named process-tier gate assertion (§8), and the M-1/M-4 window is explicitly
   closed by the spawn-time gateway booking + the `boot_ticks_p99`-covers-dial horizon.
3. **`VD_PEERS` = ancestor closure + anchors only** (M-6 fix): siblings/crossing peers resolve
   lazily via (2). Restores deterministic env content and O(closure) not O(N) per spawn.
4. **Auto-pusher (cloud only):** `spawn_peer_resolver` re-plumbs a *booked* peer's changed IP but
   never invents a dial target; process-tier `VD_PEER_HOSTS` unset ⇒ clean no-op.

---

## §8 — Slice plan (per-slice tests, 100% Tier-A, process-tier gate)

Every slice keeps `just gate` green (fmt + clippy `-D warnings` + tests + `coverage-fast` Tier-A
100% region+branch). RLM tuning stays inert (`RlmTuning::default()`) until 5f, so every slice
through 5e is byte-identical to the current tree for all existing gates. `SpawnCore` decision logic
is monomorphic + fake-`LaunchBackend`-covered in `vd-node` (Tier-A); the OS shim is exempted (§1.1).

- **5a — `NodeKind::Shard(profile)` + `VD_OWN_COORD` + capability-inertness proof.** Retire the
  `shard.rs:44` `StubShard` hardcode; feed `StubConfig.own_coord` from `VD_OWN_COORD`; re-gate the
  StubShard sites. *Tests (Tier-A):* `profile_for` round-trip totality over `ProfileKind`; a
  `VD_OWN_COORD`-Galaxy shard carries `signal_relay`; **the capability-inertness gate** — byte-
  identical authored frames + grants vs `StubShard` at the same synced tick for every `ProfileKind`
  (§3.2). *Gate:* every existing rig (upgraded to emit `VD_OWN_COORD`) byte-identical.
- **5b — `SpawnCore` + `LaunchBackend` (decision half, Tier-A, fake backend).** Mint/kill/live/
  rehydrate against the fake backend + `orch_control`. *Tests (Tier-A):* F2 monotone (mint never
  reuses a killed id, even after a simulated rehydrate); `kill_realm` two-arm taxonomy via
  `expect_err`; `LaunchFailed` drives no state change; `live_nodes()` drops a backend-dead child +
  caches orphan probes; `update_peer_addr` called once per spawn; rehydrate reconstructs survivors +
  resumes `next_node`/`next_port` from durable high-water (not max-survivor); `VD_HELD_REALMS`
  unset. 100% region+branch.
- **5c — `ProcLaunchBackend` (OS shim, Tier-B) + coord-driven env.** `spawn_node` launch, `try_wait`
  poll, `pid`/`pgid` signal teardown, durable intent fsync, `admin_get_body`+cookie probe. Exercised
  by the process gate; syscall lines exempted. *Test:* a spawned Planet boots `RealmId::Planet(seed)`
  with the planet profile; a spawned Galaxy carries `signal_relay`; child's authored frames byte-
  identical to a static one at the same synced tick (Cat-A DET-1).

> **5c VET AMENDMENTS (workflow `wf_12f77998`, GO_WITH_FIXES — these SUPERSEDE the 5c bullet above and
> the §1.3/§1.5/§4 detail where they conflict).** The vet confirmed 10 defects in the first-draft 5c
> design; the corrected, buildable 5c is:
>
> 1. **Owned-path only — orphan/adopt DESCOPED to 5e (D4).** 5c ships `launch`, Owned `is_alive`
>    (`try_wait`), Owned `teardown`, `book_peer`, `mint_cookie`, and the `/whoami` echo plumbing
>    (dormant). `AdoptSpec`, `Slot::Orphan`, `ProcLaunchBackend::adopt`, the orphan cookie-probe cadence,
>    and the cookie-guarded teardown branch move to **5e**, where the kill-9-then-rehydrate PROCESS gate
>    actually constructs orphans and drives both arms of the pid-reuse guard. 5c's rehydrate stays the 5b
>    `book_peer`+insert shape (reading the new fields) and MUST NOT delete a `pid: Some` survivor's
>    intent.
> 2. **Reap on teardown (D2).** `teardown` moves the owned `Child` into a detached thread → SIGTERM(group)
>    → bounded `try_wait` poll → SIGKILL → **final `child.wait()` to REAP**. Dropping the `Child` without
>    `wait()` would leak a `<defunct>` zombie per spin-down (a self-inflicted `EAGAIN` DoS on the up/DOWN
>    churn that is RLM's whole point).
> 3. **Mint the cookie PRE-FORK; ONE `Option` in the durable intent (D3).** `LaunchBackend` grows
>    `mint_cookie(node) -> IncarnationCookie` (fake: deterministic; real: std-entropy, no `rand` dep).
>    The kernel mints BEFORE the fork and persists it in the write-ahead (v1) record. `LaunchIntent`
>    grows `cookie: IncarnationCookie` (NON-optional) + `pid: Option<u32>` + `probe_port: u16` — exactly
>    ONE `Option`, both arms reachable (`None` = crash between v1 and the post-launch v2 commit;
>    `Some` = completed spawn). No impossible `(Some,None)`/`(None,Some)` cross-product ⇒ HR5 100% branch
>    attained naturally. `launch(&LaunchSpec) -> Result<u32 /* pid */, String>`; `LaunchSpec` gains
>    `probe: SocketAddr` + `cookie: IncarnationCookie` (passed IN). Two-port stride allocator (bind +
>    probe; exhaustion when `cursor + 1 > u16::MAX`).
> 4. **Shared redb handle, TWO barriers — NOT one (D1).** The "single group-commit barrier / intent and
>    head can never diverge" claim is UNATTAINABLE (`RedbStore` is single-writer, exclusive-lock, and
>    already moved into `register_orchestrator_with_store`) and is DROPPED. `RedbStore` becomes a
>    shareable handle; `SpawnCore` shares that ONE physical redb file but commits in a SEPARATE barrier
>    from `commit_barrier`. The v1→v2 (fork-done, crash-before-pid) window + the intent/head divergence
>    window are LEDGERED to 5e (alongside the crash-recovery it already owns) — DEFERRED, not owed in 5c.
> 5. **DRY-lift scope (D5/D6).** Lift ONLY the node-shape helpers `vd-devcluster` and `spawn_node` share
>    (`sibling_binary`, `pid_alive`, `signal_group`, and a new `spawn_node_grouped(exe, common, node_env,
>    log)` — log passed as a PARAM). Leave `vd_bins::spawn_node` byte-identical (12+ callers). EXCLUDE the
>    three `render_*_smoke` client spawns (full arg vector + inherited stdio + `ChildGuard` — not
>    expressible in the env-only shape). Keep `vd-devcluster::down()`'s BATCHED TERM-all/wait-once/KILL-all
>    loop (a per-pid `teardown_group` in a loop would serialize an 8-node `--forest` ~3s→~24s).
> 6. **No DET-1 label in 5c (D7).** An occupant-less Walk-scale Planet emits no `RealmSnap` rows (empty
>    vs empty proves nothing) and the F2 node id differs from a static rig's hand-picked id (head value
>    not byte-identical). 5c's process test asserts only: boots the right realm+profile, self-grants the
>    expected head (head APPEARS), teardown exits AND is reaped. The content-keyed, source-node-stripped,
>    tick-aligned frame-parity DET-1 proof moves to **5g**.
> 7. **`IncarnationCookie` in `vd_core::incarnation` (pure data, no rng/clock)** — minted by the backend,
>    stored by the kernel as opaque bytes. `spawn_probe_server` grows a `cookie: Option<String>` param;
>    gateway + orchestrator pass `None` (D8); shard reads `VD_INCARNATION_COOKIE` and installs the
>    `/whoami` echo; the `probe.rs` "no body" invariant doc is updated to acknowledge `/whoami`'s
>    non-sensitive per-incarnation nonce (the guard relies on the true child HOLDING the probe port, not
>    on cookie secrecy — D9).
> 8. **No dead coverage exemptions (D10).** `vd-bins` is in NO 100% recipe, so NO
>    `#[cfg_attr(coverage_nightly, coverage(off))]` / `coverage-exemptions.toml` entry on `proc_launch.rs`
>    — the Tier-B posture is proven by the §8 process gate; exemptions stay reserved for Tier-A code.
>
> **Build order — AS BUILT (2 gate-able stages; 5c-3 REFOLDED into 5e):** 5c-1 = vd-core
> `IncarnationCookie` + the vd-node seam growth (Tier-A 100% against the one `FakeBackend`) — COMMITTED
> `1a57d33`. 5c-2 = the vd-bins DRY lift + `/whoami` (5c-2a `6a04107`) + `ProcLaunchBackend` + the process
> smoke gate (5c-2b `bc6e1af`, Tier-B). **5c-3 (the shared-`RedbStore` refactor + the orchestrator-bin
> wire) is DEFERRED to 5e**, decided during implementation: `SpawnCore` is INERT under
> `RlmTuning::default()` (the reconciler never sweeps ⇒ it never writes its store), so landing the durable
> store-share in 5c-3 would wire durability into a path NOTHING exercises — and the share is an invasive
> change to the depth-1 single-writer D-6 durability core (shared monotone seq, two channel senders, a
> drop-order/join hazard) that the D-6 kill-9 crash proptests exercise. Its correct home is **5e**, where
> the launch-ledger rehydrate is BUILT and the kill-9-then-rehydrate gate PROVES the shared-writer
> coordination + drop order don't break durability — and 5e precedes 5f (which arms `RlmTuning`), so the
> spawner is never armed over an unshared store. The `MemSpawner` placeholder at `orchestrator.rs:239`
> stays until 5e swaps it for `SpawnCore<ProcLaunchBackend>` over the shared redb. The real spawner itself
> (kernel + hands) is COMPLETE and PROVEN forking real `vd-shard`s as of 5c-2b.
- **5d — CA-1 addressing + idempotency-by-coord.** `update_peer_addr` push at spawn (orch + dest→
  gateway); `VD_PEERS` = ancestor closure; reply-on-connection convergence. *Tests:* orch reaches a
  spawned shard; an older shard reaches a newer via a learned reply lane (pure-loopback repro, D-18
  discipline); a re-issued `SpinUp` on a live coord is a no-op (idempotency keyed on `coord.path`,
  not incarnation — the C-1 guard fix); absent `VD_PEER_HOSTS` ⇒ clean no-op.

> **5d VET AMENDMENTS (workflow `wf_9e6e54c7`, GO_WITH_FIXES) — AS BUILT.**
> 1. **Idempotency-by-`coord.path` ALREADY EXISTS** in the reconciler kernel (`rlm.rs` `launch_live`/
>    `LaunchLedger.minted`, path-keyed). 5d adds NO kernel code — only a paired assertion on the existing
>    `drive_..._holds_while_launching` test proving a RE-ISSUED `SpinUp` on a live/launching coord is a
>    no-op (path-keyed, not incarnation-keyed). **D2 caveat:** the *running/absent* half keys the directory
>    head on the LOSSY `coord.lowered()`, so C-1 is galaxy-correct only until the D-41 path-indexed
>    directory lands (ledgered to D-41, NOT re-keyed here).
> 2. **`VD_PEERS` ancestor closure — the frozen `RealmSpawner` seam is UNTOUCHED.** The closure rides the
>    INTERNAL `LaunchSpec.peers`, computed per-spawn by a monomorphic free helper `closure_peers(coord,
>    live, anchors)` in `vd-node` (Tier-A, 100% — 3 tests incl. the walked-but-not-live arm, D3). The
>    static anchors (ORCH/GATEWAY) are a `SpawnCore` construction field (`anchor_peers`); real values wire
>    at 5e, a fixture until then. `ProcLaunchBackend::child_env` emits `VD_PEERS = book(&spec.peers)` (the
>    ONE formatter, reused). Closure = live ancestors (`coord.parent()`→root) ∪ anchors; excludes
>    self/siblings/descendants/not-yet-live ancestors.
> 3. **Gateway→shard is REPLY-ON-CONNECTION, not a push.** The spec's literal "dest→gateway
>    `update_peer_addr` push" is a REACHABILITY NO-OP (`update_peer_addr` creates no dial lane; the sole
>    lane-create is the boot loop over `cfg.peers`) — the child (which lists GATEWAY as an anchor) DIALS the
>    gateway, which learns it on accept (`LearnedPeers`). This mechanism is ALREADY PROVEN by the existing
>    mesh tests (`..._on_connection_reaches_a_peer_not_in_the_book`, the D-18 symmetric-datagram test) — 5d
>    adds NO mesh code, so no new loopback test (DRY); it only DRIVES the existing machinery. `book_peer` is
>    belt-and-suspenders bookkeeping only.
> 4. **DEFERRED (ledgered, D-RLM-6):** a child's `VD_PEERS`/dial lanes are FIXED at boot — an ancestor
>    absent at spawn, or one that RESTARTS under a new incarnation while a descendant stays live, is NOT
>    reachable by that running descendant (reply-on-connection can't repair it: needs the descendant to be
>    the dialer, which has no addr for the new incarnation). Correct ONLY under parent-first spawn ordering
>    + no ancestor churn under a live descendant. The refresh policy is an OPEN 5e/5f decision (two
>    candidates: reconciler re-spawns the descendant subtree on ancestor-incarnation-change [likely
>    HR-clean, zero new wire]; or a runtime lane-re-key control path [new wire, joint investigation]) with a
>    **5f HARD gate**: ancestor crash+respawn while a descendant stays live → descendant re-reaches within
>    bounded ticks. **D4** first-contact input-drop (Unreliable `Input` before the dest is learned) is a
>    **5f HARD gate** too, not a 5d assertion.
> **As-built 5d = (1) `closure_peers` Tier-A + 3 tests; (2) the path-keyed idempotency assertion; (3)
> `LaunchSpec.peers` + `SpawnCore.anchor_peers` (fixture); (4) `child_env` `VD_PEERS` emit + the smoke test
> carries a peers fixture.** Real anchors + the spawner-internal coord-scan short-circuit fold to 5e;
> end-to-end reachability + D1/D4 hard gates to 5f.
- **5e — crash re-discovery (the C-1/H-1/H-4 capstone) + the ABSORBED 5c-3 durable wiring.** FIRST lands
  what 5c-3 deferred (see the amendment): the shared-`RedbStore` handle (shared monotone seq + the single
  writer/join, so `SpawnCore`'s launch-ledger and the saga WAL share ONE physical redb in SEPARATE
  barriers) and the orchestrator-bin swap of the `MemSpawner` placeholder for
  `SpawnCore<ProcLaunchBackend>` (built with the retained `Arc<MeshControl>` + a `ProcSpawnTuning` whose
  anchors are the orchestrator's own boot env). THEN the crash machinery on top: the orphan-adopt path
  (`ProcLaunchBackend::adopt` + `Slot::Orphan` + the `/whoami` cookie-probe liveness, all descoped from 5c
  per D4), `LaunchLedger::rehydrate` + `DirectoryCore` rehydrate before the first sweep, durable id/port
  high-water, orphan-head sweep, cookie-guarded orphan reap. *Gate tests (process-tier):* **kill-9 the
  orchestrator mid-spawn (before head commit) → rehydrate repopulates `minted`+directory → re-drive → NO
  double-spawn, NO headless zombie, NO F2 id-reuse**; a stranded head with no ledger cell is adopted or
  reaped after the freeze; a killed-then-crashed id is never re-minted. **This gate is ALSO where the
  shared-writer durability is PROVEN** — re-run the full D-6 crash proptest suite (`R-6d4-A` both-ends
  restart) against the shared-handle refactor. Extends the 4b crash-replay proptest with a real-launcher
  variant. (Absorbing 5c-3 here is deliberate: the store-share is unexercised until the reconciler is
  armed at 5f, and 5e both PRECEDES the arm and OWNS the crash test that proves the share.)

> **5e VET AMENDMENTS (workflow `wf_71fd3e40`, GO_WITH_FIXES + 2 user decisions 2026-07-25).**
> - **STORE: SEPARATE `launch.redb`, NOT a shared-writer (user decision; supersedes the "share ONE file"
>   line above).** redb takes a process-exclusive lock, and the single-barrier atomicity that a shared file
>   would buy was ALREADY surrendered by D1's two-barrier design — so `SpawnCore`'s launch ledger gets its
>   OWN redb file + writer (a fresh instance of the SAME proven store; ZERO edits to the depth-1
>   single-writer/seq/Drop core the D-6 kill-9 proptests pin). This DECOUPLES launch fsyncs from the
>   per-tick universe-clock barrier — a warp-burst subtree spin-up can never stall the clock (the D4
>   seamless-freeze hazard the shared writer created). Cost: one extra writer thread + two-file boot.
> - **D1 CRITICAL (blocking): write-ahead must be DURABLE BEFORE THE FORK.** `RedbStore::commit` is
>   block-on-PRIOR (submits this batch to the off-tick writer, returns before ITS fsync), so `spawn_realm`
>   forking right after the v1 commit races v1's durability → a kill-9 in that window = headless zombie +
>   F2 id-reuse + double-spawn. FIX: the `Store` seam grows `flush(&mut self)` (block until every committed
>   write is durable; `MemStore` = no-op, `RedbStore` = wait-durable-through-last-submitted); `spawn_realm`
>   does commit(v1) → `store.flush()` → `backend.launch` → commit(v2). A crash then degrades to at most a
>   `pid:None` partial (D-RLM-5), never a no-intent orphan with a reusable id.
> - **D-RLM-6 = C (lazy resolve-on-miss)** — 5e PICKS + ledgers only (DEFERRED.md D-RLM-6); 5f builds the
>   miss-trigger + re-resolve + the `PeerLocate` addr arm (validated in the no-DNS process gate).
> - **D3:** `SpawnInner` gains a `path_index: BTreeMap<RealmPath, NodeId>`; `closure_peers` does an
>   exact-key lookup per ancestor (O(depth·log L), not the O(depth·L) full scan) — done in 5e since 5e
>   wires the spawn path LIVE.
> - **D2 (scope, ledger):** the launch ledger is full-`RealmPath`-keyed ⇒ NO double-spawn galaxy-wide; but
>   every DIRECTORY-HEAD-keyed reconciler decision (absent/teardown/force-reap + C's resolve) keys the
>   lossy `coord.lowered()` and is single-galaxy-correct until D-41's path-indexed directory — pin it with
>   a two-same-`lowered()`-id case in the kill-9 gate; gate WARP (5f/Step 7) on D-41.
> - **D5:** the rehydrated `minted` is a SUPERSET of pre-crash `minted` (the launch reconcile drains a
>   head-up entry that the live-projection re-seeds), drained on sweep-1 — the proptest invariant is
>   superset-+-sweep-1-drain, NOT byte-identical equality.
> - **Sub-slices (each gate-able):** 5e-1 flush seam + the D1 fix; 5e-2 orch-bin wiring (launch.redb +
>   `SpawnCore<ProcLaunchBackend>`, `RlmTuning::default()` inert → byte-identical); 5e-3 rehydrate seed +
>   the `path_index`; 5e-4 orphan-adopt (`LaunchBackend::adopt` + `Slot::Orphan` + slow-cadence cookie-probe
>   with a post-promotion probe [D7] + per-sweep cap [D8]); 5e-5 kill-9 process gate (ADOPT with full shard
>   anchors + a rehydrate-DISABLED control arm that DOES double-spawn [D6], MID-FSYNC asserting fork-did-not-
>   happen [D1], the two-`lowered()` pin [D2]) + the 4b real-launcher proptest arm (each new op paired with
>   a deterministic example test [D9]).
> - **5e-4 DONE (uncommitted at time of write).** `LaunchBackend` grew `adopt(node, pid, cookie, probe)`
>   (added to the trait AFTER `launch`); `SpawnCore::rehydrate` calls it per `pid:Some` survivor (was a
>   manual `alive`-prime in the Tier-A tests — adopt is now the ONE thing that marks a re-owned child live).
>   Tier-B `ProcLaunchBackend`: `OwnedSlot`→`enum Slot{Owned,Orphan}`; `adopt` inserts an `Orphan{pid,cookie,
>   probe,cache:Some((now,true))}` (optimistic-alive seed defers the FIRST post-rehydrate probe = D8 no-storm);
>   `is_alive` = Owned `try_wait` / Orphan cadence cookie-probe (`admin_get_body(probe,"/whoami",Some(
>   probe_timeout)) == Some(cookie.to_env_string())`, cached within `orphan_probe_interval`); `teardown` =
>   Owned reap / Orphan signal-only (init reaps the reparented corpse). `ProcSpawnTuning` grew
>   `orphan_probe_interval` (`VD_REALM_ORPHAN_PROBE_MS`, 1000) + `probe_timeout` (`VD_REALM_PROBE_TIMEOUT_MS`,
>   500). Gate: 15 Tier-A kernel tests + the process gate (extended with an ADOPT+cookie-probe leg: launch a
>   real shard → adopt into a SECOND backend → `is_alive` true via `/whoami` equality → kill → `is_alive`
>   false once the pid is gone). **DELIBERATE DEVIATION from §1.5's "promote to Owned-equivalent on first
>   contact, leave the probe path":** the orphan is probed at the SLOW cadence for its whole life, NOT
>   promoted-then-untracked — because RLM's ONLY crash-detection signal is `live_nodes → is_alive`, so a
>   promoted-and-untracked orphan that later crashed would report alive forever (a player in a dead realm
>   gets no shard — a correctness bug). The cadence cache still satisfies the anti-storm intent (D8) AND
>   retains continuous death detection (strictly more robust). The head-layer death detection §1.5 assumes
>   is a P6/warp-era concern, not wired in RLM yet.
- **5f — `--demand` launcher + bootstrap seed-demand + non-inert `RlmTuning` + realm-aware
  `select_rehome_target`.** Root-chain-only boot; orchestrator seeds the bootstrap-containing-realm
  demand from the stored spawn pos (§2.2); AoI grows the rest. `--static-forest` retained (temporary
  scaffold). Measure process boot p99 → set `boot_ticks_p99`; validate `min_dwell ≥ boot_ticks_p99 +
  settle`. *Tests (process-tier):* a walk into a neighbor realm spins its shard UP on demand (no
  pre-baked forest) and a walk out spins it DOWN with hysteresis (no thrash) — the process-tier
  repro the mem twin masks.
- **5g — retire the static forest + process-parity capstone.** Delete `ClusterShape::Forest`/the
  literal `Vec`/fixed roster NodeIds/`expected_realms(Forest)`/`--static-forest` once every gate runs
  on `--demand`. Delete `NodeKind::StubShard` after 5a's inertness proof (or keep only as the
  `ProfileKind::Stub` test path). **Capstone gate (`process_parity.rs` extended), release-only band
  (like SPIKE-3a, NOT `coverage-fast`):** on-demand shard byte-identical in output to a static one;
  spawned Galaxy carries `signal_relay`; first-contact ordering holds; the kill-9-mid-spawn no-orphan/
  no-double proof (5e); the demand walk-up/down anti-thrash (5f). *Gate:* full `just gate` green;
  DEFERRED.md D-RLM-1 (k8s) + D-RLM-2 (Cat-C) + warm-pool confirmed as the only Step-5-adjacent
  ledger entries.

**Multithreading (working-method):** 5a and 5b are independent and parallelizable; 5c gates 5e; 5f
gates 5g. Report global progression at arc end (check against the full end-goal: a spun-up shard runs
the ONE author-and-ship law and is ready to host voxels/ships/signals/warp without rework — §3/§5/§6
confirm none of that is cornered).

---

## §9 — Risk register + OPEN QUESTIONS (user decisions)

### Risk register
| # | Risk | Mitigation (in-spec) |
|---|---|---|
| R1 | Crash mid-spawn double-spawn (C-1) | §4.1 durable intent domain + `LaunchLedger::rehydrate` + `DirectoryCore` rehydrate before first sweep; idempotency keyed on `coord.path` |
| R2 | F2 id reuse across crash (H-1) | §4.2 durable monotone high-water, reserve-before-launch, never `max(survivors)` |
| R3 | Unreapable/leaked orphan (H-3, L-1) | §1.4/§4.3 persist `pid`+`pgid`, cookie-guarded reap, group signal |
| R4 | `live_nodes()` probe storm (H-3′) | §1.5 cache + slow orphan cadence + promote-on-first-contact; trait doc states per-sweep |
| R5 | Stranded directory head (H-4) | §4.4 orphan-head sweep behind the freeze |
| R6 | HR5 launcher escapes the gate (H-4′) | §1.1 decision half in Tier-A `vd-node`; only the syscall shim in `vd-bins`, exempted |
| R7 | 5a not byte-identical (H-2) | §3.2 capability-inertness proof as the 5a acceptance gate, not an assertion |
| R8 | Demand loop inert (H-3″) | §1.3 `VD_OWN_COORD` + `VD_BOOT_TICKS_P99` plumbed into `StubConfig` |
| R9 | O(N²) peer booking (M-6) | §7 book ancestor closure + anchors only; siblings via reply-on-connection |
| R10 | Process-tier scale ceiling (M-2) | §6 explicitly dev/CI single-host; k8s is the scale path; nothing in reconciler assumes cheap spawn |
| R11 | PID reuse aliases wrong process (M-3/M-5) | §4.3 IncarnationCookie identity token; ports reclaimed only from confirmed-dead slots |
| R12 | Bootstrap has no seed demand (M-8) | §2.2 minimal seed-demand injector in Step 5 (not deferred to Step 6) |

### OPEN QUESTIONS — RESOLVED (user, 2026-07-25)
- **OQ-1 (frozen-taxonomy edit) → DECIDED: ADD `SpawnError::LaunchFailed { reason }`.** A real/k8s
  spawn can fail meaningfully; returning `UnknownNode` for a launch refusal is a semantically-false
  value the no-corners mandate forbids. Additive; does not disturb `UnknownNode`/`AlreadyKilled`;
  each variant independently reachable (no HR5 dead region); `exec_spinup` already absorbs `Err(_)`
  into the backoff. Lands in slice 5b (the decision kernel returns it; 5c's real backend produces it).
- **OQ-2 (warm-spare-pool) → DECIDED: DEFER to Step 7, seam kept open.** Step 5 ships the predictive-AoI
  (F7) latency term, which hides cold-boot latency generically. The warm pool is a pure OPTIMISATION on
  top, and its right size/policy is only measurable against a real warp fly-by (Step 7). When built it
  MUST be the generic form — a pool of **profile-agnostic BLANK shards**, "spawn" = assign-a-coord to a
  waiting blank, entirely behind the frozen `RealmSpawner` contract (no caller change, no mode branch,
  no per-kind pool). **LEDGERED in `docs/design/DEFERRED.md` (D-RLM-4) so it is not forgotten (user:
  "don't forget about it").** The `LaunchBackend` seam (§1.1) is the drop-in point.
- **OQ-3 (NEW library) → DECIDED: NO new dependency in Step 5.** Orphan reap uses the admin-probe +
  IncarnationCookie handshake (§4.3), not `nix`/`libc` raw signalling. A k8s apiserver client stays
  deferred to `K8sSpawner`/#123. Step 5 uses only `std::process::Command`/`spawn_node` + the existing
  mesh + redb. (If a future wedged-pod case proves the admin-probe insufficient, `nix`/`libc` returns
  as an investigate-together item — not adopted now.)
- **OQ-4 (bootstrap policy) → DECIDED: ORGANIC.** The generator resolves the bootstrap occupant's
  stored pose to the deepest containing realm and injects it into the SAME per-realm AoI demand path
  everything else uses (§2.2) — one generic mechanism, exercises the real production path, no
  bootstrap-only shortcut.

---

## Hard-constraint ledger

- **Only generic/scalable/robust, no scope reduction:** one generic backend behind the frozen trait;
  O(closure) spawn, `BTreeMap` live set; no per-realm-kind code. Warm-pool + k8s deferred with clean
  seams (§5/§6), not scoped-out hacks. Every critic-found crash hole resolved in-spec (§4).
- **NO Dormant/Active state machine:** every spawned shard runs the identical `has_synced`-gated
  author-and-ship law; an occupant-less ancestor is the same code with nothing to integrate (§3.3).
  `ProcSpawner` has no mode branch; the only lifecycle axis is up-vs-down (Step-3 reconciler).
- **NO static-forest residue:** `--static-forest` is a TEMPORARY migration scaffold deleted in 5g;
  the end state is forest-free with a self-starting demand loop (§2).
- **HR1 sealed shards:** the spawner (both halves) lives OUTSIDE the sealed shard; it chooses the
  address, the shard constructs its own `MeshTransport` (§1.3). No new inter-shard bytes; demands
  ride the frozen `RealmDemand` arm.
- **HR3 one-tooling / no match-on-kind:** the single kind→profile match is `profile_kind_of`
  computed into the shard's coord-derived profile; `reconcile`/`drive`/features never match on kind
  (§3.1). One `vd-shard` binary, `ShardProfile` capability configs.
- **HR5 100% Tier-A:** decision logic in Tier-A `vd-node` covered via a fake `LaunchBackend`; the OS
  shim is an explicit exemption (§1.1). Object-safe trait, no monomorphization debt.
- **F2 monotone NodeId, never reused:** durable high-water, reserve-before-launch, resumed above the
  durable max (§4.2) — never `max(survivors)`.
- **New library ⇒ flag for the user:** OQ-3. Step 5 adds none.
