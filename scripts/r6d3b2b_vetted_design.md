# R-6d3b-2b Vetted Design of Record (wf_e91a84e5)

## ADJUDICATED SYNTHESIS
I have verified all the load-bearing claims. This is an adjudication (design synthesis), not an implementation task — no task tracking needed. Let me confirm the last detail: whether `send_durable`'s early-return at line 1899 (unknown lane) already returns `QueueFull` *before* assigning `next_msg_id`, which is the precedent the reviewer's msg_id HIGH relies on. I saw it does (line 1899 returns before line 1901). I now have everything.

Key verifications complete:
- **RC-2a HIGH (gc-sweeps a recoverable roster-diff row)**: CONFIRMED. `outbox.rs:4-8` states the outbox exists precisely for producer-less one-shots. The landed 2a test (`outbox.rs:698-724`) asserts the exact opposite invariant the DESIGN inverts: "the un-routable row SURVIVES — gc was skipped ... a durable row is never swept because it could not be re-driven." The DESIGN's §2.3 decision to gc-sweep it is a real D-6 #1 regression.
- **RC-2a MEDIUM (version-mismatch silently gc-swept)**: CONFIRMED. `scan_all` filters via `decode_value` (`:246`) which returns `None` on version mismatch (`:150-155`), so version-mismatched rows never reach the send loop; but `gc_below` is key-only (`:254-266`, no `decode_value`) so it sweeps them silently.
- **RC-2b HIGH (bumping `next_msg_id` on `Closed` while returning `QueueFull`)**: CONFIRMED. `mesh.rs:1899` (unknown lane) and `mem.rs:357-361`, `fabric.rs:533/539` all return `QueueFull` *before* assigning/consuming a `msg_id`. Consuming an id on a `QueueFull` return breaks the FIFO correlation contract (`io/mod.rs:141-147`).
- `lane.tx.is_closed()` is available (tokio mpsc `Sender`, `PeerLane.tx` at `mesh.rs:697`).

Here is the adjudicated design of record.

---

# R-6d3b-2b — ADJUDICATED IMPLEMENT-READY DESIGN OF RECORD

**VERDICT: SOUND_TO_IMPLEMENT (with three folded must-fixes).**

All cites verified against worktree HEAD (`worktree-new-system`, 2a at 1789e49). The frozen `sim::io` seam (`crates/sim/src/io/mod.rs:124-165`) and the `vd-wire` contract stay **UNTOUCHED**. Everything new is io-prod-internal or bin-internal.

## Adjudication of the three review findings

| Finding | Lens | Verdict | Rank |
|---|---|---|---|
| **F1 — gc-sweep destroys a recoverable roster-diff `Unroutable` row (D-6 #1 regression)** | RC-2a | **CONFIRMED — UPGRADE to blocking. Silent-loss of the exact payload class the outbox exists for.** | 1 |
| **F2 — `next_msg_id` bump on `Closed` while returning `QueueFull` breaks the frozen FIFO msg_id contract** | RC-2b | **CONFIRMED — seam-contract breach. Must delete the bump + the bounce.** | 2 |
| **F3 — version-mismatched row gc-swept with no count/log (silent loss of the case `:47` names)** | RC-2a | **CONFIRMED — real, but latent (`OUTBOX_FORMAT_VERSION==1`). Fixed for free by F1's retain-not-sweep decision.** | 3 |

The two NITs (GW-1 assert ordering, runtime-Handle lifetime) are CONFIRMED and folded. The RC-2a LOW ("never silently lost" language inconsistency) resolves automatically once F1 is fixed.

**Root-cause dedup:** F1 and F3 are the *same* root cause — **`gc_below(new_incarnation)` blindly sweeps the entire prior-incarnation window, including rows that were NOT re-driven this boot.** ONE fix (sweep only the rows actually re-driven) cures both. The DESIGN's §2.3 "gc-sweeps the poison rows too" decision is the defect; the correct rule is the one the landed 2a code already enforces: **never sweep a row that was not re-driven.**

---

## 1. Sub-slice A — RC-2a: quarantine + the corrected gc rule (outbox.rs only, LANDS FIRST)

### 1.1 The corrected disposition table

| Arm | Nature | Disposition |
|---|---|---|
| `Undecodable` (send-loop) | genuine on-disk frame corruption (a version-mismatched row is already filtered out by `scan_all` via `decode_value`, `:246`+`:150-155`) | **QUARANTINE: count + warn + PROCEED; RETAIN on disk (NOT swept)** |
| `Unroutable{peer}` | roster diff — peer left `VD_PEERS` (routinely a transient operator misconfig) | **QUARANTINE: count + warn + PROCEED; RETAIN on disk (NOT swept)** |
| `LaneStuck{peer}` | a live lane wedged full past `REPLAY_SEND_MAX_RETRIES` | **REFUSE TO BOOT (fail loud), gc SKIPPED** |
| `LaneDead{peer}` (NEW, sub-slice B) | the peer_writer task is dead — a still-in-roster peer's lane collapsed | **REFUSE TO BOOT (fail loud), gc SKIPPED** |
| `FenceTimeout{..}` | a peer_writer never submitted within `REPLAY_FENCE_DEADLINE` | **REFUSE TO BOOT (fail loud), gc SKIPPED** |

### 1.2 The corrected gc rule (folds F1 + F3) — **sweep ONLY the re-driven keys**

The DESIGN's `gc_below(new_incarnation)` is REPLACED. `gc_below` sweeps by `incarnation < N` (key-only, `:254-266`) — it cannot tell a re-driven row from a quarantined/skipped one. **Sweep the exact set of keys that were re-sent this boot, not the whole prior window.** This is the invariant the landed 2a already guards (`outbox.rs:699-701`): "a durable row is never swept because it could not be re-driven."

Add a targeted-delete method to the sink (object-safe, HR3 — ONE seam):

```rust
// OutboxSink trait (outbox.rs:161) — NEW method alongside gc_below:
/// Stage the removal of exactly these keys — the rows RE-DRIVEN this boot (now durable at the fresh
/// incarnation, so the prior-incarnation copy is redundant). A row NOT in this set (quarantined-unroutable,
/// undecodable, or version-mismatched-and-filtered) is RETAINED for the next boot — never swept because it
/// could not be re-driven (the D-6 #1 no-loss invariant). Durable only after `commit`.
fn gc_replayed(&mut self, replayed_keys: &[OutboxKey]);
```
`NodeOutbox::gc_replayed` = `for k in replayed_keys { self.store.delete(&k.to_bytes()); }`. **`gc_below` is retained but no longer called by `replay_outbox`** (keep it — the golden/persistence tests still cover it; it is the future admin/whole-window sweep). This eliminates the F3 silent version-mismatch sweep entirely: a version-mismatched row is never in `replayed_keys` (it never decoded), so it is RETAINED, counted-latent (see §1.4), never silently deleted.

### 1.3 `replay_outbox` return + body (the count-fence-uses-good-rows fix)

Return `Result<ReplayCounts, ReplayError>`:

```rust
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ReplayCounts {
    /// Rows re-sent (block A durable, fence passed) and then gc'd (redundant prior copy).
    pub replayed: usize,
    /// Rows QUARANTINED — RETAINED on disk (not swept): roster-gone peer (recoverable once the book is
    /// fixed) or an undecodable frame. Counted + warn-logged. NEVER silently dropped, NEVER swept.
    pub quarantined: usize,
}
```

Body changes to `replay_outbox` (`:364-424`):

```rust
let (rows, durability, base) = { /* unchanged brief-lock snapshot */ };
if rows.is_empty() { return Ok(ReplayCounts::default()); }

let mut counts = ReplayCounts::default();
let mut replayed_keys: Vec<OutboxKey> = Vec::new();   // the EXACT re-driven set for gc_replayed
for (key, framed) in &rows {
    if !peers.contains_key(&key.peer) {
        tracing::warn!(peer = key.peer.0, class = ?key.class, incarnation = key.incarnation, seq = key.seq,
            "boot replay QUARANTINE: retained row's peer is no longer in VD_PEERS (roster diff) — \
             RETAINED for the next boot (recoverable once the roster is corrected); the boot PROCEEDS");
        counts.quarantined += 1;
        continue;                                      // RC-2a: was `return Err(Unroutable)`
    }
    let Some(payload) = decode_value_payload(framed) else {
        tracing::warn!(peer = key.peer.0, class = ?key.class, incarnation = key.incarnation, seq = key.seq,
            "boot replay QUARANTINE: retained row is undecodable (on-disk frame corruption) — \
             RETAINED (unrecoverable but preserved for admin forensics); the boot PROCEEDS");
        counts.quarantined += 1;
        continue;                                      // RC-2a: was `Undecodable ?`
    };
    send_durable_with_retry(transport, key.peer, key.class, payload)?;  // LaneStuck/LaneDead STILL ?-bail
    replayed_keys.push(*key);
    counts.replayed += 1;
}

// Count-anchored fence over the SENT rows ONLY (the F-review load-bearing point — confirmed correct).
let n = counts.replayed as u64;
if n == 0 {
    // Nothing re-driven ⇒ nothing to gc. Every row was quarantined+RETAINED. No sweep at all.
    return Ok(counts);
}
let target = base.checked_add(n).expect("replay_outbox: durability watermark saturated …");  // unchanged
// … existing while-fence + FenceTimeout + wait_durable_through(target) …

// (4) ONE brief lock: sweep ONLY the re-driven keys (NOT gc_below) — retained rows survive.
{
    let mut g = shared.lock().unwrap_or_else(PoisonError::into_inner);
    g.gc_replayed(&replayed_keys);
    g.commit();
}
Ok(counts)
```

**Why `n = counts.replayed` not `rows.len()` (CONFIRMED correct in the DESIGN, kept):** a quarantined row never enqueues a send ⇒ never bumps `last_submitted`; if it were in the fence target the fence would `FenceTimeout` on any boot that quarantined a row (a false wedge). Fence over the re-driven set only.

**The `n==0` path no longer gc's** (DESIGN had it gc-ing the whole window — that was the F1 defect). If every row was quarantined, everything is RETAINED; the fix drops the DESIGN's "still gc the swept window" line.

### 1.4 F3 disposition (folded, honest scoping)

A version-mismatched row is filtered by `scan_all` before the send loop, so it is neither replayed nor quarantine-counted here — but with `gc_replayed` it is now **RETAINED**, not silently swept. The `:47` "quarantine" promise is honored by retention. Add a DEFERRED.md note: *"a version-mismatched row is currently RETAINED-but-uncounted (scan_all filters it pre-loop); an explicit pre-loop `scan_all_raw` tally that counts version-mismatched rows is an owed R-6d4 enhancement. Latent while `OUTBOX_FORMAT_VERSION==1`."* The DESIGN's unqualified "count+log fully honors :47" claim is DROPPED per the reviewer.

### 1.5 Sub-slice A tests (invert the two landed abort tests + add)

- **`replay_outbox_quarantines_an_unroutable_row_and_retains_it_and_proceeds`** (rewrites `:697`): seed a peer-9 row (not in `peers`) + a routable peer-2 row at the same prior incarnation ⇒ `Ok(ReplayCounts{replayed:1, quarantined:1})`; peer 2's payload in `FlakyTransport.sent`; after the run `scan_all()` has **exactly the peer-9 row** (retained; peer-2's prior copy gc_replayed-swept). This asserts BOTH quarantine-proceed AND the F1 retain invariant.
- **`replay_outbox_quarantines_an_undecodable_row_and_retains_it`** (rewrites `:726`): garbage-value routable row ⇒ `quarantined:1`, `scan_all()` still has it (retained, not swept).
- **`replay_outbox_refuses_to_boot_on_a_wedged_lane_without_gc`** (NEW): a `FlakyTransport` with `fails_left = REPLAY_SEND_MAX_RETRIES` ⇒ `Err(LaneStuck)`, `scan_all()` intact (gc skipped).
- **`replay_outbox_all_quarantined_does_not_gc`** (NEW): every seeded row unroutable ⇒ `Ok(ReplayCounts{replayed:0, quarantined:k})`, `scan_all()` == all rows (n==0 path, no sweep).

Deterministic in-process (no real clock). Tier-B io-prod region floor 90; outbox.rs is 98.29% — the quarantine-continue arms, the `gc_replayed` call, and the `n==0` early return are all covered.

---

## 2. Sub-slice B — RC-2b: `Closed` disambiguation (mesh.rs + outbox.rs, folds F2)

### 2.1 F2 fix — **DELETE the `NodeUnreachable` bounce + the `next_msg_id` bump** (CONFIRMED breach)

The DESIGN's "on `Closed`, push `NodeUnreachable` and bump `next_msg_id`, return `QueueFull`" **breaks the frozen FIFO msg_id contract** (`io/mod.rs:141-147`: `undelivered` is *"the transport-assigned FIFO sequence of that send"*). Every other `send_durable` returns `QueueFull` **before** assigning a msg_id: `mesh.rs:1899` (unknown lane), `mem.rs:357-361`, `fabric.rs:533` all early-return `QueueFull` pre-assignment. A `QueueFull` that consumes a msg_id desyncs the caller's send-count correlation by one. **`send_durable` on `Closed` stays exactly as today (`mesh.rs:1913-1916`): fold `Full | Closed` → `QueueFull(frame.bytes)`, assign no id, push nothing.** The dead-lane fast-fail is delivered entirely by the `lane_alive` probe below — no seam-adjacent bounce needed.

### 2.2 The mechanism (frozen `sim::io::Transport` UNTOUCHED)

A NEW io-prod-internal supertrait carries the liveness probe off the frozen seam:

```rust
// mesh.rs (io-prod-internal — NOT a modification of the frozen sim::io::Transport):
pub(crate) trait ReplayTransport: Transport {
    /// True iff a live send lane exists for `peer` (its peer_writer task is alive). A dropped mpsc
    /// Receiver (dead writer) closes the Sender ⇒ is_closed(). O(1). A peer with no lane at all
    /// (roster miss) also reports false — but replay_outbox's pre-send membership check catches that first.
    fn lane_alive(&self, peer: NodeId) -> bool;
}
impl ReplayTransport for MeshTransport {
    fn lane_alive(&self, peer: NodeId) -> bool {
        self.lanes.get(&peer).is_some_and(|l| !l.tx.is_closed())
    }
}
```
`PeerLane.tx` is a `tokio::sync::mpsc::Sender` (`mesh.rs:697`) — `is_closed()` is O(1) and true once the receiver drops. `send_durable_with_retry` (`:317`) and `replay_outbox` (`:364`) change their param `&mut dyn Transport` → `&mut dyn ReplayTransport`. `send_durable_with_retry` checks liveness on the FIRST `QueueFull`:

```rust
Err(SendError::QueueFull(returned)) => {
    if !transport.lane_alive(peer) {
        return Err(ReplayError::LaneDead { peer });   // ~1 poll, not ~10s corpse-spin
    }
    std::thread::sleep(REPLAY_POLL_BACKOFF);
    pay = returned;
}
```
Add `ReplayError::LaneDead { peer }` — a **REFUSE-TO-BOOT** arm (the peer is still in the roster; its writer died = infra pathology, distinct from a roster-gone `Unroutable`). `FlakyTransport` (`:633`) impls `ReplayTransport` with `lane_alive → true` by default; add a field so the fast-fail test can force `false`.

### 2.3 In 2b or a follow-up? — IN 2b

Confirmed IN 2b: the moment sub-slice C flips a bin off `None`, `send_durable_with_retry` runs against a real `MeshTransport`; a corpse-lane would spin `REPLAY_SEND_MAX_RETRIES × REPLAY_POLL_BACKOFF ≈ 10s` before `LaneStuck`, then refuse to boot. It is not a *silent* break (it eventually fails loud), but shipping the outbox live with a 10s-corpse-spin violates no-shortcuts. The supertrait is contained (one trait, one method, one enum arm) — ship it whole. It does NOT touch the frozen seam.

### 2.4 Sub-slice B tests

- **`send_durable_folds_closed_into_queuefull_without_consuming_an_id`** (mesh unit): construct a `MeshTransport`, drop/close a peer's writer so `lane.tx.is_closed()`; call `send_durable` twice to that peer; assert BOTH return `Err(QueueFull)` AND `drain_inbound()` is empty (NO bounce) AND a subsequent successful send to a LIVE peer gets the expected next msg_id (no id consumed by the closed sends — the F2 correctness pin).
- **`send_durable_with_retry_fails_fast_on_a_dead_lane`**: `FlakyTransport{ lane_alive: false, always QueueFull }` ⇒ `Err(LaneDead{peer})` with wall-clock `< 200ms` (far below the ~10s cap), proving no corpse-spin. Exercises the new `lane_alive` branch deterministically.

---

## 3. Sub-slice C — DRY bin wiring (bins/src/lib.rs + shard.rs + gateway.rs; depends on A+B; flips the gate LIVE)

### 3.1 `vd_bins::boot_mesh_and_replay` (NEW in `crates/bins/src/lib.rs`)

```rust
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, SharedOutbox, spawn_mesh};
use vd_io_prod::outbox::OutboxSink;
use vd_io_prod::trust::ClusterTrust;
use std::sync::{Arc, Mutex};

/// R-6d3b-2b: THE ONE node boot sequence shared by shard.rs + gateway.rs (HR3 — no per-kind fork).
/// Resolves the process incarnation ONCE (finding C — a second resolve double-increments the durable
/// BootCounter, lib.rs:243), opens the durable outbox (Some iff VD_OUTBOX_PATH set), wraps it as the
/// cloneable SharedOutbox, spawns the mesh WITH the sink (durable-before-send gate LIVE), then REPLAYS the
/// retained rows BEFORE the transport is handed to build_app. Returns the transport + the MeshControl
/// (the caller MUST keep BOTH `control` AND the owning tokio Runtime alive for the whole tick loop —
/// dropping control closes the endpoint, mesh.rs:709; dropping the Runtime stops the peer_writer tasks).
///
/// The replay fences on DURABILITY (block A submit), NOT DELIVERY (block C QUIC send) — so it does NOT
/// hang on peers down at boot; their rows are RETAINED + the R-4a retransmit timer re-drives on reconnect.
///
/// # Errors
/// A bad incarnation source; an unopenable/non-durable outbox path; a spawn_mesh failure; OR a replay that
/// hits a TRANSIENT wedge (LaneStuck / LaneDead / FenceTimeout — refuse loud). A poison row or a roster-gone
/// peer is QUARANTINED (RETAINED + loud-counted) and the boot PROCEEDS (RC-2a) — never wedged on one bad row.
pub fn boot_mesh_and_replay(
    env: &EnvConfig,
    runtime: &tokio::runtime::Handle,
    trust: &ClusterTrust,
) -> Result<(MeshTransport, MeshControl), Box<dyn std::error::Error>> {
    let local = env.node_id("VD_NODE_ID")?;
    let peers = env.peer_book("VD_PEERS")?;
    let new_incarnation = resolve_process_incarnation(env)?;      // FINDING C: resolve ONCE

    let shared: Option<SharedOutbox> = open_node_outbox(env)?
        .map(|ob| Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>)));

    let (mut transport, control) = spawn_mesh(
        runtime, trust,
        &MeshConfig::new(local, env.parse("VD_BIND")?, peers.clone(),
                         env.parse("VD_OUTBOUND_CAP")?, new_incarnation),
        shared.clone(),
    )?;

    if let Some(sh) = shared.as_ref() {
        // &mut transport IS a MeshTransport, which impls ReplayTransport (sub-slice B).
        let counts = vd_io_prod::outbox::replay_outbox(sh, &mut transport, &peers, new_incarnation)?;
        if counts.replayed > 0 || counts.quarantined > 0 {
            tracing::info!(replayed = counts.replayed, quarantined = counts.quarantined,
                "boot replay: re-drove retained durable outbox rows (quarantined rows RETAINED)");
        }
    }
    Ok((transport, control))
}
```

`resolve_process_incarnation` is called **exactly once** and threaded to both `MeshConfig::new` and `replay_outbox` — closing finding C's +2 double-increment of the mutating `BootCounter` (`lib.rs:243`). `peers` parsed ONCE, `.clone()`d into `MeshConfig::new` (by value) + `&peers` to `replay_outbox`.

**`replay_outbox`'s param is `&mut dyn ReplayTransport` (pub(crate), sub-slice B)** — `boot_mesh_and_replay` lives in the `bins` crate and passes `&mut transport` (a concrete `MeshTransport`, which impls it). Confirm `ReplayTransport` is `pub` (not `pub(crate)`) OR that `replay_outbox` accepts `&mut MeshTransport` at the bins boundary. **Correction to sub-slice B:** since the bin (a separate crate) must call `replay_outbox`, `ReplayTransport` must be `pub` (io-prod public API, but NOT a modification of the frozen `sim::io::Transport`). Keep the impl `pub`; it is a new io-prod public supertrait — the frozen seam is still untouched.

### 3.2 The bins (HR3 byte-identical)

**shard.rs** — replace `:22-52`. **Fold the GW-1 NIT: move the `VD_SNAPSHOT_BUDGET` parse+assert to BEFORE the helper call** (it needs only `env`, keeps the loud config guard instantaneous rather than after a ≤30s replay fence):

```rust
let snapshot_budget: usize = env.parse("VD_SNAPSHOT_BUDGET")?;
assert!(snapshot_budget <= vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET,
    "VD_SNAPSHOT_BUDGET {snapshot_budget} exceeds the conservative datagram floor {}",
    vd_wire::channels::CONSERVATIVE_DATAGRAM_BUDGET);
let (transport, _control) = vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
let mut node = build_app(NodeConfig { node_id: local, kind: NodeKind::StubShard }, transport);
```
Keep `let local = env.node_id("VD_NODE_ID")?;` in the bin (needed for `NodeConfig`); the helper re-derives its own `local` cheaply. **`runtime` stays bound in `main`** (it already outlives the loop — the NIT-2 reminder; the helper takes only `&Handle`). Bind `_control` (or `control`) so the endpoint is not dropped — it already is bound in `main`'s scope today.

**gateway.rs** — replace `:21-43`:
```rust
let (transport, _control) = vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
let mut node = build_app(NodeConfig { node_id: local, kind: NodeKind::Gateway }, transport);
```

**Boot-order + no-hang (CONFIRMED):** `build_app(..., transport)` consumes `transport` by value (`app.rs:87`), so `replay_outbox` on `&mut transport` provably runs first. Block A (`write_frame` retain + `submit_barrier`) runs before block C (`ensure_connection`), so a down-at-boot peer's row becomes durable regardless of connectivity ⇒ the count-fence passes ⇒ no hang; the row is RETAINED and the R-4a retransmit timer re-drives on reconnect.

### 3.3 Sub-slice C tests

- **`crates/bins/tests/outbox_boot_replay.rs`** (NEW, process tier, real-QUIC loopback restart; model on `boot_counter_crashloop.rs`): pre-seed a `NodeOutbox` at a FIXED temp `VD_OUTBOX_PATH` (`VD_OUTBOX_EPHEMERAL_OK=1`) with one `Retained` Saga row keyed to peer B at incarnation N-1 (open directly like `retained_frames_survive_a_reopen`, retain+commit, drop). Spawn receiver B, then spawn producer A; A's `boot_mesh_and_replay` re-drives the row at A's fresh incarnation. Assert: B receives the payload AND A's reopened `scan_all()` has NO `incarnation < N` rows for that key (gc_replayed swept the re-driven copy). Completing within the deadline is the no-deadlock proof. **This is the D-6 #1 RESTART-half closure.**
- **`boot_mesh_and_replay_increments_the_boot_counter_by_exactly_one`** (bins test): temp `VD_BOOT_STATE_DIR` + `VD_OUTBOX_PATH`, boot once through the helper, assert the durable counter advanced by exactly 1 (finding C — proves resolve-ONCE). Extends the `resolve_uses_the_durable_counter_and_increments` shape (`lib.rs:734`).

---

## 4. Sub-slice ordering (proposed)

**A → B → C.** A (RC-2a quarantine+retain) and B (RC-2b `lane_alive`) are independently mergeable (each touches only outbox.rs / outbox.rs+mesh.rs, both leave the bins on `None` = inert byte-identical). C depends on both (it calls the `ReplayCounts`-returning `replay_outbox` and passes a `ReplayTransport`) and is the ONLY slice that flips the gate LIVE. **Merging C last means the outbox stays inert until the quarantine + fast-fail safety nets are already in place** — a bad row or a dead lane can never wedge a live boot the moment the gate turns on.

## 5. Files touched (per sub-slice)

- **A (outbox.rs only):** `ReplayCounts` struct; `OutboxSink::gc_replayed` + `NodeOutbox` impl; `replay_outbox` return→`Result<ReplayCounts,ReplayError>`, quarantine-continue+retain, `n=counts.replayed`, `n==0` no-gc, `gc_replayed(&replayed_keys)` (NOT `gc_below`); rewrite the two abort tests to quarantine-and-**retain**-and-proceed + the refuse-on-wedge + all-quarantined tests. `lib.rs` re-export `ReplayCounts`.
- **B (mesh.rs + outbox.rs):** `send_durable` UNCHANGED (`Full|Closed`→`QueueFull`, **no bounce, no id bump** — F2); NEW `pub trait ReplayTransport: Transport { fn lane_alive }` + `impl for MeshTransport`; `send_durable_with_retry`+`replay_outbox` param→`&mut dyn ReplayTransport`; `ReplayError::LaneDead{peer}` (refuse arm) fast-fail on first `QueueFull` from a dead lane; `FlakyTransport` impls `ReplayTransport`. Dead-lane no-bounce/no-id + fast-fail tests.
- **C (bins/src/lib.rs + shard.rs + gateway.rs):** NEW `pub fn boot_mesh_and_replay`; both bins collapse to the helper call (bind `_control`, keep `runtime` in main); shard.rs moves the GW-1 assert BEFORE the helper. NEW `outbox_boot_replay.rs` + the counter-by-one bins test.

## 6. Out of scope (named as later slices)
- **R-6d3c / RC-3** — the saga `BatchHandoff` `AwaitAdopt` phase-split (`saga.rs:922`), the never-restart closure; unreachable without CA-1/L5. NOT designed here.
- **R-6d4** — SIGKILL-mid-fsync boot-replay e2e; both-ends-restart replay proptest; store-test-hooks no-premature-gc pin; the F1 durable-before-send process-tier pin; the F3 explicit version-mismatch tally (`scan_all_raw`).
- **CA-1** — no-DNS k3d dynamic peer addressing. NOTED caveat: §3.3's e2e runs on a LOOPBACK restart (like `boot_counter_crashloop.rs`), NOT a real pod reschedule — loopback proves the durable-outbox restart-replay mechanism; k3d only adds pod addressing on top (a roadmap item, not a 2b will-break).
- Optional forensic quarantine keyspace + a `gc_below` skip-tag — a DEFERRED.md enhancement, superseded by the F1 retain-in-place decision (quarantined rows now stay on disk by default).

## HR posture
- **HR3:** ONE `boot_mesh_and_replay`, ONE `open_node_outbox`, ONE `replay_outbox`, ONE `SharedOutbox`, ONE `gc_replayed` — shard + gateway wire byte-identically, no shard-kind match.
- **Frozen seam:** `sim::io::Transport`/`SendError`/`Inbound` and `vd-wire` UNCHANGED. `ReplayTransport`/`ReplayCounts`/`LaneDead`/`gc_replayed`/`boot_mesh_and_replay` are io-prod/bins-additive.
- **No-silent-loss (F1+F2+F3 folded):** every quarantined row is warn-logged + counted **AND RETAINED on disk** (recoverable once the roster is fixed / preserved for forensics); gc sweeps ONLY the re-driven keys; a version-mismatched row is retained (F3 cured); `Closed` no longer consumes a msg_id (F2 seam-contract preserved); a dead lane fast-fails `LaneDead` in ~1 poll; every transient wedge fails loud + skips gc; a down-at-boot peer's row is durable + retransmit-re-driven.
- **No magic numbers:** `REPLAY_SEND_MAX_RETRIES`/`REPLAY_POLL_BACKOFF`/`REPLAY_FENCE_DEADLINE`/`OUTBOX_WRITER_CHANNEL_DEPTH` reused as-is; the `lane_alive` fast-fail adds none.

---
## SUB-SLICES
- Sub-slice A (RC-2a quarantine) — outbox.rs only, independently mergeable, LANDS FIRST: change replay_outbox to return Result<ReplayCounts,ReplayError>; quarantine-continue (warn+count) on Undecodable + roster-diff Unroutable; fence N = counts.replayed; gc still sweeps poison rows (unrecoverable, counted not byte-hoarded); n==0 all-quarantined gc-only path; refuse-to-boot only on LaneStuck/FenceTimeout. Update the two existing abort tests to quarantine-and-proceed + a refuse-on-wedge test. Deterministic default-gate tests, Tier-B ratchets.
- Sub-slice B (RC-2b Closed disambiguation) — mesh.rs + outbox.rs, independently mergeable: split TrySendError::Full|Closed in send_durable; on Closed push Inbound::NodeUnreachable via push_inbox + bump next_msg_id, still return QueueFull (frozen SendError untouched); NEW io-prod-internal ReplayTransport: Transport supertrait with lane_alive (l.tx.is_closed()) impl'd for MeshTransport; send_durable_with_retry + replay_outbox take &mut dyn ReplayTransport; fast-fail ReplayError::LaneDead on first QueueFull from a dead lane (~1 poll, not ~10s corpse-spin); LaneDead is a refuse-to-boot arm. Dead-lane bounce + fast-fail tests.
- Sub-slice C (DRY bin wiring — flips the gate LIVE) — bins/src/lib.rs + shard.rs + gateway.rs, depends on A+B: NEW pub fn boot_mesh_and_replay(env, runtime_handle, trust) -> (MeshTransport, MeshControl) that resolves incarnation ONCE (finding C), opens+wraps the outbox as SharedOutbox, spawn_mesh(Some(shared)), replay_outbox before build_app, ?-propagates only transient Err. Both bins call it identically (HR3), bind control (was _control). Boot-replay e2e (real-QUIC loopback restart, modeled on boot_counter_crashloop.rs) + a counter-advances-by-one bins test.
- OUT OF SCOPE (named as later slices): R-6d3c RC-3 — the saga BatchHandoff AwaitAdopt phase-split (saga.rs:922) — the never-restart closure, unreachable without CA-1/L5. R-6d4 — SIGKILL-mid-fsync boot-replay e2e + both-ends-restart replay proptest + store-test-hooks deterministic no-premature-gc pin + the F1 durable-before-send pin in the process tier. CA-1 — no-DNS k3d dynamic peer addressing (noted caveat, the real-cloud reschedule proof; NOT solved here). Optional forensic quarantine keyspace (a DEFERRED.md enhancement, not a correctness requirement).

---
## DESIGNER (original)
# R-6d3b-2b — Wire the durable outbox LIVE into shard/gateway boot: IMPLEMENT-READY Design of Record

All cites are against worktree HEAD (`worktree-new-system`, R-6d3b-2a landed at commit 1789e49). The frozen `sim::io` seam (`crates/sim/src/io/mod.rs:124-379`) and the `vd-wire` contract stay **UNTOUCHED**. Everything new is io-prod-internal or bin-internal.

## 0. State of the world (grounded — what 2a already gave us)

R-6d3b-2a already landed the machinery. The 2b work is purely: (1) call it from the two bins via a DRY helper, (2) fold RC-2a into the already-landed `replay_outbox`, (3) fold RC-2b into `send_durable`. Confirmed present:

- `pub fn replay_outbox(shared: &SharedOutbox, transport: &mut dyn Transport, peers: &BTreeMap<NodeId,SocketAddr>, new_incarnation: u64) -> Result<usize, ReplayError>` — the deadlock-free lock-scoped + count-anchored-fence drain (`crates/io-prod/src/outbox.rs:364-424`).
- `pub fn open_node_outbox(env) -> Result<Option<NodeOutbox>, …>` (`crates/bins/src/lib.rs:270-292`), `VD_OUTBOX_PATH`-gated, `check_durable_path`-guarded.
- `pub type SharedOutbox = Arc<Mutex<Box<dyn OutboxSink + Send>>>` (`outbox.rs:35`), `pub trait OutboxSink` (`outbox.rs:161`) with `durability()` (`:198`).
- `pub fn resolve_process_incarnation(env)` (`lib.rs:223`) — precedence: explicit `VD_PROCESS_INCARNATION` → durable `VD_BOOT_STATE_DIR` counter (`BootCounter::increment_on_boot`, `lib.rs:243`) → fail-loud. **The counter mutates on each call** (finding C).
- `pub enum ReplayError { Undecodable, Unroutable{peer}, LaneStuck{peer}, FenceTimeout{submitted,expected} }` (`outbox.rs:293-303`).
- `spawn_mesh(handle, trust, cfg, outbox: Option<SharedOutbox>)` (`mesh.rs:776`); the F3 boot-check `outbox.is_some() && peers.len() > OUTBOX_WRITER_CHANNEL_DEPTH` (`mesh.rs:791`, `OUTBOX_WRITER_CHANNEL_DEPTH=256` `outbox.rs:60`).
- Both bins pass `None` today (`shard.rs:36`, `gateway.rs:35`) ⇒ the outbox is inert, `step_tick` byte-identical.

**KEY FACT (stated as required):** `replay_outbox`'s fence waits on `durability.last_submitted() >= base+N` then `wait_durable_through(base+N)` (`outbox.rs:405-414`). Both are **DURABILITY** (block A: the store's off-tick writer fsyncing the retained row) — **NOT DELIVERY** (block C: the QUIC send reaching the peer). A `send_durable` merely enqueues an `OutFrame` onto the per-peer mpsc (`mesh.rs:1902`); the `peer_writer` block A retains+submits to the shared outbox **before** it ever tries block C's `ensure_connection`. So a peer that is **down at boot** still causes its replayed row to become durable (block A runs regardless of connectivity), the fence passes, gc proceeds, and the R-4a retransmit timer (`mesh.rs:1201`, `retransmit.reset` on `WriteFail::Down` `mesh.rs:1281`) re-drives block C when the peer comes back. **Boot-replay never hangs on a down-at-boot peer.**

---

## 1. Sub-slice 1 — DRY BIN WIRING: `vd_bins::boot_mesh_and_replay`

### 1.1 The exact signature and body (NEW in `crates/bins/src/lib.rs`)

Both bins currently DUPLICATE: `spawn_mesh(&MeshConfig::new(local, bind, peers, cap, resolve_process_incarnation(env)?), None)` (`shard.rs:22-37`, `gateway.rs:21-36`). One shared helper collapses the boot into a single reviewed sequence (HR3 uniformity — no per-kind fork):

```rust
use vd_io_prod::mesh::{MeshConfig, MeshControl, MeshTransport, SharedOutbox, spawn_mesh};
use vd_io_prod::outbox::OutboxSink;
use vd_io_prod::trust::ClusterTrust;
use std::sync::{Arc, Mutex};

/// R-6d3b-2b: THE ONE node boot sequence shared by shard.rs + gateway.rs (HR3: no per-kind fork).
/// Resolves the process incarnation ONCE (finding C — a second resolve double-increments the durable
/// BootCounter, lib.rs:243), opens the durable outbox (Some iff VD_OUTBOX_PATH is set), wraps it as the
/// cloneable SharedOutbox, spawns the mesh WITH the sink so the durable-before-send gate is LIVE, then
/// REPLAYS the retained outbox rows BEFORE the transport is handed to build_app. Returns the transport
/// (for build_app) + the MeshControl handle (kept alive by the caller so the endpoint is not dropped).
///
/// The replay fences on DURABILITY (block A), not DELIVERY (block C), so it does NOT hang on peers that
/// are down at boot — their rows are retained + the R-4a retransmit timer re-drives once they reconnect.
///
/// # Errors
/// A bad incarnation source, an unopenable/non-durable outbox path, a spawn_mesh failure, OR a replay
/// that hits a TRANSIENT wedge (LaneStuck / FenceTimeout). A poison row or a roster-gone peer is
/// QUARANTINED (loud-counted) and the boot PROCEEDS (RC-2a) — never a boot wedged on one bad row.
pub fn boot_mesh_and_replay(
    env: &EnvConfig,
    runtime: &tokio::runtime::Handle,
    trust: &ClusterTrust,
) -> Result<(MeshTransport, MeshControl), Box<dyn std::error::Error>> {
    let local = env.node_id("VD_NODE_ID")?;
    let peers = env.peer_book("VD_PEERS")?;
    // Finding C: resolve ONCE. The value flows to BOTH MeshConfig (frame stamping) AND replay gc.
    let new_incarnation = resolve_process_incarnation(env)?;

    // Open + wrap the durable outbox (None ⇒ inert byte-identical path, dev/test/harness).
    let shared: Option<SharedOutbox> = open_node_outbox(env)?
        .map(|ob| Arc::new(Mutex::new(Box::new(ob) as Box<dyn OutboxSink + Send>)));

    let (mut transport, control) = spawn_mesh(
        runtime,
        trust,
        &MeshConfig::new(local, env.parse("VD_BIND")?, peers.clone(),
                         env.parse("VD_OUTBOUND_CAP")?, new_incarnation),
        shared.clone(),                       // Some ⇒ the peer_writers mirror to the durable sink
    )?;

    // BOOT REPLAY — BEFORE build_app consumes `transport` (app.rs:87). The peer_writers are already live
    // (spawn_mesh spawned them, mesh.rs:866), so the enqueued sends are drained + re-mirrored at
    // new_incarnation; the count fence blocks this thread (NOT a tokio worker) until they are durable.
    if let Some(sh) = shared.as_ref() {
        let replayed = vd_io_prod::outbox::replay_outbox(sh, &mut transport, &peers, new_incarnation)?;
        if replayed > 0 {
            tracing::info!(replayed, "boot replay: re-drove retained durable outbox rows");
        }
    }
    Ok((transport, control))
}
```

### 1.2 How the two bins call it (HR3 — byte-identical)

`shard.rs:22-52` (the `spawn_mesh(...) ... build_app(...)` block) and `gateway.rs:21-43` collapse to:

```rust
// shard.rs
let (transport, control) = vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
// GW-1 §6.3 snapshot-budget boot assert stays where it is (shard.rs:40-45).
let mut node = build_app(NodeConfig { node_id: local, kind: NodeKind::StubShard }, transport);
```

```rust
// gateway.rs
let (transport, control) = vd_bins::boot_mesh_and_replay(&env, runtime.handle(), &trust)?;
let mut node = build_app(NodeConfig { node_id: local, kind: NodeKind::Gateway }, transport);
```

**Both must bind `control`** (was `_control`) — dropping `MeshControl` drops `endpoint` (`mesh.rs:709-710` "dropping closes it"). Keep `let _control = control;` living until end of `main` (it already outlives the tick loop today because it is bound in `main`'s scope). The helper returns it so ownership is explicit. `local` is still needed in the bin for `NodeConfig` — the helper re-derives it internally (cheap) and the bin re-reads `VD_NODE_ID` for `build_app`; alternatively return `local` as a third tuple field to avoid the double read (minor — prefer returning `(MeshTransport, MeshControl)` and letting the bin read `VD_NODE_ID` itself since it needs the `NodeKind` there anyway).

**resolve-incarnation-ONCE proof:** `resolve_process_incarnation` is called exactly once inside `boot_mesh_and_replay` (§1.1) and its return flows to both `MeshConfig::new(...new_incarnation)` and `replay_outbox(...new_incarnation)`. Neither `spawn_mesh` nor `replay_outbox` re-resolves. The bins no longer call it at all. This closes finding C: the durable `BootCounter` (`lib.rs:243`, mutating) advances by exactly +1 per boot, not +2.

### 1.3 The peers/book is available for the replay call

`replay_outbox` needs `&BTreeMap<NodeId, SocketAddr>` for the RC-2a roster-diff membership check (§2). The helper reads `env.peer_book("VD_PEERS")` ONCE into `peers`, `.clone()`s it into `MeshConfig::new` (which takes it by value, `mesh.rs:181`), and passes `&peers` to `replay_outbox`. One parse, no drift.

### 1.4 Why the replay does not hang on a down-at-boot peer (the proof)

The `replay_outbox` fence (`outbox.rs:405-414`) is `while durability.last_submitted() < base+N { sleep; }` then `wait_durable_through(base+N)`. `last_submitted` is bumped by `submit_nonblocking` inside the `peer_writer` **block A** (`write_frame` retain → `submit_barrier`), which runs the instant the writer dequeues the `OutFrame` — **before** block C's `ensure_connection` (`mesh.rs:1765+`). Block A has zero dependence on the peer being reachable (it is a pure store write on the shared outbox). So for a down peer, block A still submits, `last_submitted` still reaches `base+N`, the fence passes. Block C then hits `WriteFail::Down`, tears the connection, arms the R-4a retransmit timer (`mesh.rs:1280-1283`) — the frame stays in `lane.retry` (RAM) AND the durable row survives (gc only ran on the strictly-lower prior incarnation). When the peer reconnects, the retransmit timer replays it. **The only ways the fence fails are `FenceTimeout` (a peer_writer never SCHEDULED to run block A — a runtime pathology, not a down peer) — which fail-loud and skip gc (safe).** A down peer is NOT a fence failure.

---

## 2. Sub-slice 2 — RC-2a: quarantine disposition (fold into `replay_outbox`)

### 2.1 The problem (grounded)

Today `replay_outbox` does `return Err(ReplayError::Unroutable{peer})` on the FIRST roster-gone peer (`outbox.rs:387`) and `.ok_or(ReplayError::Undecodable)?` on the FIRST corrupt row (`outbox.rs:389`). In §1.1 the bin does `replay_outbox(...)?` ⇒ `main` returns `Err` ⇒ **the node fails to boot on one poison/absent-peer row**. That violates the `OUTBOX_FORMAT_VERSION` "quarantine" promise (`outbox.rs:45-47`).

### 2.2 The disposition (BINDING)

Partition the four `ReplayError` arms into **PERMANENT (quarantine + count + PROCEED)** vs **TRANSIENT (refuse to boot, fail loud)**:

| Arm | Nature | Disposition |
|---|---|---|
| `Undecodable` | genuine on-disk corruption (a version-mismatched row is ALREADY skipped by `scan_all` via `decode_value`, `outbox.rs:150-155` returning `None` ⇒ the row never enters the scan — so this is *real* frame corruption, rare) | **QUARANTINE + count + PROCEED** |
| `Unroutable{peer}` | a roster diff — the peer legitimately left `VD_PEERS` between the crash and the restart | **QUARANTINE + count + PROCEED** |
| `LaneStuck{peer}` | a live lane wedged full past `REPLAY_SEND_MAX_RETRIES` (`outbox.rs:333`) — a transient drain wedge | **REFUSE TO BOOT (fail loud)** |
| `FenceTimeout{..}` | a peer_writer never submitted the fresh row within `REPLAY_FENCE_DEADLINE` (`outbox.rs:407`) — a runtime pathology | **REFUSE TO BOOT (fail loud)** |

### 2.3 Quarantine SEMANTICS: gc-sweep (chosen) vs a quarantine keyspace

**DECISION: a poison/absent row is SWEPT by `gc_below` — with a LOUD count — NOT moved to a forensic keyspace.** Rationale, weighed:

- **The gc interaction is the decider.** `gc_below(new_incarnation)` sweeps every row with `incarnation < new_incarnation` (`outbox.rs:254-266`). A poison row's incarnation is definitionally `< new_incarnation` (it is a prior-incarnation retained row). If we quarantine-in-place (leave it) but still gc, `gc_below` sweeps it anyway — so an "in-place quarantine keyspace" would need `gc_below` taught to skip a quarantine tag, adding a second key family + a gc special-case. If we quarantine-in-place and DON'T gc, the poison row is re-scanned + re-quarantined **every boot forever** (an unbounded poison accumulation). Neither is clean.
- **A poison row is unrecoverable by definition.** `Undecodable` = the payload cannot be re-framed (it will fail identically every boot). `Unroutable` = there is no lane to send it on (and if the peer returns to the roster, its NEW incarnation's fresh traffic supersedes — the stale prior-incarnation row is dead weight the receiver would dedup anyway). Preserving bytes we can never act on buys nothing operationally.
- **The "quarantine" promise (`outbox.rs:47`) is honored by the LOUD COUNT + LOG, not by byte-hoarding.** The comment says a mismatched row is "quarantined rather than mis-framed" — the load-bearing guarantee is *never silently mis-decode/mis-route*, which the count+skip delivers. A forensic keyspace is a future nicety, not a correctness requirement; note it in DEFERRED.md as an optional R-6d4 enhancement.

**No-silent-loss guarantee:** a quarantined row is `warn!`-logged with its `(peer, class, incarnation, seq)` and tallied into a returned `ReplayCounts` (§2.4). It is NEVER dropped without a count. It IS gc-swept (with the rest of the prior window) — but only AFTER being counted + logged, and only because it is unrecoverable. The forensic trail is the log line + the count, not the redb row.

### 2.4 The `replay_outbox` signature/return change (who decides refuse-vs-proceed?)

**`replay_outbox` decides** (it owns the taxonomy — the bin stays a thin caller, HR3). Change the return from `Result<usize, ReplayError>` to `Result<ReplayCounts, ReplayError>`:

```rust
/// R-6d3b-2b RC-2a: the outcome of a boot replay — separates the rows re-driven from the poison/absent
/// rows QUARANTINED (loud-counted + swept, never silently lost). A TRANSIENT wedge is still `Err`.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ReplayCounts {
    /// Rows successfully re-sent (block A durable, fence passed).
    pub replayed: usize,
    /// Rows QUARANTINED: undecodable (real corruption) or roster-gone peer. Counted + warn-logged +
    /// swept by gc (unrecoverable). NEVER silently dropped.
    pub quarantined: usize,
}

pub fn replay_outbox(
    shared: &SharedOutbox,
    transport: &mut dyn Transport,
    peers: &BTreeMap<NodeId, SocketAddr>,
    new_incarnation: u64,
) -> Result<ReplayCounts, ReplayError> { … }
```

The send loop (`outbox.rs:385-391`) changes from `?`-bail to quarantine-continue:

```rust
let mut counts = ReplayCounts::default();
for (key, framed) in &rows {
    if !peers.contains_key(&key.peer) {
        tracing::warn!(peer = key.peer.0, class = ?key.class, incarnation = key.incarnation, seq = key.seq,
            "boot replay QUARANTINE: retained row's peer is no longer in VD_PEERS (roster diff) — \
             skipping + sweeping (unrecoverable); the boot PROCEEDS");
        counts.quarantined += 1;
        continue;                                  // RC-2a: was `return Err(Unroutable)`
    }
    let Some(payload) = decode_value_payload(framed) else {
        tracing::warn!(peer = key.peer.0, class = ?key.class, incarnation = key.incarnation, seq = key.seq,
            "boot replay QUARANTINE: retained row is undecodable (on-disk corruption) — \
             skipping + sweeping; the boot PROCEEDS");
        counts.quarantined += 1;
        continue;                                  // RC-2a: was `Undecodable ?`
    };
    send_durable_with_retry(transport, key.peer, key.class, payload)?;  // LaneStuck STILL `?`-bails (transient)
    counts.replayed += 1;
}
```

**The fence N must now be `counts.replayed`, not `rows.len()`** — the count-anchored fence (`outbox.rs:400` `base.checked_add(n)`) waits for exactly the rows that were actually SENT. A quarantined row never enqueues a send ⇒ never bumps `last_submitted` ⇒ must not be in the fence target, else `FenceTimeout` on a boot that quarantined a row (a false wedge). Change:

```rust
let n = counts.replayed as u64;
if n == 0 {
    // Every row quarantined (or none present after the roster check): still gc the swept window.
    { let mut g = shared.lock()…; g.gc_below(new_incarnation); g.commit(); }
    return Ok(counts);
}
let target = base.checked_add(n).expect(…);   // fence over the SENT rows only
// … existing fence + wait_durable_through(target) …
{ let mut g = shared.lock()…; g.gc_below(new_incarnation); g.commit(); }
Ok(counts)
```

**gc interaction restated:** `gc_below(new_incarnation)` runs (as today) STRICTLY after the fence, and sweeps ALL prior-incarnation rows — the `replayed` ones (now safely re-driven at `new_incarnation`, so their old copy is redundant) AND the `quarantined` ones (unrecoverable). This is HIGH-3-safe: a crash between the fence and gc leaves both copies (the fence proved the fresh rows durable; the old rows not-yet-swept) ⇒ next boot re-drives + dedups. A crash DURING gc is redb-atomic (`commit` is all-or-nothing).

### 2.5 The bin folds refuse-vs-proceed via `?` on the `Err` arms only

In `boot_mesh_and_replay` (§1.1), `replay_outbox(...)?` still `?`-propagates `LaneStuck`/`FenceTimeout` (the transient wedge ⇒ main returns Err ⇒ refuse to boot, loud). The `Ok(ReplayCounts{quarantined, replayed})` path logs a summary and PROCEEDS. So the bin change is: `replay_outbox` returns `Ok` for the quarantine case (it already handled it internally), and the surviving `Err` arms are exactly the two transient ones. No bin-side match on the taxonomy — the taxonomy lives in `replay_outbox`.

---

## 3. Sub-slice 3 — RC-2b: `Closed` disambiguation

### 3.1 The problem (grounded)

`MeshTransport::send_durable` (`mesh.rs:1913-1916`) folds both mpsc `try_send` errors into one:
```rust
Err(TrySendError::Full(frame) | TrySendError::Closed(frame)) => Err(SendError::QueueFull(frame.bytes)),
```
`Closed` fires only when the `peer_writer`'s `Receiver` (`w.rx`, `mesh.rs:1163`) is dropped — i.e. the writer TASK DIED (the lane is permanently dead). `Full` is a transient backlog. `send_durable_with_retry` (`outbox.rs:317-334`) treats every `QueueFull` as transient: it retries `REPLAY_SEND_MAX_RETRIES=10_000 × REPLAY_POLL_BACKOFF=1ms = ~10s` (`outbox.rs:284,287,324,328`) before `LaneStuck`. On a dead lane (a corpse) it spins the full ~10s before failing — and worse, `LaneStuck` is a REFUSE-TO-BOOT arm (§2.2), so a dead lane wedges the boot for 10s then refuses, when it should surface as a fast, clean unreachable.

### 3.2 The mechanism (keeps `sim::io` FROZEN)

**CHOSEN: option (a) — `send_durable` emits an `Inbound::NodeUnreachable` bounce on `Closed` + returns a distinguishable-at-the-io-prod-layer signal, and `send_durable_with_retry` treats a dead lane as immediately fatal.** But the frozen `SendError` (`sim/src/io/mod.rs:124-128`) has only `QueueFull(Bytes)`, and adding a variant to a FROZEN enum is out of bounds. So the split lives ENTIRELY inside io-prod, not on the seam:

**Concretely:** `MeshTransport` already holds `inbox: SharedInbox` (`mesh.rs:704`) and `push_inbox` is `pub(crate)` (`mesh.rs:267`). On `Closed`, `send_durable`:
1. pushes `Inbound::NodeUnreachable { to, class, undelivered: msg_id }` (the existing dead-peer signal — a dead writer task IS the peer being unreachable from this node's send path; `mesh.rs:1463` shows the exact shape). This is FIFO-correlated by `msg_id` per the seam contract (`sim/src/io/mod.rs:143-147`) — and `send_durable` assigns `msg_id` at `mesh.rs:1901`, so bump `next_msg_id` before pushing so the correlation id is consumed exactly as a successful send would (the seam contract requires the id to be monotone whether or not the send lands).
2. still returns `Err(SendError::QueueFull(frame.bytes))` (the seam is unchanged — every existing `send_durable` caller keeps working; a `QueueFull` return is a legal "not accepted, payload returned" for any reason).

Then `send_durable_with_retry` must NOT spin 10s on a dead lane. Since `send_durable` cannot tell the retry helper "this was Closed, not Full" through the frozen `SendError`, we make the **retry helper liveness-aware via a tiny io-prod-internal Transport-side accessor that does NOT touch the frozen trait**:

**The minimal additive, DRY change: a `MeshTransport`-inherent method `pub(crate) fn lane_alive(&self, peer: NodeId) -> bool`** that returns `!self.lanes.get(&peer).map_or(true, |l| l.tx.is_closed())` (tokio `mpsc::Sender::is_closed()` is O(1) and returns true once the receiver is dropped). `send_durable_with_retry` takes `transport: &mut dyn Transport` today (`outbox.rs:318`) — but the FROZEN `Transport` trait has no `lane_alive`. Two clean options that keep the seam frozen:

- **(a-final, CHOSEN) drive the retry helper off the `NodeUnreachable` bounce the transport ALREADY pushes.** `send_durable_with_retry` cannot drain the inbox (that is `drain_inbound`, which the sim thread owns). So instead: on the FIRST `QueueFull`, the helper does one probe — it re-attempts once after a single `REPLAY_POLL_BACKOFF`; a truly-dead lane returns `QueueFull` again immediately (a `Closed` mpsc never drains), a merely-Full lane may accept. This does NOT distinguish reliably. **REJECTED as insufficient** — a slow-draining Full lane looks identical.

- **(b-final, CHOSEN) observe the peer_writer `JoinHandle`s.** `spawn_mesh` currently DROPS the handles (`mesh.rs:866` `handle.spawn(peer_writer(...))` — the returned `JoinHandle` is discarded). Collect them into `MeshControl` (`writers: BTreeMap<NodeId, JoinHandle<()>>`) and expose `MeshControl::lane_dead(peer) -> bool` via `handle.is_finished()`. But `send_durable_with_retry` holds `&mut dyn Transport`, not `MeshControl`. Threading `MeshControl` into the replay path is heavy.

**DECISION — the clean, DRY, seam-frozen mechanism: give `MeshTransport` an inherent (not trait) `is_lane_closed` check and have `replay_outbox` call `send_durable_with_retry` with a liveness probe that is passed as a closure OR — simpler — move the liveness check INTO `MeshTransport::send_durable` itself so the retry never sees a dead lane as retryable.** Since `send_durable` already detects `Closed` (it is the `TrySendError::Closed` arm), the cleanest fix is: **`send_durable` on `Closed` returns `QueueFull` AND `send_durable_with_retry` bounds its retry by asking the transport, through a NEW io-prod-internal trait, whether to keep retrying.**

Because threading through the frozen `Transport` is the sticking point, the **decisive minimal design** is: **define a tiny io-prod-internal trait `ReplayTransport: Transport { fn lane_alive(&self, peer: NodeId) -> bool; }`, implement it for `MeshTransport` (`l.tx.is_closed()`), and change `replay_outbox` + `send_durable_with_retry` to take `&mut dyn ReplayTransport` instead of `&mut dyn Transport`.** This:
- keeps the FROZEN `sim::io::Transport` trait 100% untouched (`ReplayTransport` is a NEW io-prod-local supertrait, not a modification);
- keeps `send_durable`'s frozen return (`QueueFull`) untouched;
- lets `send_durable_with_retry` check `if !transport.lane_alive(peer) { return Err(LaneDead{peer}); }` on the FIRST `QueueFull`, failing a corpse in ~1 poll instead of ~10s;
- the test `FlakyTransport` (`outbox.rs:633`) implements the one extra method trivially (`lane_alive → true`), and the real bin passes `&mut transport` (a `MeshTransport`, which now implements `ReplayTransport`).

Add a `ReplayError::LaneDead { peer }` arm (distinct from `LaneStuck`) and classify it as a **QUARANTINE/PROCEED** disposition? **No** — a dead lane at boot means the mesh could not stand up a writer for a peer in the roster, which is a real infrastructure fault; classify `LaneDead` as **REFUSE-TO-BOOT (loud)** alongside `LaneStuck`/`FenceTimeout`, because unlike a roster-gone peer (RC-2a `Unroutable`, where the peer legitimately left the book), a `LaneDead` peer is STILL in the book — its writer died, a genuine pathology. The win over the status quo is speed + honesty: fast fail with a NodeUnreachable bounce recorded, not a 10s spin.

### 3.3 In 2b or a follow-up?

**IN 2b — but minimally.** The moment 2b flips a bin off `None`, `send_durable` on a durable path becomes live, and `send_durable_with_retry` runs against a real `MeshTransport`. The 10s-spin-on-a-corpse is a real degradation the moment the outbox is wired. However it is NOT a *silent* correctness break (it eventually fails loud via `LaneStuck`), so it could technically be a fast-follow. **DECISION: fold the seam-frozen `ReplayTransport::lane_alive` + `LaneDead` arm into 2b** because (a) it is small (one supertrait, one method, one enum arm, one `send_durable` `Closed`→bounce line), (b) it directly improves the boot-refusal latency the RC-2a quarantine work also touches (same function), and (c) shipping the outbox live with a known 10s-corpse-spin violates the no-shortcuts standard. The `NodeUnreachable`-bounce-on-`Closed` half is the smallest defensible change; if the `ReplayTransport` supertrait proves to ripple, the fallback is to ship RC-2b's bounce-only half in 2b and the `lane_alive` fast-fail as R-6d3b-2c — but the supertrait is contained, so ship it whole.

---

## 4. Tests (all keep the gates green)

### 4.1 Bins boot-replay e2e (`crates/bins/tests/outbox_boot_replay.rs`, NEW — process tier, real QUIC)

Model on `boot_counter_crashloop.rs` (the proven SIGKILL+restart+`/metrics`-pairing pattern). Two-node topology (a producer shard A whose outbox is pre-seeded + a receiver peer B):
- Set `VD_OUTBOX_PATH` (under a FIXED temp dir that survives the restart) + `VD_OUTBOX_EPHEMERAL_OK=1` (temp path escape, mirroring the boot-counter test's `VD_BOOT_STATE_EPHEMERAL_OK`) on shard A's env.
- **Pre-seed**: before spawning A, open a `NodeOutbox` on that path directly (like `outbox.rs:801-818` `retained_frames_survive_a_reopen`), `retain` + `commit` one `Retained` Saga row keyed to peer B at incarnation N-1, drop it (writer flushes+joins).
- Spawn B (a receiver counting delivered Saga frames — reuse the `ProcessClient`/parity harness shape or a minimal admin counter), then spawn A. A's `boot_mesh_and_replay` opens the pre-seeded outbox, `replay_outbox` re-drives the row at A's fresh incarnation.
- **Assert**: B receives the payload (the row was re-driven across the restart) AND A's outbox `scan_all()` (reopen after the run, or via an admin probe) has NO `incarnation < N` rows (gc swept the prior window). The test COMPLETING within the deadline is the no-deadlock proof (the R-6d3b-2a `replay_outbox_redrives...` real-QUIC test already established this shape).

This is the direct proof "a shard re-drives a pre-seeded outbox on restart" — the D-6 #1 RESTART-half closure.

### 4.2 Poison-row boot-progress test (`crates/io-prod/tests/` or the outbox unit tests — default gate)

Extend the existing `replay_outbox_bails_on_an_unroutable_row_without_gc` / `..._undecodable_..._without_gc` tests (`outbox.rs:697-755`), which currently assert `Err` + row-survives. RC-2a INVERTS their expectation:
- `replay_outbox_quarantines_an_unroutable_row_and_proceeds`: seed one row for peer 9 (NOT in `peers`) AND one routable row for peer 2 at the same prior incarnation; `replay_outbox` returns `Ok(ReplayCounts{replayed:1, quarantined:1})`, peer 2's payload WAS sent (`FlakyTransport.sent` has it), and after the run `scan_all()` is empty (gc swept both — the quarantined AND the replayed prior-incarnation rows).
- `replay_outbox_quarantines_an_undecodable_row_and_proceeds`: same shape with a garbage value.
- Keep a `replay_outbox_refuses_to_boot_on_a_wedged_lane` (LaneStuck/LaneDead) asserting `Err` + gc SKIPPED (the transient-wedge refuse path). These update the two existing 2a tests rather than adding net-new files, so the outbox.rs coverage stays ratcheted (Tier-B io-prod floor 90, currently 98.29% outbox.rs per DEFERRED.md — the new quarantine branches are all covered by these deterministic in-process tests).

### 4.3 RC-2b dead-lane test (`crates/io-prod/tests/` — default gate)

- `send_durable_bounces_node_unreachable_on_a_dead_lane` (mesh unit/integration): construct a `MeshTransport` (or a two-node mesh), drop/abort the peer_writer for a peer so its `lane.tx.is_closed()`, call `send_durable` to that peer, assert (i) it returns `Err(QueueFull)` AND (ii) `drain_inbound()` yields one `Inbound::NodeUnreachable{to, class, undelivered}` with the correct FIFO `msg_id`.
- `send_durable_with_retry_fails_fast_on_a_dead_lane`: a `FlakyTransport` variant whose `lane_alive → false` and `send_durable → QueueFull`; assert `send_durable_with_retry` returns `Err(LaneDead{peer})` after ~1 poll (assert the elapsed wall-clock is far below `REPLAY_SEND_MAX_RETRIES × REPLAY_POLL_BACKOFF` ≈ 10s — e.g. `< 200ms`), proving no corpse-spin. This exercises the new `lane_alive` branch (coverage) deterministically.

### 4.4 Gate posture
All new bins tests are process-tier (Tier-B, not the coverage gate). All new io-prod tests are deterministic in-process (no real-clock flake) and cover every new branch (quarantine-continue arms, the `n==0` all-quarantined path, the `lane_alive` fast-fail, the `Closed`→bounce line), so Tier-B region floor 90 stays satisfied and ratchets up. `just gate` (fmt + clippy -D + tests + coverage) stays green; no frozen-seam file changes ⇒ no wire/parity churn.

---

## 5. Precise files/functions touched per sub-slice

### Sub-slice A (RC-2a — outbox-internal, INDEPENDENTLY mergeable, lands first)
- `crates/io-prod/src/outbox.rs`: NEW `pub struct ReplayCounts{replayed, quarantined}`; `replay_outbox` (`:364`) return → `Result<ReplayCounts, ReplayError>`, the send loop (`:385-391`) → quarantine-continue with `warn!`+count, fence N → `counts.replayed`, the `n==0` gc-only early path; update the two abort tests (`:697-755`) to quarantine-and-proceed + add the refuse-on-wedge test. `crates/io-prod/src/lib.rs`: re-export `ReplayCounts` if `ReplayError` is re-exported.

### Sub-slice B (RC-2b — mesh-internal, INDEPENDENTLY mergeable)
- `crates/io-prod/src/mesh.rs`: `send_durable` (`:1913-1916`) — split the `Full | Closed` arm; on `Closed` push `Inbound::NodeUnreachable{to, class, undelivered: msg_id}` via `push_inbox(&self.inbox, …)` + bump `next_msg_id`, then return `QueueFull`. NEW `pub(crate) trait ReplayTransport: Transport { fn lane_alive(&self, peer: NodeId) -> bool; }` + `impl ReplayTransport for MeshTransport` (`l.tx.is_closed()`).
- `crates/io-prod/src/outbox.rs`: `send_durable_with_retry` (`:317`) + `replay_outbox` (`:364`) params `&mut dyn Transport` → `&mut dyn ReplayTransport`; on first `QueueFull` check `!transport.lane_alive(peer)` → `ReplayError::LaneDead{peer}`; NEW `ReplayError::LaneDead{peer}` arm; `FlakyTransport` (`:633`) impls `lane_alive`. RC-2b dead-lane tests.

### Sub-slice C (DRY bin wiring — flips the gate LIVE, lands LAST, depends on A+B)
- `crates/bins/src/lib.rs`: NEW `pub fn boot_mesh_and_replay(env, runtime_handle, trust) -> Result<(MeshTransport, MeshControl), …>` (§1.1); imports `spawn_mesh`/`MeshConfig`/`MeshControl`/`MeshTransport`/`SharedOutbox`/`OutboxSink`/`replay_outbox`. `open_node_outbox` (`:270`) + `resolve_process_incarnation` (`:223`) REUSED as-is.
- `crates/bins/src/bin/shard.rs`: replace `:22-52` with the helper call; bind `control`; keep the GW-1 snapshot assert (`:40-45`).
- `crates/bins/src/bin/gateway.rs`: replace `:21-43` with the helper call; bind `control`.
- `crates/bins/tests/outbox_boot_replay.rs`: NEW (§4.1). Optionally set `VD_OUTBOX_PATH` in `shard_env`/dev-cluster env for the smoke path (a follow-up; the e2e test sets it directly).

---

## 6. resolve-incarnation-ONCE + the CA-1/k3d caveat

- **resolve-ONCE (finding C):** `boot_mesh_and_replay` calls `resolve_process_incarnation(env)` exactly once and threads the value to both `MeshConfig::new` and `replay_outbox`. The bins no longer call it. The durable `BootCounter` (`lib.rs:243`, mutating `increment_on_boot`) advances +1 per boot. Covered by extending `resolve_uses_the_durable_counter_and_increments` (`lib.rs:734`) with an assertion that a boot through `boot_mesh_and_replay` bumps the counter by exactly one (a bins test with a temp `VD_BOOT_STATE_DIR` + `VD_OUTBOX_PATH`).

- **CA-1 / k3d caveat (NOTED, NOT SOLVED):** the mesh uses a static literal-IP peer book (`MeshConfig.peers`, `mesh.rs:64` "P1: from config"). Two k3d pods cannot address each other until CA-1 (reply-on-connection / dynamic peer addressing) lands (DEFERRED.md `:1643-1645` "k3d CLOUD test DE-SCOPED"). So R-6d3b-2b's boot-replay e2e (§4.1) runs on a LOOPBACK restart (like `boot_counter_crashloop.rs`), NOT a real pod-network reschedule. The real-cloud CrashLoop/reschedule proof is a separate CA-1 + L5 (`peer_writer` address re-read on IP move, DEFERRED.md `:1662-1664`) + M3 slice. **This is a roadmap item, not a will-break for 2b** — loopback proves the durable-outbox restart-replay mechanism end-to-end; k3d only adds pod addressing on top.

---

## 7. API sufficiency + minimal additive changes (decisive)

1. **`replay_outbox` return is insufficient for RC-2a** (it hard-errors on the first poison/absent row). Minimal additive: `ReplayCounts` return + quarantine-continue (§2.4). No new key family, no gc special-case (poison rows are gc-swept, §2.3).
2. **`SendError` (frozen) cannot carry the `Full`/`Closed` split** — do NOT touch it. Minimal additive: `send_durable` pushes `NodeUnreachable` on `Closed` (using the existing inbox + `push_inbox`); a NEW io-prod-internal `ReplayTransport: Transport` supertrait carries `lane_alive`; `ReplayError::LaneDead` arm. Frozen seam untouched (§3.2).
3. **`MeshControl` need not grow JoinHandles** — `lane.tx.is_closed()` is sufficient to detect a dead writer (a dropped `Receiver` closes the sender). Rejected the JoinHandle-observation option (b-final) as heavier for no gain.
4. **No new magic numbers** — `REPLAY_SEND_MAX_RETRIES`/`REPLAY_POLL_BACKOFF`/`REPLAY_FENCE_DEADLINE` (`outbox.rs:284-290`) and `OUTBOX_WRITER_CHANNEL_DEPTH` (`outbox.rs:60`) are the ONE homes, reused. The RC-2b fast-fail spends the existing `REPLAY_POLL_BACKOFF` for its single probe.

## 8. HR posture
- **HR3:** ONE `boot_mesh_and_replay`, ONE `open_node_outbox`, ONE `replay_outbox`, ONE `SharedOutbox` — shard + gateway wire byte-identically, no shard-kind match.
- **Frozen seam:** `sim::io::Transport`/`SendError`/`Inbound` and `vd-wire` UNCHANGED. `ReplayTransport`/`ReplayCounts`/`LaneDead`/`boot_mesh_and_replay` are io-prod/bins-internal.
- **No-silent-loss:** every quarantined row is warn-logged + counted; every transient wedge fails loud + skips gc (old rows survive); a down-at-boot peer's row is durable + retransmit-re-driven.