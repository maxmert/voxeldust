# Transfer Protocol & Authority Model — Voxeldust Greenfield (REVISED / FINAL, hardened)

This document is self-contained. The old repo (`ecs-system`, present as a full copy in the worktree) is referenced only to (a) name failure modes R1–R10 the design structurally prevents and (b) extract proven pure functions. Every code path described is NEW.

Confirmed against the actual worktree code while hardening:
- `shard-common/src/heartbeat_sender.rs` heartbeat is `Duration::from_secs(2)` fire-and-forget `send_to` UDP with no ack — REPLACED here.
- `shard-common/src/client_listener.rs` mints `SessionToken(rand_u64())` / `SessionToken(42)`, never validated — REPLACED (sec 4.5).
- `shard-common/src/quic_transport.rs` is `open_uni`/`accept_uni` only (no bidi, no datagrams) — the replication transport in sec 1.4 is NEW code, not an extension.
- `core/src/system.rs::compute_planet_position` calls hand-rolled `solve_kepler` (Newton-Raphson) + `brahe::coordinates::cartesian::state_koe_to_eci` (iterative) — the determinism trap (sec 6) is real.
- `core/src/system.rs::compute_soi_radius` (planet Hill sphere, mass ratio) and `core/src/galaxy.rs::system_soi_radius` (`BASE_SOI_RADIUS + luminosity*SOI_LUMINOSITY_SCALE`) are DIFFERENT formulas — band table now distinguishes them (sec 1.3).
- `core/src/handoff.rs::PlayerHandoff` has 12+ mutually-exclusive `Option` fields (R4) — replaced by typed per-class messages (sec 5).
- All shard crates are bin-only (`main.rs`, no `lib.rs`) — migration plan sec 0.6 fixes this first.

---

## 0. Three-Plane Separation (foundation, P2)

```
                         +--------------------------------------------+
                         |            ORCHESTRATOR (control)          |
                         |  +----------------+  +------------------+  |
                         |  | Ownership Dir   |  | Transfer         |  |
                         |  | (versioned +    |  | Coordinator      |  |
                         |  |  LEASE-EPOCH    |  | (per-key-serial  |  |
                         |  |  fenced, redb)  |  |  saga, redb WAL)  |  |
                         |  +----------------+  +------------------+  |
                         |  Ephemeris Authority (publishes body poses)|
                         |  Universe Epoch (persisted, monotonic)     |
                         |  Auth service (Ed25519) + Revocation feed   |
                         +---+---------------------------+------------+
                             | control: ONE reliable QUIC stream/peer  |
        +--------------------+-----------+      +--------+-------------------+
        | SHARD A (sim authority plane)  |<====>| SHARD B (sim authority)    |
        |  bevy_ecs + rapier3d           | ghost|  bevy_ecs + rapier3d        |
        |  Authority{Owned|Ghost|Frozen} | DGRAM|  Authority{...}             |
        +---------------+----------------+ +ctrl +--------------+-------------+
                        | snapshot frames (version-fenced)      |
                        +-------------->  GATEWAY  <------------+
                          (connection plane; validates lease-epoch
                           on every frame; ONE QUIC conn / client)
                                            |
                                         CLIENT
```

- **Connection plane = Gateway.** Holds the single client QUIC connection for the whole session (P1). The client NEVER re-homes (kills R2). A transfer is a silent server-side route swap. Gateway crash handling: sec 4.6 (reconnect-to-session, not reconnect-to-shard).
- **Authority plane = shards.** `bevy_ecs` + `rapier3d`. Holds `Owned` and `Ghost` entities. Snapshot frames it sends to the gateway are tagged with the lease-epoch they were produced under (sec 4.3); the gateway drops stale-epoch frames (the fence).
- **Persistence plane = redb.** Distinct stores: orchestrator (directory + saga WAL + epoch + auth), per-shard WAL (transfer-leg + checkpoints), per-region/per-ship block-delta store. Off-tick world-snapshot writes use the proven drain-queue idiom (never fsync on the 20Hz tick). **Exception (sec 3.4):** transfer-leg WAL and saga WAL use a synchronous group-commit point, NOT the lossy drain queue.

---

## 1. Entity Authority Model

### 1.1 Authority states

```rust
enum Authority {
    Owned { lease_epoch: u64 },                  // this shard simulates; writes frames tagged lease_epoch
    Ghost { source: ShardId, since_tick: u64 },  // read-only kinematic mirror; collides, never integrates
    Frozen { transfer_id: TransferId },          // mid-transfer; no input, no integration; fenced at gateway
}
```

Invariant (enforced by directory + the ordered hand-off of sec 2.1): the set of `Owned` placements is a cluster-wide partition at every committed instant — never two owners, and (outside an in-flight saga) never zero.

### 1.2 EntityId — globally unique, stable, NOT time-derived

`EntityId(u128) = {kind:u8, mint_shard:u32, seq:u64, rand:u24}`. Stable for the entity's whole life across every shard. Never reused, never derived from `SystemTime` (the R7 root cause). Bound to (but not equal to) the cryptographic session identity (sec 4.5). Three distinct ids throughout: `SessionTicket` (identity/authz), `EntityId` (the moved thing), `TransferId` (saga correlation).

### 1.3 The overlap band — geometry-driven hysteresis, velocity-scaled, per-class radius

A ghost exists iff the entity is inside the overlap band. Bands are geometric (not timeouts → kills R8) but **velocity-scaled** so a fast body cannot skip the band in one tick.

| Class | Radius source (named, tested) | Inner edge (ghost CREATE) | Outer edge (ghost DESTROY) |
|---|---|---|---|
| Planet SOI (descent) | `core::geometry::planet_soi(planet,star)` = old `compute_soi_radius` (Hill sphere, mass ratio) | `R_soi*1.15` | `R_soi*1.30` |
| System SOI (warp/interstellar) | `core::geometry::system_soi(star)` = old `system_soi_radius` (`BASE + lum*scale`) — DISTINCT formula | `R_sys*0.95` | `R_sys*1.05` |
| Ship hull proximity | `hull_aabb` Minkowski | `+ max(3m, v_rel*dt*K_safety)` | `+ 2×inner_pad` |
| Sector edge (planet face seam) | `band_blocks` (seed-derived) | within `band_blocks` | beyond `band_blocks*2` |
| System↔Galaxy warp | event-gated (warp charge engaged ∧ heading→star) | event | warp aborted/arrived |

`R_soi` uses the Hill formula `sma * (m_planet/m_star).powf(0.4)`, evaluated ONCE in `core::geometry` (kills R6 duplication). **Both SOI functions are exposed as separately named, doc-commented, tested functions** with a CI test asserting they differ by the expected order of magnitude (a future refactor cannot collapse them — closes the un-merged-SOI finding). `WarpDeparture` messages carry the SYSTEM radius; `SoiEntry` carries the PLANET radius (sec 5).

**Velocity-scaled band-width invariant (closes the despawn/PREPARE race + thin-3m-band finding):** at band-membership evaluation, `(outer_edge - inner_edge) >= v_rel * tick_dt * K_safety` (K_safety ≥ 2). The hull band's inner pad is `max(3m, v_rel*dt*K)`, so above ~60 m/s relative velocity the band widens automatically. Band-crossing detection sub-samples the entity's swept segment within the tick (segment-vs-shell intersection), so a body moving 5.5 km/tick cannot tunnel the band undetected.

**Hysteresis + dwell:** ghost CREATE uses inner edge; DESTROY uses outer edge; transfer COMMIT fires only when the entity crosses the commit line (geometric midpoint) **with inward radial velocity sustained for `N_entry` ticks** (closes the tangential-graze ping-pong). After any commit, a **minimum-dwell cooldown of `K_dwell` ticks** suppresses re-triggering regardless of geometry (guardrail, not primary mechanism — does not reintroduce R8). Metric `transfer_pingpong` alerts on opposite-direction transfer of the same entity within `K_dwell`.

### 1.4 Ghost lifecycle — transport: datagrams + one reliable control stream (closes HOL-blocking + cross-stream-ordering)

The old `quic_transport.rs` is `accept_uni` only; this is NEW transport. Per ordered peer pair:
- **GhostDelta** (20Hz, lossy-tolerant): QUIC **DATAGRAMs**, multiplexed on the existing shard↔shard connection. No per-pair stream lifecycle (closes O(shards²) stream-sprawl + churn from ship-shards coming/going). Each delta carries `{entity, source_lease_epoch, tick, seq}`.
- **GhostSpawn / GhostDespawn** (rare, must-deliver): the single **reliable control stream** to the orchestrator-mediated peer, acked.

**Cross-channel ordering is defined by data, not by stream guarantees (kills the R2 "ordering enforced only by comments" bug):**
- A `GhostDelta` whose `tick < spawn.since_tick` is ignored.
- A `GhostDespawn` **fences** the entity: every later `GhostDelta` for that entity is dropped until a new `GhostSpawn` with a higher `since_tick`.
- **A `GhostDespawn` is REFUSED while the directory shows `in_transfer = Some` for that entity** (the directory field, previously decorative, is now enforced — closes the despawn-races-PREPARE attack). Ghost lifecycle and saga state for a given entity are serialized by the per-key lock in the Coordinator (sec 2.0).

```
crosses INNER edge      -> owner: GhostSpawn(EntityId, snapshot) [reliable, acked]
                           -> neighbor inserts Ghost + kinematic collider, consumes deltas
inside band             -> owner: GhostDelta each tick [datagram]; lost deltas tolerated (next supersedes)
crosses OUTER edge      -> if NOT in_transfer: GhostDespawn [reliable, acked]; else deferred until saga ends
owner shard dies        -> lease lapses -> orchestrator-driven recovery (sec 3). Ghosts NEVER removed on a
                           missed-heartbeat timer (the R8 mistake); only on explicit despawn or directory event.
```

Ghosts carry: pose (f64 in the ghost's frame, transformed by the source per sec 6), lin+ang velocity, orientation, a small replicated blob (health/shield/anim/pilot-flags), `source`, `source_lease_epoch`, `last_applied_tick`. Ghosts carry NO authoritative game-logic state (no inventory mutation, no block edits) — pure render/collision mirrors.

---

## 2. Transfer State Machine

### 2.0 Per-key serialization (foundation for several fixes)

The Coordinator holds a **per-EntityId (and per-ShipId) lock**. At most one saga per key is in flight. Ghost despawn, band re-evaluation, and saga transitions for that key are all serialized behind this lock. This single mechanism closes: despawn-vs-PREPARE race, double-saga, and the re-parent ordering (sec 8).

### 2.1 Happy path — ORDERED Demote-before-Promote (kills split-brain double-ownership, the #1 fatal)

The original "dest becomes Owned; version++; source→Ghost" treated three messages to three processes as atomic. They are not. Hand-off is now strictly serialized and lease-epoch-fenced:

```
IDLE --(commit-line crossed inward N_entry ticks, fresh ghost on dest, dwell ok)--> PREPARE
   PREPARE: write saga PREPARE (WAL, write-ahead); dest verifies ghost fresh + spatial precond;
            dest begins buffering input. Dest does NOT become Owned.
PREPARE --(dest PrepareAck{ready, spatial_ok})--> FLUSH
   FLUSH: source freezes entity (Authority::Frozen); computes DEST pose authoritatively (sec 6);
          sends FlushComplete{dest_pose, final_auth_state, input_backlog, last_tick}.
FLUSH --(source FlushComplete)--> FENCE_DEMOTE
   FENCE_DEMOTE: Coordinator writes OwnerRecord.version = N+1, owner = dest, lease_epoch = E+1 to redb (linearizable).
                 Sends Demote(N+1) to SOURCE; WAITS for source DemoteAck.
                 On DemoteAck: source has flipped Owned->Ghost and STOPPED emitting Owned frames
                 (its old lease_epoch E frames are now dropped at the gateway — the fence).
FENCE_DEMOTE --(source DemoteAck)--> PROMOTE
   PROMOTE: Coordinator sends Promote(N+1, lease_epoch=E+1) to DEST.
            Dest PULLS the directory head, confirms version==N+1 (pull-through, not push-trust),
            applies final_auth_state idempotently (sec 3.4), becomes Owned{lease_epoch=E+1}.
PROMOTE --(dest CommitAck)--> ROUTE_SWAP
   ROUTE_SWAP: Coordinator -> gateway RouteSwap{client, dest, lease_epoch=E+1}.
               Gateway atomically swaps the feeding shard; from now ONLY E+1 frames pass.
ROUTE_SWAP --(gateway RouteSwapped)--> CLEANUP
   CLEANUP: saga Committed. Source ghost persists until entity leaves overlap band.
```

At no instant are two shards emitting accepted frames for the entity: between `Demote` and `Promote` the entity is `Frozen` on source (fenced) and not-yet-`Owned` on dest; the gateway, still routing to source, sees source emit nothing for it and the client interpolates across the (sub-second) gap. Cost: one extra RTT (Demote→DemoteAck before Promote). This is the deliberate price of a true partition.

For warp, IDLE diverts to AWAIT_PROVISION before PREPARE (sec 7).

### 2.2 Transitions, triggers, timeouts

| From | Event | Action | Timeout | On timeout |
|---|---|---|---|---|
| IDLE | commit-line sustained inward, dest ghost fresh, dwell ok | WAL PREPARE; send PrepareTransfer | — | — |
| IDLE | commit-line, dest absent (warp) | AWAIT_PROVISION | T_prov=30s | ABORT→source |
| PREPARE | PrepareAck{ready, spatial_ok=true} | →FLUSH; freeze source | T_prep=2s | ABORT (dest unresponsive) |
| PREPARE | PrepareAck{spatial_ok=false} | →ABORT_SPATIAL; typed TransferRejected{reason} | — | — |
| FLUSH | FlushComplete{dest_pose, final_state, backlog, last_tick} | →FENCE_DEMOTE | T_flush=1s | **FENCE-THEN-FORCE (sec 2.4)** |
| FENCE_DEMOTE | source DemoteAck | →PROMOTE | T_demote=1s | source unreachable: bump+fence already done, proceed to PROMOTE (source frames already dropped at gateway) |
| PROMOTE | dest CommitAck | →ROUTE_SWAP | T_promote=2s | retry idempotently ×3 → ALERT, hold (entity safe Frozen-on-dest, fenced) |
| ROUTE_SWAP | gateway RouteSwapped | →CLEANUP | T_swap=2s | retry; gateway rebuilds from directory on restart |
| CLEANUP | source DemoteAck already had; GC | saga Committed | T_clean=5s | sweeper finishes later |

### 2.3 ABORT — player always alive, authoritative, consistent

| Abort from | Cause | Action | Player ends up |
|---|---|---|---|
| AWAIT_PROVISION | provision timeout/failure | source keeps Owned; **child-saga compensation de-provisions (sec 7)** | Source; warp visual reverses |
| PREPARE | dest unresponsive (T_prep) | source keeps Owned; dest ghost untouched | Source; no state left source |
| PREPARE | spatial_ok=false (hull trap) | typed TransferRejected{reason} | Source, with feedback ("hatch obstructed") |
| FENCE_DEMOTE/PROMOTE | dest crashed after fence | directory already head=N+1 owner=dest; recovery (sec 3) promotes dest from its WAL/ghost; if dest truly gone, **roll forward** only if dest WAL shows accepted, else roll back version to source (un-fence) | Dest if it accepted; else Source |
| any pre-FENCE | source crashed | recovery: dest has fresh ghost; if FlushComplete never arrived, see 2.4 | per 2.4 |

### 2.4 Bounded forced resolution — FENCE-THEN-FORCE (closes the "force-commit doesn't actually freeze source" major)

The original force-committed on a 1s timeout that cannot distinguish "source crashed" from "FlushComplete delayed/dropped" — a live-but-slow source could keep integrating (R1/R2 reborn). Fixed:

1. **The fence comes first, always.** On T_flush, the Coordinator does NOT immediately promote dest. It first writes `version=N+1, lease_epoch=E+1` to the directory and pushes the new lease-epoch to the gateway. **From this instant the gateway drops every frame tagged with source's old lease_epoch E.** Even a live, partitioned, still-integrating source can no longer affect the client. The freeze is enforced by the fence, not by cooperative in-memory state.
2. **Crash vs slow is distinguished by the lease/heartbeat (sec 4.3), not by T_flush.** If the source's lease is still live, the Coordinator waits the full `T_prep`-class budget for a late FlushComplete before forcing; if the lease has lapsed (true crash), it forces immediately using the freshest ghost delta.
3. **Input backlog is bounded and acknowledged, never silently dropped (closes the "1s of input vanishes" sub-finding).** The buffered input on source has a bounded ring (size `= ceil(T_flush * 20Hz)`). On forced resolution the Coordinator forwards whatever backlog FlushComplete delivered; if none arrived, it sets `input_gap=true` in the COMMIT and the client receives a typed `InputGap{from_tick,to_tick}` so interpolation does not present stale authoritative state as live. The gap is bounded (`< T_flush`) and surfaced, not hidden.

This removes the "<1s of fine input" hand-wave: the gap is fenced, bounded, and reported.

---

## 3. Saga Durability & Crash Recovery

### 3.1 Where state persists (single owner of the lifecycle — kills R1)

- **Orchestrator saga WAL (redb):** authoritative saga record per TransferId — state, idempotency key, participants, lease_epoch, lossy/input_gap flags, correlation_id, `universe_epoch_id`. Written **before** the corresponding message is sent (write-ahead). Single owner (no state smeared across 4 maps).
- **Source & dest shard WAL (transfer_leg, redb):** each shard durably records its leg ("froze E for T at tick K, lease_epoch E", "accepted E for T at version N+1") **before applying any world-mutating effect** (sec 3.4).
- **Idempotency key = (TransferId, version).** Every message carries it; every handler is idempotent. Re-applying PrepareTransfer/Promote is a no-op.

### 3.2 Tiered crash-recovery (closes the "27-cell matrix is over-engineered for day one" major)

Recovery is tiered by frequency, so the solo dev hardens the paths that actually fire:

- **Tier 1 (day one, exercised constantly):** happy path; FLUSH FENCE-THEN-FORCE; PREPARE spatial-abort.
- **Tier 2 (Milestone 3):** source/dest crash recovery via WAL re-drive (the table below).
- **Tier 3 (later; mitigated operationally):** orchestrator/gateway crash. **Initial guarantee: on orchestrator restart, ABORT all in-flight sagas back to last-known directory owner** (player snaps to source — always safe per sec 2.3) rather than resume mid-saga. Resume-mid-saga is a later optimization, not a foundation. Orchestrator is a fast-restart singleton; directory rebuilds from the saga WAL + shard heartbeats.

### 3.3 Tier-2 recovery matrix (component × state)

| State \ Crash | Source crash | Dest crash | Gateway crash | Orchestrator crash |
|---|---|---|---|---|
| PREPARE | source committed nothing → ABORT to source; entity still source-owned in directory | dest restart re-receives PrepareTransfer (idempotent), rebuilds ghost, re-ACKs | route unchanged (no transfer state yet) | Tier-3: ABORT to source |
| FLUSH | KEY: source froze then died. Lease lapsed → FENCE-THEN-FORCE: bump+fence, promote dest ghost→Owned (input_gap). Player lands on dest | dest restart: rebuild ghost, re-ACK; re-send flush forward | route unchanged | Tier-3: ABORT to source |
| FENCE_DEMOTE | fence already durable (version N+1 in redb); proceed to PROMOTE; source frames already dropped at gateway | dest restart re-PULLs head; if WAL shows accepted, re-ACK; else Coordinator rolls version back to source | route unchanged | Tier-3: directory shows N+1 head → roll forward to dest if dest WAL accepted, else N |
| PROMOTE | source already demoted/fenced; harmless | dest re-applies Promote idempotent via WAL ("accepted E for T?") | route unchanged | Tier-3: re-drive ensure route correct |
| ROUTE_SWAP | nothing critical; ghost GC'd | dest already Owned; harmless | **gateway restart reloads route table from directory; client reconnects via sec 4.6 handshake** | Tier-3: finish bookkeeping |
| AWAIT_PROVISION | source still Owned; abort path | n/a | unchanged | Tier-3: ABORT + de-provision child saga |

**Universe epoch durability + cross-store consistency (closes the dev-loop footgun):** epoch persisted in orchestrator redb at first init, NEVER regenerated from `SystemTime` (kills R7's epoch-reset desync). **Every persisted record (saga legs, transfer_legs, block-deltas) is tagged with `universe_epoch_id` + `epoch_schema_version`.** On recovery, a leg whose epoch_id mismatches the current epoch is **discarded/aborted, not resumed** (fail-safe — no entity placed at a stale celestial position). The join handshake asserts epoch_id + schema match or the shard refuses to start. `dev-cluster.sh` gets an atomic "wipe ALL state stores together" command so partial wipes are impossible by tooling.

### 3.4 Write-ahead on EVERY participant + idempotent effects (closes the double-apply-side-effects major)

The original specified write-ahead only for the orchestrator. Now mandatory on dest:
- Dest must **durably journal `accepted E for T at version N+1` (synchronous commit point) BEFORE applying any world-mutating side effect** (spawning the entity with inventory, applying in-flight block edits, health).
- Side-effect application is itself **idempotent keyed on (TransferId, version)**: block-edit deltas are tagged with the producing TransferId and deduplicated on apply; inventory transfer is an idempotent set-to-final-state, not an additive op.
- **The transfer-leg WAL does NOT use the lossy off-tick drain queue** (which lags many ticks and widens the double-apply window). It uses a synchronous commit point. World-snapshot persistence keeps the drain queue; transfer legs do not.

### 3.5 Saga fsync under load — group commit (closes the fsync-convoy major)

The original's per-transition write-ahead = 4–6 synchronous fsyncs per transfer through redb's single writer. Under "everyone lands at the same zone" load this convoys and blows T_prep/T_flush, causing spurious forced commits. Fixed:
- **Group-commit:** saga transitions occurring within one orchestrator tick are batched into ONE redb transaction (fsync amortized across all concurrent transfers, like a DB WAL group commit). Messages for that tick send after the one shared fsync.
- **Lease renewals never touch redb** (sec 4.3): leases are soft state rebuildable from heartbeats; only lease ASSIGNMENTS are persisted, not renewals.
- **Separate redb file for saga WAL vs directory** so they don't contend on one writer lock.
- A synthetic "everyone lands at once" load test measures fsync p99 before the 1–2s timeouts are trusted; timeouts are config, not magic numbers.

---

## 4. Ownership Directory

Single authoritative answer to "who owns E / S / R" (P5). Orchestrator-hosted, redb-backed, with a replicated read-cache on gateway/shards used **only as a hint** — authority decisions pull-through (sec 4.2).

### 4.1 Record shape

```
OwnerRecord {
  key: Entity(EntityId) | Region(RegionId) | Ship(ShipId),
  owner: ShardId,
  version: u64,                 // ++ on every authority change
  lease_epoch: u64,             // ++ on every owner change; the FENCE token (sec 4.3)
  lease_deadline: Tick,         // soft; renewed by heartbeat (not persisted per-renewal)
  effective_tick: u64,
  in_transfer: Option<TransferId>,  // ENFORCED: blocks GhostDespawn + concurrent saga (sec 1.4, 2.0)
}
```

### 4.2 Consistency model — fence, not cached hint (closes the #1 fatal)

Writes are linearizable (single redb writer). The replicated cache is a **hint only**:
- **Gateway validates `lease_epoch` on every snapshot frame.** A frame tagged with a lease_epoch < the directory head for that key is **dropped**. This makes the version bump a true fence: a stale or partitioned owner physically cannot affect the client after the bump, regardless of cache-push reordering.
- **Authority decisions are pull-through.** Dest, on Promote, PULLS the directory head and confirms `version==N+1` before becoming Owned (never trusts a pushed cache value). Source demotes on explicit `Demote(N+1)` and DemoteAck, ordered BEFORE Promote (sec 2.1).
- Monotonic per-key version on the cache still holds (a reader never acts on v-1 for a key it has seen at v), but the SAFETY no longer rests on cross-reader cache ordering — it rests on the gateway fence + ordered Demote-before-Promote.

### 4.3 Leases — owner-fault vs control-plane-fault distinction (closes the correlated-failure SPOF major)

Each shard holds a lease per owned key, renewed by a **reliable, acked control-stream RTT probe** (NOT the old 2s fire-and-forget UDP heartbeat — that mechanism is deleted). Renewal measures real RTT and clock skew (sec 6.2).

**Critical distinction (closes "orchestrator blip orphans every player"):** a lapsed lease is interpreted as ownership loss **only when the orchestrator confirms the owner is unreachable**, not when the orchestrator itself is unreachable. A separate liveness signal (the shard↔shard mesh and the gateway's view of which shards are sending frames) lets the system tell "owner failed to renew" (owner fault → recovery) from "control plane unavailable" (orchestrator fault → **freeze recovery, do not mass-orphan**). During an orchestrator outage, existing routes keep working off the gateway's last-known route table; no new transfers, but no mass eviction. "A connection is never evicted by a timer" no longer silently inverts into "every connection dies."

### 4.4 How gateway & shards consume it

- **Gateway:** materializes `route_table (ClientId/EntityId → (ShardId, lease_epoch))`. On ROUTE_SWAP it atomically swaps the feeding shard and the accepted lease_epoch; thereafter only ≥E+1 frames pass. Client sees nothing (P1). On restart, route table rebuilt from directory.
- **Shards:** resolve forwarding targets from the cached directory (hint) but make authority moves pull-through. A shard NEVER spawns a hoped-for entity (kills R3) — it only promotes an existing ghost on a directory-confirmed version bump.

### 4.5 Cryptographic session identity + refresh + revocation (fixes R7 fully)

`SessionTicket` = Ed25519-signed `{player_id, entity_id, issued_epoch, expiry, nonce, conn_binding}` issued by the orchestrator auth service at login. NOT `hash(SystemTime)`; NOT reused as correlation key (kills the R7 triple-overload).

Closing the freshness/revocation/replay sub-findings:
- **Refresh:** the gateway transparently renews the ticket from the auth service before `expiry` and pushes the new ticket to shards. **Transfers never fail on expiry** — there is no time-bomb disconnect. Shards re-verify on every transfer against the current (refreshed) ticket.
- **Revocation:** the orchestrator pushes `revoked player_ids` to gateway + shards over the reliable control stream (bans/compromise take effect mid-session).
- **Replay binding:** `conn_binding` = hash of the QUIC connection id; a ticket replayed on a different connection is rejected.
- **Transport:** mandate **mTLS on all inter-process links** so tickets cannot be sniffed off a shard↔gateway hop.

### 4.6 Gateway crash → reconnect-to-session, not reconnect-to-shard (closes the "gateway is sole conn holder" major)

QUIC state is not transferable across a process crash, and the client deliberately cannot re-home to a shard (that was R2). Resolution:
- The client has exactly ONE behavior on connection loss: **reconnect to the gateway service endpoint (a stable VIP / k8s Service), re-presenting its signed SessionTicket.** This is reconnect-to-identity, not the old observer/promotion re-homing.
- The new (or restarted) gateway verifies the ticket, looks up the server-side session by `player_id`, rebuilds the route table from the directory, and re-binds the fresh QUIC connection to the existing session. No shard-level state moves; the player's entity never left its shard.
- This is the ONE sanctioned reconnect path and is explicitly NOT client-side shard selection.

**Directory availability:** for dozens-of-players dev scale, the single-writer orchestrator is an accepted SPOF with a **fast-restart SLO (<2s to ready, rebuilding directory from saga WAL + heartbeats)** and "no new transfers during restart" documented as a known limitation (existing routes survive per 4.3). The thousands-scale path (range-partition the directory by RegionId, one coordinator per region) is flagged as a **known scaling seam** — and sec 8's atomic N+1-key re-parent is explicitly noted as the cross-key transaction that would block partitioning, so it is kept within a single ship's region.

---

## 5. Typed Per-Transition Messages (kills the R4 god-struct)

Replaces `core/src/handoff.rs::PlayerHandoff` (12+ Option fields). Shared envelope, one payload per class:

```
TransferEnvelope<P> {
  transfer_id: TransferId,
  universe_epoch: EpochId,    // mismatch => reject (sec 3.3)
  schema_version: u16,        // postcard control-plane version
  version: u64,               // directory version this message targets (idempotency)
  entity: EntityId,
  payload: P,                 // exactly one typed variant
}

enum TransferClass {
  ShipToPlanetDisembark, PlanetToShipBoard, EvaExit, EvaBoard,
  SoiEntry, SoiExit, WarpDeparture, WarpArrival, ShipSystemToSystem,
}
```

Each payload carries ONLY its own fields: `PlanetToShipBoard{target_ship, hatch_local}`, `SoiEntry{planet_index, soi_radius /* planet_soi */}`, `WarpDeparture{target_star_index, system_soi /* system_soi, NOT planet */}`. The compiler forbids a boarding message carrying warp velocity. Control plane serializes with **postcard** (stable, versioned). High-frequency GhostDelta/snapshot use **bitcode** (same-version, compact) over datagrams. FlatBuffers not adopted.

---

## 6. Coordinate-Frame Transfer — source-authoritative, NOT cross-binary recomputed (kills R6 + the determinism fatal)

ONE tested function in shared `core`, but the authority model changed to kill the brahe-Kepler determinism trap.

```
StampedPose {
  frame: PlanetCentered(seed) | ShipLocal(ship) | SystemSpace(seed) | GalaxySpace,
  pos: DVec3, vel: DVec3, orient: DQuat,
  epoch_tick: u64,            // tick against the SHARED persisted Universe Epoch
}
fn transfer_frame(src: &StampedPose, dst_frame: FrameId, dst_tick: u64, ctx: &FrameContext) -> StampedPose;
```

### 6.1 Source computes, dest uses (the determinism fix)

Iterative/transcendental math (`solve_kepler` Newton-Raphson + `state_koe_to_eci`) is NOT bit-reproducible across heterogeneous binaries (LLVM version, FMA contraction, build profile, brahe minor bump during rolling deploy). So:

- **The SOURCE shard computes the destination `StampedPose` via `transfer_frame` and ships it in `FlushComplete.dest_pose`. The DEST shard USES that pose; it does NOT independently recompute the transform for authority.** This eliminates the cross-binary determinism dependency for the authoritative placement entirely.
- **Ephemeris Authority:** celestial body positions used by `FrameContext` are computed by a **single source** — the orchestrator's Ephemeris Authority publishes planet/star poses at each tick; shards CONSUME them (single-writer fact, not a recompute-and-hope determinism assumption). This is the primary mechanism; per-shard local computation is only a fallback for ghost RENDERING (where a sub-meter discrepancy is cosmetic, not an authority bug).
- Spatial-precondition checks at PREPARE compare against the published ephemeris (same numbers on both sides), so `spatial_ok` cannot spuriously disagree from float drift.

### 6.2 Clock — monotonic-disciplined tick, never raw wall-clock floor (closes the clock-guardrail major)

The old `epoch_tick = floor((wall_now - epoch_ms) * 20)` is non-monotonic (chrony slew / leap-second step moves `wall_now` backward → `epoch_tick` goes backward → breaks the `u64` monotonic assumption + interpolation). Fixed:

- Each shard derives `epoch_tick` from a **local monotonic counter (`Instant`-based), periodically disciplined toward the orchestrator's tick via bounded slew, NEVER stepped backward.** Monotonicity is guaranteed structurally.
- **Skew measured by the reliable acked control-stream RTT probe** (sec 4.3), not 2s fire-and-forget UDP — no stale-2s-window TOCTOU.
- **"Degraded" is a lease attribute the Coordinator reads at the MOMENT of PREPARE (read-through), not a 2s-old measurement** — closes the TOCTOU where a just-stepped shard is still a valid destination.
- In-flight transfers are skew-immune anyway: `transfer_frame` time-advances analytically to the discrete `dst_tick` both shards agree on (zero skew in the transform itself); skew affects only WHEN a shard reaches tick K, not WHERE the entity sits at K.
- **Abort backpressure:** when a popular destination is excluded, transfers to it queue with bounded backpressure rather than thundering-herd aborting (closes the abort-storm cascade). Hosts run NTP as a guardrail, not the mechanism.

### 6.3 f64 precision budget

- Galaxy ~1e16 m: f64 → ~0.5 m abs. Mitigation: galaxy frame is integer light-year cells + f64 offset within a cell; positions stored relative to nearest-star anchor.
- System ~1e11 m: f64 → ~0.01 mm. Fine.
- Ship-local / planet-surface (≤200k-block radius): f64 trivially exact.
- **Cross-binary determinism CI gate:** `core` compiled into a tiny standalone binary, run on two target-cpu/profile builds, asserts bitwise-equal `transfer_frame` outputs on seed+tick fixtures. **FMA contraction forbidden in `frames.rs`** (compiler flag). glam/brahe versions pinned + asserted identical across all shard images. (Even though sec 6.1 removes the authority dependency on this, the gate protects ghost-render quality and catches drift early.)

---

## 7. Provisioning-Aware Transfers (Warp) — child saga with compensation (closes the orphan-shard major)

```
IDLE -> (warp engaged, heading->star resolves) -> AWAIT_PROVISION
   Coordinator opens a PROVISION CHILD SAGA keyed by provision_request_id, tied to TransferId:
     POST /provision {system_seed, provision_request_id, transfer_id}
     - shard exists -> skip to PREPARE
     - in flight -> stream ProvisionProgress{pct,eta} -> gateway -> client warp visual STRETCHES to eta
     bounded by T_prov=30s
   on provisioned + registered + lease active -> PREPARE (dest exists; ghost streamed in before flip — P3)
   on T_prov / provisioner error -> ABORT + run child-saga COMPENSATION
```

**Compensation (kills the half-born zombie shard):**
- Aborting AWAIT_PROVISION enqueues a durable `de-provision(provision_request_id)`.
- **The dest shard's self-registration is GATED:** before activating its lease, a freshly-provisioned pod asks the Coordinator "am I still wanted for saga T / provision_request_id P?" A pod that finds its saga aborted **self-terminates** and never registers. So a pod that wins the schedule race AFTER abort cannot become a zombie owner.
- Provisioning is tied to TransferId, so a stale `transfer_leg` from a first aborted attempt is discarded on epoch/transfer mismatch (sec 3.3) — the second warp's new TransferId cannot mis-correlate with it.
- Idle-GC sweeps any `provisioned-but-unclaimed` shard.

Provisioning failure can never strand the player (kills R1): they were never removed from source.

---

## 8. Ship Transfers — Compound Case (versioned bundle, closes the re-parent ordering major)

A ship is its own shard (interior = private rapier3d world). What transfers is the ship's EXTERIOR authority (who owns the hull rigid body in the host frame), via the standard saga on the `Ship(ShipId)` key. The interior world is owned by the ship-shard throughout and untouched — passengers feel zero discontinuity ("walk inside while flying").

### 8.1 Atomic re-parent, consistently observed

Passengers are `ChildOf(ShipId)` sub-entities whose authoritative frame is `ShipLocal(ship)`. Their OwnerRecords are SLAVED to the ship's: when exterior authority commits to a new host, the directory bumps the ship key AND re-parents all `ChildOf(ShipId)` records in ONE redb transaction. One saga, one commit, N+1 keys, serialized by the ship's per-key lock (sec 2.0). **This atomic write is kept within a single galaxy region** so it never becomes the cross-key transaction that blocks future directory partitioning (sec 4.6 seam).

**External observers see no teleport (closes the major):** the original streamed ship-exterior pose and child poses as independent deltas that could reorder, making a passenger jump relative to the hull for one tick. Fixed: **ship + children stream as ONE versioned bundle.** Every ghost delta is tagged with the directory `version` it was computed against; an observer composes a passenger's render pose through the ship's exterior pose **only using a single consistent version** for (ship_exterior_pose, child_anchor) — the newer of the two is buffered until both are at the same version. No "new ship pose + stale anchor" tick is ever rendered.

### 8.2 Sub-cases
- **Warp (system A→B):** exterior host changes A→B; `transfer_frame` (source-authoritative, sec 6.1) re-expresses the ship's single exterior pose SystemSpace(A)→GalaxySpace→SystemSpace(B). Passenger ShipLocal poses are frame-invariant (don't move relative to hull) — only the one exterior pose transforms. Atomic bundle.
- **Surface proximity (system→planet):** exterior authority hands system-shard→planet-shard; `transfer_frame` SystemSpace→PlanetCentered. Same atomic bundle.

### 8.3 Passenger ghosts during ship transfer
Outside observers see ship + passengers as one versioned ghost bundle before/during/after — no visibility pop, because the ghost set is maintained continuously by the overlap band (sec 1.3) independent of the authority flip, and the bundle versioning (8.1) prevents intra-tick inconsistency.

---

## 9. Cross-Shard Visibility — same machinery, observable

Two players on different shards see each other via the SAME ghost/datagram stream that powers transfers. No separate 1Hz digest (the lossy R8 system is deleted).

- When A (surface) is in the planet↔system overlap band, the planet shard publishes A as a ghost to the system shard, and vice-versa.
- Each observer keeps a per-observer interest set (P6) with add/remove diffs as ghosts enter/leave the band; **ghost count is capped per observer** (datagram budget metric). The gateway forwards (owned entities on the client's shard) + (relevant ghosts) as one coherent snapshot stream; the client never knows some entities are cross-shard.
- Interpolation: all entities (owned + ghost) interpolate against the shared epoch timeline by `epoch_tick`; cross-shard entities don't blink. NO client-side prediction anywhere — pure server-authoritative interpolation (binding standard).
- **Observability on the steady-state path (closes the "ghost debugging is grep again" minor):** per-ghost structured state queryable on each shard — `ghost{entity, source, last_applied_tick, stream_seq, age_ms}`; metrics `ghost_staleness_seconds` histogram, `ghost_spawn/despawn_total`, `replication_datagrams_dropped`, `ghost_count{shard}`. A dev-only gateway command "dump all ghosts visible to client X with source shard + staleness" makes a misbehaving ghost diagnosable from one query.

---

## 10. Observability (kills R10)

- **Correlation ID = TransferId** in every `TransferEnvelope` across gateway/source/dest/orchestrator. Tracing span `transfer{id, class, src, dst, entity}` opened at IDLE→PREPARE, closed at CLEANUP/ABORT, propagated as span context across QUIC control messages.
- **Saga WAL IS the post-mortem log:** every transition is an immutable appended event `{transfer_id, from_state, to_state, epoch_tick, reason, lossy, input_gap, lease_epoch}`. "Show transfer T's full timeline across all 4 processes" is one redb scan.
- **Ghost deltas carry `(shard_pair, seq)`** so a wedged replication path is diagnosable without a transfer in flight (closes the "deltas have no correlation id" minor).
- **Metrics (Prometheus):** `transfers_total{class, outcome}`, `transfer_duration_seconds{class, state}`, `transfers_in_flight`, `ghost_count{shard}`, `ghost_staleness_seconds`, `abort_total{cause}`, `forced_commit_total`, `input_gap_total`, `transfer_pingpong_total`, `clock_skew_ms{shard}`, `directory_version_lag`, `saga_fsync_p99`. Alert on `forced_commit_total` rate, `transfer_pingpong`, and any `transfers_in_flight` exceeding `T_promote*3` (a wedged saga — the thing that must never silently happen).

---

## 11. Testability (kills R5) + cross-binary harness

- All shard logic in **library crates** (`*-core` lib + thin bin wrapper) so behavioral tests can `use` them. Closes the bin-only-untestable reality (all current shard crates have only `main.rs`).
- Tick loop reads time from an injectable `Clock` trait (real `Instant`-disciplined in prod, `VirtualClock` in tests). No hard 50ms/`Instant::now()` in gameplay code. Saga timeouts driven by the virtual clock so every PREPARE→…→CLEANUP and ABORT/recovery path is a deterministic unit test with fault injection (drop FlushComplete, kill source between freeze and DemoteAck, partition source and assert the gateway fence drops its frames).
- **Single-binary virtual-clock tests CANNOT catch cross-binary float drift or live packet-loss stutter.** Two additional harnesses are mandated: (a) the cross-binary determinism CI gate (sec 6.3); (b) a **two-real-shards integration harness that injects packet loss on the ghost datagram path** (where stutter bugs live and unit tests structurally can't see them).

---

## 0.5 Strangler-fig Migration (closes the "greenfield is fiction" fatal)

The worktree is a full copy of the old 365-file repo, NOT empty. Demolition order, each step compile-green:

1. **Start a genuinely empty cargo workspace** with ONLY: `core` (lib), one `stub-shard` (lib+bin), `gateway`, `orchestrator`. Delete `voxydust-next/shard_transition.rs` first (its existence means the client still re-homes — hard gate) and the `PlayerHandoff` god-struct.
2. **Port proven PURE functions with their tests first:** `compute_soi_radius`, `system_soi_radius` (kept distinct), `celestial_time_from_epoch`, greedy meshing. These already have unit tests.
3. **Single-shard, no-transfer, connect+walk loop GREEN** (gateway owns one connection, dev-static SessionTicket, in-memory directory) before ANY authority/ghost/saga code exists.
4. Sequence transfer machinery as Milestone 2+. Explicit "delete these files in this order" checklist; `shard_transition.rs` deletion is the gate proving the client no longer re-homes.

## 0.6 Tiered Milestones (closes the "monolith / no walking skeleton" fatal)

- **Milestone 1 — Vertical slice MVP:** ONE transition class (`SoiEntry`), same datacenter, both shards always alive, no warp/provisioning, no crypto (dev-static ticket), in-memory directory, datagram ghosts + one reliable control stream, NO crash recovery. Ordered Demote-before-Promote + gateway lease-epoch fence ARE in from day one (they're cheap and are the core safety property). First green committing transfer under a virtual-clock test — a real feedback loop, not rebuild+fly+grep.
- **Milestone 2 — Durability:** redb directory + group-commit saga WAL; per-shard transfer-leg WAL with write-ahead; Tier-1 recovery (FENCE-THEN-FORCE, spatial abort).
- **Milestone 3 — Tier-2 recovery + leases:** source/dest crash WAL re-drive; reliable-RTT leases; clock discipline.
- **Milestone 4 — Crypto identity:** Ed25519 auth service, ticket refresh, revocation, mTLS, conn-binding.
- **Milestone 5 — Warp + provisioning child saga.**
- **Milestone 6 — Ship compound (versioned bundle re-parent).**
- Tier-3 orchestrator/gateway crash recovery uses abort-to-source throughout until/unless resume-mid-saga is justified.

---

## Attack resolutions (each fatal/major/minor, how resolved)

**[FATAL] Directory read-cache split-brain double-ownership in COMMIT window** → Sec 2.1 ordered Demote-before-Promote (wait DemoteAck before Promote) + sec 4.2 gateway validates `lease_epoch` on every frame (true fence, not cached hint) + dest pull-through on Promote. Two shards never both emit accepted frames. Cost: one RTT (accepted trade-off).

**[FATAL] brahe iterative Kepler not byte-identical → transfer_frame determinism false** → Sec 6.1: source computes dest pose authoritatively and ships it; dest USES it (no independent recompute for authority). Ephemeris Authority publishes body poses as a single-writer fact; shards consume. Cross-binary determinism downgraded to ghost-render cosmetics + a CI gate (sec 6.3). Authority no longer depends on float reproducibility.

**[MAJOR] FORCED COMMIT on T_flush re-introduces R1 (source not actually frozen; input silently dropped)** → Sec 2.4 FENCE-THEN-FORCE: fence first (gateway drops old-lease_epoch frames so even a live-slow source can't affect the client); distinguish crash vs slow via lease/heartbeat not T_flush; bound + acknowledge input backlog via typed `InputGap`.

**[MAJOR] Ghost despawn at OUTER edge races a started PREPARE; cross-stream ordering** → Sec 1.4 datagrams + one reliable stream with version-fenced ordering (despawn fences later deltas; delta < spawn.since_tick ignored); GhostDespawn REFUSED while `in_transfer=Some` (directory field enforced); sec 2.0 per-key serialization; sec 1.3 velocity-scaled band + swept-segment crossing detection (no tunneling); 3m hull band auto-widens with relative velocity.

**[MAJOR] Ship atomic re-parent observed via reordered independent pushes → passenger teleports for external observers** → Sec 8.1 ship+children stream as ONE versioned bundle; observer composes child-through-ship using a single consistent directory version; newer delta buffered until both match.

**[MAJOR] Clock guardrail uses SystemTime::now() (R7 survives); TOCTOU; self-eviction cascade** → Sec 6.2 monotonic-disciplined tick (never steps backward); skew via reliable acked RTT probe; "degraded" read-through at PREPARE moment (no TOCTOU); abort backpressure (no thundering herd).

**[MAJOR] AWAIT_PROVISION abort leaks half-born shard that later claims ownership** → Sec 7 provision child saga with durable de-provision compensation; dest self-registration gated on "am I still wanted for saga T?"; tied to TransferId; idle-GC for unclaimed pods.

**[MAJOR] Gateway is sole conn holder; orchestrator blip orphans every player** → Sec 4.6 reconnect-to-session handshake (re-present signed ticket to stable gateway VIP; re-bind to existing session, NOT re-home to shard); sec 4.3 owner-fault vs control-plane-fault distinction (freeze recovery during orchestrator outage; existing routes survive); fast-restart SLO; directory-partition flagged as future seam.

**[MAJOR] Idempotency keyed on TransferId doesn't make EFFECTS idempotent → double-applied block edits/inventory** → Sec 3.4 write-ahead on dest (durable "accepted E for T" before any world mutation, synchronous commit point, NOT the lossy drain queue); effects idempotent keyed on (TransferId, version); block deltas dedup-tagged; inventory is set-to-final not additive.

**[MINOR] SessionTicket expiry/replay/revocation gaps** → Sec 4.5 transparent gateway refresh before expiry (no time-bomb); orchestrator revocation feed; conn-binding rejects replay on a different connection; mTLS on all inter-process links.

**[MINOR] Replication stream couples reliable+unreliable; HOL-blocking stalls visibility** → Sec 1.4 datagrams for GhostDelta + separate reliable stream for spawn/despawn; explicit version/tick ordering between them; bounded self-healing loss like the digest it replaces but authoritative.

**[FATAL] "Greenfield" is fiction; day-one is demolition** → Sec 0.5 strangler-fig migration with explicit delete order, compile-green checkpoints, `shard_transition.rs`-deletion gate.

**[FATAL] Full machinery front-loaded as a monolith; no walking skeleton** → Sec 0.6 tiered milestones; Milestone 1 vertical slice (one class, in-memory dir, dev-static identity, single bidi-equivalent, no recovery) green under virtual-clock test before anything else.

**[MAJOR] redb saga write-ahead fsync convoy under load** → Sec 3.5 per-tick group-commit; lease renewals out of redb; separate saga vs directory redb files; synthetic-load fsync p99 measured before trusting timeouts.

**[MAJOR] Per-tick GhostDelta over per-shard-pair streams: allocation churn + O(shards²) stream sprawl** → Sec 1.4 datagrams multiplexed on the existing connection (no per-pair stream lifecycle); sec 11 reused encode buffers (no per-tick alloc); sec 9 ghost cap + per-pair metrics; (shard_pair, seq) tag for diagnosability.

**[MAJOR] 27-cell recovery matrix over-engineered; rare paths bit-rot** → Sec 3.2 tiering: Tier-1 day one, Tier-2 Milestone 3, Tier-3 abort-to-source (simpler, equally safe) until justified.

**[MAJOR] Geometric hysteresis still thrashes; freezing during FLUSH removes the motion geometry relies on** → Sec 1.3 minimum-dwell cooldown + sustained-inward-crossing entry; `transfer_pingpong` metric/alert; FENCE-THEN-FORCE means a healthy source completes FLUSH sub-tick (frozen <1 tick), only crashed-source hits the 1s timeout.

**[MAJOR] transfer_frame byte-identical determinism untestable in single-binary loop** → Sec 6.1 source-authoritative pose removes the authority dependency; sec 6.3 cross-binary CI gate + pinned glam/brahe + FMA-contraction forbidden; sec 11 two-real-shards harness.

**[MINOR] Single orchestrator SPOF gates all transfers/warp** → Sec 4.6 accepted for dozens-scale with fast-restart SLO + lease-renewals-off-redb; range-partition-by-RegionId flagged as future seam; sec 8 atomic re-parent kept within one region to not block partitioning.

**[MINOR] No observability on steady-state ghost path** → Sec 9 per-ghost structured state + staleness metrics + dev dump command; sec 11 packet-loss integration harness.

**[MINOR] Two un-merged SOI functions; "one shared function" is unstarted with a semantic split** → Sec 1.3 band table splits Planet-SOI (`planet_soi`, Hill/mass) vs System-SOI (`system_soi`, luminosity); both exposed as named tested functions with a CI test asserting order-of-magnitude difference; per-class messages carry the correct radius.

**[MINOR] Durable epoch dev-loop footgun / partial-wipe desync** → Sec 3.3 tag every persisted record with `universe_epoch_id` + `epoch_schema_version`; mismatch → discard/abort (fail-safe); `dev-cluster.sh` atomic "wipe all stores together"; join handshake asserts epoch + schema match.

---

## Explicit trade-offs (genuine tensions, with recommendation)

1. **Ordered Demote-before-Promote adds one RTT to every commit.** Tension: latency vs partition safety. **Recommendation: accept it.** At 20Hz with sub-second control RTT, one extra RTT is invisible to the player (entity is Frozen and interpolated across it), and it is the only way to guarantee a true Owned-partition without a stale owner affecting the client. Double-ownership is unacceptable; latency is not.

2. **Single-writer orchestrator caps concurrency.** Tension: simplicity/correctness now vs thousands-scale. **Recommendation: keep single-writer now**, with the explicit non-negotiable that no cross-key transaction (sec 8 re-parent) ever spans regions, preserving the future range-partition seam. Do NOT build partitioning now.

3. **Source-authoritative pose vs fully symmetric determinism.** Tension: a malicious/buggy source could ship a bad dest pose. **Recommendation: source-authoritative + dest sanity-bounds the received pose against the published ephemeris** (reject poses outside a generous physical band → ABORT_SPATIAL), getting determinism-independence without blindly trusting the source.