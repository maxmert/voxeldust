# Identity, Persistence & World-State Foundations — REVISED FINAL (hardened)

All paths are for the **new empty greenfield workspace**. `OLD:` citations are reference-only and were re-verified against branch `ecs-system` while hardening (e.g. `shard-common/src/quic_transport.rs:382 SkipServerVerification` + `:369 with_no_client_auth`, `core/src/block/chunk_delta.rs:23 HashMap` "order not guaranteed", `gateway/src/main.rs:151 SessionToken(rand_u64())`, `planet-shard/src/main.rs:3292 /tmp/voxeldust-planet-{id}.redb`, `core/src/system.rs:586 mean_anomaly = mean_anomaly_epoch + mean_motion*time_s`).

This document is self-contained. Where an attack changed a decision, the change is stated inline and again in **§A Attack resolutions**.

---

## 0. Crate / module layout

```
crates/
  vd-identity/    accounts, session tickets (HMAC day-1 / PASETO later), key rotation, revocation epoch
  vd-trust/       cluster trust: static PSK cert day-1, CA+SPIFFE later; rustls mTLS configs
  vd-store/       redb table defs, codecs, versioned records, dev-wipe, two-phase WAL checkpoint
  vd-fence/       lease_version + gateway_gen + saga-step fencing types (the linearizability core)
  vd-time/        DurableUniverseClock: write-ahead tick-ceiling, monotonic clamp, analytic clock read
  vd-worldspec/   determinism boundary (Analytic vs Physics), FrameRef, pose re-eval, transition payloads
  vd-saga/        saga state machine as a PURE function (testable; no I/O)
  shard-core/     all shard logic as a lib (lib+thin-bin rule, fixes R5)
services/
  orchestrator/   directory, sagas, universe clock, accounts, admin endpoint
  auth/           SEPARATE small service: Argon2id verify (isolates unbounded CPU from saga timers)
  gateway/        owns the ONE client QUIC connection; ticket validation
  shard/          one bin, ShardKind param; fn main(){ shard_core::run(Config::from_env()) }
```

**R5 structural fix:** every shard is `lib + thin bin`; tests `use shard_core`. The saga (`vd-saga`) and clock (`vd-time`) are pure over an injected `Clock` trait → virtual-clock behavioral tests with deterministic fault injection, no k3d required.

---

## 1. The linearizability core (`vd-fence`) — the single fix unifying the fatal attacks

Three of the fatal attacks (lease/saga split-brain, COMMIT-not-atomic, gateway dual-input, plus the deadline split-brain) are the **same bug**: two processes act on stale beliefs about who is authoritative. The fix is one primitive used everywhere:

```rust
/// Monotonic generation number. The ONLY linearizable fact about authority.
struct Fence(u64);

struct OwnerRecord {            // directory value, see §3.5
  authority: AuthorityRef,      // shard_id OR (gateway_id for sessions)
  fence: Fence,                 // bumped on EVERY ownership change; never reused, never decreases
  lease_expires_universe_ms: u64,
}
```

Rules (enforced at every boundary, not by comments — fixes R2's "ordering by comment"):
1. **Every authoritative action and every inter-shard/gateway message carries the `fence` it was issued under.** A receiver (peer, orchestrator, gateway) rejects any message whose `fence < highest_seen_for(subject)`. This is the Stoppable-token / generation-number pattern.
2. **The directory write is the single commit point.** Authority is *derived from the directory*, never from a peer notification. A shard simulates entity/realm X **iff** `directory[X].authority == self && directory[X].fence == my_held_fence`. A gateway forwards session S **iff** `directory[session:S].authority == self`.
3. **CAS on transitions.** COMMIT and ABORT are each a compare-and-swap on `directory[X].fence` with `expected == current`. Exactly one wins; the loser observes `fence != expected` and becomes a **no-op**. This makes COMMIT-vs-ABORT mutually exclusive by construction (fixes the deadline split-brain).
4. **Self-fence before reassign.** A shard/gateway that fails lease renewal MUST hard-stop all authority (stop simulating, stop ACKing as owner) within `self_fence_grace`. The orchestrator only reassigns after `lease_ttl + max_self_fence_grace` (lease-before-grant). Therefore a paused-then-resumed zombie finds its held fence stale and self-fences instead of dual-simulating.

---

## 2. Identity & Sessions

### 2.1 Accounts (orchestrator/auth redb, table `accounts`)
```
AccountId = u128 (random, non-PII). Account{ account_id, username(case-folded unique),
  pw_hash(Argon2id PHC), created_ms, universe_epoch_id, banned, session_validity_epoch:u32, schema_version }
```
Names are display-only; `AccountId` is the durable principal (old repo evicted by `player_name` — `OLD: planet-shard/src/main.rs:852`). **Argon2id verification runs in the separate `auth` service** (or at minimum `tokio::task::spawn_blocking`), never on the orchestrator runtime threads that drive saga deadlines — a reconnect-loop login flood can no longer inject 100 ms stalls into the saga/timer path (fixes the Argon2-on-hot-path finding).

### 2.2 Session tickets (fixes R7)
Replaces `SessionToken(rand_u64())` (`OLD: gateway/src/main.rs:151`).

**Day-1: HMAC-SHA256 signed ticket** (the original doc's own "acceptable simpler fallback"). PASETO-v4-local is deferred (§staging). HMAC fully satisfies R7 (unforgeable) at our scale; the validator is only the gateway.

```
SessionTicket (opaque, client-held) = base64( payload || HMAC(key[key_id], payload) )
SessionClaims{ v:u8, session_id:u128, account_id:u128, universe_epoch_id:u64,
  session_validity_epoch:u32, issued_ms, expires_ms, key_id:u32, scope }
```
`session_id` (NOT the ticket) is the cross-shard principal. Shards never see the ticket; they accept a `session_id` only over the authenticated inter-shard plane (§3) and only with a valid `gateway_gen` fence (§2.4). A client cannot forge a value any shard accepts.

### 2.3 Key rotation + revocation (fixes the key-purge / replay / no-revocation finding)
- `session_keys` table: `SessionKey{ key_id, key_bytes:[u8;32], created_ms, retire_after_ms }`.
- **Renewal MINTS A FRESH TICKET signed by the current active key** (new `key_id`, new short `expires_ms`). Sliding renewal never extends an old ticket's `key_id`, so a key can always be retired on schedule.
- **Absolute session lifetime cap** independent of sliding renewal → guarantees old keys drain.
- **Revocation set:** `account.session_validity_epoch` (and an explicit `revoked_sessions` set keyed by `session_id`). Logout/ban/credential-change/universe-epoch-roll bumps the epoch; the gateway checks the ticket's `session_validity_epoch == current` on resume, hard-invalidating outstanding tickets before their natural expiry.
- **Epoch-rollover migration:** a ticket minted just before a universe-epoch roll is **re-minted on first contact** (gateway sees old epoch but valid HMAC → issues fresh ticket for the new epoch + redirects the session) rather than silently rejected mid-session.

### 2.4 Gateway-owned connection + session-ownership LEASE (fixes the fatal dual-gateway race)
The client holds exactly ONE persistent QUIC connection to a gateway for the whole session; all shard traffic is multiplexed over it (decision #2). A shard transfer is a server-side route swap the client transport never observes.

**Session ownership is itself a directory lease** keyed `session:<session_id>` with a `gateway_gen` fence (§1):
```
LOGIN   client→gateway(creds)→auth verify→orchestrator mints ticket(session_id)
        directory CAS: session:S → {gateway:A, gateway_gen:G}
RESUME  client reconnects to gateway B, presents ticket → B validates (HMAC, expiry,
        key_id, universe_epoch, validity_epoch) → B requests session lease:
        directory CAS bumps gateway_gen G→G+1, owner→B. Orchestrator PUSHES revoke to A.
        A self-fences: stops forwarding S immediately.
EXPIRE  no activity past cap → drop SessionState, require fresh LOGIN.
```
**Every inter-shard input message carries `(session_id, gateway_gen)`.** Shards drop any input whose `gateway_gen < highest_seen` for that session (monotonic fence). Even if old gateway A's half-open QUIC path lingers (idle timeout is seconds), A's stale-gen input is rejected and A self-fences on lease loss. Gateway failover is now linearizable. A shard crash does **not** disconnect the player (the connection to the gateway is untouched); the ticket-resume path is only for client-side disconnect or gateway failover.

### 2.5 Who validates what
| Component | Validates |
|---|---|
| Gateway | Full ticket: HMAC, `expires_ms`, `key_id` known, `universe_epoch_id`==current, `session_validity_epoch`==current. Only ticket validator. Holds session lease. |
| Auth svc | Account credentials (Argon2id), off the orchestrator hot path. |
| Orchestrator | Mints tickets; owns directory/leases/fences; serves keyset; arbitrates session + entity ownership via CAS. |
| Shard | Nothing about the ticket. Accepts a `session_id` only over mTLS from a peer whose cert SAN matches the directory-current authority AND whose `fence`/`gateway_gen` is not stale. |

---

## 3. Inter-Shard Trust & Transport (fixes R2, R9, R10)

### 3.1 Day-1 trust: static cluster PSK (fixes the 6–8-week-infra finding)
Replaces `SkipServerVerification` + `with_no_client_auth()` (`OLD: quic_transport.rs:382,369`).

**Day-1:** one shared cluster cert + key in a k8s secret; rustls mTLS with `with_client_cert_verifier(rootstore)` requiring the shared cert on BOTH ends. ~100 lines, gives the same forgery guarantee at dozens-of-players scale: no unauthenticated process can inject control messages. **Deferred** (behind a real multi-tenant trust boundary): cluster CA, per-shard ed25519 leaf certs, SPIFFE SAN per-shard identity, CSR flow, 24h rotation, keyset-encryption-at-rest. The peer's claimed identity is day-1 carried in the authenticated `Envelope.from` and **cross-checked against the directory** (the directory, not the cert SAN, is the day-1 source of "who is shard X"); SAN-asserted identity is a later hardening, not a correctness prerequisite given the directory fence.

### 3.2 Envelope (fixes R9, R10, R2-ordering)
```
Envelope{ correlation_id:u128, from:Ref, to:Ref, seq:u64, causal_after:Option<u64>,
          fence:Fence,                 // §1 — authority generation this msg was issued under
          step_id:u32,                 // idempotency at the side-effect level (see §3.4)
          kind:ControlKind, body:Bytes /* postcard */ }
```
Delivered over **bidirectional QUIC streams with application-level ACKs** (replaces `try_send`). Unacked messages retry with backoff; the saga (§4) is the durable backstop so a lost ACK can never wedge a transfer (fixes R1, R9). `seq`/`causal_after` enforce ordering within a live connection only.

### 3.3 Durable dedupe across restart (fixes the seq-resets-on-restart finding)
`seq` is **not** a correctness mechanism across restarts. Correctness is per-side-effect idempotency: **every side-effecting step is keyed `(correlation_id, step_id)`** and recorded durably in the shard's `applied_steps` redb table **in the same write txn as the side effect**. On restart, a replayed message whose `(correlation_id, step_id)` is already in `applied_steps` is a no-op. So ghost-spawn, demote, lease-renew, GhostUpdate-apply cannot double-apply after a crash. `seq` is retained only for in-connection ordering/fast-dedupe.

### 3.4 Message authorization
A shard receiving a transfer-initiation for entity E accepts it only if the sender is the **directory-current authoritative owner of E at the carried `fence`** (cached + fence-versioned). A non-owner or stale-fence message is rejected and logged with `correlation_id` (structural fix for R3 "spawn-and-hope," layered on cert auth).

### 3.5 Serialization
- **postcard** for all control/saga/checkpoint payloads — stable wire format + explicit `schema_version` per type → safe cross-version handoff.
- **bitcode** for same-version 20 Hz world snapshots (codec boundary defined here; snapshots not in this charter).
- FlatBuffers dropped (deletes the `OLD: core/src/handoff.rs:9` 3-step FB↔struct↔spawn manual-sync footgun).

---

## 4. Transfer Saga (fixes R1, R3, R4; P4, P7)

The transfer lifecycle has a **single durable owner**: the orchestrator. The saga is a **pure state-transition function** in `vd-saga` (testable without I/O); the orchestrator wraps it with persistence and at-least-once delivery.

### 4.1 Typed transitions (fixes R4)
Replaces the `OLD: core/src/handoff.rs:48-94` god-struct (~12 mutually-exclusive Options: target_star_index, galaxy_context, target_planet_seed, target_ship_id, ship_system_position, warp_target_star_index, target_system_eva, …).
```
enum TransitionKind { InitialSpawn, ShipToSurface, SurfaceToShip, ShipToEva, EvaToShip,
                      PlanetToSpace, SpaceToPlanet, SystemWarp }
enum TransitionPayload { ShipToSurface(ShipToSurfacePayload), SystemWarp(SystemWarpPayload{...}), ... }
```
Each variant carries only its fields — impossible to construct an invalid combination (type-level enforcement). **The compiler forces exhaustive handling, so transitions are added incrementally (§staging): ship the engine + InitialSpawn + ShipToSurface with full ABORT and a chaos test FIRST; SystemWarp (needs on-demand dest provisioning) LAST.**

### 4.2 Saga record (orchestrator redb, table `transfer_saga`)
```
TransferSaga{ correlation_id:u128, idempotency_key:u128, session_id:u128, kind:TransitionKind,
  source:ShardRef, dest:ShardRef, state:SagaState, step_counter:u32,   // monotonic liveness proof
  expected_fence:Fence,                  // the CAS expected-version for COMMIT/ABORT
  ghost_ready:bool, source_frozen_at_tick:Option<u64>,
  adaptive_deadline_universe_ms:u64,     // = now + k*measured_RTT (NOT a magic constant)
  hard_deadline_universe_ms:u64,         // unconditional: forces DONE or ABORT, never indefinite resume
  attempts:u16, payload:TransitionPayload, schema_version:u16 }
```

### 4.3 State machine (P4: two-phase + abort, hardened)
```
        PREPARE ──ghost-ready-ack──► FREEZE ──flush+CAS-commit──► COMMIT ──► DEMOTE ──► DONE
           │                            │                            │
           └──precondition-fail──► ABORT ◄────deadline/CAS-loser─────┘
                                        │
                                        └─ compensation: source un-freeze (durable), ghost torn down
```
- **PREPARE:** dest instantiates a **read-only ghost** of E (P3) in the overlap/hysteresis band; ACKs `ghost_ready`. Entity exists on dest before authority flips (fixes R3 eviction landmine + visibility pop).
- **FREEZE+FLUSH (off the critical path — see §6 seamlessness):** source freezes E and **persists an explicit durable freeze marker** into `player_ckpt`: `frozen:true, frozen_at_universe_ms, pre_freeze_velocity, pre_freeze_pos`. It sends the **exact frozen pose+velocity+`source_frozen_at_tick`** to dest as the authoritative handoff state (NOT the last lossy GhostUpdate). The durable checkpoint write happens **lazily after** authority flips — the ghost already holds the pose, so the freeze window is sub-tick.
- **COMMIT:** **CAS on `directory[E].fence` (expected == `expected_fence`)**. Winner sets owner=dest, fence+1, fsync — *this single linearizable write is the commit point.* Dest derives its authority by reading the directory (it does not wait for a "you are authoritative" message; if the orchestrator crashes right after the CAS fsync, dest still becomes authoritative on its next directory read — fixes the COMMIT-not-atomic fatal). Dest promotes the ghost to the **frozen handoff state**, then **re-evaluates forward** from `source_frozen_at_tick` to its current tick using the deterministic solve (analytic for celestial/ballistic; never re-stepping rapier — §7). Commit-time reconciliation is explicit; a ghost is never promoted to its last broadcast pose (fixes the stale-velocity pop).
- **DEMOTE:** source converts its copy to a ghost for `demote_after_ticks`, broadcasting `GhostUpdate` so neighbors see no gap, then despawns. The durable freeze marker is cleared.
- **ABORT:** any failed spatial precondition (explicit abort path, our addition over the Improbable double-handshake) or `hard_deadline` breach → the terminal abort clears the directory transfer-lock **FENCE-NEUTRALLY** (`DirectoryCore::abort_clear`): authority STAYS with the source and the fence does **NOT** advance — an abort is not an ownership change, so `directory.fence == source.entity_fence` is preserved (FENCE-9). *(Reconciled from the original "CAS re-asserts source ownership (fence+1)" — a spurious bump strands the surviving source one fence behind the directory and permanently wedges its post-abort logout `LeaseRevoke{fence}`; and no in-flight crossing ever exists to fence out at the abort point, since crossings are emitted only post-CAS and post-commit phases never abort. See DEFERRED.md D-1/D-6 abort-path.)* COMMIT and ABORT remain mutually exclusive on the same fence — the loser of the race no-ops; a separate FENCE-MOVING `abort_cas` arm exists ONLY for an abort that genuinely races a live commit, never for a terminal abort. Compensation un-freezes the source. **Because the freeze is a durable marker, a replacement source provisioned mid-abort reads `frozen:true` from `player_ckpt` and deterministically completes the abort** (un-freeze, re-advance ballistically from `frozen_at_universe_ms` using the analytic closed-form, not a lost rapier integration) — fixes the "ABORT after FREEZE has no compensation if source died" major. **FREEZE is forbidden for entities in unconstrained free-fall unless `pre_freeze_velocity` + analytic ballistic state is checkpointed** (so re-advance is closed-form, not lost).
- **Liveness proof:** `step_counter` is strictly monotonic; `hard_deadline_universe_ms` is unconditional. Every restart re-drives steps idempotently `(correlation_id, step_id)` to DONE or ABORT — never indefinite resume (fixes the COMMIT-progress-liveness fatal).

### 4.4 At-most-one in-flight transfer per authority unit (fixes the concurrent-saga finding)
The directory enforces a **per-entity and per-session transfer mutex**: the orchestrator refuses to open a new saga for any entity/session that has a non-terminal saga, **checked in the same txn that creates the saga**. A retry after ABORT may start a new saga only once the prior reaches a terminal state proven by directory state. This prevents two ghosts on the same dest / colliding commits (the ship-transferring-while-player-EVAs case).

### 4.5 Persistence cadence (fixes the orchestrator-fsync-bottleneck finding)
The saga runs **in memory**; it persists durably at only the two crash-critical points: **(a) after PREPARE-ack** (ghost exists) and **(b) the COMMIT CAS** (authority flipped). FREEZE/DEMOTE are reconstructable from those + the durable freeze marker. Lease renewals and the tick-ceiling checkpoint are batched into one periodic transaction. This collapses orchestrator write-txn/s from "every state transition" to ~2 per transfer + 1 batched periodic — keeping redb's single writer off the saga critical path so `adaptive_deadline` timers don't fire spuriously on a slow fsync.

---

## 5. Persistence Architecture (EntityGraph-lite) — three planes (P2)

Simulation authority (shard ECS) ≠ client connection (gateway) ≠ persistence (redb) — independently recoverable.

### 5.1 Per-realm redb, single-writer ownership (fixes the RealmId multi-writer fatal-adjacent)
**Explicit ownership unit:** a realm's block store is owned by **exactly one shard process at a time** — the directory's `RealmRef` owner. redb is single-process single-writer; two shards cannot open the same file. Therefore: **block edits originating in a ghost/overlap region are FORWARDED (as fenced control messages) to the realm owner, never written locally.** RealmId-keyed redb is then safe. (If horizontal planet sharding is ever needed, shard the store by `(RealmId, region-range)` so each shard writes a disjoint file with the directory tracking region ownership — deferred; not needed at current scale.)

Tables (per realm-owning shard):
```
meta          : &str -> VersionedRecord                  // table versions, shard identity, last_tick
block_wal     : u64(lsn) -> WalRecord                    // append-only block edits
chunk_snapshot: ChunkKey -> CompactedChunk               // periodic compaction
snapshot_ckpt : () -> DurableCheckpointLsn               // two-phase WAL prune marker (§5.3)
config_state  : BlockKey -> FunctionalBlockConfig        // functional-block config (shard-agnostic)
grants        : GrantId -> RemoteAccessGrant
player_ckpt   : SessionId -> PlayerCheckpoint            // incl. durable freeze marker (§4.3)
applied_steps : (CorrelationId,StepId) -> ()             // §3.3 durable idempotency
ship_design   : ShipId -> ShipGrid
```
`ChunkKey{ realm:RealmId, chunk:ChunkAddress }`, `RealmId = planet_seed | ship_id | system_seed`. Same `vd-store::wal` code persists planet, ship-interior, and any future voxel realm (satisfies "block logic shard-agnostic"; fixes the SPEC §8.4 gap). DBs live on a **persistent volume keyed by `RealmId`**, not `shard_id` (old repo used `/tmp/...{shard_id}.redb` — `OLD: planet-shard/src/main.rs:3292`, ephemeral, edits lost on restart), so a re-sharded planet keeps its edits.

### 5.2 Block-edit WAL
```
WalRecord{ lsn, chunk:ChunkKey, edits:Vec<BlockEdit>, tick }
BlockEdit{ bx:u8,by:u8,bz:u8, new_type:BlockId(u16) }   // 5 bytes, dedup-to-final
CompactedChunk{ base_seed_version:u32, palette:Vec<BlockId>, indices:PackedBitset, up_to_lsn:u64 }
```
Off the tick (learned from `OLD: grant_persistence.rs:53` "don't fsync on the 20 Hz tick"): gameplay pushes edits to an in-memory `ChunkDelta` (dedup-to-final). **`ChunkDelta::drain()` sorts by chunk-local key before emitting** so WAL byte output is deterministic — fixes the `OLD: chunk_delta.rs:23` HashMap "order not guaranteed" determinism hole. A dedicated async writer batches deltas into `block_wal` every ~250 ms, one fsync/batch.

### 5.3 Compaction = WAL checkpoint discipline (fixes the prune-race lost-edits major)
Snapshot-write and WAL-prune are **never the same naive write**. Standard three-step durable checkpoint:
1. Write new `chunk_snapshot` (with `up_to_lsn = N`), fsync.
2. Record `snapshot_ckpt = N` (durable checkpoint LSN), fsync.
3. **Only then** prune `block_wal` records strictly `<= snapshot_ckpt`.

Recovery: load snapshot, **replay all WAL with `lsn > snapshot.up_to_lsn`**, and **tolerate WAL records older than the snapshot** (replay is idempotent last-writer-wins). Pruning is never assumed complete. A crash between any two steps loses nothing (at worst double-applies an idempotent edit).

### 5.4 Player checkpoint + tiered cadence (fixes R6 budget; §8)
```
PlayerCheckpoint{ session_id, account_id, realm:RealmId, frame:FrameRef,
  pos:DVec3, vel:DVec3, rot:DQuat, universe_time_ms:u64,
  health:f32, shield:f32, stance:u8, inventory:Inventory,
  // durable freeze marker (§4.3):
  frozen:bool, frozen_at_universe_ms:u64, pre_freeze_velocity:DVec3, pre_freeze_pos:DVec3,
  ckpt_tick:u64, schema_version:u16 }
```
- **Hot tier (pose/health):** every 20 ticks (1 s), all players on a shard coalesced into **one redb txn/s** (~12 KB, 1 fsync/s).
- **Warm tier (inventory/stance):** write-on-change, debounced ~5 s.
Cadence tiering is **required, not optional** (the concrete answer to write-amplification — naive 20 Hz × 100 players × COW B-tree = 10–50× amplification = unacceptable).

### 5.5 FrameRef (fixes R6 — kills per-shard frame reconstruction)
```
FrameRef{ kind:Realm|ShipLocal|SystemSpace, anchor_realm:RealmId, valid_at_universe_ms:u64 }
```
The frame and its valid-at time travel **with** the pose. No shard re-derives frames from hand-matched formulas + clock-skew terms (R6). The receiver reconstructs world position as a pure function of `FrameRef` + the shared analytic clock (§7). One `vd-worldspec` function used everywhere replaces the per-shard duplication that was R6.

---

## 6. Seamlessness under interpolation-only + no-prediction (fixes the "freeze is visible" major)

The user forbids client prediction and mandates server-authoritative interpolation. "Seamless" therefore has a **hard real-time definition: the FREEZE window must be sub-frame (< 50 ms, target one tick)** so the client's interpolation buffer never sees a gap longer than its lookahead. Enforced by architecture:
- FREEZE does **not** block on redb fsync. The final checkpoint is written **lazily after** authority flips (§4.3); COMMIT flips authority using the pose the ghost already holds from PREPARE/FREEZE-handoff-state.
- Commit-time forward re-evaluation (§4.3) regenerates the intermediate poses analytically across the known, bounded `source_frozen_at_tick → dest_current_tick` delta, so there is no missing keyframe to snap across.
- The freeze-window budget is a **tested invariant** (chaos test asserts window < one tick under injected latency). This is the actual operational definition of "seamless" here; the warp masking visual (system shrinks to a dot) covers SystemWarp's longer dest-provisioning latency without violating the underlying seamless machinery.

This bounds the residual to **interpolation error, not an uncorrected jump** — which dovetails with §7's "don't bound seamlessness on absolute-clock agreement."

---

## 7. Universe Time & Determinism (fixes R6 precision + the three time-related majors/fatal)

### 7.1 Determinism boundary (fixes the "stack isn't bit-deterministic" major)
`vd-worldspec` declares two categories explicitly:
- **Category A — Analytic (bit-deterministic, reconstructable on ANY host):** terrain/biomes/geology (`PlanetParams::from_seed`), planet params (radius 100k–200k blocks, gravity, rotation period via `t_freefall=√(R³/GM)` × calibrated multiplier — `OLD: planet_rotation.rs`), **orbital state** `mean_anomaly = mean_anomaly_epoch + mean_motion·universe_time` → Kepler solve (`OLD: system.rs:586`), galaxy layout, and **closed-form ballistic motion under constant gravity**.
- **Category C — Physics (NOT bit-deterministic: rapier3d FP/contact-ordering, bevy_ecs `multi_threaded` system ordering):** all rapier poses/velocities. **Category C state is ALWAYS carried by checkpoint/snapshot and NEVER reconstructed by re-simulation on a different host.** §7.4-step-3 free-fall advance is permitted **only via the Category-A closed-form ballistic solve**, never a rapier step. bevy_ecs system ordering must be explicitly constrained anywhere it touches persisted state. (Two shards recovering the same checkpoint thus never diverge.)

### 7.2 Durable monotonic universe clock (fixes the "1 s checkpoint rewinds time" fatal)
`universe.genesis_unix_ms` + `universe_epoch_id` are written once at genesis, never reset. **Universe time is derived from a write-ahead-reserved tick ceiling** (Lamport/HLC durable-clock technique):
- The orchestrator **reserves tick `N + R` durably** (write-ahead, fsync) and only hands out ticks **below the reserved ceiling**. On restart it resumes at the persisted **ceiling**, never behind real progress.
- A restart may cause a **forward jump** (deterministic, safe — Category-A positions are pure functions of time so jumping forward just advances orbits analytically). **Backward motion is forbidden:** every shard clamps its `UniverseClock` to be **monotonic non-decreasing**; a sync reply that would slew backward is ignored. This preserves the monotonicity invariant the whole determinism story rests on.

### 7.3 Analytic clock vs local sim tick (fixes the "tick-rate divergence blows the skew budget" major + the Cristian-0.1 ms major)
The design **separates two clocks the original conflated**:
1. **Analytic clock** (drives all Category-A/celestial math): a single **orchestrator-authoritative wall-derived universe time**, read by every shard via sync. **Celestial position is computed from this synced analytic clock, NOT from a shard's local `ticks_since_genesis`.** So a CPU-starved shard that falls behind on simulation ticks still places planets/ships correctly — it just renders local physics slightly behind, which interpolation tolerates.
2. **Local simulation tick** (drives local rapier physics) — per shard, may lag.

**Seamlessness is NOT bound on absolute-clock agreement.** Every cross-shard pose is strictly relative: ghosts/handoffs carry `(FrameRef, valid_at_universe_ms)` and the **receiver re-evaluates at its own analytic clock**. For the celestial frame, both shards compute identical positions for identical `universe_ms` because the inputs are **seed-derived orbital elements + integer epoch** — the only things that must agree are the seed and the integer tick, exchanged as **exact integers with a causal barrier**, never as estimated wall-ms. This makes the original "≤0.1 ms agreement" requirement obsolete: Cristian-over-QUIC on a k3d overlay (asymmetric latency, 0.1–1 ms jitter, 50 ms tick-quantized replies) **cannot** hit 0.1 ms, and it no longer has to. The sync's job shrinks to keeping the analytic clock within a slew band; residual skew bounds interpolation error, not a jump. (If absolute sub-ms agreement is ever truly needed, NTP-disciplined PHC/PTP — not Cristian — is the tool; not required by this design.)

### 7.4 Leap catch-up on recovery (deterministic, O(1))
A shard down 5 min: (1) sync analytic clock to current `universe_ms`; (2) recompute all celestial/orbital state **directly at current time** — pure function of `(seed, universe_ms)`, no integration to replay (analytic Kepler jumps straight to correct state); (3) restore players from `player_ckpt`, and if a player was in free-fall, advance from `frozen_at`/`ckpt_tick` to now using the **Category-A closed-form ballistic solve** (never a rapier re-step), bounded and explicit. Recovery is exact: universe = f(time) + persisted deltas, time is authoritative and monotonic.

---

## 8. State Sizes & Budgets
| Item | Size | Notes |
|---|---|---|
| Player checkpoint (hot) | ~140 B | pose 80 B + health/stance + ids + FrameRef + freeze marker |
| Player checkpoint (warm) | 0.5–2 KB | on change only |
| Block edit | 5 B | dedup-to-final, sorted |
| Edited chunk delta | 0.25–2.5 KB | most chunks have zero edits (terrain derived) |
| CompactedChunk | ~1–7.7 KB | bounded by edits, not world size |
| Saga record | ~300–800 B | typed payload |
| Session ticket | ~120 B | HMAC, opaque |

**Write budget:** shard ≈ 1 fsync/s (hot ckpt coalesced) + 1 fsync/250 ms (WAL) + occasional compaction ≈ 4–5 fsync/s, within one NVMe/PV. **Orchestrator budget is now separately analyzed (was unanalyzed):** ~2 saga writes/transfer + 1 batched periodic (leases + tick-ceiling) → handful of fsync/s at dozens of players; the admin endpoint exposes `orchestrator_write_txn_latency` so this is measured, not asserted.

---

## 9. Crash Recovery
| Crashes | Player experience | Mechanism |
|---|---|---|
| Shard | **No disconnect** (gateway holds QUIC); brief realm freeze | Orchestrator detects missed lease → waits `ttl + max_self_fence_grace` (old shard self-fences first) → provisions replacement for same `RealmId` → opens RealmId-keyed redb, loads snapshot, replays WAL (`lsn > up_to_lsn`, tolerating older), reloads config/grants/ship_design, restores players from `player_ckpt` (honoring durable freeze markers → completes pending abort/commit deterministically), re-registers + renews lease with **fresh fence**, resumes. In-flight sagas driven to COMMIT/ABORT via idempotent `(correlation_id, step_id)` re-drive. |
| Gateway | QUIC drops; client reconnects to a gateway, presents ticket; no re-login/reload | New gateway validates ticket, **acquires session lease (bumps gateway_gen, revokes old)**, rebinds. Shards drop stale-gen input. session_id unchanged. |
| Orchestrator | **No disconnect**; transfers pause then resume | Reads `universe.genesis` + `universe_epoch_id`; **resumes tick at persisted write-ahead ceiling (never behind)**; recovers directory + sagas; **freezes lease expiry while it was unreachable** (shards keep authority on the held fence; restart does NOT trigger a reassignment storm — grace proportional to downtime); re-drives each saga to terminal honoring `hard_deadline`. |
| Client | Ticket-resume (as gateway row); expired ticket → fresh login restored from `player_ckpt`. |

Graceful shutdown: SIGTERM → drain (bounded tick) → final hot+warm checkpoint → WAL fsync → compaction checkpoint → lease release (fence retired) → directory deregister.

---

## A. Attack resolutions (every fatal/major, explicit)

**[fatal] Lease reassignment + zombie source = split-brain.** Resolved by §1: fencing tokens on every action/message; **self-fence-before-grant** (lease_ttl < self_fence_grace; orchestrator reassigns only after `ttl + max_self_fence_grace`); reassign-on-lease-loss and the saga share the directory fence — a reassign bumps the fence, which aborts/quiesces any in-flight saga on that shard (stale-fence messages no-op). A paused-then-resumed shard finds its held fence stale and self-fences.

**[fatal] COMMIT not atomic with directory write.** Resolved by §1+§4.3: the **directory CAS is the single commit point**; **dest authority is DERIVED from the directory**, not from a peer notification — if the orchestrator crashes after the CAS fsync, dest still becomes authoritative on its next directory read. All saga steps re-drive idempotently `(correlation_id, step_id)`; `step_counter` + unconditional `hard_deadline` give the liveness proof (always DONE or ABORT).

**[fatal] Gateway-resume dual authoritative input.** Resolved by §2.4: **session ownership is a directory lease with a `gateway_gen` fence**; resume bumps the gen and revokes the old gateway, which self-fences. Every input carries `(session_id, gateway_gen)`; shards drop sub-max-gen input. Linearizable failover.

**[fatal] 1 s tick checkpoint rewinds time.** Resolved by §7.2: **write-ahead-reserved tick ceiling**; restart resumes at the ceiling (never behind); forward jumps allowed, **backward slew forbidden** (monotonic clamp at every shard).

**[fatal] (set-2) deadline forced-resolution wedges the player / COMMIT-ABORT split-brain.** Resolved by §4.3+§4.5: COMMIT and ABORT are each a **CAS on the same `directory.fence`** — mutually exclusive, loser no-ops; `deadline` is **adaptive (k·measured_RTT, no magic constant)**; saga runs in memory with only 2 durable points so a slow fsync doesn't fire spurious aborts; on breach, ABORT-to-source only if source is provably authoritative via the CAS, else drive forward. Chaos test asserts exactly one authoritative owner at every instant.

**[major] ABORT after FREEZE has no compensation if source dies.** Resolved by §4.3: freeze is a **durable marker in `player_ckpt`** (`frozen, frozen_at_universe_ms, pre_freeze_velocity, pre_freeze_pos`); a replacement source reads it and deterministically completes abort (un-freeze + closed-form ballistic re-advance) or commit. FREEZE forbidden for unconstrained free-fall unless full analytic ballistic state is checkpointed; abort timeout bounded << ballistic-divergence budget.

**[major] Cristian can't hit 0.1 ms.** Resolved by §7.3: **seamlessness is not bound on absolute-clock agreement**; poses are relative, receiver re-evaluates at its own clock; celestial agreement needs only seed + integer tick (exact, causal-barriered). The 0.1 ms requirement is retired.

**[major] WAL prune races recovery → lost edits.** Resolved by §5.3: three-step durable checkpoint (write snapshot+fsync → record durable LSN+fsync → prune strictly `<=` that LSN); recovery tolerates WAL older than the snapshot (idempotent replay); pruning never assumed complete.

**[major] Ghost stale velocity → pop at COMMIT.** Resolved by §4.3: FREEZE sends the **exact frozen pose+velocity+tick** as authoritative handoff state; dest promotes to that and **re-evaluates forward analytically** to its current tick; never promotes to last broadcast pose.

**[major] seq dedupe resets on restart → replay.** Resolved by §3.3: **durable `(correlation_id, step_id)` idempotency** recorded in `applied_steps` in the same txn as the side effect; `seq` is in-connection-only.

**[major] key rotation / cross-epoch replay / no revocation.** Resolved by §2.3: renewal **mints a fresh ticket on the current key**; absolute lifetime cap drains old keys; **revocation set + `session_validity_epoch`** hard-invalidates; explicit epoch-rollover re-mint.

**[major] concurrent sagas per session/entity.** Resolved by §4.4: **per-entity/per-session transfer mutex in the directory**, checked in the saga-creating txn; new saga only after prior is terminal.

**[major] orchestrator SPOF / reassignment storm.** Partially resolved + surfaced as trade-off (§B): **lease expiry freezes while orchestrator unreachable** (shards keep authority on held fences); restart does not storm (grace ∝ downtime); Argon2 moved off orchestrator (§2.1). Full replication is an explicit deferred trade-off.

**[major] 6–8 weeks of infra before a cube.** Resolved by staging (§staging): **HMAC ticket + static cluster PSK + grant-pattern versioning + dev-wipe** day-1; PASETO/CA/SPIFFE/migration-registry deferred.

**[major] orchestrator redb single-writer bottleneck.** Resolved by §4.5: saga in memory, 2 durable points; batched lease+tick txn; high-churn leases can be memory-soft (rebuilt on restart from fences) separate from durable accounts/keyset.

**[major] no-prediction + 20 Hz makes freeze visible.** Resolved by §6: **sub-tick freeze window as a tested invariant**; FREEZE off the fsync path; analytic forward re-eval fills the bounded gap.

**[major] stack not bit-deterministic.** Resolved by §7.1: explicit Analytic-vs-Physics boundary; Category-C never reconstructed by re-sim; free-fall advance is closed-form; `ChunkDelta::drain()` sorts; bevy_ecs ordering constrained where it touches persisted state.

**[major] refuse-on-future-version bricks the multi-worktree dev loop.** Resolved by §staging: `VD_DEV_WIPE_ON_SCHEMA_MISMATCH` (regenerable data) instead of crash-loop; ship the proven `grant_persistence` leading-version-byte + default-fill pattern day-1; defer the SCHEMA_REGISTRY/Migrator framework until a real rolling deploy is on the roadmap; each migration wrapped in one redb txn that updates `meta` atomically (no mid-migration corruption).

**[major] 2am debuggability unchanged (grep loop).** Resolved by §operability: **day-1 orchestrator read-only admin endpoint** (reuses the existing `OLD: orchestrator/tests/http_api_test.rs` HTTP surface) dumping live sagas by `correlation_id` (state, deadline, attempts, step), directory ownership + fence for a `session_id`, per-shard lease health; Prometheus `saga_stuck_count`, `abort_rate`, `orchestrator_write_txn_latency`. Converts 2am debugging from rebuild-and-grep into `curl`. **This outranks PASETO/SPIFFE in sequencing.**

**[major] 8 transitions × 5 states day-1 = god-struct complexity moved not removed.** Resolved by §staging: build the engine + **InitialSpawn + ShipToSurface** with full ABORT + chaos test first; the typed enum's exhaustiveness makes incremental addition compiler-safe; SystemWarp (on-demand provisioning) last.

**[minor] Argon2 stalls saga driver.** Resolved by §2.1: separate `auth` service / `spawn_blocking`.

---

## B. Explicit trade-off: single-orchestrator durability

The orchestrator remains a non-replicated singleton owning directory, sagas, accounts, keyset, and the universe clock. **Recommendation:** ship day-1 as a **documented single-orchestrator limitation** with: (1) redb periodic backup + WAL-shipping to the PV for a bounded RTO; (2) **lease-freeze-on-unreachable** so shards do NOT lose authority during an orchestrator blink (decouples lease liveness from the coordinator — the highest-value mitigation); (3) restart grace ∝ downtime to prevent reassignment storms. **Defer** Raft-replicated directory/saga log (or an external replicated store for the authoritative state) until the player count justifies the complexity — but design the directory/saga records as an append-only fenced log from day-1 so that replication is a later additive change, not a rewrite. The genuine unsolvable-at-this-scale tension is "MMO-grade durability from day one" vs "solo dev, dozens of players": the fences + durable saga + lease-freeze give correctness (no split-brain, no lost transfers) even on the singleton; only *availability* during an orchestrator outage is traded, and that is the right trade for a solo dev's k3d cluster.

---

## C. Staging / sequencing (the build order)
- **Phase 0 (foundation, no gameplay payoff minimized):** static cluster-PSK mTLS; HMAC tickets; `vd-fence`; directory + leases + fences (in-memory + periodic durable); grant-pattern versioning + `VD_DEV_WIPE`; durable monotonic clock (write-ahead ceiling); **orchestrator admin endpoint + metrics**.
- **Phase 1 (first seamless loop):** saga engine (`vd-saga` pure) + **InitialSpawn + ShipToSurface** with PREPARE/FREEZE(off-path)/COMMIT(CAS)/DEMOTE/ABORT + durable freeze marker + chaos test asserting single-owner + sub-tick freeze window.
- **Phase 2:** SurfaceToShip, ShipToEva, EvaToShip, PlanetToSpace, SpaceToPlanet (each compiler-forced exhaustive).
- **Phase 3:** SystemWarp + on-demand dest provisioning; SCHEMA_REGISTRY/Migrator (gated on rolling deploys); PASETO + cluster CA + SPIFFE leaf certs + 24h rotation (gated on a real trust boundary); orchestrator replication (gated on scale).

---

## D. Root-cause traceability
| RC | Structural fix |
|---|---|
| R1 no transfer owner | Orchestrator saga; CAS commit; adaptive+hard deadline; idempotent re-drive; durable freeze marker (§4) |
| R2 client re-homes / cert-skip / ordering-by-comment | Gateway-owns-connection + session lease/gen fence (§2.4); static-PSK→CA mTLS (§3.1); fenced envelope seq/causal_after (§3.2) |
| R3 spawn-on-promote | Ghost-first; frozen-state handoff; analytic forward re-eval (§4.3) |
| R4 god-struct | Typed TransitionKind/Payload (§4.1) |
| R5 no tests / bin-only | lib+thin-bin; pure `vd-saga` + injectable `vd-time`; chaos tests (§0,§4) |
| R6 frame reconstruction / clock skew | FrameRef-with-pose + analytic synced clock; relative re-eval (§5.5,§7.3) |
| R7 forgeable identity | HMAC (→PASETO) tickets + rotation + revocation; shards trust only fenced authenticated session_ids (§2) |
| R8 lossy digests / hand timeouts | Fence-versioned lease directory; interest diffs (§1,§5.1) |
| R9 try_send | ACKed retried streams + durable saga backstop + durable step idempotency (§3) |
| R10 no correlation / observability | correlation_id+step_id in every envelope; admin endpoint + metrics (§3,§operability) |