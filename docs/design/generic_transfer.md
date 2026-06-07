
# Charter A (REVISED FINAL) — Generic Entity Transfer, Hardened

Self-contained. Binding vocabulary: NODE, SHARD, REALM (`RealmId = planet_seed|system_seed|ship_id`), AUTHORITY {Owned, Ghost, Frozen}, `Fence(u64)` (the one generation primitive; receivers reject `fence < highest_seen`), `TransferId` (the one correlation id), `(TransferId, step_id)` idempotency in per-realm `applied_steps`, `universe_tick` (analytic), `source_tick` (per-shard, on every message), `StampedPose{frame_ref,pos,vel,orient,universe_tick}` (source-computed, dest sanity-bounds). COMMIT = a single linearizable directory CAS on an OwnerRecord fence (integration blocking-resolution #1). Directory key space = `Session | Entity(EntityId) | Realm(RealmId) | Ship(ShipId)` (integration key-space resolution). All node code is built against the `ShardIo`/`Transport` seam (test_harness). One node lib `build_app(NodeKind,cfg,io)`. postcard v1 everywhere; codec flag bit reserved.

This revision keeps the charter's thesis — **the registry is the only per-KIND surface, the saga/directory/gateway/ghost transport are untouched** — but corrects the two fatal foundations and five majors. The phrase "ONE machinery" is now made literally true (a real, cheap, durable commit point for every transfer including transients) rather than rhetorically true.

---

## A0. Core principle (unchanged in spirit, tightened in form)

`EntityId(u128) = {kind:u8, mint_shard:u32, seq:u64, rand:u24}` (transfer_protocol §1.2). The `kind:u8` indexes a static registry, generalizing the proven `BlockDef`-by-`BlockId` table (`core/src/block/block_def.rs`: "one entry per BlockId in the registry") and the versioned `CharacterStateBlob`. The registry is the single point of per-kind variation.

```rust
// core/src/entity_kind.rs — shard-agnostic, pure, in `core`
#[repr(u8)]
pub enum EntityKind { Player=0, Ship=1, NamedConstruction=2,
                      Debris=10, DroppedBlock=11, Projectile=12, Rocket=13 }

pub struct KindDef {
    pub class: DurabilityClass,     // Durable | Transient
    pub ghost_policy: GhostPolicy,  // Always | InBand | NeverTransferOnly
    pub continuity: ContinuityModel,// Frozen | BallisticReadvance | Guided | RealmAnchored
    pub loss_budget: LossBudget,    // Transient only; Durable = ZERO
    pub blob_schema: SchemaId,      // TLV envelope id (§A1)
    pub max_state_bytes: u16,
}

/// ONE trait per kind. The serialize/spawn surface is a TYPED struct, not raw Bytes (§A9 HR1 guard).
pub trait TransferableKind: 'static {
    const KIND: EntityKind;
    const DEF: KindDef;
    type Blob: TlvBlob;                                            // typed; CI-asserted no undeclared field
    fn serialize(world: &World, e: Entity) -> Self::Blob;
    fn spawn(world: &mut World, blob: &Self::Blob, pose: &StampedPose,
             prov: SpawnProvenance) -> Entity;                     // idempotent on (TransferId,step_id)
    fn precondition(dest: &World, blob: &Self::Blob, pose: &StampedPose)
             -> Result<(), SpatialReject>;
    fn rebind_refs(blob: &mut Self::Blob, ctx: &RefRebindCtx);     // EntityId rebind (§A5)
}
```

A new kind = (1) enum variant, (2) `KindDef` const, (3) `impl TransferableKind`, (4) `register!()` — **iff** it reuses an existing `(DurabilityClass, ContinuityModel, GhostPolicy)` triple. A kind needing a NEW triple is first-class engineering + a crash-matrix multiplication (§A8 honesty fix).

---

## A1. Blob evolution — TLV envelope + version-floor handshake (FATAL #2 fix)

**The original claim "older shards ignore unknown trailing fields" is FALSE for the binding codec.** Verified: `core/src/character/handoff_blob.rs` uses bincode and `decode()` returns `CharacterStateBlob::default()` on any parse error. `#[serde(default)]` only lets a NEWER reader fill a MISSING field; it does NOT let an OLDER reader skip unknown TRAILING bytes a newer producer appended — postcard/bincode are positional, non-self-describing, with no field framing to skip. For a Durable blob this silently zeroes a player (inventory→Default) on a rolling deploy where dest is older than source. Two structural fixes:

**(1) TLV-framed blob.** A `TlvBlob` is `header || repeated { tag:u16, len:u32, bytes }`. The decoder reads tags it knows and **skips `len` bytes for tags it does not** (real forward skip the format lacks at the postcard level is reintroduced at the TLV level). Each field is an explicit tag; CI asserts `serialize` emits only declared tags (the HR1 by-construction guard, §A9). Header = `{ schema_id, writer_max_tag:u16, codec_flags:u8 }`.

```
TlvHeader{ schema_id: SchemaId, writer_max_tag: u16, codec_flags: u8 }
TlvField { tag: u16, len: u32, bytes: [u8; len] }   // bytes = postcard of that field's type
```

**(2) Version-floor capability handshake; decode-to-Default BANNED for Durable.**
- At cluster join, each shard advertises, per registered kind, its `max_known_tag`. The orchestrator records a per-kind cluster **version floor** = min over live shards. A Durable transfer is **refused at PREPARE** if `dest.max_known_tag < source.writer_max_tag` for any *required* (non-skippable) tag (capability negotiation). Optional additive tags are skip-tolerant by construction; required tags are gated by the floor. This makes a rolling deploy safe: the orchestrator never routes a Durable entity to a dest that cannot represent its required state.
- For Durable kinds, a decode error is a hard ABORT_SPATIAL (`TransferRejected{reason: BlobUndecodable}`) — the player stays on source, alive — **never** a silent Default. `decode→Default` is forbidden for Durable; allowed only for Transient *cosmetic* tags within their loss budget.
- CI gate (replaces the false one): a `kind_blob_evolution` test constructs a writer at tag-set N+1 and a reader at tag-set N and asserts (a) the reader skips the unknown tag and recovers all known fields, and (b) a reader missing a *required* tag returns `Err`, not Default. This actually tests older-reads-newer, which the original same-version round-trip test could not.

---

## A2. Durability tiering on ONE machinery, with a REAL commit point for transients (FATAL #1 + MAJOR #5 fix)

The saga FSM is unchanged: `IDLE → PREPARE → FLUSH → FENCE_DEMOTE → PROMOTE → ROUTE_SWAP → CLEANUP`, mapped to the gateway command vocabulary (PrepareSubscribe/RequestCut/FreezeSource/CommitAuthority/…). `DurabilityClass` selects which *effects* fire. **Crucially, every transfer — Durable and Transient — has a single linearizable orchestrator go-decision on a directory Fence; transients do not skip the commit point, they share a cheap batched one.** This is what makes "ONE machinery" literally true and keeps NO-UNILATERAL-COMMIT (test_harness §5a) intact without amending it.

### Durable class (Player, Ship, NamedConstruction)
Exactly the existing hardened design. Per-`Entity(EntityId)` OwnerRecord; full saga persistence at the two crash-critical points; per-realm `applied_steps(TransferId, step_id)` write-ahead before any world mutation; full crash matrix; `LossBudget::ZERO`; gateway lease-epoch/Fence frame-drop; per-entity `in_transfer` mutex. Untouched.

### The transient authority problem and its fix (FATAL #1)
The original grounded transient authority on "the directory already has a Realm(RealmId) owner record; a transient is Owned by whichever shard owns the realm containing its position." **This is unsound:** integration collapsed Region INTO RealmId, and `RealmId = planet_seed|system_seed|ship_id` is a WHOLE planet/system/ship. The system realm and the planet realm OVERLAP in world space precisely in the SOI band — that overlap *is the band*. A bullet in the planet-SOI band is inside the system realm's space AND the planet realm's space; "the realm containing its position" is multi-valued exactly where handover happens, so two shards each correctly conclude they own it (len==2), and the ControlOracle, finding no `Entity` row for the bullet, cannot evaluate ownership at all.

**Fix — explicitly-HELD transient authority anchored to a shared region-shard lease Fence:**

1. **A transient is Owned by the shard that holds it in its in-memory `owned_transients` set — full stop. There is no spatial point-to-owner function.** Ownership is held state, not derived state. The single source of truth for "who owns this bullet" is the at-least-once handover ack, not a spatial query.

2. **No per-bullet directory row, but a real fenced anchor.** Every shard already holds, in the directory, the `Realm(RealmId)` OwnerRecord for each realm it runs, carrying a `Fence`. That Fence is the **region-shard lease**: the linearizable, orchestrator-issued proof "shard S authoritatively runs realm R at fence F." A `TransientTransferBatch` carries the SOURCE's held `src_realm_fence` and the DEST's last-observed `dst_realm_fence`. The bullet inherits NOTHING spatial; it inherits the *handover's* authorization, which is the realm-lease Fence. This costs zero per-entity directory writes (the §A2 cost goal is preserved) while giving a Fence the oracle can read.

3. **A cheap, batched, durable orchestrator go-token keeps NO-UNILATERAL-COMMIT (MAJOR #5).** The transient handover is NOT a unilateral spawn. Before the dest may promote a received batch to Owned, the saga records — group-committed, one fsync amortized across ALL concurrent batches that tick — a `TransientGo{TransferId, from_realm@src_fence, to_realm@dst_fence, n_items}` token. This is the "single linearizable go-decision durably recorded by the orchestrator" that NO-UNILATERAL-COMMIT demands. It is per-BATCH, not per-item, so 200 bullets/tick = 1 token, 1 amortized fsync. The token's CAS is on a lightweight `Transfer(TransferId)` saga key, not on a per-bullet `Entity` key — so the "no per-transient directory write" cost win holds, and the invariant holds. **Result: there are NOT two machines; there is one FSM whose Transient branch collapses the ghost-subscribe handshake but still passes through the same orchestrator commit point (a Fence-CAS go-token), just batched.** The integration COMMIT-is-a-CAS resolution is satisfied for transients too.

4. **AUTHORITY-UNIQUE for transients is redefined and assigned to the right monitor.** The ControlOracle cannot read a per-bullet directory row (there is none, by design). So:
   - **ControlOracle (ground truth)** asserts `TRANSIENT-AUTHORITY-HELD`: the union of all shards' `owned_transients` sets contains each live transient `EntityId` **exactly once** (len==1 across held-sets), AND every accepted batch has a matching durable `TransientGo` token (no unilateral promotion). The oracle reads node-reported held-sets (a `Inspect::transient_owned()` hook) + the saga's go-tokens — both ground truth.
   - **WireMonitor (delivered bytes)** asserts `TRANSIENT-CONSERVATION`: across a handover a transient appears in exactly one shard's snapshot per tick (no double-render), and is delivered-to-terminal OR lost-within-`loss_budget`. Duplication is always a failure.
   The design's original conflation of "no directory row" with "directory-derivable ownership" is resolved: ownership is HELD and proven by the go-token + the held-set, not derived from a spatial map that does not exist.

### Transient handover message (revised)
```rust
struct TransientTransferBatch {
  transfer_id: TransferId,                  // ONE id; correlates the go-token
  from_realm: RealmId, to_realm: RealmId,
  src_realm_fence: Fence,                    // SOURCE's held region-shard lease (authority proof)
  dst_realm_fence: Fence,                    // DEST realm-lease fence last observed by sender (staleness guard)
  source_tick: u64, universe_epoch: EpochId,
  items: Vec<TransientItem>,                 // each {entity, kind, TlvBlob, StampedPose}; bounded (§A6)
}
```
Accept rule on dest D for realm R: D accepts iff (a) it currently holds `Realm(R)` at `fence >= dst_realm_fence` (it is the live owner, not stale), AND (b) the orchestrator has (or, in the same tick, will group-commit) the `TransientGo` token for this `transfer_id` — D pulls/awaits the token before promoting items to its `owned_transients`. On accept D emits `TransientTransferAck{transfer_id}`; on ack the source deletes the items from its `owned_transients`. The flip is atomic at the ack; the go-token makes it non-unilateral.

### Exactly-once + dest-crash recovery (MAJOR #4 fix)
Receiver dedup: bounded in-memory `recent_transient_accepts: HashMap<EntityId, Fence, FixedSeedHasher>` (NOT redb). Redelivered item already present → no-op + re-ack.

The original "two shards can never both claim region R because the partition is directory-fenced single-owner" is true for *static* ownership but breaks under **dest-realm-owner crash + reassignment** (identity_persistence §9: orchestrator provisions a replacement for the same RealmId at a fresh, higher fence). Sequence: S sends batch with `dst_realm_fence=F`; D crashes; D' takes realm R at `F+1` with an EMPTY dedup map and no knowledge of the batch; S's retry carries stale `dst_realm_fence=F`; D' rejects (`F < F+1`); S cannot hand off and (per the original) has not deleted the items → wedge or wider-than-declared loss. **Fix:**

1. **Staleness-guard recovery (one hint refresh, not a CAS):** on a `dst_realm_fence` rejection, the source does ONE directory **hint read** of `Realm(to_realm)`'s current fence (a head read, no CAS, transient-class, carries no fence expectation so a mid-flip never rejects it), refreshes `dst_realm_fence`, and retries the batch ONCE. The go-token's CAS is what serializes; the hint read just unblocks the retry.
2. **Declared loss surface (made honest):** if the retry still fails (realm genuinely unavailable) the batch is **dropped-with-reason and counted against `loss_budget(kind)`** — never wedges S's bounded `transient_queue`. The original framed the loss budget around "the OWNING shard crashing"; this revision explicitly adds **"a DEST realm-owner crashing mid-handover loses the in-flight batch within the SAME declared budget"** to the budget definition. `recent_transient_accepts` being in-memory means a dest crash forfeits exactly-once for *in-flight* (not-yet-accepted) batches — acceptable only because it is now declared, asserted, and chaos-tested.
3. **New chaos cell (assigned to §A7):** dest-realm-owner crash+reassign during an in-flight `TransientTransferBatch` — assert items either redeliver to D' within budget OR drop-with-reason, NEVER wedge the pair's queue, and the concurrent Durable saga is byte-identical (DURABLE-UNAFFECTED-BY-BURST).

### Why this is ONE machinery (HR3, now literally)
Same FSM states; same `TransferId`; same envelope family; same Fence; same `GhostDelta` transport; same gateway (transients are never gateway-routed to a client — they appear in whichever shard's snapshot the client subscribes to, satisfying SEAMLESS). **Same orchestrator commit point** (a Fence-CAS go-token), batched for transients. The only table-driven differences by `DurabilityClass`: {per-`Entity` directory row vs realm-lease-anchored held-set?, full saga fsync per entity vs batched go-token?, per-entity `applied_steps` vs in-memory dedup?, full crash matrix vs declared loss budget on crash?}. A policy fan-out inside the shared `step()`, not a parallel implementation — and now it genuinely shares the commit semantics the integration mandates.

---

## A3. Continuity per kind (ContinuityModel) — with the fence-stale guidance case (MAJOR #3 fix)

| Model | Kinds | Cross-shard behavior |
|---|---|---|
| `Frozen` | Player, Ship | Source computes dest `StampedPose`; dest uses it, interpolated across the sub-tick freeze. Full durable state. |
| `BallisticReadvance` | Debris, Projectile, spent Rocket | No freeze, no ghost-collider continuity. Blob carries `{pos, vel, spawn_universe_tick}` at `source_tick`. Dest re-advances **closed-form** under the dest realm's analytic gravity (Category-A, identity_persistence §7.1) from `source_tick` to dest's current tick. Never re-stepped in rapier across the boundary. Continuity exact because motion is analytic. |
| `Guided` | Rocket under guidance | `BallisticReadvance` kinematics PLUS `target: EntityId`. Guidance resolves target pose by a **four-way** branch (see below). |
| `RealmAnchored` | DroppedBlock at rest, placed construction | In flight = `BallisticReadvance` transient; on rest = a DURABLE ack'd conversion to realm block-store data (§A3.1). |

**Guided four-way target resolution (the missing fourth case):** the original three-way branch (in-band ghost / last-known-extrapolated / unresolvable→ballistic) has no case for **"present but frozen/fence-stale,"** which is exactly what happens when the target is a Ship whose exterior authority is mid-host-transfer. During FENCE_DEMOTE→PROMOTE the ship is Frozen on source and not-yet-Owned on dest; `GhostDespawn` is refused while `in_transfer=Some`, but `GhostDelta` for a Frozen entity STOPS — so the rocket's guidance sees the ghost *present but not advancing*. Reading that pose guides toward where the ship WAS, with no staleness flag firing (the EntityId resolves, the ghost exists). Fix:

```
1. target ghost in-band AND source_tick advancing  -> use live ghost pose.
2. target ghost in-band BUT source_tick stalled > N_stale ticks
   (the frozen/fence-stale case)                   -> treat as UNRESOLVABLE:
                                                       degrade to ballistic-with-staleness-flag. HONEST.
3. target resolvable via directory but no in-band ghost -> last-known extrapolated pose, staleness-flagged.
4. target EntityId unresolvable anywhere            -> ballistic, no target. Never error.
```
Additionally: **transient-originated directory reads (guidance target-resolve, damage routing) carry NO fence expectation and read the directory HEAD** (they are not authoritative actions), so a mid-flip CAS never rejects them; they may read a transiently-inconsistent owner and **must tolerate one-tick misroute (redeliver)**. `N_stale` lives in `TransferTuning` (§A6), not a magic number. This is covered by a new scenario crossing `GuidedRocketTargetOnOtherShard` WITH a concurrent ship-host transfer of the target (§A7).

### A3.1 DroppedBlock rest-conversion — durable ack boundary (MAJOR #6 fix)
The original had `on_rest()` "emit a forwarded BlockEdit and despawn the entity" — a non-atomic seam between two durability domains (transient in-memory, `loss_budget>0`) and `block_wal` (durable, ZERO loss). If shard X crashes after the despawn-decision but before the forwarded `BlockEdit` is durably in realm-owner R's `block_wal`, the block is GONE (despawned, never durable); if X resends on recovery without recording it emitted, the edit double-lands. The transient had no durable `applied_steps` row, so there is no `(TransferId, step_id)` to dedup against across the X-crash. Fix — **the conversion is a durable two-step on the REALM OWNER, keyed by the block's stable `EntityId`:**

1. The transient owner X forwards `LandBlock{ entity_id, block_pos, block_id }` to realm owner R. **`entity_id` (durable/unique by §1.2) IS the idempotency key** — not a TransferId the transient never had.
2. R writes the resulting `BlockEdit` to `block_wal` **idempotently keyed by `entity_id` (recorded in `applied_steps` as `(entity_id-as-cid, step=Land)`) BEFORE acking.**
3. X despawns the transient **only on R's durable ack.** A redelivered `LandBlock` for an `entity_id` already in `applied_steps` is a no-op + re-ack.

This makes landing exactly-once and crash-survivable independent of the transient's lossy tier. The conversion point is explicitly documented as **"where a Transient becomes Durable and therefore MUST cross a durable ack boundary."** `DroppedBlockLands` becomes a real permanent durability gate.

---

## A4. Ghost policy per kind (GhostPolicy) and NO-VANISH

| Policy | Kinds | Ghosting | NO-VANISH meaning |
|---|---|---|---|
| `Always` | Player, Ship | Continuous kinematic-collider ghost across the whole overlap band; never integrates. | Client never sees the entity disappear > K ticks while a transfer involving it/the observer is in-flight (test_harness §5b). |
| `InBand` | Rocket, named-construction debris | Ghosted only inside the overlap band (cross-shard collision/visibility near a boundary). | In band: as Always. Outside: absence is correct, not a vanish. |
| `NeverTransferOnly` | Projectile, bulk Debris, dropped block in flight | NO ghost ever. Exists on one shard; handed over by the batch; appears on the new shard next tick. | NO-VANISH RELAXED to **TRANSIENT-CONSERVATION** (§A7): exactly-once OR dropped-with-reason within budget; never on two shards' snapshots simultaneously. A one-tick interpolation seam at the boundary is acceptable for a bullet. Bullets pay zero ghost bandwidth. |

**Bullet hitscan vs projectile:** hitscan is never an entity and never transfers (instantaneous server-authoritative ray on the firing shard; a cross-region hit queries the neighbor's ghosts and becomes a `DamageEvent` addressed to the victim's `EntityId`). Projectile bullet is a `BallisticReadvance` transient that transfers mid-flight via the batch path.

---

## A5. Ordering / causality / damage across transfer — serialized through the transfer barrier (MAJOR #7 fix)

Stable references: the projectile blob carries `owner: EntityId` and (guided) `target: EntityId`; these survive any number of transfers (HR2). Attribution survives because `source_owner: EntityId` is data.

**The damage-vs-transferring-player hole (MAJOR #7):** the original said damage is "applied idempotently set-to-final-with-version." But damage is inherently ADDITIVE (`health -= amount`); set-to-final requires the sender to know the final value, which it cannot. Across a transfer the owning shard CHANGES, so a per-event dedup key applied on "the owning shard" is applied on different shards before/after the CAS. Concretely: projectile on A computes `DamageEvent(target=P, amount=20, event_id=E, hit_tick=100)`; P transfers A→B; if E lands on A just before the CAS flip, A applies `-20` but A's FlushComplete already shipped P's health to B computed PRE-damage → the `-20` is LOST; if E lands on B just after, and B also received the late-applied state from A, it is DOUBLE-counted. `event_id` dedup only works if A and B share a durable dedup log for P, which they do not (B reads P from the handoff blob, not A's damage log). Fix — **serialize damage through the entity's transfer barrier, using the COMMIT Fence as the cut:**

1. The handoff blob carries a per-victim **`applied_damage: HashSet<DamageEventId>`** (small, bounded, pruned by `hit_tick` age) AND the authoritative post-drain health. At FLUSH the source **drains and applies all pending `DamageEvent`s with `hit_tick < freeze_tick`** into the health it ships, and includes their `event_id`s in `applied_damage`.
2. The directory routes NEW damage by the cut: a `DamageEvent` with `hit_tick < freeze_tick` is delivered to the **source-final state path** (already folded into the blob; a late arrival is deduped against `applied_damage` on the dest, which inherited the set → exactly once); a `DamageEvent` with `hit_tick >= freeze_tick` is routed to **dest only** (post-CAS owner) and applied there, recorded in dest's `applied_damage`.
3. The CAS Fence/`freeze_tick` is the explicit cut. `set-to-final-with-version` is retained only for the health VALUE (idempotent re-apply of the same final), while the additive `DamageEvent`s are made exactly-once by the carried `applied_damage` dedup set crossing the barrier with the entity. The hit either lands on source-final (folded) or on dest (post-cut) — exactly one authoritative application, no double-count, no loss.

Ordering uses `hit_universe_tick` on the victim shard; a redelivered `DamageEvent` (dedup by `event_id` against `applied_damage`) is a no-op. New scenario: `BulletAcrossBoundary` crossed WITH the player-transfer scenario, asserting the cut.

---

## A6. Performance budget and backpressure (never affects Durable)

```rust
struct TransferTuning {
  durable_max_in_flight: u16,
  transient_batch_max_items: u16,
  transient_batch_max_bytes: u16,
  transient_max_batches_per_tick_per_pair: u16,
  transient_queue_depth: u32,
  guidance_stale_ticks_n: u16,           // N_stale for the Guided fence-stale case (§A3)
  transient_go_group_commit: bool,       // batched go-token amortization (§A2.3)
  // adaptive saga deadlines (k*RTT + hard cap) also live here
}
```
- Durable capped by `durable_max_in_flight`; transients by `transient_max_batches_per_tick_per_pair × pairs`, each ≤ `transient_batch_max_items`. A 256-item debris batch ≈ ~20 KB on the reliable control stream, application-paced under the SPIKE-3a datagram budget.
- **Burst (1000 debris/tick):** source enqueues into a **bounded `transient_queue` (`transient_queue_depth`)** draining at the per-pair batch budget → spread over a few ticks. On saturation (`SendError::QueueFull`, the synchronous backpressure of test_harness §1) excess transients are **dropped-with-reason within budget** (LIVENESS preserved). The go-tokens for a 1000-debris burst are themselves batched, so the burst adds at most a handful of amortized fsyncs, not 1000.
- **Structural isolation from Durable:** the transient queue, dedup map, batch budget, and go-token group are physically separate from the Durable saga path (orchestrator saga + its own deadlines). The ONLY shared resource is raw link bandwidth, where Durable control rides a higher-priority `MsgClass` than transient batches (transients drain after `Saga`). `DURABLE-UNAFFECTED-BY-BURST` (§A7) is the executable proof.

---

## A7. Harness additions (revised invariants + scenarios)

New invariants (test_harness §5):
- **TRANSIENT-AUTHORITY-HELD (ControlOracle, ground truth).** The union of all shards' `owned_transients` held-sets contains each live transient `EntityId` exactly once (len==1 across held-sets, catching both double-own and zero-own), AND every accepted batch maps to a durable `TransientGo` token (no unilateral promotion → NO-UNILATERAL-COMMIT holds for transients too). Reads node `transient_owned()` + saga go-tokens (both ground truth). **Replaces** the original's unprovable "owning shard == directory-current owner of the realm containing position."
- **TRANSIENT-CONSERVATION (WireMonitor, delivered bytes only).** Across a handover a transient appears in exactly one shard's snapshot per tick (never two → no double-render); over flight it is delivered-to-terminal OR absent-after-handover at most `loss_budget(kind)` times. Duplication is ALWAYS failure; loss beyond budget is failure; loss within budget allowed.
- **DURABLE-UNAFFECTED-BY-BURST (ControlOracle).** Under a scripted 1000-transient burst (incl. a dest-realm-owner crash mid-handover), every concurrent Durable saga's deadline/outcome is byte-identical to the no-burst run.
- **NO-UNILATERAL-COMMIT (unchanged, now also covers transients).** A node promotes a received transient to Owned ONLY after the orchestrator durably recorded the batch's `TransientGo` token. The invariant is NOT amended; the transient path was made to satisfy it.

New/changed scenarios:
- `DebrisFieldCrossing`: N-debris batch; TRANSIENT-CONSERVATION; batch count `⌈N/batch_max⌉`; exactly-once via dedup; ack-loss redelivery dedups; go-token recorded.
- `BulletAcrossBoundary` (projectile) **crossed with the player-transfer scenario**: ballistic re-advance continuity AND the §A5 damage-cut (hit_tick < freeze_tick folded into source-final; >= routed to dest; `applied_damage` dedup across the barrier).
- `GuidedRocketTargetOnOtherShard` **with a concurrent ship-host transfer of the target**: the §A3 four-way branch — frozen/fence-stale target → honest ballistic-with-staleness-flag, never stale-tracking; transient directory reads tolerate one-tick misroute.
- `DroppedBlockLands`: transient crosses then rests → durable two-step on the realm owner keyed by `entity_id`; assert it is in `block_wal` exactly once and survives an X-crash between forward and ack.
- `TransientDestOwnerCrash` (NEW): dest-realm-owner crash+reassign during an in-flight batch → redeliver to D' within budget OR drop-with-reason; never wedge the pair queue; DURABLE-UNAFFECTED-BY-BURST holds.
- `kind_blob_evolution` (NEW, replaces the false CI test): writer at tag N+1, reader at tag N skips unknown tag and recovers known fields; a reader missing a *required* tag returns Err (Durable), never Default.
- Chaos: transient handover under drop/dup/reorder/partition — dup is no-op (dedup), drop covered by redelivery within budget, partition → lost-within-budget (NOT wedged), Durable on the same link still satisfies NO-UNILATERAL-COMMIT.

---

## A8. Roadmap amendments (slotted into the binding P0–P11)

The binding roadmap stages transitions before voxels and puts SystemWarp last. Charter A folds in as deltas (it does NOT introduce a parallel track):

- **P0:** Add `core/src/entity_kind.rs` (EntityKind, KindDef, DurabilityClass, GhostPolicy, ContinuityModel, LossBudget, `TransferableKind` with a TYPED `Blob` assoc-type, registry, **TLV envelope codec + version-floor handshake types**). Add `TransientTransferBatch` + `TransientGo` token to the FROZEN `wire` crate `TransferEnvelope<P>` family (contract-tested seam). Add the new harness invariant hooks (`transient_owned()`, go-token read). Add the `kind_blob_evolution` CI gate (the corrected one). **Cap the day-one behavioral surface to TWO triples:** `{Durable, Frozen, Always}` (the player, already designed) and `{Transient, BallisticReadvance, NeverTransferOnly}` (debris) — DEFER Guided, RealmAnchored, InBand.
- **P2:** When the saga drives its first transition class, `step()` branches its action list by `DurabilityClass` from the start (only Player/Durable registered) — the tier is structural, not retrofitted. The Transient branch's batched `TransientGo` go-token path is stubbed-but-present so the commit point exists uniformly.
- **P3:** Add the stub `Debris` transient kind and the full `TransientTransferBatch` + `TransientGo` path end-to-end on stub shards alongside the Durable classes. Turn on TRANSIENT-CONSERVATION + TRANSIENT-AUTHORITY-HELD + DURABLE-UNAFFECTED-BY-BURST + `TransientDestOwnerCrash` in the crash/chaos matrix. Proves generic+tiered machinery under kill -9 BEFORE any voxel.
- **P5:** `Debris` gains rapier kinematics but transfers via Category-A closed-form ballistic re-advance (Category-C never recomputed cross-host).
- **P6:** Add `RealmAnchored` triple: `DroppedBlock::on_rest()` → durable two-step `LandBlock` ack'd against `block_wal` keyed by `entity_id`. `DroppedBlockLands` becomes a permanent durability gate. (This is a NEW triple → a real crash-matrix multiplication, scheduled here deliberately.)
- **P9:** **Cross-realm gameplay signals (thrust→hull, the HR1 example) are designed HERE as a typed channel** — NOT entity transfer, but reusing the same fenced-control-message + `(SignalId, step_id)` idempotency mechanism as block-edit forwarding (§A9). The shared mechanism is specified, not merely asserted.
- **P10:** Add `Guided`/`InBand` triples (Rocket) — the §A3 four-way resolution + the GuidedRocketTargetOnOtherShard×ship-transfer scenario. (NEW triples → crash-matrix multiplication, deliberately last with warp.)
- **P11:** Register `Projectile`/`Bullet` (`NeverTransferOnly`, `BallisticReadvance`) plus cross-shard `EntityId`-addressed damage with the §A5 transfer-barrier cut. They land as registry entries because the transient machinery exists from P3 — concrete proof of HR2/HR3.

---

## A9. How each Hard Rule is satisfied (restated honestly)

- **HR1 (no data leaks):** Inter-shard ENTITY MOVEMENT flows ONLY through typed `TlvBlob`s reached through `TransferableKind::serialize` returning a **typed `Blob` struct, not raw `Bytes`** — a CI test asserts no undeclared tag crosses (by-construction, not by-convention; this is the §A1 guard doing double duty). The dest reconstructs via `spawn` from declared tags only. **Honest scope:** the transfer charter satisfies HR1 *for entity movement*. The user's concrete HR1 example (ship thrust → system shard moves the hull) is a **gameplay signal between live co-existing shards, not entity transfer.** It is DEFERRED to the signals charter, but the SHARED mechanism is specified here concretely enough to be "implemented once": a `Signal{ signal_id: SignalId, from_realm, to_realm, src_realm_fence, payload: TypedSignal, source_tick }` fenced control message, idempotent on `(SignalId, step_id)` in `applied_steps`, riding the identical envelope family and realm-lease-Fence authorization as block-edit forwarding and the transient go-token. HR1 is satisfied by transfer + signals JOINTLY; the transfer charter does not over-claim sole satisfaction.
- **HR2:** Any kind implementing `TransferableKind` transfers — rockets, debris, blocks, constructions, bullets — through the identical batch/saga path. No per-kind transfer code (only per-kind blob/spawn/precondition/rebind, which IS the registry surface).
- **HR3:** One FSM, one envelope family, one Fence, one ghost transport, one registry, **one commit point (the Fence-CAS go-decision, batched for transients).** Durable vs Transient is a policy fan-out, not a fork — now literally, because transients pass through the same orchestrator go-decision. All shard types run identical transfer code, parameterized only by realm/registry data.
- **HR4:** Block placement / functional blocks / dropped-block conversion / guidance / signals all route through the same shard-agnostic registry + realm-owner-forwarding + realm-lease-Fence mechanisms, so a feature implemented once works on every shard type. **Honest scope:** the feature actually demonstrated working-everywhere by THIS charter is transfer + the dropped-block conversion + the signal mechanism; block-placement/functional-blocks/HUDs are routed through the same mechanisms by design (P6/P9) and gated by the shard-agnostic identical-fixture test (roadmap P9), not asserted.

---

## Attack resolutions (every fatal/major/minor, explicit)

**[FATAL] Region-ownership-implied transient authority violates AUTHORITY-UNIQUE (RealmId is whole-planet/system; realms overlap in the band; len==2; oracle can't read a non-existent row).** → §A2: transient authority is explicitly HELD in `owned_transients`, anchored to the shared region-shard-lease `Fence` (no per-bullet row, but a real linearizable anchor). A batched durable `TransientGo` go-token is the commit point. AUTHORITY-UNIQUE for transients is REDEFINED as len==1 across held-sets + a matching go-token, checked by ControlOracle on ground-truth held-sets/tokens; no-double-render checked by WireMonitor on delivered bytes. The spatial point-to-owner map the resolved key space cannot provide is no longer relied upon.

**[FATAL] Version discipline FALSE for postcard/bincode (older shards can't skip unknown trailing fields; cited decode() returns Default = silent data loss).** → §A1: TLV-framed blobs (tag/len/bytes; unknown tags skipped by `len`) + version-floor capability handshake (orchestrator refuses to route a Durable entity to a dest below the required-tag floor). decode→Default BANNED for Durable (hard ABORT_SPATIAL instead). New `kind_blob_evolution` CI gate tests older-reads-newer (the old same-version test could not). Verified against `core/src/character/handoff_blob.rs`.

**[MAJOR] Guided rocket reads frozen/fence-stale target across the host-transfer window → silent stale-tracking, not honest ballistic degrade.** → §A3: four-way resolution adds the "present but source_tick stalled > N_stale" case → degrade to ballistic-with-staleness-flag. Transient directory reads carry no fence expectation, read HEAD, tolerate one-tick misroute. New scenario crosses GuidedRocket with a concurrent ship-host transfer.

**[MAJOR] Batch exactly-once breaks on dest-realm-owner crash+reassign (stale dst_fence rejected by D'; in-memory dedup lost; items wedge or lost wider than declared).** → §A2: staleness-guard recovery (one HEAD hint-read refresh + retry once), then drop-with-reason within an explicitly-WIDENED loss budget ("a dest realm-owner crashing mid-handover loses the in-flight batch within budget"), never wedge the queue. New `TransientDestOwnerCrash` chaos cell.

**[MAJOR] "Same FSM, branches collapse" is a fork; transient path has no CAS/commit point → violates NO-UNILATERAL-COMMIT.** → §A2.3: a cheap, batched, group-committed orchestrator `TransientGo` Fence-CAS go-token per batch IS the single linearizable go-decision NO-UNILATERAL-COMMIT requires. Per-batch, not per-item, so the cost win holds. The invariant is NOT amended; the transient path was built to satisfy it. ONE machinery becomes literal.

**[MAJOR] DroppedBlock transient→durable conversion is non-atomic (despawn before BlockEdit durable = lost; resend without record = double-land; no durable idempotency key).** → §A3.1: durable two-step on the REALM OWNER keyed by the block's stable `entity_id` (durable/unique); R writes `block_wal` idempotently before acking; X despawns only on ack. Conversion is documented as the Transient→Durable boundary that MUST cross a durable ack.

**[MAJOR] Bullet damage across a transferring player double-counts or drops (additive damage made "idempotent set-to-final" is hand-waved; owning shard changes across the CAS).** → §A5: serialize damage through the transfer barrier using the COMMIT Fence/freeze_tick as the cut. Source drains+folds `hit_tick < freeze_tick` damage into shipped health and carries an `applied_damage: HashSet<DamageEventId>` in the blob; dest applies only `hit_tick >= freeze_tick`, deduped via the inherited set. Exactly one authoritative application.

**[MINOR] Solo-dev feasibility: the registry bounds SCHEMA variation, not BEHAVIORAL variation (4 ContinuityModels × 3 GhostPolicies × 2 DurabilityClasses = a Cartesian product of real code paths, each crash-tested).** → §A0/§A8: reframed honestly. Adding a kind WITHIN an existing triple is 4 lines; each NEW triple is a first-class engineering+test effort and a crash-matrix multiplication. Day-one surface capped to TWO triples ({Durable,Frozen,Always}, {Transient,BallisticReadvance,NeverTransferOnly}); Guided/RealmAnchored/InBand deferred to P6/P10 where their crash-matrix cost is budgeted.

**[MINOR] HR1 by-convention not by-construction; the thrust→hull signal example is scoped OUT and handed to an undesigned charter.** → §A9: `serialize` returns a TYPED `Blob` (not raw Bytes) with a CI no-undeclared-tag assertion (by-construction). The HR1 claim is tightened to "entity movement flows only through typed blobs." The signal mechanism is specified concretely here (fenced `Signal{SignalId,...}` control message, `(SignalId, step_id)` idempotency, same envelope/Fence as block-edit forwarding) so it is genuinely implemented-once, while honestly deferring the signals *charter*. HR1 is satisfied by transfer + signals jointly, stated plainly.

---

## Explicit trade-offs (genuine tensions, with recommendation)

1. **Transient exactly-once vs zero per-entity durable state.** A transient is in-memory by design; a dest-realm-owner crash mid-handover forfeits exactly-once for the in-flight batch. **Recommendation: accept, because it is now DECLARED, asserted (TRANSIENT-CONSERVATION budget), and chaos-tested.** Making bullets fully durable would reintroduce the 200-fsync/tick cost the tier exists to avoid; the loss is bounded to in-flight batches during a neighbor's crash and is gameplay-acceptable for bullets/debris (never for players, which are Durable). The boundary is the per-kind `DurabilityClass`, not a hidden assumption.

2. **The batched go-token adds one amortized orchestrator fsync per tick-with-transients.** Tension: a true commit point vs the "no orchestrator round-trip" cost goal. **Recommendation: accept the batched go-token.** It is per-batch and group-committed (one fsync amortized across all concurrent batches), so the per-entity cost is ~0, while it buys uniform NO-UNILATERAL-COMMIT and makes HR3's "ONE machinery" literal rather than rhetorical. The alternative — amending the binding invariant to scope transients out — weakens a safety property to save a fraction of an fsync, which is the wrong trade.

3. **Version-floor handshake can REFUSE a Durable transfer during a rolling deploy.** Tension: availability of transfers vs never silently zeroing a player. **Recommendation: refuse-and-keep-on-source (the player stays alive on the source shard) over decode→Default.** A briefly non-transferable player is recoverable; a Default-zeroed inventory is data loss. The floor is per-required-tag, so additive evolution never refuses; only a genuine required-field gap does, which is exactly when refusal is correct.
