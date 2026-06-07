## Connection Plane — FINAL hardened design (Voxeldust greenfield)

Scope: gateway, client netcode, client<->shard data flow. This plane OWNS the wire and the route table. It CONSUMES (does not implement) the Transfer Saga, Authority Directory, and identity-ticket format. Those three are made first-class, versioned, contract-tested seams (Attack-2.2 resolution). Reference code grounding the root causes was read and confirmed: `gateway/src/main.rs` (stateless redirector + `rand_u64()` from `SystemTime` = R7), `shard-common/src/quic_transport.rs` (`SkipServerVerification`, `recv_loop_sourced` writes sender ShardId rather than deriving from address = the pattern we generalize), `shard-common/src/client_listener.rs:164-175` (the IP-heuristic `peer_addr.ip()==udp_src.ip()` that silently drops all input = R2/R7), `core/src/handoff.rs` (`PlayerHandoff` god-struct + `ShardHandoff` client-promote + `GhostUpdate` create-and-hope = R2/R3/R4), `shard-common/src/handoff_pipeline.rs` (`DEFAULT_SOURCE_DEMOTE_TICKS=15` magic), `shard-common/src/harness.rs:176` + `celestial_clock.rs` (`(SystemTime::now()-epoch)*time_scale` per shard = the literal R6 clock-skew mechanism). quinn pinned at 0.11.9 (datagrams + fallible `send_datagram` + negotiated `max_datagram_size()`).

---

## 0. Topology — the one-connection invariant

```
                       ONE persistent QUIC conn per session
  +--------+   (multiplexed: CONTROL bidi + 1 input-datagram-class + N BULK + per-sub EVENTS + snapshot datagrams)
  | CLIENT |<====================================================>+---------+
  +--------+                                                       | GATEWAY |
       ^  client transport NEVER sees a shard, never opens /       |  (Gn)   |
       |  promotes / demotes a connection, never learns a          +----+----+
       |  shard IP, never participates in a transfer (kills R2,R3)      |
       |                                          pooled long-lived QUIC (>=1 conn per gateway<->shard
       |                                          pair, sessions hashed across a small pool; SessionId in-frame)
       |                  +---------+   +---------+   +---------+   +---------+
       |                  |  SHIP   |   | PLANET  |   | SYSTEM  |   | GALAXY  |
       |                  |  shard  |   |  shard  |   |  shard  |   |  shard  |
       |                  +---------+   +---------+   +---------+   +---------+
       v
  Client composites entities from 2-3 subscribed shards, each tagged by a monotonically
  increasing (never-reused) SubscriptionId, each carrying its reference_frame_id.
```

- **C1 (single connection):** opened once at login, closed only at logout / fatal error. A shard transfer is a server-side route swap; the client transport sees at most a cosmetic CONTROL notice.
- **C2 (no client transfer participation):** no client `Promote`/`Demote`/`ObserverConnect`/UDP-hole-punch. The entire `ConnectionMode`/`ControllerEvent`/observer machinery (`client/src/net/connection.rs`, 1022 lines) is deleted by design.
- **C3 (no IP-derived identity, anywhere):** server-side routing is always by an in-frame `SessionId`, never by source address. This permanently removes the `client_listener.rs:164` IP heuristic.

---

## 1. QUIC stream / datagram layout (REVISED — input is now datagrams)

### 1.1 Channel allocation

| Channel | QUIC object | Reliability | Dir | Count | Carries |
|---|---|---|---|---|---|
| CONTROL | bidi stream | reliable, ordered | both | exactly 1 | session lifecycle, auth, version negotiation, `SubscriptionOpened/Closing`, `AuthorityChanged`, `ReferenceFrameDef`, `TransferCosmetic`, the input **CUT_MARKER echo**, heartbeats, errors/close |
| INPUT | **datagrams** | **unreliable, seq-stamped** | C->G | n/a | per-tick `InputFrame{seq,...}` (latest-wins) + the in-band `is_cut_marker` token |
| BULK[sub] | uni stream (G->C) | reliable, ordered | G->C | one per active subscription | chunk snapshots/deltas, catalogs, block-config, grants, terminal scrollback |
| EVENTS[sub] | uni stream (G->C) | reliable, ordered | G->C | **one per active subscription** | per-subscription discrete gameplay events (damage, destroyed, chat) |
| SNAPSHOT | datagrams | unreliable | G->C | n/a | 20Hz `SnapshotFrame` per subscribed shard, tagged `SubscriptionId` + `frame_id` + `celestial_tick` |

**INPUT is now datagrams (Attack-1.7 / Attack-2.5 fatal-class resolution).** Rationale: input is absolute state (not deltas), idempotent, latest-wins — the exact datagram profile. A reliable ordered stream converts a single packet loss into a multi-tick HoL stall that then delivers a *burst of stale inputs* (visible rubber-band, strictly worse than UDP for a no-prediction movement game). With datagrams, loss means "skip a tick, apply the next fresh frame," which is what the old working UDP path did. The `seq` still gives the gateway dedup-on-reconnect and clean transfer cuts. The ONE thing that must not be lost — the transfer CUT_MARKER — is made loss-tolerant by being sent 3x consecutively AND echoed/confirmed on the reliable CONTROL stream (see 1.5 and 5).

**EVENTS is now per-subscription (Attack-1.minor EVENTS-HoL resolution).** A single session-global reliable stream let one shard's chat burst HoL-block another shard's time-critical `Destroyed`, and let `Destroyed{E}` (EVENTS) overtake `SubscriptionOpened{sub}` (CONTROL) so the client referenced an entity in a render layer that did not yet exist. Fix: each subscription owns its own EVENTS[sub] uni stream, opened by the gateway *causally after* it has sent `SubscriptionOpened{sub_id}` on CONTROL and *before* it forwards any event for that sub. Cross-stream invariant **X1**: no `EventMsg` for sub_id X is deliverable to the client before `SubscriptionOpened{X}` was processed; the client enforces this by only attaching the EVENTS[sub] reader after it has the sub in its map. Chat is BULK-only (scrollback) or EVENTS-only (live line) — never both; the double-listing is removed.

### 1.2 Framing

Reliable streams (length-delimited):
```
StreamFrame := u32_be total_len | u8 codec_flags | payload[total_len-1]
   codec_flags bit0 = serializer (0=postcard, reserved 1=bitcode-future)   bit1 = lz4 (bulk only)
```
We keep the 4-byte BE length + flag-byte shape from `core/src/wire_codec.rs`. **v1 always sets serializer=0 (postcard)** (Attack-2.4 resolution); the bit is reserved so bitcode can be added later without a wire break.

Datagrams (QUIC already delimits them):
```
InputDatagram    := u8 kind=INPUT    | u64_le seq | u8 flags(bit0=is_cut_marker) | u64_le client_tick | payload
SnapshotDatagram := u8 kind=SNAPSHOT | u32_le sub_id | u32_le frame_id | u64_le celestial_tick | u8 codec | payload
```
`sub_id` is **monotonically increasing per session, NEVER reused** (Attack-1.6 resolution): a 64-bit-effective space a session cannot exhaust. Plus the client drops any datagram whose `(sub_id, frame_id)` pair is not its current mapping, so a stale in-flight datagram is discarded, not miscomposited.

### 1.3 Prioritization, flow control, and BULK pacing (REVISED)

- quinn stream priorities: CONTROL > EVENTS[*] > BULK[*]. INPUT and SNAPSHOT are datagrams (no stream FC).
- Per-stream flow control isolates a stalled BULK (slow client draining chunks) from CONTROL/EVENTS; ordering is QUIC-enforced per stream, not comment-enforced.
- **BULK is application-paced per RTT (Attack-1.minor congestion-control + Attack-2.10 thrash resolution).** A fresh-subscribe chunk burst (multi-MB) is NOT dumped into the congestion window at once; it is spread over many ticks via a `bulk_budget_bytes_per_rtt` derived from quinn's live path RTT + bandwidth estimate (`Connection::rtt()`, congestion stats), NOT a hand-set window. This is the real fix — QUIC will pace datagrams behind a CC-filling stream burst otherwise, exactly during a warp/transfer when smooth snapshots matter most. The plane MEASURES and asserts in CI the datagram delivery ratio during a 4 MB BULK burst (real-quinn harness, Section 10). We do not assert "datagrams are not flow-controlled against streams" (false — they share the CC); we assert "BULK pacing keeps datagram delivery >= target ratio under a fresh-subscribe burst."

### 1.4 HoL analysis (REVISED)

| Scenario | Old | New |
|---|---|---|
| Chunk burst while moving | TCP HoL blocked all | BULK[sub] paced + isolated; INPUT/CONTROL/SNAPSHOT unaffected |
| Lost snapshot | UDP drop, fine | datagram drop, fine (latest-wins) |
| Lost input | UDP drop -> silent wedge (R1) | **datagram drop -> skip a tick, apply next fresh frame; seq dedup on reconnect; impossible to wedge AND no stale-burst** |
| Lost CUT_MARKER | n/a | sent 3x + confirmed on CONTROL; transfer cannot proceed without confirmation (5.2) |
| Control behind bulk | comment-ordered (R2) | separate prioritized stream; QUIC-enforced |
| Event overtakes its subscription | n/a | per-sub EVENTS opened causally after SubscriptionOpened (X1) |

### 1.5 The CUT_MARKER (new primitive, central to transfer correctness)
A CUT_MARKER is an `InputDatagram` with `is_cut_marker=1` at a specific `seq`. Because it travels on the SAME ordered logical input stream as movement frames (ordered by `seq`, not by wire arrival), the gateway routes by seq relative to the marker with zero cross-stream race (Attack-1.1 resolution). It is the ONLY input datagram that must not be lost; loss-tolerance is provided by triple-send + CONTROL-stream confirmation handshake (5.2).

---

## 2. Gateway internals

### 2.1 Per-session state (the route table) — REVISED for lock-free hot path

```rust
struct SessionState {
    session_id: SessionId,                 // 128-bit random (kills R7)
    session_gen: u32,                      // monotonic fence; bumped on every adoption (Attack-2.7)
    player_id: PlayerId,
    conn: quinn::Connection,
    // --- 20Hz hot-path state: lock-free atomic snapshot, read wait-free ---
    route: arc_swap::ArcSwap<RouteSnapshot>,  // {authority: ShardRef, seq_cut: Option<SeqCut>}
    last_input_seq: AtomicU64,             // dedup cursor (authoritative value owned by Directory on resume)
    // --- cold state: under a tokio Mutex, NEVER held across an await on the hot path ---
    cold: tokio::sync::Mutex<SessionCold>,
}
struct RouteSnapshot { authority: ShardRef, cut: Option<SeqCut> }   // SeqCut { marker_seq: u64, dest: ShardRef }
struct SessionCold {
    subscriptions: BTreeMap<SubscriptionId, Subscription>,  // <= ~4
    epoch: UniverseEpoch,
    transfer_in_progress: Option<TxnId>,   // mirrors Directory; gates reconnect (Attack-1.8/2.7)
}
struct Subscription { sub_id, shard: ShardRef, bulk_stream, events_stream, frame_id, state: Active|Draining }
```

**Hard rule (Attack-2.4 lock-discipline resolution):** the 20Hz input route decision and snapshot re-tag take NO async lock that any control path also takes. They read `route` via `ArcSwap::load` (wait-free) and `last_input_seq` via atomics. Commit publishes a new `RouteSnapshot` with a single atomic `store`. Control sends (Freeze/Resume) happen AFTER the store, off the hot path, awaited but holding no session lock. Torn-read is structurally impossible: authority + seq_cut live in ONE `Arc<RouteSnapshot>` swapped atomically (resolves the Attack-1.minor torn-read concern at the source).

Per-session memory target < 4 KB excluding quinn buffers.

### 2.2 Route swap during a transfer (REVISED — phased, compensatable, no atomic 2-of-2 fan-out)

The gateway does NOT decide transfers; the Transfer Saga (owner, P7) does. The gateway exposes idempotent, **separately-acked** commands, one per saga phase. There is NO single command that fans Freeze + Resume to two shards "atomically" (that was a 2-of-2 distributed write with no atomicity — Attack-1.3 fatal):

```
TransferControl (saga -> gateway):                  ack (gateway -> saga):
  PrepareSubscribe { txn, session, dest }    -->    Prepared { ghost_ready }      // dest ghost mirror open
  RequestCut       { txn, session }          -->    CutConfirmed { marker_seq }   // client emitted CUT_MARKER (5.2)
  FreezeSource     { txn, session }          -->    SourceFrozen { drained_seq }  // source applied <= marker, drained
  CommitAuthority  { txn, session }          -->    Committed                     // atomic route store -> dest
  ThawSource       { txn, session }          -->    SourceThawed                  // COMPENSATOR for FreezeSource
  AbortTransfer    { txn, session }          -->    Aborted                       // tears down dest ghost+buffer
  ReleaseSubscribe { txn, session, src }     -->    Released                      // closes src sub after demote grace
```

**Every phase has a defined compensator** (Attack-1.3 resolution): `FreezeSource`'s compensator is `ThawSource` (the missing Unfreeze the original lacked). The saga state machine is `{prepared -> cut-confirmed -> frozen -> committed -> released}` and on failure at any edge runs the compensator chain back to a safe state. The saga never advances to Commit until it holds `SourceFrozen{drained_seq}`; it never frees the source until it holds `Committed`. If `CommitAuthority` to dest fails (dest conn died), the saga runs `ThawSource` and `AbortTransfer` — the player is thawed on source, never stranded.

**Commit ordering (in-flight packets):**
1. Pre-Commit: dest already has the ghost (passive mirror, 4.2) and an open per-session command buffer. Input still flows to source.
2. `RequestCut` -> gateway tells client (CONTROL) "emit cut after your next input" -> client emits `InputDatagram{is_cut_marker}` at `marker_seq`. Gateway confirms `marker_seq` (5.2).
3. `FreezeSource`: gateway sets `route.cut = Some({marker_seq, dest})` via atomic store but keeps `authority=source`. Input routing: `seq <= marker_seq -> source`, `seq > marker_seq -> dest command buffer (not applied)`. Source applies through `marker_seq`, drains, acks `SourceFrozen{drained_seq=marker_seq}`.
4. `CommitAuthority`: single atomic store `route = {authority: dest, cut: None}`. Gateway sends dest `ResumeFrom{marker_seq+1}`; dest applies its buffered frames in seq order. **Directory records the authority flip ONLY now, on the Committed ack** (Attack-1.9 / Attack-2.7 resolution) — a crash before this leaves the Directory pointing at source, no split-brain.
5. In-flight snapshots: source ghost keeps the same sub_id; client keeps rendering source as authoritative until `AuthorityChanged` flips the authoritative sub for that player (4.3 invariant A1).

### 2.3 Backpressure
- C->G INPUT datagrams: never buffered; oversize -> dropped (latest-wins) and counted on a metric.
- G->C BULK: per-subscription byte budget from `TransportTuning` (2.7); exceeding for > T -> mark sub unhealthy -> notify saga (typed error, never `try_send`). The dwell/debounce rules (4.4) prevent the thrash-cascade where this aborts a healthy transfer.
- G->C SNAPSHOT datagrams: `send_datagram` is FALLIBLE in quinn. `SendDatagramError::TooLarge` is treated as a HARD BUG surfaced on a metric + loud log, never silent (Attack-2.3 resolution); `ConnectionClosed`/`Disabled` end the session path; a full send queue is the only "expected drop" (latest-wins). Startup asserts `conn.max_datagram_size() >= datagram_budget` and FAILS LOUD if the k3d/CNI overlay (VXLAN ~50B) clamped it below budget.

### 2.4 Crash recovery + resume ticket (REVISED — single-use, fenced, checkpoint-authoritative)

Gateways are soft-stateful; all authoritative truth lives in Authority Directory + shard state + persistence. The session map is reconstructable.

```
ResumeTicket := { session_id, player_id, epoch, session_gen, resume_nonce, issued_at }  ||  HMAC_gatewaykey(...)
```
- HMAC (cluster gateway secret, rotated) lets any gateway validate -> horizontal scale.
- **Single-use freshness (Attack-1.4 resolution):** the Directory stores a current `resume_nonce` per session. Resume requires `ticket.resume_nonce == Directory.current_nonce`; a successful resume CAS-bumps the nonce, invalidating every prior ticket. Replay of a captured/stale ticket is rejected. The Ed25519 login identity (or a short-lived derived session key) is RE-VALIDATED on Resume, not trusted on HMAC alone.
- **Generation fence against split-brain adoption (Attack-2.7 resolution):** the Directory's ownership answer includes `session_gen`. The adopting gateway CAS-bumps `session_gen`; shards and the prior gateway reject any input/command carrying a stale `session_gen`. This fences a stalled-but-alive old gateway during an L4-LB failover/partition: two connections cannot both drive the session, so C1 holds even under partition. Adoption is Directory-mediated eviction, not merely "pull-based on reconnect."
- **Input cursor is checkpoint-authoritative, NOT client-claimed (Attack-1.4 part 2 resolution):** on resume the gateway reads the authoritative resume seq from the shard's durable checkpoint via the Directory, and resets `last_input_seq` to it. It never trusts `ResumeTicket.last_input_seq` (which is why that field is removed). This prevents the "checkpoint rolled back to N, ticket says N+50, 50 ticks silently dropped by dedup" bug.
- **Reconnect during an in-flight transfer (Attack-1.8 / Attack-1.9 resolution):** if `Directory.transfer_in_progress(session) = Some(txn)`, the adopting gateway does NOT autonomously re-derive subscriptions. It registers with the saga ("I am the new gateway for session S, txn T") and the saga replays remaining steps THROUGH it. Single recovery authority = the saga. Combined with "authority recorded only at Committed" (2.2 step 4) and the durable seq-range interval map (5.3), recovery deterministically assigns every seq range to exactly one shard.

Reconnect is without game-state replay: the client re-subscribes and shards send a fresh snapshot + only the bulk it lacks (4.4 cache).

### 2.5 Horizontal scaling
N gateways behind an L4 LB. Soft affinity (sticky by connection); any gateway adopts via ResumeTicket + generation CAS. No gateway<->gateway chatter. Shared: HMAC secret (k8s secret), Authority Directory, orchestrator. Capacity unit = concurrent conns + aggregate datagram bandwidth.

---

## 3. Gateway<->shard transport (REVISED — small pool, saga-attested sessions)

**QUIC, a small POOL of long-lived connections per gateway<->shard pair** (not exactly one), sessions hashed across the pool, multiplexing per-session logical channels keyed by in-frame `SessionId`.

```
GatewayShardConn (pooled, K per (gateway,shard), K small, default from TransportTuning):
  control bidi : per-session lifecycle (AttachSession{saga_binding}, DetachSession, Freeze/Resume, prepare/commit acks)
  input  uni   : multiplexed InputFrame { session_id, session_gen, seq, payload }   gw->shard
  bulk   uni   : per-session-subscription, shard tags session_id          shard->gw
  events uni   : per-session-subscription                                  shard->gw
  snapshot dgr : { session_id, session_gen, sub_tag, celestial_tick, frame_id, payload }  shard->gw
```
- **Pool, not single conn (Attack-2.minor SPoF/HoL resolution):** one connection per pair concentrates risk — a shard redb fsync stall in its send path back-pressures ALL its sessions. We use a small pool (sessions hashed by `session_id`) so connection-level HoL blast radius is 1/K, and we keep the SessionId-in-frame routing unchanged so K is a pure tuning knob. We ALSO mandate the shard decouples redb persistence onto a separate task/thread so an fsync stall never stalls its QUIC send task, and expose per-(gateway,shard,conn) RTT + FC-blocked-time metrics so the 2am symptom is attributable.
- **SessionId + session_gen are the routing keys everywhere server-side**, written by the sender (the proven `recv_loop_sourced` pattern from `quic_transport.rs`, generalized). Kills the IP heuristic permanently.
- **mTLS** between gateway and shards (cluster CA), replacing `SkipServerVerification`.
- **Shard rejects spoofed sessions (Attack-1.minor gateway-trust resolution):** the shard maintains its own `SessionTable` seeded ONLY by the saga's `AttachSession{saga_binding}` (a saga-signed session binding). It rejects any input whose `session_id` is not in an Attached state OR whose `session_gen` is stale. A compromised/buggy gateway can therefore only spoof sessions the saga already attached to THAT shard — not arbitrary players. We document the residual blast radius (a compromised gateway impersonates only sessions currently homed on it) as inherent to the proxy pattern.
- Shard-side, the old `ClientRegistry{clients,observers,session_observers,pending_udp}` + IP heuristic collapses to `SessionTable: HashMap<SessionId, SessionSlot>` where a slot is `Authoritative | Ghost`. Promotion = a slot flip on `CommitAuthority`, not a write-half move.

---

## 4. Multi-shard subscription model (REVISED — bootstrap path, hysteresis, dwell, frozen ghosts)

### 4.1 Who decides + bootstrap (Attack-1.5 resolution)
Steady state: the player's AUTHORITATIVE shard computes interest (per change, debounced) and emits an `InterestSet` to the gateway, which diffs against current subs (P6).
**Bootstrap (the circular-dependency fix):** at first login and at warp into an on-demand-provisioned destination, there is no authoritative shard yet. A stateless **Spawn Resolver** (a pure function provided by the saga/Directory seam) computes the initial InterestSet from the player's durable checkpoint position. Once the first shard attaches and becomes authoritative, it takes over interest computation. Invariant **B1**: interest is always computed by SOMEONE — Spawn Resolver until first attach, then the authoritative shard.

### 4.2 Lifecycle + frozen-ghost rule (Attack-1.2 resolution)
```
InterestSet diff -> gateway:
  + add(shard): AttachSession{ghost, saga_binding} -> shard acks ghost-ready
                -> gw assigns NEXT monotonic sub_id, sends CONTROL SubscriptionOpened{sub_id,kind,frame_id}
                -> opens BULK[sub] + EVENTS[sub] -> begins forwarding snapshots tagged sub_id
  - remove(shard): CONTROL SubscriptionClosing{sub_id} -> stop datagrams -> drain+close BULK/EVENTS -> DetachSession
```
**A Ghost slot is FROZEN — kinematic, NO independent physics integration.** It advances ONLY by replaying authoritative pose deltas forwarded from the source via the gateway during prepare; it is a passive mirror, not a second simulation. This kills the "visibility fork" where source and dest both ran rapier3d on the player and the client received two divergent positions for one `player_id` across the whole hysteresis band (Attack-1.2). The ghost only becomes a live simulation at `CommitAuthority`.

### 4.3 Tagging, frames, and the single-authoritative-sub invariant
- Every snapshot datagram carries `sub_id`, `frame_id`, `celestial_tick`. Client maps `sub_id -> RenderLayer`.
- **Invariant A1 (Attack-1.2 resolution):** at most one `sub_id` is authoritative for any `player_id` at any instant from the client's compositing view. The authoritative sub for each `player_id` is carried in CONTROL `AuthorityChanged{player_id, sub_id}` and mirrored in the snapshot entity record; the client SUPPRESSES the ghost copy of that player from every other sub. No forked avatar, for self or for nearby observers.
- **Coordinate frames are time-parameterized functions, not sampled snapshots (Attack-2.minor R6-relocation resolution + R6):** the server sends, on CONTROL, a `ReferenceFrameDef{frame_id, orbital_elements | rigid_transform, validity_range}` — a deterministic function of the celestial-clock tick `t` (orbital elements via brahe, or a rigid parent transform). The client EVALUATES the frame at its exact interpolated render instant, so same elements + same `t` = same frame everywhere, with no wall-clock skew. This is the structural replacement for the old `(SystemTime::now()-epoch)*time_scale` per-shard formula (`harness.rs:176`) that produced the 110 km/s x ms = km error. Every `SnapshotFrame` binds to one authoritative `celestial_tick`; ALL subscribed shards share one celestial clock (required for determinism anyway — it is an integer tick counter advanced by the simulation, NOT read from each shard's wall clock). Frame updates carry a `validity_range` so a late CONTROL `ReferenceFrameDef` is applied at the correct tick, not on arrival. To composite ship-interior vs planet-surface at one render time, the client evaluates each frame's function at the common render tick (interpolated within the 100-150 ms buffer) — deterministic, sub-meter (asserted by a multi-shard skew-injection test, Section 10).

### 4.4 Hysteresis, dwell, and chunk cache (Attack-2.10 resolution)
- **Hysteresis is first-class:** interest uses `enter_radius > exit_radius` (P3 overlap band), entity/seed-derived, so a player hovering at a boundary does not flap the InterestSet.
- **Minimum subscription dwell:** a sub is held for `min_dwell_ticks` after it leaves the interest set before remove is honored (debounced at the gateway, like the demote grace). This stops the multi-MB re-stream + stream open/close churn at up to 20Hz for one oscillating player.
- **Client-side chunk cache keyed by `(shard, region, edit_epoch)`:** re-subscribe within the dwell window does NOT re-stream chunks the client already has — only the bulk it lacks. The resume path's "only the bulk it lacks" promise becomes the UNIVERSAL subscribe path.
- **[Failed] subscription recovery:** `Preparing -> Failed` (attach failed / dest not yet provisioned) triggers retry-with-backoff against the orchestrator; a saga-level timeout aborts the warp and returns the player to source. Invariant **B2 (Attack-1.5 resolution):** source authority is RETAINED until the dest `Committed` ack — source is never torn down on speculation, so a failed/slow provisioning returns the player to a live source, never "warps into nothing."

---

## 5. Input routing + seq-range ownership during a transfer (REVISED — durable interval map)

```
Client INPUT datagrams: InputFrame { seq:u64, is_cut_marker:bool, client_tick, payload }   (unreliable, latest-wins)

Gateway per InputFrame (lock-free, reads route via ArcSwap, dedups via AtomicU64):
  if seq <= last_input_seq: drop (dedup)                     // reconnect replay safety
  last_input_seq = max(last_input_seq, seq)
  let r = route.load();
  match r.cut {
    None                         => send to r.authority
    Some({marker_seq, dest}) =>
       if seq <= marker_seq      => send to source (FreezeSource bounds source apply at marker_seq)
       else                      => send to dest command buffer (applied only after CommitAuthority -> ResumeFrom)
  }
```

### 5.1 Exactly-once across the flip
The CUT_MARKER is a token IN the ordered input stream, so source gets everything strictly `<= marker_seq` and dest everything `>`, with zero cross-stream race (Attack-1.1 resolution). Wire is at-least-once + seq dedup; application is at-least-once-with-dedup across a crash (idempotent — input is absolute state, confirmed in `core/src/handoff.rs`/old `empty_input`). Worst case is re-applying one already-applied absolute input.

### 5.2 CUT_MARKER loss tolerance
Input is now datagrams (lossy), so the marker is protected: client emits it 3x consecutively at `marker_seq`, and the gateway CONFIRMS `marker_seq` back on the reliable CONTROL stream (`CutConfirmed`). The saga does not advance past `RequestCut` until it holds `CutConfirmed{marker_seq}`. If all three copies are lost, the client re-emits on the CONTROL-stream request retry. The cut cannot silently fail.

### 5.3 Durable seq-range ownership interval map (Attack-1.9 fatal resolution)
The cut point is the single source of truth for "which shard owns which seq range," and it is **durable in the saga/Directory, not just gateway RAM**. The Directory holds, keyed by `(session_id, txn)`, an interval map: `[.., marker_seq] -> source`, `(marker_seq, ..] -> dest`, recorded at the FreezeSource/Commit points. A given seq range is owned by exactly one shard for all time. On reconnect, the gateway reloads `(authority, cut, interval-map)` from the Directory — never from the client's ticket. Replay of frames `1007..1012` on a thawed source after a crash is REJECTED because the durable map already assigns that range to dest (or, if commit never recorded, the range was never owned by dest and source still owns it — deterministic either way). Per-shard `(session_id, seq)` dedup is retained as defense-in-depth, but correctness no longer depends on it spanning shards. "Exactly-once-ish" becomes "exactly-one-authority-per-seq-range, durably."

### 5.4 Abort path
If the saga ABORTs before Commit: gateway clears `route.cut` (atomic store back to `{authority:source, cut:None}`), discards the dest command buffer, runs `ThawSource` if it had frozen. Source continues from where it froze (or never froze). The interval map for that `txn` is discarded. Explicit, compensated, no stranded player.

---

## 6. Wire protocol

### 6.1 Taxonomy (four typed channel families, each a closed enum)
```
enum ControlMsg (CONTROL, reliable):
  C->G: Hello{login_ticket, proto_major, proto_minor, build_id} | Resume{resume_ticket} | InterestAck | CutEmitted{marker_seq} | Pong | Bye
  G->C: Welcome{session_id, session_gen, epoch} | SubscriptionOpened{sub_id, shard_kind, frame_id} |
        SubscriptionClosing{sub_id} | AuthorityChanged{player_id, sub_id} |
        ReferenceFrameDef{frame_id, kind(orbital_elements|rigid), params, validity_range} |
        RequestCut{txn} | CutConfirmed{marker_seq} | TransferCosmetic{kind} | Ping | Error{code} | Close{reason}
InputDatagram (unreliable): { seq, is_cut_marker, client_tick, movement, look, action_bits, seat_values }
enum BulkMsg (BULK[sub], reliable):
  G->C: ChunkSnapshot | ChunkDelta{edit_epoch} | BlockConfigState | Catalog | GrantsSnapshot | TerminalScrollbackDelta | SeatBindings
enum EventMsg (EVENTS[sub], reliable): G->C: Damage | Destroyed | Notice | ChatLine
SnapshotDatagram (unreliable): { sub_id, frame_id, celestial_tick, entities[], bodies[], lighting? }
```
The old single ~50-variant `ServerMsg`/`ClientMsg` (`core/src/client_message.rs`, 4613 lines) is replaced by these typed channels. The transfer messages smeared into `ServerMsg` (`ShardRedirect`/`ShardHandoff`/`ShardPreConnect`/`ShardDisconnectNotify`, all in `core/src/handoff.rs`) DO NOT EXIST on the client wire — they are internal saga<->gateway<->shard control. The `PlayerHandoff` god-struct (R4) is not a connection-plane type; the plane carries only opaque `TxnId` correlation (P8).

### 6.2 Serialization + versioning (REVISED — one serializer for v1)
- **v1: postcard everywhere** — CONTROL, INPUT, BULK, EVENTS, SNAPSHOT, gateway<->shard, saga (Attack-2.4 resolution). At dozens of players the bitcode size win is noise; postcard's `#[serde(default)]` field evolution is the safety you want while iterating. Maintaining two live codecs doubles the schema-evolution mental model and creates the mis-set-flag-bit footgun the design itself condemned (the `protocol-fb` drift that silently dropped `ShipColliderSync`, `quic_transport.rs:22-30`).
- `codec_flags` bit reserved (always 0) so bitcode can be added WITHOUT a wire break once a profiler proves snapshot bandwidth is the bottleneck. **FlatBuffers is removed from the plan entirely** until a funded non-Rust client requirement exists.
- **Version negotiation:** `Hello{proto_major, proto_minor, build_id}`. Gateway accepts `proto_major == own`; minor is forward/back compatible via postcard defaults. `proto_major` mismatch -> `Error{IncompatibleVersion}` + Close. High-freq channels are pinned to the negotiated version for the connection's life; a rolling upgrade drains old connections, new connections pick the new version — no mid-connection schema change ever.

### 6.3 MTU / datagram config (Attack-2.3 resolution)
- quinn datagram config is pinned + asserted at startup: `enable_datagrams`, `max_udp_payload_size`, `datagram_send_buffer_size` from `TransportTuning`; startup ASSERTS `conn.max_datagram_size() >= datagram_budget`, FAILING LOUD if the CNI overlay clamped it.
- `SnapshotFrame` is sized to a conservative budget (default 1200B, configurable). Oversize snapshots are split BY CONTENT into multiple independent `SnapshotDatagram`s (entities partitioned), same `frame_id`+`celestial_tick`, each self-contained latest-wins. **Partitioning lives in ONE shared `connection-plane` library function all shard types call** (preserves shard-agnostic standard; one fuzz test over entity counts vs budget) — not duplicated per shard. `SendDatagramError::TooLarge` is a hard bug on a metric, never silent.
- Inherently large bulk (chunk snapshots) goes on reliable BULK, never datagrams.

### 6.4 Encryption / auth integration points
- Client<->gateway: TLS 1.3, real verified server cert (no `SkipServerVerification`). App auth = Ed25519 `login_ticket` in `Hello` (identity charter owns format), validated before any subscription. Fixes R7.
- Gateway<->shard: mTLS (cluster CA). SessionId+session_gen conveyed in-frame, trusted only because the channel is mTLS-authenticated AND the saga attested the session to that shard (Section 3).
- Three explicit seams with typed errors: (1) `Hello` ticket -> auth-service pubkey; (2) ResumeTicket HMAC + nonce -> gateway secret + Directory; (3) per-frame SessionId trust -> mTLS peer + shard SessionTable.

---

## 7. Latency budget
Per-direction intra-cluster: added gateway<->shard ~0.3-1.0 ms one-way; gateway processing (lock-free route load + forward) ~5-50 us; snapshot re-tag is a header rewrite ~1-5 us. Total added ~<=1 ms each way, ~2 ms RTT worst case = <4% of a 50 ms tick, below the tick quantization floor. No client-side prediction (mandate): client renders interpolated state on a 100-150 ms buffer; +1-2 ms is inside the buffer noise. **Honest accounting:** the client<->gateway hop is WAN and IS where loss happens — which is precisely why INPUT and SNAPSHOT are now datagrams (fresh-under-loss), not reliable streams (stale-burst-under-loss). The gateway removes the old per-transition TCP reconnect/promote cost (5-50 ms handshakes per transition, `connection.rs`), so transition latency improves dramatically.

## 8. Failure-mode table (REVISED)

| Failure | Old | New |
|---|---|---|
| Lost input (R1) | UDP drop -> wedge | input datagram drop -> skip+next-fresh; seq dedup; no wedge, no stale-burst |
| Reliable-input HoL stall | n/a | eliminated — input is datagrams (1.1) |
| Client re-homes (R2) | races, IP heuristic | client never re-homes; SessionId+gen in-frame; no IP heuristic |
| Spawn-on-promote (R3) | eviction, pop | frozen ghost exists pre-commit (4.2); commit = slot flip |
| Visibility fork (Attack-1.2) | n/a | ghost is passive mirror; A1 single-authoritative-sub |
| seq_cut cross-stream race (Attack-1.1) | n/a | in-band CUT_MARKER token on ordered input stream |
| Freeze without Thaw strands player (Attack-1.3) | wedge (R1) | phased saga; ThawSource compensator; no atomic 2-of-2 fan-out |
| seq applied on two shards (Attack-1.9) | n/a | durable seq-range interval map per txn (5.3) |
| Resume replay / stale (Attack-1.4) | forged token (R7) | single-use nonce + gen fence + Ed25519 re-validate; cursor from checkpoint |
| Split-brain adoption (Attack-2.7) | n/a | monotonic session_gen CAS fence; Directory-mediated eviction |
| Reconnect vs in-flight saga (Attack-1.8) | n/a | saga is sole recovery authority; transfer_in_progress gate; authority recorded at Commit |
| sub_id reuse miscomposite (Attack-1.6) | (shard_type,seed) routing | never-reused monotonic sub_id + (sub_id,frame_id) drop guard |
| EVENTS HoL / overtake (Attack-1.minor) | n/a | per-sub EVENTS opened causally after SubscriptionOpened (X1) |
| Coordinate skew (R6 / Attack-2 relocation) | wall-clock per shard | one celestial tick + time-parameterized frame functions |
| Subscription thrash (Attack-2.10) | n/a | hysteresis + min dwell + chunk cache |
| Bootstrap deadlock (Attack-1.5) | n/a | stateless Spawn Resolver; source retained until dest Commit (B2) |
| Dropped inter-shard control (R9) | try_send | all gw<->shard control awaited + bounded timeout -> typed error -> saga |
| Forged identity (R7) | time-hash token | Ed25519 login + HMAC/nonce resume + 128-bit SessionId |
| Lossy 1Hz visibility (R8) | digests + hand timeouts | authoritative InterestSet diffs; subs are leases |
| Single gw<->shard SPoF (Attack-2.minor) | n/a | small pool + redb off send-path + health metrics |
| Datagram silent drop / MTU clamp (Attack-2.3) | n/a | pinned config + startup assert + TooLarge-is-a-bug metric |
| Lock-on-hot-path wedge (Attack-2.4) | n/a | ArcSwap/atomic route; no lock across awaited sends |
| No correlation/trace (R10) | grep | SessionId + TxnId on every frame/span (P8) |

## 9. State machines (REVISED)

Session (gateway): `New -> [Hello/Resume valid] -> Authenticating -> [auth ok] -> Active (self-loop: InterestSet diffs open/close subs) -> [conn lost/Bye/fatal] -> Draining(resume-grace) -> Closed`. Resume re-enters at Authenticating with nonce+gen check.

Subscription (gateway): `None -> [add] Preparing -> [ghost-ready] Active -> [remove after min_dwell | shard down] Draining -> [drained] Closed`. `Preparing -> [attach fail] Failed -> [backoff retry] Preparing | [saga timeout] aborts warp, source retained`.

Transfer/authority (per session, saga-driven, every phase compensatable):
```
Auth=Src
  -> PrepareSubscribe -> Prepared(dest frozen-ghost mirror)
  -> RequestCut -> CutConfirmed(marker_seq)
  -> FreezeSource -> SourceFrozen(<= marker_seq)              [compensator: ThawSource]
  -> CommitAuthority -> Committed(atomic route store; Directory records flip)  Auth=Dest
  -> ReleaseSubscribe(after demote grace) -> Released
any failure pre-Commit -> ThawSource (if frozen) + AbortTransfer -> Auth=Src (source never torn down, B2)
```

## 10. Test / observability (REVISED — two tiers + contract tests)
- **connection-plane is a LIBRARY crate**; the gateway binary is a thin shell (old shards were bin-only, untestable, R5).
- **Tier 1 — in-memory deterministic transport + virtual `Clock`:** drives saga route-swaps, aborts, gateway-crash-resume, cut correctness, fault injection (drop/delay/reorder) with zero k3d rebuild. Good for LOGIC; explicitly NOT trusted for wire behavior.
- **Tier 2 — real-quinn loopback harness with a lossy/MTU-clamped UDP shim, in CI (Attack-2.6 resolution):** asserts concrete wire guarantees — BULK stall does not delay CONTROL/datagrams; oversize datagram is rejected LOUD; backpressure budget triggers saga notification; negotiated datagram size >= budget; datagram delivery ratio under a 4 MB BULK burst meets target; input-loss yields skip-not-stall. Tier 1 never solely gates a release.
- **Contract tests both-sided (Attack-2.2 resolution):** a single shared crate defines `TransferControl`/`InterestSet`/ownership-query/`ResumeTicket` types; a conformance suite both the gateway and the (future) saga must pass. Until the real saga exists, an in-process saga STUB lives in the same binary so transfers are end-to-end exercisable without mocking the dangerous edges (lost CommitAuthority, saga crash between phases, stale Directory ownership). Sequencing: identity ticket frozen before gateway auth; Directory query API frozen before resume; saga command set frozen before any route swap.
- **Multi-shard compositing test:** injects skewed `celestial_tick`s across subs and asserts sub-meter frame alignment (4.3).
- Every log/span carries `session_id` and, during transfers, `txn_id` (P8/R10), via tracing structured fields.

## 11. Staged build plan (NEW — Attack-2.1 resolution: no front-loaded distributed-systems cliff)
The wire format is designed fully NOW; the gateway IMPLEMENTATION is staged so gameplay feedback arrives in week 1:
- **M0 walking skeleton:** gateway forwards exactly ONE subscription, ONE authority, NO transfer, NO resume, NO mTLS (plaintext dev cert), snapshots forwarded verbatim with `sub_id=0`, input datagrams forwarded verbatim. A player walks in one shard.
- **M1 multi-sub + interest:** InterestSet diffs, monotonic sub_ids, per-sub BULK/EVENTS, hysteresis+dwell, chunk cache.
- **M2 transfer saga integration:** PrepareSubscribe/RequestCut/FreezeSource/CommitAuthority/ThawSource/Abort/Release behind the contract-tested seam; frozen ghosts; durable interval map.
- **M3 durability:** ResumeTicket (nonce+gen), Directory-mediated adoption, checkpoint-authoritative cursor, mTLS, gateway-pool.
Each milestone is separately landable and testable behind the same frozen wire shapes.

## 12. TransportTuning (NEW — Attack-2.minor no-magic-numbers resolution)
A single typed `TransportTuning` struct (loaded from config/env, validated at startup) owns EVERY operational constant: `datagram_budget`, `interp_buffer_ms`, `reconnect_grace_ticks`, `per_sub_byte_budget` + `unhealthy_T`, `min_dwell_ticks`, `enter_radius`/`exit_radius` derivation, `gw_shard_control_timeout`, `resume_nonce_ttl`, `pool_size_K`, `bulk_budget_bytes_per_rtt` derivation, quinn datagram buffers, `proto_minor` window. Standard interpretation amended explicitly: "no magic numbers" governs WORLD/SIMULATION params (seed-derived) and ENTITY props (per-entity fields); TRANSPORT/OPERATIONAL params are NOT seed-derivable (MTU is a network property) and live in ONE reviewed, externally-tunable struct with documented derivation and startup-validated invariants (e.g. assert `datagram_budget <= max_datagram_size`). 2am tuning is a config change, not a recompile; the reviewer has a single audit surface.

## 13. What the client transport may know (enforces C2)
Knows: its one connection, opaque `session_id`, `sub_id -> {shard_kind, frame_id}` map, cosmetic transfer notices. Does NOT know: any shard IP, which shard is authoritative (beyond the cosmetic `AuthorityChanged` for camera/audio), when a transfer happens at transport level, how to open/promote/demote a connection. The client net module is ~one task: read CONTROL + per-sub BULK/EVENTS streams + SNAPSHOT datagrams, demux by sub_id into render layers, evaluate frame functions at render time, write INPUT datagrams at 20Hz, emit CUT_MARKER on request. No `ConnectionMode`, no controller, no observer logic — ~10x reduction from `connection.rs` (1022 lines).

---

## Attack resolutions (explicit)

- **[fatal] seq_cut wrong-clock cross-stream race (Attack-1.1):** RESOLVED. The cut is a client-emitted in-band `CUT_MARKER` token on the ordered input stream (1.5, 5.1). The gateway routes by seq relative to the marker; no cross-stream relationship between INPUT and saga-control is required. seq_cut is client-authoritative, totally ordered with movement frames.
- **[major] two authoritative writers / visibility fork (Attack-1.2):** RESOLVED. Ghost slots are FROZEN passive mirrors with no independent physics (4.2); invariant A1 names exactly one authoritative sub per player_id, carried in `AuthorityChanged` + snapshot record, client suppresses ghost copies (4.3).
- **[fatal] Freeze/Commit/Abort no mutual exclusion -> abort-after-freeze strands player (Attack-1.3):** RESOLVED. Phases are separate, separately-acked saga steps; `FreezeSource` has compensator `ThawSource`; Freeze and Resume are never fanned out as one atomic 2-of-2 write (2.2). Every partial state has a defined compensator.
- **[major] ResumeTicket replay / no freshness (Attack-1.4):** RESOLVED. Single-use `resume_nonce` in Directory (CAS-bumped per resume), Ed25519 re-validation on Resume, input cursor read from durable checkpoint not the ticket; `last_input_seq` removed from the ticket (2.4).
- **[major] InterestSet bootstrap deadlock (Attack-1.5):** RESOLVED. Stateless Spawn Resolver computes initial interest from the durable checkpoint until first attach (B1); `[Failed]` recovery with backoff + saga timeout; source retained until dest Commit (B2, 4.4).
- **[major] sub_id reuse miscomposite (Attack-1.6):** RESOLVED. sub_id is monotonically increasing, never reused; client drops datagrams whose `(sub_id, frame_id)` is not its current mapping (1.2).
- **[major] reliable INPUT HoL stall / stale-burst (Attack-1.7 / Attack-2.5):** RESOLVED. INPUT is now unreliable seq-stamped datagrams (latest-wins); loss = skip+next-fresh, not a stale burst (1.1). The false "reapply last input" claim is dropped; the only must-not-lose token (CUT_MARKER) is triple-sent + CONTROL-confirmed (5.2).
- **[major] gateway-crash-mid-transfer split recovery (Attack-1.8):** RESOLVED. Saga is sole recovery authority; `transfer_in_progress` gate; adopting gateway registers with the saga and replays through it; authority recorded in Directory only at Commit (2.4, 2.2).
- **[major] per-shard dedup lets one seq apply on two shards (Attack-1.9):** RESOLVED. Durable per-txn seq-range interval map in the Directory is the single source of truth; reconnect reloads ownership from Directory not ticket; replay on the wrong shard is rejected (5.3).
- **[minor] in-frame SessionId trusts a buggy/compromised gateway (Attack-1.minor):** MITIGATED + documented. Shard SessionTable seeded only by saga-attested AttachSession; rejects unattached/stale-gen sessions; residual blast radius (sessions homed on that gateway) documented as inherent (Section 3). Torn-read closed by single-Arc atomic route snapshot (2.1).
- **[minor] EVENTS HoL + Destroyed-before-SubscriptionOpened (Attack-1.minor):** RESOLVED. Per-subscription EVENTS streams opened causally after SubscriptionOpened; chat double-listing removed; invariant X1 (1.1).
- **[minor] connection-FC "sized so" magic / CC pacing (Attack-1.minor):** RESOLVED. BULK is application-paced per measured RTT (1.3); we assert measured datagram delivery ratio in CI rather than the false "datagrams not flow-controlled against streams" claim; budget derived from live RTT/bandwidth, not a hand-set window.
- **[major] gateway scope cliff (Attack-2.1):** RESOLVED via the M0-M3 staged build plan (Section 11); wire frozen now, implementation staged, gameplay in week 1.
- **[major] correctness defined in terms of nonexistent charters (Attack-2.2):** RESOLVED. Saga/Directory/identity seams are first-class shared-crate types with both-sided conformance tests + an in-process saga stub (Section 10).
- **[major] quinn datagram semantics glossed (Attack-2.3):** RESOLVED. Pinned datagram config + startup assert on negotiated size; `TooLarge` is a metric-tracked hard bug; content-partition in one shared library fn (6.3, 2.3).
- **[major] three serializers + codec_flags footgun (Attack-2.4):** RESOLVED. One serializer (postcard) for v1; bit reserved for future bitcode without a wire break; FlatBuffers removed (6.2).
- **[major] session lock across awaited sends (Attack-2.4 lock):** RESOLVED. Route is a lock-free `ArcSwap` snapshot + atomics; no lock held across network sends; hard rule documented (2.1).
- **[major] resume/adoption duplicate-authority window (Attack-2.7):** RESOLVED. Monotonic `session_gen` CAS fence; Directory-mediated eviction of the prior gateway; shards/old gateway reject stale-gen commands (2.4).
- **[major] in-memory harness false confidence (Attack-2.6):** RESOLVED. Tier-2 real-quinn loopback harness with lossy/MTU shim in CI; Tier-1 never solely gates release (Section 10).
- **[major] reference-frame relocates clock skew (Attack-2.8):** RESOLVED. Frames are time-parameterized orbital-element/rigid functions evaluated at one shared integer `celestial_tick`; validity ranges on late frame defs; multi-shard skew test asserts sub-meter (4.3). The old wall-clock `celestial_time_from_epoch` mechanism is abolished.
- **[major] per-tick interest thrash (Attack-2.10):** RESOLVED. First-class hysteresis (enter>exit), min subscription dwell, client chunk cache keyed by edit_epoch (4.4).
- **[minor] no-magic-numbers unenforceable for transport (Attack-2.minor):** RESOLVED. Single typed `TransportTuning` struct + amended standard interpretation (Section 12).
- **[minor] single gw<->shard conn = SPoF/HoL (Attack-2.minor):** RESOLVED. Small connection pool (1/K blast radius), redb persistence off the send-path, per-conn health metrics (Section 3).

## Unsolvable-tension trade-offs (surfaced with recommendation)
1. **Gateway-as-trust-boundary (inherent to the proxy pattern).** A compromised gateway can impersonate sessions currently homed on it; per-session cryptographic proof to shards would require the client to sign every input (re-introducing per-frame crypto cost at 20Hz and a client-held shard-facing key, contradicting C2). RECOMMENDATION: accept it, bound the blast radius via saga-attested SessionTable + session_gen (Section 3), monitor gateway integrity at the infra layer. Do not push per-input signing onto the client.
2. **Datagram input freshness vs guaranteed delivery of discrete actions.** Movement/look want latest-wins datagrams; block edits/interactions want exactly-once reliable. RECOMMENDATION (already in design): continuous movement/look + CUT_MARKER on INPUT datagrams; discrete world-mutating actions (block place/break, seat enter) carried as reliable BULK[sub] C->G action frames OR on CONTROL with their own seq+ack — so a lost block-edit is retried, while a lost movement frame is simply superseded. This is a minor v1.1 addition; M0 can ship movement-only on datagrams.