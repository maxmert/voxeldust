# Holistic /goal audit — full base vs end-goal (Jul 2026, post-SPIKE-3a HEAD ab8fae8)

**Workflow:** wf_43d2db17 — 7 opus audit lenses → adversarial verification (vs DEFERRED) → adjudicated synthesis.
**Verdict: NO CRITICAL FINDINGS — consider-done.** All 3 critical/high candidates that reached adversarial review
resolved to `is_real=true` but `is_critical=false` AND `is_already_ledgered=true`. The transfer-first inversion is
genuinely realized and structurally sound; the base is NOT painted into a corner on any end-goal pillar.

## Confirmed strengths (structural, not aspirational)

- **HR1 egress structurally sealed** — `InterShardFlow` is the SOLE shard-bound byte producer, closed enum with NO
  wildcard in `effect_class`/`durability_class`, guarded by three nested compile-time tripwires (arm/payload/ghost)
  forcing every new arm to be classified. The old R1/R9 fire-and-forget-with-no-owner class is UNREPRESENTABLE.
- **HR3 one-tooling by construction** — `build_app` inserts by capability never by NodeKind; saga branches on
  `DurabilityClass` only (ONE FSM); verified ZERO match-on-shard-kind/profile/geometry in sim/node/connection-plane
  feature code. "Shard type" is validated DATA (private fields, single fail-loud ctor, branchless `satisfies()`).
- **Kind registry exemplary table-driven DRY** — one `KindDef` per kind, exhaustive `def()/from_tag()`, machine-checked
  coherence + a registry-vs-enum drift tripwire. A new kind reusing a triple is 4 lines.
- **Transfer is genuinely ONE FSM** — pure `saga.rs` (proptested) vs impure `saga_runtime.rs` driver; Durable-vs-Transient
  is a policy branch inside the one `step()`, not a fork (ports-and-adapters; made the kill-9/chaos matrix buildable).
- **Untrusted-input decode uniformly cap-before-alloc, total, non-panicking** across every wire ingress; `mesh.rs` has
  zero prod-region unwrap/expect/panic. A forged oversize length header cannot OOM. The #1 property for a cloud MMO.
- **Redelivery ladder (incarnation/epoch/seq)** — pure, fail-loud, exhaustive at-least-once receiver; a restarted sender
  can't be silently deduped; a gap surfaces as `Verdict::Gap` not a silent dedup.
- **Fence discipline airtight** — `cas_next` is the sole commit point (saturating/total), source authority retained
  until dest wins, fence-neutral vs fence-bumping abort split pin-tested (FENCE-9).
- **Split-brain prevented by a config-time-VALIDATED typed timing invariant** — a mis-tuned deploy fails at BOOT not in
  production. **Crash safety uniformly fail-loud** — write-tmp→fsync→rename→fsync-dir, monotone boot-counter incarnation
  (immune to wall-clock rewind / sub-second CrashLoop), boot replay QUARANTINES orphans.
- **Every network pose sanitized** (NaN/Inf→0, non-finite orient→identity) at the single decode-ingress chokepoint.
- **Snapshot fan-out Arc-shared with a zero-decode varint re-tag** (5-byte splice, not a re-serialize) — per-subscriber
  cost is a fence compare + refcount bump + Vec push, NOT O(entities) re-encode. The remaining D-9 wall is byte-VOLUME.
- **Transport has no node-wide send serialization** — bounded mpsc + dedicated writer per peer, per-peer re-keyed RX
  ledger, wait-free ArcSwap reads (~625ns p99), bounded loud shedding (degrades by shedding, not unbounded growth).
- **Fail-loud cloud preflight** (allow-list, run first in every bin) + **partition-aware drain-safe readiness/liveness**
  dual to the D-3 self-fence (rcu-monotone drain edge, no Service flap).
- **No-prediction hard-collision authority structurally enforced** — only Owned entities integrate, ghosts are
  position-fed kinematic mirrors, a discrete authority event on a lossy datagram is UNCOMPILABLE (R3 two-owners class
  unrepresentable). The correct netcode foundation for contested cross-boundary PvP is already in the tree.
- **End-goal anti-corner posture real across EVERY pillar** — FrameRef(ShipLocal/PlanetCentered/SystemSpace/GalaxySpace)
  + RealmId carry all future frames; transfer_frame is FramePlacement-generic with the Coriolis term; VoxelGeometry
  Spherical/Cartesian is the ONE geometry seam; the client keys per-(SubId,EntityId) with a single world_pos
  compositing chokepoint (multi-mesh already shaped); `InterShardFlow::Signal` is a reserved forced-classifier arm.

## End-goal readiness (per pillar — all READY-additive, none cornered)

- **SIGNALS** — reserved `InterShardFlow::Signal` arm + capability-derived signal_graph + GalaxyRelay NodeKind; all
  three legs (cross-shard/client/scopes) additive & named (generic_transfer.md). Watch: P6's reliable client→shard
  discrete-action carrier (D-39.1) MUST be the same infra the P9 signal client-leg reuses (HR3).
- **BLOCKS** — blocks-live-in-voxels is a construction invariant (fail-loud); block_edit/surfaces/seats are capability
  booleans identical on Spherical+Cartesian (HR4 run-anywhere). Owed: the literal G-IDENTICAL fixture (D-38) with P6.
- **PLANETS** — VoxelGeometry is the sole geometry seam; FrameSpace/reanchor prose-planted (P4). Terrain unbuilt-by-design.
- **SHIPS (walk-inside-flying)** — ShipLocal frame + Coriolis + EffectFree coupling + compound-transfer envelope shaped;
  P8 adds bodies not reshapes. SPIKE-10a dual-frame interior physics is the gating de-risk.
- **MULTISHARDING** — most owed but additive: N-shard needs D-32 (coordinator_of/partition-by-region) + D-34 (per-session
  home field). Land D-34 before the k3d multi-shard load.
- **MULTI-MESH RENDER** — READY: client keys per-(SubId,EntityId), composes nested frames through ONE world_pos
  chokepoint; D-35 (version-matched compositing) is a wire field-append.
- **PvP** — no-prediction hard-collision authority enforced now; lag-comp rewind (D-42,P11) additive (rewind timestamps
  already on the wire).
- **HUNDREDS-IN-ONE-LOCATION** — the ONE pillar with real load-bearing debt, ledgered + additive: D-9 AoI wall + D-21
  single-thread rapier ceiling + D-41 intra-realm region split. Client wire stays frozen through the D-9 reshape;
  Arc-share survives. NOT cornered — but the owed N-in-one-realm load fixture (today's largest is 32 sessions) must
  land before any hundreds/dense-PvP soak claim, and the D-21/D-41 headcount SPIKE must precede the P5 physics commit.

## Non-critical findings (all ledgered — planned, not defects)

1. **D-9 within-realm AoI wall (HIGH, ledgered P6, coupled D-39.5)** — whole-realm O(entities×clients) snapshot broadcast;
   per-client bytes scale with realm POPULATION not visible-neighbor count. Additive P6 reshape (one-body-per-cell),
   client wire stays frozen. **Early-win PROBED + DECLINED:** the `partition_entities` double-encode (throwaway
   `to_allocvec` for `.len()`) could be a byte-count pass, but postcard 1.1.3 exposes `serialized_size` ONLY under its
   `experimental` namespace — inappropriate to depend on from the FROZEN wire crate — and the stable alternative
   (`serialize_with_flavor::<_, ser_flavors::Size, _>`) is verbose/less elegant. Not a bottleneck (SPIKE-3a proved the
   path sub-ms) and it dies in the D-9 reshape anyway → left ledgered; do it inside the D-9 P6 refactor (or if a clean
   stable API lands), not via an experimental dep now.
2. **D-41 region seam is prose-only (MEDIUM, ledgered, self-flagged guard-gap)** — the >1-anchor / region-range
   DirectoryKey / inter-region-band reservations are pinned only in doc-comments with NO exists-to-be-flipped tripwire
   test, so a P6 implementer could add a single-cluster hard error against a green suite and silently corner crowd-scaling.
   Fix (cheap): add the tripwire tests WITH the P4/P5 re-centering work, before single-* assumptions harden.
3. **No auto-pusher for a rescheduled BOOKED peer's new pod IP (LOW, ledgered S5/S6)** — `update_peer_addr` has no
   production caller; a booked peer that reschedules to a new IP is dialed stale forever until it re-dials first
   (bites only at a real non-k3d CNI; happy-path already gate-enforced by the live `l5_a_rescheduled_peer` test). Land
   the auto-pusher (in-process DNS re-resolve or an orchestrator watch) before the first real-cloud deploy.

**Action taken:** none required — NO critical findings. All 3 non-critical findings are ledgered at their correct
phase (D-9 double-encode → P6 reshape; D-41 tripwire tests → P4/P5; auto-pusher → pre-real-cloud-deploy). The D-9
early-win was probed and declined (experimental-only clean API; not a bottleneck; dies in the reshape). Base is
AAA-sound and not cornered on any pillar → consider done.
