//! RLM Step 5b — `SpawnCore`, the DECISION half of the real-process realm spawner.
//!
//! The [`vd_sim::io::RealmSpawner`] contract has two independent halves (spec
//! `scripts/rlm_step5_real_spawner_spec.md` §1.1): a DECISION kernel that owns the F2 monotone id/port
//! allocator, the durable launch ledger, and the live/killed bookkeeping (this module, Tier-A, vd-node);
//! and a LAUNCH BACKEND that actually forks/execs a `vd-shard` (or, later, admits a k8s pod) and books it
//! into the mesh (Tier-B, vd-bins, slice 5c). The seam between them is [`LaunchBackend`] — an object-safe
//! port so the decision logic is exercised to 100% against a deterministic fake here, and the OS shim
//! carries NO branching worth covering. This is the same DECIDE-vs-EXECUTE split the reconciler already
//! uses ([`crate::rlm_runtime`]) and the same layering fix the spec's §1.2 correction demands: vd-node
//! cannot reach `vd-io-prod`'s `MeshControl`, so peer-booking is delegated THROUGH the backend
//! ([`LaunchBackend::book_peer`]) rather than held as a mesh handle in the kernel.
//!
//! # What 5b owns
//!
//! - **F2 monotone allocation** — every realm gets a fresh `NodeId` and dev bind port from a cursor that
//!   only ever advances. A torn-down (or crashed) id is retired forever, so a stale dead-node latch can
//!   never shadow a new incarnation. The cursor is a DURABLE high-water ([`StoreKey::RlmWater`]) persisted
//!   BEFORE each mint and NEVER re-derived from `max(survivors)` — so even after every survivor's intent is
//!   deleted, the next mint still lands strictly above every id this spawner ever produced.
//! - **A write-ahead launch ledger** — one durable intent per realm ([`StoreKey::RlmLaunch`]), staged
//!   before the launch, so a rebuilt orchestrator reconstructs EXACTLY the children it minted pre-crash
//!   (the `live_nodes` "recognize my pre-crash pods" contract), then reconciles each against backend
//!   ground truth.
//! - **Backend-truth liveness** — `live_nodes` is the reconcile read: it probes each believed-live child
//!   against [`LaunchBackend::is_alive`] and PRUNES crashers (a k8s-kubelet-style ground-truth reconcile),
//!   caching the orphan so a corpse is never re-probed.
//!
//! # What 5b does NOT own (deferred, by the slice plan)
//!
//! - The OS process shim + pid/incarnation-cookie re-adoption (slice 5c).
//! - The ancestor-closure peer set + real host:port addressing across pods (slice 5d).
//! - Rebuilding the reconciler's `LaunchLedger` (node→path) from the recovered intents on restart (slice
//!   5e) — 5b delivers and proves the durable node→coord SUBSTRATE ([`SpawnCore::live_slots`]); 5e consumes
//!   it.
//!
//! Coverage: Tier-A — 100% region + branch (HR5). The generic `SpawnCore<B>` has a SINGLE instantiation in
//! this crate (the test `FakeBackend`), so covering every branch against the fake covers the whole
//! monomorphization; the real backend is a separate Tier-B binary.

use std::collections::{BTreeMap, BTreeSet};
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use serde::{Deserialize, Serialize};

use vd_core::incarnation::IncarnationCookie;
use vd_core::realm_coord::RealmCoord;
use vd_core::realm_path::RealmPath;
use vd_core::{NodeId, UniverseTick};
use vd_sim::io::{RealmSpawner, SpawnError, Store};

use crate::saga_runtime::{encode, rlm_launch_prefix, rlm_launch_store_key, rlm_water_store_key};

/// The operational parameters of the dev-scale allocator — the ONE config struct (no inline magic
/// numbers). Production/5d overrides every field; [`SpawnTuning::dev`] is the single-host loopback default.
#[derive(Clone, Copy)]
pub struct SpawnTuning {
    /// The first `NodeId` the F2 cursor mints. Chosen ABOVE the hand-picked scenario ids so a minted id
    /// never collides with a statically-planted one (same guidance as `MemSpawner::new`'s `first_node`).
    pub first_node: u64,
    /// The first dev bind port. On a single loopback host the cursor walks `u16`s; 5d spreads realms across
    /// pods/IPs so per-host port pressure never approaches exhaustion at real scale.
    pub first_port: u16,
    /// RLM 5f-4g: the EXCLUSIVE end of the bind-port band (a `u32` so it can name 65536 = "the whole u16
    /// space", the byte-identical `dev()` default). The F2 allocator refuses LOUDLY once a `(bind, probe)`
    /// pair would reach it — so a demand cluster's monotone allocator cannot walk out of its per-slot band
    /// into a neighbouring slot's ports (the launcher sets this from `SlotPorts::spawn_port_end`). Because
    /// the allocator never frees a port, this is the LIFETIME cap, not a concurrent one.
    pub port_limit: u32,
    /// The dev bind host every port hangs off (loopback). 5d generalizes to real per-pod addresses.
    pub bind_host: Ipv4Addr,
}

impl SpawnTuning {
    /// Base of the dev `NodeId` range — above any scenario's statically-picked ids.
    const DEV_FIRST_NODE: u64 = 1_000;
    /// Base of the dev loopback port range.
    const DEV_FIRST_PORT: u16 = 42_000;

    /// The single-host loopback defaults for local `just run`/tests. `port_limit` is the WHOLE u16 space
    /// (65536), so the raw default is byte-identical to the pre-5f-4g allocator; a demand LAUNCHER passes a
    /// bounded per-slot `port_limit` (`SlotPorts::spawn_port_end`) to keep the band inside its slot.
    #[must_use]
    pub const fn dev() -> SpawnTuning {
        SpawnTuning {
            first_node: Self::DEV_FIRST_NODE,
            first_port: Self::DEV_FIRST_PORT,
            port_limit: u16::MAX as u32 + 1,
            bind_host: Ipv4Addr::LOCALHOST,
        }
    }
}

/// The launch arguments the decision kernel hands the backend for one realm. The backend derives the
/// child's `ShardProfile` from `coord` (its `VD_OWN_COORD` env — same path `shard.rs` already walks in 5a),
/// so no profile is passed separately; `addr` is the kernel-allocated dev bind, `probe` the admin/health
/// bind, and `cookie` the pre-fork incarnation nonce the child echoes on its `/whoami` probe (Step 5e's
/// pid-reuse guard). All are minted by the kernel and passed IN so the backend carries zero policy.
pub struct LaunchSpec {
    /// The freshly minted id this realm will run under.
    pub node: NodeId,
    /// The realm's full lineage coord (→ the child's `VD_OWN_COORD`).
    pub coord: RealmCoord,
    /// The dev bind address the child listens on and the kernel books into the mesh.
    pub addr: SocketAddr,
    /// The admin/health bind (→ the child's `VD_PROBE_ADDR`); the Step-5e cookie-probe targets it.
    pub probe: SocketAddr,
    /// The pre-fork incarnation nonce (→ the child's `VD_INCARNATION_COOKIE`); persisted in the
    /// write-ahead intent BEFORE the fork, so a restart can identify this exact incarnation (Step 5e).
    pub cookie: IncarnationCookie,
    /// The child's `VD_PEERS` book (RLM Step 5d): the ANCESTOR CLOSURE — every live shard on `coord`'s
    /// `parent()`-to-root chain — plus the fixed anchors (orchestrator + gateway). Computed per-spawn by
    /// [`closure_peers`] from the kernel's live map (the ONLY component holding every live node's coord AND
    /// addr), so the child can dial its parent chain up to root without DNS. Excludes self/siblings/
    /// descendants (they resolve lazily via reply-on-connection). Rides this INTERNAL spec, NOT the frozen
    /// `RealmSpawner` seam.
    pub peers: Vec<(NodeId, SocketAddr)>,
}

/// The launch/teardown/liveness/booking port (RLM Step 5b) — object-safe, so [`SpawnCore`]'s decision logic
/// is covered against a deterministic fake and the real OS shim (5c) carries no coverable branching. Every
/// method is side-effecting against the outside world (fork/exec, signal, mesh booking); NONE decides
/// policy — the policy is entirely in [`SpawnCore`].
pub trait LaunchBackend: Send + Sync {
    /// Mint a fresh incarnation nonce for `node`, called by the kernel BEFORE the fork so it can be
    /// persisted in the write-ahead intent (Step 5c D3). The real backend draws std entropy; the fake is
    /// deterministic. Kept OUT of the kernel so the deterministic sim/node core never touches rng.
    fn mint_cookie(&self, node: NodeId) -> IncarnationCookie;

    /// Bring `spec`'s shard up. `Ok(pid)` is a confirmed-launched child whose OS pid the kernel records in
    /// the post-launch intent (Step 5e teardown/identity); `Err(reason)` is a REAL launch failure
    /// (fork/exec / admission) the kernel surfaces as [`SpawnError::LaunchFailed`] (→ the reconciler's
    /// backoff). The in-process fake never fails unless told to; the OS shim maps a failed spawn.
    fn launch(&self, spec: &LaunchSpec) -> Result<u32, String>;

    /// Re-adopt a survivor recovered from the durable launch ledger after an orchestrator RESTART (RLM
    /// Step 5e): a DIFFERENT process launched it, so the backend has no `Child` handle — it reconstructs an
    /// ORPHAN entry keyed on the persisted `(pid, cookie, probe)`, and `is_alive` then confirms it via the
    /// `/whoami` cookie-probe (the pid-reuse guard) rather than `try_wait`. Called by [`SpawnCore::rehydrate`]
    /// for each confirmed (`pid: Some`) survivor BEFORE the first `live_nodes` sweep. The fake records it;
    /// the OS shim builds the orphan slot.
    fn adopt(&self, node: NodeId, pid: u32, cookie: IncarnationCookie, probe: SocketAddr);

    /// Ground-truth liveness of a previously launched/adopted node — a crashed child returns `false`. Reads
    /// external state (a pid `try_wait`, or an adopted orphan's `/whoami` cookie-probe); the kernel
    /// reconciles `live_nodes` against it.
    fn is_alive(&self, node: NodeId) -> bool;

    /// Best-effort teardown of a node (signal / pod delete). The reconciler is the SOLE kill authority, so
    /// the kernel only calls this from `kill_realm`.
    fn teardown(&self, node: NodeId);

    /// Announce `node`'s reachable `addr` to the mesh (real: `MeshControl::update_peer_addr`; fake:
    /// record). Called ONCE per successful spawn and once per survivor on rehydrate — the layering fix that
    /// keeps the mesh handle OUT of the kernel (vd-node cannot depend on vd-io-prod).
    fn book_peer(&self, node: NodeId, addr: SocketAddr);
}

/// The durable launch-intent record (write-ahead of the launch). Self-describing (carries its own `node`)
/// so the rehydrate `scan` reconstructs the live set without parsing keys. Written TWICE per spawn (Step 5c
/// D3): a v1 record BEFORE the fork with `pid: None` (the write-ahead — a crash here cannot orphan a child
/// we never recorded), then a v2 record with `pid: Some` AFTER the launch confirms. `pid` is the ONE
/// `Option` (both arms reachable: `None` = crash between v1 and v2; `Some` = completed) — no impossible
/// cross-product, so rehydrate's match is HR5-coverable. `cookie`/`probe_port` are persisted for the
/// Step-5e cookie-probe (dormant in 5c: written, not yet read by the Owned-only rehydrate path).
/// Public (data only — no logic, so zero coverage burden) so the RLM 5e-5 process-tier crash gate can reopen
/// `launch.redb` and decode the recovered rows to assert on the ledger (row count, node id, pid) — reading
/// back through the SAME frozen postcard shape the spawner wrote, no re-implementation.
#[derive(Serialize, Deserialize)]
pub struct LaunchIntent {
    pub node: NodeId,
    pub coord: RealmCoord,
    pub port: u16,
    pub probe_port: u16,
    pub at_tick: UniverseTick,
    pub cookie: IncarnationCookie,
    pub pid: Option<u32>,
}

/// The durable F2 high-water — a SINGLE record (no payload key). Persisted BEFORE each mint so a rehydrate
/// resumes strictly above every id ever produced, independent of which survivors remain. `next_port` is a
/// `u32` cursor so the LAST valid `u16` bind port (65535) is itself allocatable — exhaustion is the mint
/// that would need 65536, not the one that uses 65535.
#[derive(Serialize, Deserialize)]
struct WaterMark {
    next_node: u64,
    next_port: u32,
}

/// One live realm's in-RAM record — the recovered node→realm binding the reconciler needs to map a
/// surviving pod back to its desired coord (slice 5e consumes it via [`SpawnCore::live_slots`]).
#[derive(Clone)]
pub struct LiveSlot {
    coord: RealmCoord,
    addr: SocketAddr,
    at_tick: UniverseTick,
}

impl LiveSlot {
    /// The realm this node serves (its full lineage coord).
    #[must_use]
    pub fn coord(&self) -> &RealmCoord {
        &self.coord
    }
    /// The node's booked mesh address.
    #[must_use]
    pub fn addr(&self) -> SocketAddr {
        self.addr
    }
    /// The universe tick the realm was spawned at (staleness accounting).
    #[must_use]
    pub fn at_tick(&self) -> UniverseTick {
        self.at_tick
    }
}

/// The mutable, lock-guarded interior — allocator cursors, the live/killed/dead sets, and the durable
/// store. All ordered collections (`BTreeMap`/`BTreeSet`): deterministic iteration, no default-hasher ban.
struct SpawnInner {
    /// F2 node cursor — only ever advances (persisted in [`WaterMark`]).
    next_node: u64,
    /// F2 dev port cursor (`u32` so the last `u16` port is usable) — only ever advances; exhaustion at
    /// `port_limit` is loud, never wrapped.
    next_port: u32,
    /// RLM 5f-4g: the EXCLUSIVE end of the bind-port band (from [`SpawnTuning::port_limit`]). A `(bind,
    /// probe)` pair that would reach it is refused LOUDLY — a demand cluster's monotone allocator stays
    /// inside its per-slot band. `dev()`'s 65536 keeps the raw default byte-identical to pre-5f-4g.
    port_limit: u32,
    /// The dev bind host every allocated port hangs off.
    bind_host: Ipv4Addr,
    /// Minted, launched, and not yet torn down — the reconcile source of truth for `live_nodes`.
    live: BTreeMap<NodeId, LiveSlot>,
    /// Commanded down via `kill_realm` (F2: an id here is retired; distinguishes `AlreadyKilled`).
    killed: BTreeSet<NodeId>,
    /// Crashed/orphaned (backend reported dead) — pruned out of `live` by the reconcile read and never
    /// re-probed (the "cache orphan probes" property). Observable via [`SpawnCore::orphan_count`].
    dead: BTreeSet<NodeId>,
    /// The durable launch ledger + high-water. Its committed contents survive an orchestrator kill-9.
    store: Box<dyn Store + Send + Sync>,
    /// The fixed anchors appended to EVERY child's `VD_PEERS` (RLM Step 5d): the orchestrator + gateway,
    /// which are never part of a realm lineage but every shard must reach. A construction input (a test
    /// fixture in 5d; the orchestrator fills the real `(ORCH, bind)`/`(GATEWAY, bind)` at 5e). The
    /// per-spawn ANCESTOR set is computed dynamically from `live`; only these static anchors are held here.
    anchor_peers: Vec<(NodeId, SocketAddr)>,
    /// RLM Step 5e (D3): a `RealmPath → NodeId` index over `live`, so [`closure_peers`] resolves each
    /// ancestor with an exact-key lookup — O(depth·log L) per spawn — instead of the O(depth·L) full scan
    /// that would go O(depth·K²) under a warp-burst subtree spin-up (`closure_peers` is on the spawn hot
    /// path once wired live). Kept in lockstep with `live` by [`SpawnInner::insert_live`]/[`remove_live`].
    /// INVARIANT: at most one live shard per realm path (the reconciler's idempotency-by-`coord.path` guard
    /// ensures it; `insert_live` overwrites, `remove_live` clears — so the index is always `live`'s
    /// path-projection).
    path_index: BTreeMap<RealmPath, NodeId>,
}

impl SpawnInner {
    /// Add a live shard, keeping `path_index` in lockstep with `live` (the ONLY insert path — so the index
    /// can never drift). Overwrites any prior entry for the coord's path (the one-shard-per-path invariant).
    fn insert_live(&mut self, node: NodeId, slot: LiveSlot) {
        self.path_index.insert(slot.coord.path().clone(), node);
        self.live.insert(node, slot);
    }

    /// Remove a live shard by id, clearing its `path_index` entry (the ONLY remove path). Callers
    /// (`kill_realm` after its `contains_key` guard, the `live_nodes` prune over `live.keys()`) ALWAYS pass
    /// a currently-live node, so the `.expect` panic body is in stdlib, not a coverable caller branch (HR5).
    fn remove_live(&mut self, node: NodeId) {
        let slot = self
            .live
            .remove(&node)
            .expect("remove_live called on a node that is not live");
        self.path_index.remove(slot.coord.path());
    }
}

/// The decision half of the real-process [`RealmSpawner`] (RLM Step 5b). Generic over the [`LaunchBackend`]
/// so the identical policy drives the deterministic fake (tests) and the OS shim (5c) — the reconciler
/// holds it as `&dyn RealmSpawner`. Cheap-cloneable handle (the state is `Arc<Mutex<..>>`), like
/// `MemSpawner`.
pub struct SpawnCore<B: LaunchBackend> {
    inner: Arc<Mutex<SpawnInner>>,
    backend: B,
}

impl<B: LaunchBackend> SpawnCore<B> {
    /// Genesis construction: an orchestrator starting with no durable launch history. Defined AS
    /// rehydrate-over-an-empty-store — one construction path (DRY): an empty store yields exactly the
    /// genesis cursors and an empty live set.
    #[must_use]
    pub fn new(
        store: Box<dyn Store + Send + Sync>,
        backend: B,
        tuning: SpawnTuning,
        anchor_peers: Vec<(NodeId, SocketAddr)>,
    ) -> SpawnCore<B> {
        SpawnCore::rehydrate(store, backend, tuning, anchor_peers)
    }

    /// Rebuild from a durable store after an orchestrator restart: resume the F2 high-water (NEVER
    /// `max(survivors)`), reconstruct the live set from the launch intents, and RE-ANNOUNCE each survivor's
    /// address to the mesh (the restart forgot peer addrs; the children themselves survived). The first
    /// `live_nodes` then reconciles the reconstructed set against backend truth, pruning any that died
    /// during the downtime.
    #[must_use]
    pub fn rehydrate(
        store: Box<dyn Store + Send + Sync>,
        backend: B,
        tuning: SpawnTuning,
        anchor_peers: Vec<(NodeId, SocketAddr)>,
    ) -> SpawnCore<B> {
        let mut store = store;
        let water = resume_water(&*store, &tuning);
        let mut live = BTreeMap::new();
        let mut path_index = BTreeMap::new();
        // v1-only intents (`pid: None`) are launches never confirmed before a crash — DROP them (Step 5c
        // D3/D4): reconstructing an unconfirmed launch risks a double-spawn, so 5c re-drives from scratch.
        // The narrow forked-but-crashed-before-v2 orphan window is DEFERRED to 5e (DEFERRED.md D-RLM-5),
        // where the cookie-probe sweep reaps it. F2 is unaffected — the high-water advanced in the v1
        // commit, so a dropped id/port is retired regardless.
        let mut partial = Vec::new();
        for (_key, value) in store.scan(&rlm_launch_prefix()) {
            let intent: LaunchIntent =
                postcard::from_bytes(&value).expect("decode persisted launch intent");
            let addr = SocketAddr::from((tuning.bind_host, intent.port));
            match intent.pid {
                Some(pid) => {
                    // RLM 5e-4: re-adopt the survivor (the backend has no `Child` for it post-restart) so
                    // `is_alive` confirms it via the `/whoami` cookie-probe; then re-announce its addr.
                    let probe = SocketAddr::from((tuning.bind_host, intent.probe_port));
                    backend.adopt(intent.node, pid, intent.cookie, probe);
                    backend.book_peer(intent.node, addr);
                    path_index.insert(intent.coord.path().clone(), intent.node);
                    live.insert(
                        intent.node,
                        LiveSlot {
                            coord: intent.coord,
                            addr,
                            at_tick: intent.at_tick,
                        },
                    );
                }
                None => partial.push(intent.node),
            }
        }
        for node in &partial {
            store.delete(&rlm_launch_store_key(*node));
        }
        if !partial.is_empty() {
            store.commit();
        }
        SpawnCore {
            inner: Arc::new(Mutex::new(SpawnInner {
                next_node: water.next_node,
                next_port: water.next_port,
                port_limit: tuning.port_limit,
                bind_host: tuning.bind_host,
                live,
                killed: BTreeSet::new(),
                dead: BTreeSet::new(),
                store,
                anchor_peers,
                path_index,
            })),
            backend,
        }
    }

    /// Rebuild the F2 ALLOCATOR CURSORS ONLY (the durable high-water) with an EMPTY live set — deliberately
    /// SKIPPING the survivor adopt-scan [`rehydrate`] runs. Resuming the high-water keeps F2 intact (the next
    /// mint is still strictly above every id ever produced, so NO id-reuse), but the empty live set means
    /// `launch_ledger_seed` comes up empty and the reconciler's minted-guard is NOT pre-seeded.
    ///
    /// This exists for the RLM 5e-5 crash-gate **D6 CONTROL arm**: a rebuilt orchestrator constructed this
    /// way re-issues a spawn for a coord whose survivor it did NOT adopt → a genuine DOUBLE-SPAWN — the
    /// falsifiable twin that proves [`rehydrate`]'s adopt path (populates `live` → seeds the guard) is
    /// LOAD-BEARING, not vacuous. It is a cursor-only constructor selected by a store-test-hooks env in the
    /// bin, NOT a shard-kind fork (HR3-neutral).
    #[must_use]
    pub fn water_only(
        store: Box<dyn Store + Send + Sync>,
        backend: B,
        tuning: SpawnTuning,
        anchor_peers: Vec<(NodeId, SocketAddr)>,
    ) -> SpawnCore<B> {
        let water = read_water(&*store).unwrap_or(WaterMark {
            next_node: tuning.first_node,
            next_port: u32::from(tuning.first_port),
        });
        SpawnCore {
            inner: Arc::new(Mutex::new(SpawnInner {
                next_node: water.next_node,
                next_port: water.next_port,
                port_limit: tuning.port_limit,
                bind_host: tuning.bind_host,
                live: BTreeMap::new(),
                killed: BTreeSet::new(),
                dead: BTreeSet::new(),
                store,
                anchor_peers,
                path_index: BTreeMap::new(),
            })),
            backend,
        }
    }

    fn lock(&self) -> MutexGuard<'_, SpawnInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The recovered node→realm bindings — the substrate slice 5e reads to rebuild the reconciler's
    /// `LaunchLedger` (node→path) after a restart. A snapshot clone (deterministic order).
    #[must_use]
    pub fn live_slots(&self) -> BTreeMap<NodeId, LiveSlot> {
        self.lock().live.clone()
    }

    /// How many minted children have been observed to crash (pruned orphans). Observability + the proof
    /// that a dead child is cached, not re-probed.
    #[must_use]
    pub fn orphan_count(&self) -> usize {
        self.lock().dead.len()
    }

    /// Project the live launch set into the reconciler's `LaunchLedger.minted` shape (`path → {node →
    /// at_tick}`) — the RLM Step 5e CRASH-RECOVERY SEED. Computed on the concrete `SpawnCore` BEFORE it is
    /// boxed as a `dyn RealmSpawner` and handed to the reconciler, so a restarted orchestrator's FIRST
    /// reconcile sweep already knows which coords are launched and does NOT re-spawn a survivor (the
    /// double-spawn guard's crash-recovery half — the RAM-only demand ledger comes up empty, so the seed is
    /// the ONLY thing suppressing a spurious re-spawn until demands re-accrue).
    ///
    /// SUPERSET of the pre-crash `minted` (vet D5), NOT byte-identical: the reconciler DRAINS `minted[path]`
    /// the moment a directory head is up, whereas `live` retains a node until it actually dies — so a
    /// survivor whose shard self-granted its head pre-crash had an empty `minted[path]` then, but its live
    /// slot re-seeds it here. Safe: the re-seeded head-up entry is drained on sweep-1 (the launch reconcile
    /// sees the head), and the empty post-crash demand ledger means no `SpinUp` is decided regardless.
    #[must_use]
    pub fn launch_ledger_seed(&self) -> crate::rlm_runtime::LaunchSeed {
        let g = self.lock();
        let mut seed: crate::rlm_runtime::LaunchSeed = BTreeMap::new();
        for (node, slot) in &g.live {
            seed.entry(slot.coord.path().clone())
                .or_default()
                .insert(*node, slot.at_tick);
        }
        seed
    }
}

impl<B: LaunchBackend> RealmSpawner for SpawnCore<B> {
    fn spawn_realm(&self, coord: &RealmCoord, at_tick: UniverseTick) -> Result<NodeId, SpawnError> {
        let mut g = self.lock();
        // F2 TWO-port allocation (bind + probe): loud on band exhaustion, NEVER wrapped (a wrap would reuse
        // a retired port → the exact stale-latch hazard F2 exists to prevent). Checked before minting, so an
        // exhausted allocator burns nothing. RLM 5f-4g: exhaustion is a `(bind, probe)` pair whose PROBE
        // (`cursor + 1`) would reach the EXCLUSIVE `port_limit` — a demand cluster's monotone allocator thus
        // cannot walk out of its per-slot band into a neighbour's ports. `dev()`'s 65536 limit keeps this
        // byte-identical to the pre-5f-4g "past `u16::MAX`" guard.
        let cursor = g.next_port;
        if cursor + 1 >= g.port_limit {
            return Err(SpawnError::LaunchFailed {
                reason: "dev port band exhausted".into(),
            });
        }
        let bind_port = cursor as u16;
        let probe_port = (cursor + 1) as u16;
        let next_port = cursor + 2;
        let node = NodeId(g.next_node);
        let addr = SocketAddr::from((g.bind_host, bind_port));
        let probe = SocketAddr::from((g.bind_host, probe_port));
        let cookie = self.backend.mint_cookie(node);

        // v1 write-ahead: persist the ADVANCED high-water + this node's intent (with `pid: None` and the
        // pre-fork cookie) BEFORE the launch, atomically. A crash between "child launched" and "pid
        // recorded" then leaves a v1-only record rehydrate DROPS (never a double-spawn). The water advances
        // even if the launch fails (the id/port pair is burned — F2 beats port thrift at dev scale).
        let water = WaterMark {
            next_node: g.next_node + 1,
            next_port,
        };
        g.store.put(&rlm_water_store_key(), &encode(&water));
        g.store.put(
            &rlm_launch_store_key(node),
            &encode(&LaunchIntent {
                node,
                coord: coord.clone(),
                port: bind_port,
                probe_port,
                at_tick,
                cookie,
                pid: None,
            }),
        );
        g.store.commit();
        // RLM 5e D1 (CRITICAL): the v1 intent+water must be DURABLE ON DISK before the child forks. `commit`
        // is block-on-PRIOR — it hands the batch to the off-tick writer and returns before ITS fsync — so
        // without this flush a kill-9 in the [fork issued .. v1 fsync] window would leave a live child with
        // NO recoverable intent (rehydrate reconstructs nothing → headless zombie) AND lose the id/port
        // water advance (→ F2 id-reuse → double-spawn). `flush` parks until v1 is fsync'd; a crash now
        // degrades to at most a `pid:None` partial rehydrate DROPS (D-RLM-5), never a no-intent orphan.
        g.store.flush();
        g.next_node += 1;
        g.next_port = next_port;

        // RLM 5d: the child's VD_PEERS book — the ancestor closure (from the live map, the ONLY holder of
        // every node's coord+addr) ∪ the fixed anchors. Computed BEFORE launch so the forked shard can dial
        // its parent chain immediately (no DNS). A branchless fill from the monomorphic helper (HR5).
        let peers = closure_peers(coord, &g.path_index, &g.live, &g.anchor_peers);
        let spec = LaunchSpec {
            node,
            coord: coord.clone(),
            addr,
            probe,
            cookie,
            peers,
        };
        match self.backend.launch(&spec) {
            Ok(pid) => {
                // v2: record the confirmed pid, so a restart can identify + tear down this exact
                // incarnation (Step 5e). Then book the peer BEFORE the node becomes visible as live, so the
                // reconciler never reads a live id the mesh cannot yet address.
                g.store.put(
                    &rlm_launch_store_key(node),
                    &encode(&LaunchIntent {
                        node,
                        coord: coord.clone(),
                        port: bind_port,
                        probe_port,
                        at_tick,
                        cookie,
                        pid: Some(pid),
                    }),
                );
                g.store.commit();
                self.backend.book_peer(node, addr);
                g.insert_live(
                    node,
                    LiveSlot {
                        coord: coord.clone(),
                        addr,
                        at_tick,
                    },
                );
                Ok(node)
            }
            Err(reason) => {
                // The write-ahead intent points at a child that never came up — delete it. The high-water
                // stays advanced (the id/port pair is retired, never retried under the same id).
                g.store.delete(&rlm_launch_store_key(node));
                g.store.commit();
                Err(SpawnError::LaunchFailed { reason })
            }
        }
    }

    fn kill_realm(&self, node: NodeId) -> Result<(), SpawnError> {
        let mut g = self.lock();
        if g.killed.contains(&node) {
            Err(SpawnError::AlreadyKilled(node))
        } else if !g.live.contains_key(&node) {
            Err(SpawnError::UnknownNode(node))
        } else {
            g.remove_live(node);
            g.killed.insert(node);
            g.store.delete(&rlm_launch_store_key(node));
            g.store.commit();
            self.backend.teardown(node);
            Ok(())
        }
    }

    /// ★ A LIVE REALM MOVED HOUSE (the ruler switch, slice 4): its persisted launch intent is
    /// rewritten with the new coord — `rehydrate` reads exactly that record, so a restart re-adopts
    /// the realm under its TRUE parent — and the live slot and the path index follow. The node, its
    /// ports and its cookie stay: the process did not move, only its place in the tree.
    fn addr_of(&self, node: NodeId) -> Option<([u8; 16], u16)> {
        let g = self.lock();
        g.live.get(&node).map(|slot| {
            let ip = match slot.addr.ip() {
                std::net::IpAddr::V4(v4) => v4.to_ipv6_mapped().octets(),
                std::net::IpAddr::V6(v6) => v6.octets(),
            };
            (ip, slot.addr.port())
        })
    }

    fn reparent_realm(&self, node: NodeId, coord: &RealmCoord) -> Result<(), SpawnError> {
        let mut g = self.lock();
        if !g.live.contains_key(&node) {
            return Err(SpawnError::UnknownNode(node));
        }
        let key = rlm_launch_store_key(node);
        let recorded = g
            .store
            .scan(&key)
            .into_iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| postcard::from_bytes::<LaunchIntent>(&v));
        let Some(Ok(intent)) = recorded else {
            return Err(SpawnError::LaunchFailed {
                reason: "no readable launch intent to rewrite".into(),
            });
        };
        g.store.put(
            &key,
            &encode(&LaunchIntent {
                coord: coord.clone(),
                ..intent
            }),
        );
        g.store.commit();
        g.store.flush();
        let mut slot = g.live.remove(&node).expect("checked live above");
        g.path_index.remove(slot.coord.path());
        slot.coord = coord.clone();
        g.insert_live(node, slot);
        Ok(())
    }

    fn live_nodes(&self) -> BTreeSet<NodeId> {
        let mut g = self.lock();
        // Reconcile the believed-live set against backend ground truth: any child the backend reports dead
        // has crashed (an orphan). Collect first (the probe borrows `self.backend` while `g` borrows the
        // state), then prune — moving each to `dead`, deleting its stale intent.
        let orphans: Vec<NodeId> = g
            .live
            .keys()
            .copied()
            .filter(|n| !self.backend.is_alive(*n))
            .collect();
        if !orphans.is_empty() {
            for n in orphans {
                g.remove_live(n);
                g.dead.insert(n);
                g.store.delete(&rlm_launch_store_key(n));
            }
            g.store.commit();
        }
        g.live.keys().copied().collect()
    }
}

/// The `VD_PEERS` book for a child at `coord` (RLM Step 5d): the ANCESTOR CLOSURE ∪ the fixed `anchors`.
///
/// Walks `coord.parent()` to the root and, for each ancestor whose `RealmPath` a `live` slot serves,
/// collects `(node, bind_addr)`; then appends `anchors` (the orchestrator + gateway — never part of a
/// lineage, always reachable). Excludes `coord` itself, siblings, descendants, and any ancestor not yet
/// live (those resolve lazily via reply-on-connection). Deterministic order (the `parent()` walk is
/// leaf→root; `live` lookups are exact; `anchors` in caller order) — no wall-clock, no rng.
///
/// MONOMORPHIC free helper (concrete types), so ALL branching (the walk loop + the found/absent arm) lives
/// OUTSIDE the generic `SpawnCore<B>` body — covered ONCE regardless of `B` (HR5). The generic `spawn_realm`
/// calls it as a straight-line fill.
///
/// SCALE (RLM 5e D3): each ancestor is resolved by an EXACT-KEY `path_index` lookup — O(depth·log L) per
/// spawn — not a full scan of `live` (which would be O(depth·L), i.e. O(depth·K²) under a warp-burst subtree
/// spin-up). `path_index` is `live`'s path-projection (kept in lockstep by `insert_live`/`remove_live`), so
/// the found slot's addr is a second exact lookup in `live`.
///
/// LIMITATION (ledgered to 5e/5f, DEFERRED.md D-RLM-6): the book is fixed at the child's boot — an ancestor
/// that is absent at spawn, or that later restarts under a NEW incarnation while this child stays live, is
/// NOT reachable by this already-running child (its dial lane is built once from `VD_PEERS`;
/// reply-on-connection cannot repair it because it needs the child to be the dialer and the child has no
/// addr/lane for the new incarnation). Correct ONLY under parent-first spawn ordering + no ancestor churn
/// under a live descendant; the refresh policy is the D-RLM-6 = C (lazy resolve-on-miss) 5f build.
fn closure_peers(
    coord: &RealmCoord,
    path_index: &BTreeMap<RealmPath, NodeId>,
    live: &BTreeMap<NodeId, LiveSlot>,
    anchors: &[(NodeId, SocketAddr)],
) -> Vec<(NodeId, SocketAddr)> {
    let mut peers = Vec::new();
    let mut ancestor = coord.parent();
    while let Some(a) = ancestor {
        if let Some(&node) = path_index.get(a.path()) {
            // `path_index` is `live`'s path-projection (the same `insert_live`/`remove_live` maintain both),
            // so the node is ALWAYS present in `live` — the `.expect` panic body is in stdlib, not a
            // coverable caller branch (HR5), so this is not an uncoverable `None` arm.
            let slot = live.get(&node).expect(
                "path_index node is always live (insert_live/remove_live keep them in lockstep)",
            );
            peers.push((node, slot.addr()));
        }
        ancestor = a.parent();
    }
    peers.extend_from_slice(anchors);
    peers
}

/// Read the durable F2 high-water, if any (the single [`StoreKey::RlmWater`] record). `None` on a virgin
/// store (genesis). A branchless helper — the `.expect` panic body is in stdlib, not a coverable branch
/// (HR5); the `.map` closure runs only when a record exists (covered by every rehydrate-with-history test).
fn read_water(store: &dyn Store) -> Option<WaterMark> {
    store
        .scan(&rlm_water_store_key())
        .into_iter()
        .next()
        .map(|(_key, value)| {
            postcard::from_bytes(&value).expect("decode persisted launch high-water")
        })
}

/// RLM 5f-4g: read the resume cursor (durable high-water, or genesis on a virgin store) and VALIDATE it
/// against the configured band — the ONE path both [`SpawnCore::rehydrate`] and [`SpawnCore::water_only`]
/// funnel through, so the band invariants are enforced at EVERY construction, never trusted. Two loud
/// guards (fail-fast on an impossible durable/config state, never a silent wrapped or out-of-band port):
///   * `port_limit <= 65536` — the exclusive band end must fit the u16 space, so every `as u16` cast in the
///     allocator is lossless (a larger limit would wrap the cursor to port 0 → the shard would bind — and so
///     GREET from — an OS-chosen ephemeral port, a mis-bound spawn).
///   * `first_port <= next_port <= port_limit` — the recovered cursor sits INSIDE its band (the top-inclusive
///     bound is a legitimately full band: the last mint leaves `next_port == port_limit`). Below the floor: a
///     store opened under a DIFFERENT (higher) `first_port` (an operator moved the band under a live store)
///     would mint BELOW its band — into a neighbouring slot's ports (a cross-slot port collision).
///     Above the ceiling (only reachable via a corrupt record — this module is the sole writer, whose max
///     `next_port` is `port_limit`): a near-`u32::MAX` cursor would overflow `cursor + 1` in a release build
///     (overflow-checks off) and wrap the guard, minting a garbage port. Both refuse LOUDLY (never clamp — a
///     retained water also carries a stale `next_node`, so a clamp would desync the (node, port) pairing).
///
/// Static assert messages (HR5: no format args → no uncoverable region); each false arm is pinned by a
/// `#[should_panic]` test, each true arm by every normal construction.
fn resume_water(store: &dyn Store, tuning: &SpawnTuning) -> WaterMark {
    assert!(
        tuning.port_limit <= u16::MAX as u32 + 1,
        "SpawnTuning port_limit exceeds the u16 port space"
    );
    let water = read_water(store).unwrap_or(WaterMark {
        next_node: tuning.first_node,
        next_port: u32::from(tuning.first_port),
    });
    assert!(
        water.next_port >= u32::from(tuning.first_port),
        "recovered launch high-water is below the configured band floor"
    );
    assert!(
        water.next_port <= tuning.port_limit,
        "recovered launch high-water is above the configured band ceiling"
    );
    water
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;
    use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};
    use vd_sim::io::mem::MemStore;

    // ---- a deterministic in-process LaunchBackend ------------------------------------------------

    #[derive(Default)]
    struct FakeInner {
        launched: Vec<NodeId>,
        /// The full `(node, bind, probe, cookie)` each `launch` received — proves the kernel threads the
        /// minted cookie + the two allocated ports INTO the backend (not re-derived there).
        launched_specs: Vec<(NodeId, SocketAddr, SocketAddr, IncarnationCookie)>,
        booked: Vec<(NodeId, SocketAddr)>,
        /// The `(node, pid, cookie, probe)` each `adopt` received — proves rehydrate re-adopts every
        /// confirmed survivor with the persisted identity (RLM 5e-4).
        adopted: Vec<(NodeId, u32, IncarnationCookie, SocketAddr)>,
        torn_down: Vec<NodeId>,
        alive: BTreeSet<NodeId>,
        probes: u32,
        fail_next: Option<String>,
    }

    /// Cheap-cloneable handle (state behind `Arc<Mutex<..>>`) so a test retains a view after moving one
    /// clone into the `SpawnCore` — mirrors `MemSpawner`/`MemHub`.
    #[derive(Clone, Default)]
    struct FakeBackend {
        inner: Arc<StdMutex<FakeInner>>,
    }

    impl FakeBackend {
        fn g(&self) -> MutexGuard<'_, FakeInner> {
            self.inner.lock().unwrap_or_else(PoisonError::into_inner)
        }
        /// Arm the NEXT `launch` call to fail with `reason`.
        fn fail_next(&self, reason: &str) {
            self.g().fail_next = Some(reason.to_string());
        }
        /// Externally kill a node (simulate a child crash): the backend now reports it dead.
        fn crash(&self, node: NodeId) {
            self.g().alive.remove(&node);
        }
        fn probes(&self) -> u32 {
            self.g().probes
        }
        fn book_count(&self) -> usize {
            self.g().booked.len()
        }
        fn last_booked(&self) -> (NodeId, SocketAddr) {
            *self.g().booked.last().expect("a booking happened")
        }
        fn torn_down(&self) -> Vec<NodeId> {
            self.g().torn_down.clone()
        }
        fn launched(&self) -> Vec<NodeId> {
            self.g().launched.clone()
        }
        /// The `(node, bind, probe, cookie)` of the most recent `launch` — proves seam threading.
        fn last_spec(&self) -> (NodeId, SocketAddr, SocketAddr, IncarnationCookie) {
            *self.g().launched_specs.last().expect("a launch happened")
        }
        /// Every `(node, pid, cookie, probe)` `adopt` received — proves rehydrate re-adopts each survivor.
        fn adopted(&self) -> Vec<(NodeId, u32, IncarnationCookie, SocketAddr)> {
            self.g().adopted.clone()
        }
    }

    impl LaunchBackend for FakeBackend {
        fn mint_cookie(&self, node: NodeId) -> IncarnationCookie {
            // Deterministic (no entropy) so the Tier-A tests are byte-stable — the real backend draws std
            // entropy. Keyed on the node so distinct spawns get distinct cookies.
            IncarnationCookie(u128::from(node.0))
        }
        fn launch(&self, spec: &LaunchSpec) -> Result<u32, String> {
            let mut g = self.g();
            if let Some(reason) = g.fail_next.take() {
                return Err(reason);
            }
            g.launched.push(spec.node);
            g.launched_specs
                .push((spec.node, spec.addr, spec.probe, spec.cookie));
            g.alive.insert(spec.node);
            // A deterministic fake pid (distinct per launch) — `unwrap_or` keeps the fn total for HR5.
            Ok(9_000 + u32::try_from(g.launched.len()).unwrap_or(0))
        }
        fn adopt(&self, node: NodeId, pid: u32, cookie: IncarnationCookie, probe: SocketAddr) {
            let mut g = self.g();
            g.adopted.push((node, pid, cookie, probe));
            // An adopted survivor is alive (the real backend confirms via the /whoami cookie-probe; the fake
            // models a successful probe) — so a rehydrate then reports it live without a manual `alive` prime.
            g.alive.insert(node);
        }
        fn is_alive(&self, node: NodeId) -> bool {
            let mut g = self.g();
            g.probes += 1;
            g.alive.contains(&node)
        }
        fn teardown(&self, node: NodeId) {
            let mut g = self.g();
            g.torn_down.push(node);
            g.alive.remove(&node);
        }
        fn book_peer(&self, node: NodeId, addr: SocketAddr) {
            self.g().booked.push((node, addr));
        }
    }

    // ---- fixtures --------------------------------------------------------------------------------

    /// A system coord `[Universe, Galaxy(g), System(s)]` (a real multi-level lineage, not a stand-in).
    fn system(g: u64, s: u64) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::Universe, 0),
            RealmLevel::new(RealmKindTag::Galaxy, g),
            RealmLevel::new(RealmKindTag::System, s),
        ]))
        .expect("3-level path has a leaf")
    }

    /// A tuning with small explicit bases so ids/ports are easy to assert on. `port_limit` is the WHOLE u16
    /// space (like `dev()`), so the allocator behaves exactly as pre-5f-4g for every existing test.
    fn tuning(first_node: u64, first_port: u16) -> SpawnTuning {
        SpawnTuning {
            first_node,
            first_port,
            port_limit: u16::MAX as u32 + 1,
            bind_host: Ipv4Addr::LOCALHOST,
        }
    }

    /// A tuning with an explicit BOUNDED port band `[first_port, port_limit)` — for the 5f-4g band-exhaustion
    /// proofs (a demand cluster passes such a per-slot band so its spawn ports stay inside its slot).
    fn banded_tuning(first_node: u64, first_port: u16, port_limit: u32) -> SpawnTuning {
        SpawnTuning {
            first_node,
            first_port,
            port_limit,
            bind_host: Ipv4Addr::LOCALHOST,
        }
    }

    fn addr(port: u16) -> SocketAddr {
        SocketAddr::from((Ipv4Addr::LOCALHOST, port))
    }

    fn core(backend: FakeBackend, tuning: SpawnTuning) -> SpawnCore<FakeBackend> {
        // Empty anchors: these kernel tests don't spawn under a live-ancestor lineage, so the VD_PEERS
        // closure is exercised directly in the `closure_peers_*` tests below (not through spawn_realm).
        SpawnCore::new(Box::new(MemStore::new()), backend, tuning, Vec::new())
    }

    const T: UniverseTick = UniverseTick(7);

    // ---- tests -----------------------------------------------------------------------------------

    #[test]
    fn genesis_is_empty_and_mints_from_the_configured_base() {
        let fake = FakeBackend::default();
        let sc = core(fake.clone(), tuning(1_000, 42_000));
        assert_eq!(sc.live_nodes(), BTreeSet::new());
        assert_eq!(sc.orphan_count(), 0);

        let a = sc.spawn_realm(&system(1, 1), T).expect("first spawn");
        assert_eq!(a, NodeId(1_000));
        assert_eq!(fake.launched(), vec![NodeId(1_000)]);
        assert_eq!(fake.last_booked(), (NodeId(1_000), addr(42_000)));
    }

    #[test]
    fn a_launched_nodes_address_is_answered_as_mapped_octets_and_a_strangers_is_not() {
        // THE PEER BOOK reads the live slot: IPv4 rides IPv4-mapped in sixteen octets.
        let fake = FakeBackend::default();
        let sc = core(fake.clone(), tuning(1_000, 42_000));
        let a = sc.spawn_realm(&system(1, 1), T).expect("spawn a");
        let (ip, port) = sc.addr_of(a).expect("launched here");
        assert_eq!(port, 42_000);
        assert_eq!(&ip[..12], &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff]);
        assert_eq!(
            std::net::Ipv4Addr::from([ip[12], ip[13], ip[14], ip[15]]),
            addr(42_000).ip()
        );
        assert_eq!(sc.addr_of(NodeId(9_999)), None, "a stranger has no slot");
    }

    #[test]
    fn spawn_mints_monotone_ids_and_books_each_peer_once() {
        let fake = FakeBackend::default();
        let sc = core(fake.clone(), tuning(1_000, 42_000));

        let a = sc.spawn_realm(&system(1, 1), T).expect("spawn a");
        let b = sc.spawn_realm(&system(1, 2), T).expect("spawn b");
        assert_eq!((a, b), (NodeId(1_000), NodeId(1_001)));

        // One booking per spawn; the two-port stride means each realm gets a bind port TWO above the last
        // (bind 42000/probe 42001 for a; bind 42002/probe 42003 for b).
        assert_eq!(fake.book_count(), 2);
        assert_eq!(
            sc.live_nodes(),
            BTreeSet::from([NodeId(1_000), NodeId(1_001)])
        );
        let slots = sc.live_slots();
        assert_eq!(slots[&NodeId(1_000)].addr(), addr(42_000));
        assert_eq!(slots[&NodeId(1_001)].addr(), addr(42_002));
        assert_eq!(slots[&NodeId(1_000)].coord(), &system(1, 1));
        assert_eq!(slots[&NodeId(1_000)].at_tick(), T);

        // The kernel minted the cookie + allocated the probe port and threaded BOTH into the backend
        // (bind and probe are the adjacent pair; the cookie is the deterministic mint of b's id).
        let (node, bind, probe, cookie) = fake.last_spec();
        assert_eq!(node, NodeId(1_001));
        assert_eq!((bind, probe), (addr(42_002), addr(42_003)));
        assert_eq!(cookie, IncarnationCookie(1_001));
    }

    #[test]
    fn launch_failure_burns_the_slot_and_touches_no_live_state() {
        let fake = FakeBackend::default();
        let sc = core(fake.clone(), tuning(1_000, 42_000));

        fake.fail_next("fork: EAGAIN");
        let err = sc.spawn_realm(&system(1, 1), T).expect_err("launch fails");
        assert_eq!(
            err,
            SpawnError::LaunchFailed {
                reason: "fork: EAGAIN".to_string()
            }
        );
        // No live state, no booking, the intent was rolled back.
        assert_eq!(sc.live_nodes(), BTreeSet::new());
        assert_eq!(fake.book_count(), 0);

        // The burned id/port pair is retired — the next spawn lands strictly above them (F2): id 1001, and
        // bind 42002 (the failed spawn consumed the 42000/42001 pair).
        let next = sc.spawn_realm(&system(1, 2), T).expect("retry spawns");
        assert_eq!(next, NodeId(1_001));
        assert_eq!(fake.last_booked(), (NodeId(1_001), addr(42_002)));
    }

    #[test]
    fn kill_is_a_two_arm_taxonomy_and_tears_down() {
        let fake = FakeBackend::default();
        let sc = core(fake.clone(), tuning(1_000, 42_000));
        let a = sc.spawn_realm(&system(1, 1), T).expect("spawn");

        assert_eq!(sc.kill_realm(a), Ok(()));
        assert_eq!(fake.torn_down(), vec![a]);
        assert_eq!(sc.live_nodes(), BTreeSet::new());

        // Killing again is AlreadyKilled; killing a never-minted id is UnknownNode.
        assert_eq!(sc.kill_realm(a), Err(SpawnError::AlreadyKilled(a)));
        assert_eq!(
            sc.kill_realm(NodeId(9_999)),
            Err(SpawnError::UnknownNode(NodeId(9_999)))
        );
    }

    #[test]
    fn live_nodes_prunes_a_crashed_child_and_caches_the_orphan() {
        let fake = FakeBackend::default();
        let sc = core(fake.clone(), tuning(1_000, 42_000));
        let a = sc.spawn_realm(&system(1, 1), T).expect("spawn a");
        let b = sc.spawn_realm(&system(1, 2), T).expect("spawn b");

        // A healthy reconcile keeps both (no orphan → no prune branch).
        assert_eq!(sc.live_nodes(), BTreeSet::from([a, b]));
        assert_eq!(sc.orphan_count(), 0);

        // A crashes out-of-band; the next reconcile drops it and records the orphan.
        fake.crash(a);
        assert_eq!(sc.live_nodes(), BTreeSet::from([b]));
        assert_eq!(sc.orphan_count(), 1);
        let probes_after_reap = fake.probes();

        // The corpse is cached — a further reconcile does NOT re-probe A (only the still-live B is probed).
        assert_eq!(sc.live_nodes(), BTreeSet::from([b]));
        assert_eq!(sc.orphan_count(), 1);
        assert_eq!(fake.probes(), probes_after_reap + 1);
    }

    #[test]
    fn a_reparented_realm_rehydrates_under_its_new_coord() {
        // The ruler switch, slice 4: the launch intent is the one durable record of a realm's parent
        // on the orchestrator's side. Rewritten at the crossing's commit, a restart reads the truth.
        let store = MemStore::new();
        let retained = store.clone();
        let live_fake = FakeBackend::default();
        let node = {
            let sc = SpawnCore::new(
                Box::new(store),
                live_fake.clone(),
                tuning(1_000, 42_000),
                Vec::new(),
            );
            let node = sc.spawn_realm(&system(1, 1), T).expect("spawn");
            sc.reparent_realm(node, &system(1, 2)).expect("reparent");
            assert_eq!(sc.live_slots()[&node].coord(), &system(1, 2));
            assert_eq!(
                sc.reparent_realm(NodeId(9_999), &system(1, 2)),
                Err(SpawnError::UnknownNode(NodeId(9_999)))
            );
            node
        };
        let sc = SpawnCore::rehydrate(
            Box::new(retained),
            FakeBackend::default(),
            tuning(1_000, 42_000),
            Vec::new(),
        );
        assert_eq!(
            sc.live_slots().get(&node).map(LiveSlot::coord),
            Some(&system(1, 2)),
            "the restart re-adopts the realm under its new parent"
        );
    }

    #[test]
    fn rehydrate_reconstructs_survivors_rebooks_them_and_recovers_the_binding() {
        let store = MemStore::new();
        let retained = store.clone(); // shares the committed WAL (the kill-9 analog: RAM dies, disk lives)
        let live_fake = FakeBackend::default();
        {
            let sc = SpawnCore::new(
                Box::new(store),
                live_fake.clone(),
                tuning(1_000, 42_000),
                Vec::new(),
            );
            sc.spawn_realm(&system(1, 1), T).expect("spawn a");
            sc.spawn_realm(&system(1, 2), T).expect("spawn b");
        } // orchestrator "dies" — RAM gone, the committed store survives via the retained clone.

        // A fresh backend (post-restart). rehydrate ADOPTS each survivor — no manual `alive` prime: adopt
        // is what marks a re-owned child live (the real backend confirms it via the /whoami cookie-probe).
        let fresh = FakeBackend::default();
        let sc = SpawnCore::rehydrate(
            Box::new(retained),
            fresh.clone(),
            tuning(1_000, 42_000),
            Vec::new(),
        );

        // Survivors recovered, each re-announced to the mesh, and the node→realm binding restored.
        assert_eq!(
            sc.live_nodes(),
            BTreeSet::from([NodeId(1_000), NodeId(1_001)])
        );
        assert_eq!(fresh.book_count(), 2);
        assert_eq!(sc.live_slots()[&NodeId(1_001)].coord(), &system(1, 2));

        // Each survivor was re-adopted with its PERSISTED identity — the pid, the minted cookie, and the
        // probe port from the ledger (the pid-reuse guard's inputs). pids are the fake's 9_000+launch-index;
        // cookies are the deterministic mint of each node id; probe ports are the odd half of each pair.
        assert_eq!(
            fresh.adopted(),
            vec![
                (NodeId(1_000), 9_001, IncarnationCookie(1_000), addr(42_001)),
                (NodeId(1_001), 9_002, IncarnationCookie(1_001), addr(42_003)),
            ]
        );

        // The F2 water resumed past the survivors — the next mint is strictly above them.
        let c = sc.spawn_realm(&system(1, 3), T).expect("spawn c");
        assert_eq!(c, NodeId(1_002));
    }

    #[test]
    fn rehydrate_resumes_past_deleted_survivors_never_maxplusone() {
        let store = MemStore::new();
        let retained = store.clone();
        let fake = FakeBackend::default();
        {
            let sc = SpawnCore::new(
                Box::new(store),
                fake.clone(),
                tuning(1_000, 42_000),
                Vec::new(),
            );
            sc.spawn_realm(&system(1, 1), T).expect("spawn a"); // 1_000
            sc.spawn_realm(&system(1, 2), T).expect("spawn b"); // 1_001
            let c = sc.spawn_realm(&system(1, 3), T).expect("spawn c"); // 1_002
            sc.kill_realm(c).expect("kill c"); // intent for 1_002 deleted
        }

        // Only A,B survive in the ledger; C's intent is gone. max(survivors)+1 would WRONGLY be 1_002.
        let fresh = FakeBackend::default();
        let sc = SpawnCore::rehydrate(Box::new(retained), fresh, tuning(1_000, 42_000), Vec::new());
        assert_eq!(
            sc.live_slots().keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([NodeId(1_000), NodeId(1_001)])
        );
        // The durable high-water resumes at 1_003 — strictly above C, the retired id, not above the
        // surviving max. This is the F2 guarantee a `max(survivors)+1` allocator would violate.
        let d = sc.spawn_realm(&system(1, 4), T).expect("spawn d");
        assert_eq!(d, NodeId(1_003));
    }

    #[test]
    fn dev_port_exhaustion_is_loud_and_changes_nothing() {
        let fake = FakeBackend::default();
        // Start the cursor so exactly ONE two-port pair fits (65534/65535) and the SECOND spawn — which
        // would need a port past u16::MAX — exhausts it.
        let sc = core(fake.clone(), tuning(1_000, u16::MAX - 1));

        let a = sc
            .spawn_realm(&system(1, 1), T)
            .expect("last port pair spawns");
        assert_eq!(a, NodeId(1_000));

        let err = sc
            .spawn_realm(&system(1, 2), T)
            .expect_err("port exhausted");
        assert_eq!(
            err,
            SpawnError::LaunchFailed {
                reason: "dev port band exhausted".to_string()
            }
        );
        // The exhausted attempt burned nothing: only the first child is live, only it was booked.
        assert_eq!(sc.live_nodes(), BTreeSet::from([NodeId(1_000)]));
        assert_eq!(fake.book_count(), 1);
    }

    #[test]
    fn a_bounded_band_exhausts_loudly_at_its_own_end_not_u16_max() {
        // RLM 5f-4g: a per-slot demand band far below u16::MAX. Band [42_000, 42_004) holds exactly TWO
        // (bind, probe) pairs — 42000/42001 and 42002/42003 — the THIRD spawn (cursor 42_004, probe 42_005
        // ≥ 42_004) must exhaust LOUDLY, never walking into a neighbouring slot's ports.
        let fake = FakeBackend::default();
        let sc = core(fake.clone(), banded_tuning(1_000, 42_000, 42_004));

        let a = sc.spawn_realm(&system(1, 1), T).expect("pair 0 fits");
        let b = sc.spawn_realm(&system(1, 2), T).expect("pair 1 fits");
        assert_eq!((a, b), (NodeId(1_000), NodeId(1_001)));

        let err = sc
            .spawn_realm(&system(1, 3), T)
            .expect_err("band exhausted below u16::MAX");
        assert_eq!(
            err,
            SpawnError::LaunchFailed {
                reason: "dev port band exhausted".to_string()
            }
        );
        // Exactly the band's capacity is live; the third attempt burned nothing.
        assert_eq!(
            sc.live_nodes(),
            BTreeSet::from([NodeId(1_000), NodeId(1_001)])
        );
        assert_eq!(fake.book_count(), 2);
    }

    #[test]
    #[should_panic(expected = "above the configured band ceiling")]
    fn an_inverted_band_cannot_be_constructed() {
        // RLM 5f-4g: port_limit < first_port is a misconfiguration — genesis water (first_port) exceeds the
        // band ceiling, so construction refuses loudly rather than yielding a dead allocator.
        let _ = core(FakeBackend::default(), banded_tuning(1_000, 42_000, 41_000));
    }

    /// Build a `MemStore` carrying a single durable high-water record (the F2 resume cursor) for the
    /// band-guard `#[should_panic]` proofs.
    fn store_with_water(next_node: u64, next_port: u32) -> MemStore {
        let store = MemStore::new();
        let retained = store.clone();
        let mut s = store;
        s.put(
            &rlm_water_store_key(),
            &encode(&WaterMark {
                next_node,
                next_port,
            }),
        );
        s.commit();
        retained
    }

    #[test]
    #[should_panic(expected = "port_limit exceeds the u16 port space")]
    fn a_port_limit_above_the_u16_space_is_rejected_at_construction() {
        // RLM 5f-4g: a hand-rolled tuning whose band end exceeds 65536 would silently wrap the `as u16` mint
        // casts to port 0 — rejected LOUDLY at construction instead.
        let _ = core(FakeBackend::default(), banded_tuning(1_000, 65_000, 70_000));
    }

    #[test]
    #[should_panic(expected = "below the configured band floor")]
    fn a_recovered_water_below_the_band_floor_is_rejected() {
        // RLM 5f-4g: a durable store opened under a HIGHER first_port (an operator moved the band) would else
        // mint below its band. The floor guard refuses loudly — never clamps (a stale next_node would desync).
        let store = store_with_water(1_005, 41_000); // next_port 41_000 < first_port 42_000
        let _ = SpawnCore::new(
            Box::new(store),
            FakeBackend::default(),
            tuning(1_000, 42_000),
            Vec::new(),
        );
    }

    #[test]
    #[should_panic(expected = "above the configured band ceiling")]
    fn a_recovered_water_above_the_band_ceiling_is_rejected() {
        // RLM 5f-4g: a corrupt high-water past the band ceiling (the sole writer never exceeds port_limit)
        // would risk a `cursor + 1` overflow wrap in a release build — refused loudly on resume.
        let store = store_with_water(1_005, 42_100); // next_port 42_100 > port_limit 42_064
        let _ = SpawnCore::new(
            Box::new(store),
            FakeBackend::default(),
            banded_tuning(1_000, 42_000, 42_064),
            Vec::new(),
        );
    }

    #[test]
    fn a_completed_spawn_persists_the_cookie_and_confirmed_pid() {
        let store = MemStore::new();
        let retained = store.clone();
        let fake = FakeBackend::default();
        let sc = SpawnCore::new(Box::new(store), fake, tuning(1_000, 42_000), Vec::new());
        let a = sc.spawn_realm(&system(1, 1), T).expect("spawn");

        // The durable intent the kernel committed carries the PRE-FORK cookie, the confirmed pid, and the
        // allocated probe port (the v2 write-back) — exactly what Step 5e needs to identify + tear down
        // this incarnation after a restart.
        let recs = retained.scan(&rlm_launch_store_key(a));
        assert_eq!(recs.len(), 1, "one intent record for the spawned node");
        let intent: LaunchIntent =
            postcard::from_bytes(&recs[0].1).expect("decode persisted intent");
        assert_eq!(intent.cookie, IncarnationCookie(1_000));
        assert_eq!(intent.pid, Some(9_001));
        assert_eq!((intent.port, intent.probe_port), (42_000, 42_001));
    }

    #[test]
    fn rehydrate_drops_partial_intents_and_keeps_completed_ones() {
        // A store as it would look after a crash: one COMPLETED survivor (`pid: Some`, both write-ahead and
        // confirm landed) and one PARTIAL (`pid: None`, crashed between the v1 and v2 commits).
        let store = MemStore::new();
        let retained = store.clone();
        {
            let mut s = store;
            s.put(
                &rlm_launch_store_key(NodeId(1_000)),
                &encode(&LaunchIntent {
                    node: NodeId(1_000),
                    coord: system(1, 1),
                    port: 42_000,
                    probe_port: 42_001,
                    at_tick: T,
                    cookie: IncarnationCookie(1_000),
                    pid: Some(9_001),
                }),
            );
            s.put(
                &rlm_launch_store_key(NodeId(1_001)),
                &encode(&LaunchIntent {
                    node: NodeId(1_001),
                    coord: system(1, 2),
                    port: 42_002,
                    probe_port: 42_003,
                    at_tick: T,
                    cookie: IncarnationCookie(1_001),
                    pid: None,
                }),
            );
            s.commit();
        }

        let fresh = FakeBackend::default();
        let sc = SpawnCore::rehydrate(
            Box::new(retained.clone()),
            fresh.clone(),
            tuning(1_000, 42_000),
            Vec::new(),
        );

        // The completed survivor is reconstructed + rebooked; the partial is NOT reconstructed (no
        // double-spawn hazard) and is booked zero times.
        assert_eq!(
            sc.live_slots().keys().copied().collect::<BTreeSet<_>>(),
            BTreeSet::from([NodeId(1_000)])
        );
        assert_eq!(fresh.book_count(), 1);

        // ONLY the completed survivor was adopted (with its persisted pid/cookie/probe) — the partial is
        // never adopted, so adopt is the one thing that distinguishes a re-owned child from a dropped one.
        assert_eq!(
            fresh.adopted(),
            vec![(NodeId(1_000), 9_001, IncarnationCookie(1_000), addr(42_001))]
        );

        // The partial's stale intent was DELETED from the durable store; only the survivor's remains.
        let remaining: BTreeSet<Vec<u8>> = retained
            .scan(&rlm_launch_prefix())
            .into_iter()
            .map(|(k, _)| k)
            .collect();
        assert_eq!(
            remaining,
            BTreeSet::from([rlm_launch_store_key(NodeId(1_000))])
        );
    }

    #[test]
    fn water_only_resumes_the_cursor_but_recovers_no_live_children() {
        // RLM 5e-5 D6 CONTROL substrate: a store carrying an ADVANCED water + one COMPLETED (pid:Some)
        // survivor. `water_only` resumes the F2 cursor ABOVE the survivor (so F2 still holds — NO id-reuse)
        // but DELIBERATELY skips the adopt-scan, so nothing is recovered live / adopted / booked.
        let store = MemStore::new();
        let retained = store.clone();
        {
            let mut s = store;
            s.put(
                &rlm_water_store_key(),
                &encode(&WaterMark {
                    next_node: 1_001,
                    next_port: 42_002,
                }),
            );
            s.put(
                &rlm_launch_store_key(NodeId(1_000)),
                &encode(&LaunchIntent {
                    node: NodeId(1_000),
                    coord: system(1, 1),
                    port: 42_000,
                    probe_port: 42_001,
                    at_tick: T,
                    cookie: IncarnationCookie(1_000),
                    pid: Some(9_001),
                }),
            );
            s.commit();
        }

        let fresh = FakeBackend::default();
        let sc = SpawnCore::water_only(
            Box::new(retained),
            fresh.clone(),
            tuning(1_000, 42_000),
            Vec::new(),
        );

        // The deliberate skip: NO live slot, NO seed entry, NO adopt, NO booking — the empty seed is what
        // makes a re-issued spawn a genuine double-spawn (the falsifiable twin of the adopt path).
        assert_eq!(sc.live_nodes(), BTreeSet::new());
        assert_eq!(
            sc.launch_ledger_seed(),
            crate::rlm_runtime::LaunchSeed::new()
        );
        assert_eq!(fresh.adopted(), Vec::new());
        assert_eq!(fresh.book_count(), 0);

        // But the F2 cursor RESUMED past the survivor (the Some arm of `read_water`) — the next mint is
        // strictly ABOVE 1000, never a reuse of the survivor's id.
        let n = sc.spawn_realm(&system(1, 2), T).expect("spawn");
        assert_eq!(n, NodeId(1_001));
    }

    #[test]
    fn water_only_over_a_virgin_store_starts_at_genesis() {
        // The `read_water` None arm: a virgin store yields the genesis cursors (base id/port), same as `new`.
        let fresh = FakeBackend::default();
        let sc = SpawnCore::water_only(
            Box::new(MemStore::new()),
            fresh,
            tuning(1_000, 42_000),
            Vec::new(),
        );
        assert_eq!(sc.live_nodes(), BTreeSet::new());
        let n = sc.spawn_realm(&system(1, 1), T).expect("spawn");
        assert_eq!(n, NodeId(1_000));
    }

    #[test]
    fn two_coords_sharing_a_lowered_leaf_get_distinct_full_path_ledger_rows() {
        // RLM 5e-5 D2 (Tier-A, authoritative): `system(2,7)` and `system(3,7)` both LOWER to
        // `RealmId::System(7)` (the single-galaxy-lossy aliasing, D-41-gated). The launch ledger is
        // FULL-`RealmPath`-keyed, so the two spawns get DISTINCT rows — no cross-galaxy double-spawn.
        let store = MemStore::new();
        let retained = store.clone();
        let sc = SpawnCore::new(
            Box::new(store),
            FakeBackend::default(),
            tuning(1_000, 42_000),
            Vec::new(),
        );

        // The lowered() collision is REAL (not a contrived fixture).
        assert_eq!(system(2, 7).lowered(), system(3, 7).lowered());

        sc.spawn_realm(&system(2, 7), T).expect("spawn a");
        sc.spawn_realm(&system(3, 7), T).expect("spawn b");

        // The seed carries TWO distinct full-path keys (not one aliased entry).
        let seed = sc.launch_ledger_seed();
        assert_eq!(seed.len(), 2);
        assert!(seed.contains_key(system(2, 7).path()));
        assert!(seed.contains_key(system(3, 7).path()));

        // The durable ledger holds TWO rows decoding to the two DISTINCT full paths.
        let rows: BTreeSet<RealmPath> = retained
            .scan(&rlm_launch_prefix())
            .into_iter()
            .map(|(_, v)| {
                postcard::from_bytes::<LaunchIntent>(&v)
                    .expect("decode intent")
                    .coord
                    .path()
                    .clone()
            })
            .collect();
        assert_eq!(
            rows,
            BTreeSet::from([system(2, 7).path().clone(), system(3, 7).path().clone()])
        );
        // The DIRECTORY-head aliasing (both → RealmId::System(7)) is by-design + D-41-gated — NOT asserted.
    }

    #[test]
    fn dev_tuning_is_loopback_with_documented_bases() {
        let t = SpawnTuning::dev();
        assert_eq!(t.first_node, 1_000);
        assert_eq!(t.first_port, 42_000);
        assert_eq!(t.bind_host, Ipv4Addr::LOCALHOST);
        // RLM 5f-4g: the raw default band is the WHOLE u16 space (exclusive end 65536), so the allocator is
        // byte-identical to pre-5f-4g — a demand LAUNCHER narrows this to a per-slot band.
        assert_eq!(t.port_limit, 65_536);
    }

    #[test]
    fn launch_ledger_seed_projects_live_into_the_reconciler_shape() {
        // RLM 5e-3b: the crash-recovery seed projector — live (node→{coord,at_tick}) → path→{node→at_tick},
        // the exact LaunchLedger.minted shape the rebuilt reconciler is seeded with.
        let sc = core(FakeBackend::default(), tuning(1_000, 42_000));
        let a = sc
            .spawn_realm(&system(1, 1), UniverseTick(5))
            .expect("spawn a");
        let b = sc
            .spawn_realm(&system(1, 2), UniverseTick(6))
            .expect("spawn b");
        let seed = sc.launch_ledger_seed();
        assert_eq!(seed.len(), 2);
        assert_eq!(
            seed[system(1, 1).path()],
            [(a, UniverseTick(5))].into_iter().collect()
        );
        assert_eq!(
            seed[system(1, 2).path()],
            [(b, UniverseTick(6))].into_iter().collect()
        );
    }

    // ---- closure_peers (RLM 5d VD_PEERS ancestor closure) ----------------------------------------

    /// A coord from a lineage of `(kind, seed)` levels.
    fn coord_of(levels: &[(RealmKindTag, u64)]) -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(
            levels
                .iter()
                .map(|(k, s)| RealmLevel::new(*k, *s))
                .collect(),
        ))
        .expect("a non-empty lineage has a leaf")
    }

    /// A live slot serving `coord`, bound at `addr(port)`.
    fn live_slot(coord: RealmCoord, port: u16) -> LiveSlot {
        LiveSlot {
            coord,
            addr: addr(port),
            at_tick: T,
        }
    }

    /// The `path_index` projection of a `live` map (what `insert_live` maintains) — so the `closure_peers`
    /// tests exercise the real exact-key lookup path.
    fn index_of(live: &BTreeMap<NodeId, LiveSlot>) -> BTreeMap<RealmPath, NodeId> {
        live.iter()
            .map(|(node, slot)| (slot.coord().path().clone(), *node))
            .collect()
    }

    #[test]
    fn closure_peers_collects_live_ancestors_leaf_to_root_then_anchors() {
        let leaf = coord_of(&[
            (RealmKindTag::Universe, 0),
            (RealmKindTag::Galaxy, 2),
            (RealmKindTag::System, 7),
        ]);
        let galaxy = coord_of(&[(RealmKindTag::Universe, 0), (RealmKindTag::Galaxy, 2)]);
        let universe = coord_of(&[(RealmKindTag::Universe, 0)]);
        let sibling = coord_of(&[
            (RealmKindTag::Universe, 0),
            (RealmKindTag::Galaxy, 2),
            (RealmKindTag::System, 8),
        ]);
        let mut live = BTreeMap::new();
        live.insert(NodeId(10), live_slot(galaxy, 5_010)); // Galaxy ancestor — INCLUDED
        live.insert(NodeId(11), live_slot(universe, 5_011)); // Universe ancestor — INCLUDED
        live.insert(NodeId(12), live_slot(sibling, 5_012)); // sibling System — EXCLUDED
        live.insert(NodeId(13), live_slot(leaf.clone(), 5_013)); // the leaf itself — EXCLUDED (never books self)
        let anchors = vec![(NodeId(1), addr(9_001)), (NodeId(2), addr(9_002))];

        // Ancestors leaf→root (Galaxy then Universe), THEN the anchors; no sibling, no self.
        assert_eq!(
            closure_peers(&leaf, &index_of(&live), &live, &anchors),
            vec![
                (NodeId(10), addr(5_010)),
                (NodeId(11), addr(5_011)),
                (NodeId(1), addr(9_001)),
                (NodeId(2), addr(9_002)),
            ]
        );
    }

    #[test]
    fn closure_peers_on_a_root_coord_is_just_the_anchors() {
        // A root coord has NO parent → the walk runs zero iterations; a live descendant is ignored.
        let root = coord_of(&[(RealmKindTag::Universe, 0)]);
        let descendant = coord_of(&[(RealmKindTag::Universe, 0), (RealmKindTag::Galaxy, 2)]);
        let mut live = BTreeMap::new();
        live.insert(NodeId(10), live_slot(descendant, 5_010));
        let anchors = vec![(NodeId(1), addr(9_001))];
        assert_eq!(
            closure_peers(&root, &index_of(&live), &live, &anchors),
            vec![(NodeId(1), addr(9_001))]
        );
    }

    #[test]
    fn closure_peers_skips_an_ancestor_not_yet_live() {
        // D3: an ancestor walked but ABSENT from `live` contributes nothing (the not-found arm). Only the
        // Galaxy ancestor is live; the Universe ancestor is missing.
        let leaf = coord_of(&[
            (RealmKindTag::Universe, 0),
            (RealmKindTag::Galaxy, 2),
            (RealmKindTag::System, 7),
        ]);
        let galaxy = coord_of(&[(RealmKindTag::Universe, 0), (RealmKindTag::Galaxy, 2)]);
        let mut live = BTreeMap::new();
        live.insert(NodeId(10), live_slot(galaxy, 5_010));
        let anchors = vec![(NodeId(1), addr(9_001))];
        // Galaxy included; Universe (absent) skipped; then the anchor.
        assert_eq!(
            closure_peers(&leaf, &index_of(&live), &live, &anchors),
            vec![(NodeId(10), addr(5_010)), (NodeId(1), addr(9_001))]
        );
    }
}
