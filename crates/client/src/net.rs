//! The client's network core: the session lifecycle, the shared §6.3 snapshot
//! decode, render-clock anchoring, and 20 Hz input assembly — over the `sim::io`
//! `Transport` seam.
//!
//! ## Shape (HR5 generic-coverage discipline)
//! [`ClientState`] is NON-generic and holds ALL the branching logic; it does its
//! I/O through `&mut dyn Transport`. [`ClientCore`]`<T: Transport>` is a branchless
//! shim that owns a concrete transport and delegates — so the bin can hold a real
//! `MeshTransport` while the logic is monomorphic and covered exactly once.
//!
//! ## Robustness
//! This is the REAL client (unlike the test-driver `ScriptedClient`, which panics
//! loudly on a protocol surprise): it NEVER panics on network input. A message from
//! a non-gateway peer, an unexpected class, an undecodable payload, or a
//! not-yet-relevant control variant is counted and ignored — the one-connection
//! invariant and forward-compatibility, enforced without a crash.

use vd_core::{EntityId, NodeId, SessionId, TickId};
use vd_devproto::{
    DevEntityRow, DevPhase, DevRealmBox, DevState, DevTransferView, DevWindowCensus, InputAction,
};
#[cfg(test)]
use vd_sim::io::Store as _;
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::channels::{
    ClientControlMsg, EventMsg, InputDatagram, RealmSnapshotDatagram, ServerControlMsg,
    SnapshotDatagram, SnapshotVerdict,
};
use vd_wire::seams::tickets::LoginTicket;
use vd_wire::version::ProtoVersion;

use std::sync::Arc;

use crate::input::InputState;
use crate::realm_scene::RealmScene;
use crate::realm_view::{RealmVerdict, RealmView};
use crate::render_clock::RenderClock;
use crate::render_snapshot::RenderSnapshot;
use crate::tuning::ClientInterpTuning;
use crate::view::DeliveredView;

/// Where the client is in its session lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientPhase {
    /// Will send `Hello` on its next step.
    Connecting,
    /// Hello sent; awaiting `Welcome`.
    AwaitingWelcome,
    /// Welcomed; awaiting `SubscriptionOpened`.
    AwaitingSubscription,
    /// Live: assembling 20 Hz input, rendering interpolated snapshots.
    Active,
    /// Closed (client-initiated `Bye` or gateway `Close`).
    Closed,
}

/// What one step did (drained / sent counts), for the driver and tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClientStepReport {
    pub received: usize,
    pub sent: usize,
}

/// The non-generic client logic + state. Drives its I/O through `&mut dyn Transport`.
pub struct ClientState {
    gateway: NodeId,
    ticket: LoginTicket,
    phase: ClientPhase,
    session: Option<SessionId>,
    /// The SET of subscriptions held (multi-realm AoI): grown on `SubscriptionOpened`, shrunk on
    /// `SubscriptionClosing`. A snapshot on ANY held sub is admitted; a pure-renderer client keeps
    /// this a SET (a player in a ship docked at a station on a planet holds several realm subs at
    /// once) but renders every entity by its `EntityId` — it never learns which node owns any sub.
    held_subs: std::collections::BTreeSet<vd_wire::channels::SubId>,
    view: DeliveredView,
    render_clock: RenderClock,
    input: InputState,
    tick: TickId,
    next_input_seq: u64,
    /// Sent-input accounting — a leak-free COUNT + the latest pair, not an
    /// unbounded per-input history (a 20 Hz session would grow that forever).
    sent_input_count: u64,
    last_sent_input: Option<(SessionId, u64)>,
    closing: bool,
    /// Whether the cluster tick rate has been learned from the wire (R1) — applied
    /// once; a duplicate `UniverseRate` must not reset the render cursor.
    tick_hz_learned: bool,
    decode_errors: u64,
    ignored: u64,
    foreign_peer_drops: u64,
    /// Snapshots accepted by the §6.3 gate (the liveness signal a `wait-until`
    /// predicate polls — distinguishes "live and receiving" from "welcomed but starved").
    snapshots_applied: u64,
    /// The freshest APPLIED universe tick (run-stable + join-independent) — what
    /// `screenshot --at-tick` aligns on; `None` before the first applied snapshot.
    latest_universe_tick: Option<u64>,
    /// The composed realm scene (proto_minor 18, §2.4): one source — the STREAM. Replaced whole
    /// on every level (`RealmRegistry`, the atomic epoch swap), patched by same-epoch deltas,
    /// shared onto the render seam via `Arc` so the per-step `render_snapshot()` clone is a
    /// pointer bump. Every drawn row's live pose rides `realm_view` and is OVERLAID at publish
    /// time; the level's own row poses make a row drawable the instant it lands.
    scene: Arc<RealmScene>,
    /// THE STAR CATALOGUE, assembling (S11). Public so the harness and the renderer can ask what the
    /// client actually holds — the S11 gate counts the stars that ARRIVED, never the absence of bytes.
    pub sky: crate::star_sky::StarSky,
    /// THE SKY, READY TO DRAW (S11) — built ONCE when the sky becomes whole, and then handed to every
    /// render snapshot by pointer.
    ///
    /// ★ WHY IT IS CACHED HERE. `StarSky::complete` allocates a fresh `Vec` of every star it holds. At
    /// the target census that is 150,000 rows, and the snapshot is published twenty times a second, so
    /// calling it per step would rebuild 7.0 MB twenty times a second for a galaxy that never moves.
    /// Built at the one moment the sky changes instead, and shared by `Arc` from then on.
    sky_draw: Option<std::sync::Arc<crate::render_snapshot::SkyDraw>>,
    /// THE SKY ANCHOR (owner ruling 2026-09-02 R1): the origin realm in the galaxy's frame, as the
    /// gateway last stated it on the per-tick realm lane. Latest-wins; cleared on a scene swap (the
    /// origin changed, so the old anchor names a realm the picture no longer stands in).
    sky_anchor: Option<vd_core::pose::StampedPose>,
    /// ★ THE ANCHOR'S TRACK (owner 2026-09-04, *"galaxy, sun and all other objects are not moving
    /// together"*): the sky anchor used to place the star cloud the instant it arrived while every
    /// body is drawn from its track 120 ms behind — so when the hull turned, the sky turned ahead of
    /// the bodies. The anchor is a stamped pose like any realm's: it rides a track and is sampled at
    /// the same render cursor as the scene, so the sky and the bodies turn together.
    sky_anchor_track: Option<crate::interp::EntityTrack>,
    /// THE GENERATION ALREADY ON DISK (S11), so a caller may ask to save every step and pay once.
    sky_cached: Option<u64>,
    /// THE SKY BEAT'S OWN CADENCE, in ticks — the watchdog's bound is derived from it, never a
    /// literal. Stated by the boot from the same tick rate the gateway beats on.
    sky_beat_cadence_ticks: u64,
    /// THE STATEMENT WAITING TO GO OUT (S11) — the generation this client holds, queued because the
    /// control handler has no transport. Drained on the next send. Latest-wins: only the newest sky is
    /// worth stating, and stating an older one would ask for work nobody needs.
    pending_sky_held: Option<u64>,
    /// THE ORIGIN MARKER (§2.7): the realm the current scene is composed in, as the level stated
    /// it. `None` before the first level. The diagnosis surface (`DevState.origin`) reads it —
    /// the pixel gates' "origin marker == home realm" assert.
    origin: Option<vd_core::pose::RealmId>,
    /// The delivered REALM view (FA-2c) — the streamed authoritative placements for the shard's
    /// moving realm boxes (an orbiting planet/station/ship). EMPTY until a `RealmSnapshot` arrives
    /// (walk scale ships none, so the published scene stays the boot `scene` — byte-identical). Read
    /// by `render_snapshot()` to OVERLAY each moving box's live pose onto the boot scene.
    realm_view: RealmView,
}

impl ClientState {
    #[must_use]
    pub fn new(gateway: NodeId, ticket: LoginTicket, tuning: ClientInterpTuning) -> ClientState {
        ClientState {
            gateway,
            ticket,
            phase: ClientPhase::Connecting,
            session: None,
            held_subs: std::collections::BTreeSet::new(),
            view: DeliveredView::default(),
            render_clock: RenderClock::new(tuning),
            input: InputState::default(),
            tick: TickId(0),
            next_input_seq: 0,
            sent_input_count: 0,
            last_sent_input: None,
            closing: false,
            tick_hz_learned: false,
            decode_errors: 0,
            ignored: 0,
            foreign_peer_drops: 0,
            snapshots_applied: 0,
            latest_universe_tick: None,
            scene: Arc::new(RealmScene::default()),
            sky: crate::star_sky::StarSky::default(),
            pending_sky_held: None,
            sky_draw: None,
            sky_anchor: None,
            sky_anchor_track: None,
            sky_cached: None,
            // Half a second at the client's own step rate, matching the gateway's keep-alive
            // derivation (`tick_hz / 2`). Never a free literal: it is the beat's cadence, and the
            // watchdog's bound is two of these plus one.
            sky_beat_cadence_ticks: ((tuning.tick_hz / 2.0) as u64).max(1),
            origin: None,
            realm_view: RealmView::default(),
        }
    }

    /// One full client step = [`pump_inbound`](Self::pump_inbound) then
    /// [`assemble_input`](Self::assemble_input). The headless driver + the parity tests
    /// call THIS at 20 Hz; the windowed renderer (Slice-3 T4) instead pumps every display
    /// frame and assembles on a 20 Hz accumulator — same two halves, so input stays
    /// byte-identical regardless of the render rate. `now_s` is wall-time fed IN by the
    /// loop (the lib reads no clock); it anchors the render cursor.
    pub fn step(&mut self, transport: &mut dyn Transport, now_s: f64) -> ClientStepReport {
        let received = self.pump_inbound(transport, now_s);
        let sent = self.assemble_input(transport);
        ClientStepReport { received, sent }
    }

    /// Drain + decode delivered messages, folding snapshots into the view and anchoring
    /// the render cursor. Idempotent w.r.t. the input clock — safe to call EVERY render
    /// frame (does not advance the input tick or send anything). Returns the count drained.
    pub fn pump_inbound(&mut self, transport: &mut dyn Transport, now_s: f64) -> usize {
        let inbound = transport.drain_inbound();
        let received = inbound.len();
        for msg in inbound {
            self.ingest(msg, now_s);
        }
        received
    }

    /// Advance the input tick and emit ONE outbound frame (Hello while connecting, a
    /// 20 Hz `InputDatagram` while active, `Bye` when closing). Call at the fixed 20 Hz
    /// input cadence, NOT per render frame. Returns the count sent.
    pub fn assemble_input(&mut self, transport: &mut dyn Transport) -> usize {
        self.tick = self.tick.next();
        self.send_outbound(transport)
    }

    fn ingest(&mut self, msg: Inbound, now_s: f64) {
        let Inbound::Wire { from, class, bytes } = msg else {
            // NodeUnreachable: the gateway will be redialed by the transport; the
            // client just keeps trying (latest-wins input tolerates the gap).
            return;
        };
        // The ONE-connection invariant: the only peer a client ever hears is its
        // gateway. Anything else is ignored (a real client never crashes on it).
        if from != self.gateway {
            self.foreign_peer_drops += 1;
            return;
        }
        match class {
            MsgClass::Control => self.on_control(&bytes),
            MsgClass::Snapshot => self.on_snapshot(&bytes, now_s),
            MsgClass::RealmSnapshot => self.on_realm_snapshot(&bytes, now_s),
            // No other class flows toward a client in P1.5.
            _ => self.ignored += 1,
        }
    }

    fn on_control(&mut self, bytes: &[u8]) {
        let Ok(msg) = postcard::from_bytes::<ServerControlMsg>(bytes) else {
            self.decode_errors += 1;
            return;
        };
        match msg {
            ServerControlMsg::Welcome { session, .. } => {
                self.session = Some(session);
                if self.phase == ClientPhase::AwaitingWelcome {
                    self.phase = ClientPhase::AwaitingSubscription;
                }
            }
            ServerControlMsg::SubscriptionOpened { sub, .. } => {
                self.held_subs.insert(sub);
                if self.phase == ClientPhase::AwaitingSubscription {
                    self.phase = ClientPhase::Active;
                }
            }
            // The reliable per-sub teardown: drop that sub's tracks so the view stays
            // bounded to the live set (vs swallowing it as `ignored` and leaking tracks), and
            // remove it from the held SET — any OTHER held sub (the dest sub of an overlap)
            // keeps routing; a later SubscriptionOpened re-arms a fresh sub id.
            ServerControlMsg::SubscriptionClosing { sub } => {
                self.view.drop_sub(sub);
                self.held_subs.remove(&sub);
            }
            // THE pure-renderer own-entity signal (minor 2): name the client's own avatar by
            // EntityId alone — NO sub, NO owning node. The client renders every entity at its
            // authoritative coordinate; this just tells it which dot is itself. It never learns
            // which shard simulates the avatar (that is the server's business, invisible here).
            ServerControlMsg::OwnEntity { entity } => {
                self.view.set_own_entity(entity);
            }
            ServerControlMsg::Close { .. } => {
                self.phase = ClientPhase::Closed;
            }
            ServerControlMsg::UniverseRate { tick_hz } => {
                self.set_tick_hz_from_wire(tick_hz);
            }
            // THE COMPOSED LEVEL (proto_minor 18, §2.4/§2.7): the complete drawable scene at one
            // tick, in the origin's frame — THE ATOMIC SWAP. The old scene rendered until this
            // landed; adopting it swaps scene + origin + epoch in one motion, forgets the old
            // epoch's tracks (the level's own row poses keep every box drawable meanwhile), and
            // replays the one-beat-held early datagram of the new epoch, if one raced the level.
            // This REPLACES the old `forget_space` inference off the entity feed. A malformed
            // level (a duplicate/cyclic realm — a server bug) is counted and the previous scene
            // kept, never a partial mutation.
            ServerControlMsg::RealmRegistry {
                origin,
                origin_epoch,
                rows,
            } => match RealmScene::from_scene_rows(&rows) {
                Ok(scene) => {
                    // ★ THE SAME ORIGIN KEEPS ITS MOTION (owner 2026-09-04): the epoch bumps for
                    // a chain change too — the pilot's hull handed from its star system to the
                    // galaxy — and then every track and the sky anchor are still positions in the
                    // origin's frame. Only a NEW origin (the pilot stepped into another realm)
                    // forgets them; the next datagram at this epoch states the new anchor.
                    // A track is forgotten only when a DELTA says its realm left the drawn set
                    // (below), never at the swap level: the swap level can be composed from a
                    // partial chain — the new hop's first level is still in flight — and a realm
                    // missing from it is coming back next tick. The owner saw that as the star
                    // blinking on the seventh flight (2026-09-04).
                    let same_origin = self.origin == Some(origin);
                    self.scene = Arc::new(scene);
                    self.origin = Some(origin);
                    if !same_origin {
                        self.sky_anchor = None;
                        self.sky_anchor_track = None;
                    }
                    if let Some(held) = self.realm_view.swap_epoch(origin_epoch, same_origin) {
                        // The held early datagram is at THIS epoch: its anchor is current too.
                        if let Some(anchor) = held.sky_anchor {
                            self.note_sky_anchor(anchor);
                        }
                        let standing_in = self.view.own_location_frame();
                        let _ = self.realm_view.on_realm_snapshot(standing_in, held);
                    }
                }
                Err(_) => self.decode_errors += 1,
            },
            // THE incremental composed update (§2.4): rows that ENTERED the drawn set and realms
            // that LEFT it, applied ONLY at the client's current epoch — a delta from a scene the
            // client no longer (or does not yet) hold is refused, counted benign (the next level
            // re-states everything). Atomic — a malformed delta is counted and the previous scene
            // kept, never a partial mutation.
            ServerControlMsg::RealmSceneDelta {
                origin_epoch,
                added,
                removed,
                ..
            } => {
                if origin_epoch != self.realm_view.epoch() {
                    self.ignored += 1;
                    return;
                }
                match self.scene.with_delta(&added, &removed) {
                    Ok(scene) => {
                        self.scene = Arc::new(scene);
                        // The realms that LEFT the drawn set take their tracks with them (the
                        // same-origin swap keeps every track, so this is the only place one ends).
                        self.realm_view.forget(&removed);
                    }
                    Err(_) => self.decode_errors += 1,
                }
            }
            // THE REMOVE MESSAGE (proto_minor 14): a reliable discrete event — today only the
            // per-entity eviction. Routed whole to `apply_event` so the client bin, the harness
            // and the unit tests all drive the ONE handler.
            ServerControlMsg::Event(event) => {
                self.apply_event(event);
            }
            // THE STAR CATALOGUE (S11): the galaxy's stars, in parts. Held unseen until the sky is
            // whole — half a catalogue is a galaxy with holes, and a hole looks exactly like a star
            // that does not exist. Every refusal is counted inside the assembler, so "the client shows
            // no stars" is explainable from the counters rather than being a report with nowhere to
            // start.
            ServerControlMsg::StarCatalogue {
                generation,
                part,
                parts,
                rows,
            } => {
                // STATE WHAT WE NOW HOLD, the moment the sky becomes whole (S11). Until the client
                // says this, the server has no way to know it has been served, and the beat keeps
                // driving a fresh request every cadence — the exchange only settles because the
                // receiver answers.
                if self.sky.accept(generation, part, parts, rows)
                    == crate::star_sky::SkyIngest::Completed
                {
                    self.state_sky_held(generation);
                }
            }
            // THE SKY'S LIVENESS BEAT (S11): the server naming the sky it believes is current, on a
            // cadence, whether or not it changed. Counted inside the assembler, because "nothing
            // changed" and "the emitter died" are the same silence and only a beat separates them.
            ServerControlMsg::SkyAlive { generation } => {
                // A beat that CONFIRMS what we hold is also the moment to re-state it, because a
                // gateway that lost our statement (a reconnect, a session that moved) would otherwise
                // go on asking for a sky we already have, for ever.
                // STAMPED WITH ITS ARRIVAL (S11): the watchdog's only input beyond the clock. The
                // universe tick is the one instant that flows through the seam — the client may not
                // read a wall clock.
                let at = self.latest_universe_tick.unwrap_or(0);
                if self.sky.beat_at(generation, at) == crate::star_sky::SkyBeat::Current {
                    self.state_sky_held(generation);
                }
            }
            // Node-AWARE legacy control a pure-renderer client no longer acts on: `AuthorityChanged`
            // (the sub re-point — superseded by `OwnEntity` + EntityId-keyed latest-wins render) and
            // `RequestCut` (the cut is server-timed now, S3 — the client stamps nothing). Both are
            // still SENT by the gateway to old (minor<2) clients, so a minor-2 client ignores them
            // (no crash) — forward/backward compatibility, counted as benign.
            _ => self.ignored += 1,
        }
    }

    /// Evict one entity's delivered copy (the reliable `EventMsg::EntityRemoved`): the
    /// server-authoritative "this entity left your view" signal a pure-renderer client needs
    /// because it keys tracks by `EntityId` (a `SubscriptionClosing` deliberately evicts nothing).
    /// Node-agnostic: it names only the entity, never a node.
    ///
    /// ROUTING (proto_minor 14 — LIVE): the gateway fans it as `ServerControlMsg::Event` on the
    /// reliable Control lane; `on_control` routes it here. `at` rides through to the view's
    /// resurrect guard (a straggler datagram row must not re-create the evicted track).
    pub fn apply_event(&mut self, event: EventMsg) {
        match event {
            EventMsg::EntityRemoved { entity, at } => self.view.remove_entity(entity, at),
            // A cosmetic notice (no state) — ignored by the headless/pure-renderer core.
            EventMsg::Notice { .. } => self.ignored += 1,
        }
    }

    fn on_snapshot(&mut self, bytes: &[u8], now_s: f64) {
        let Ok(snap) = postcard::from_bytes::<SnapshotDatagram>(bytes) else {
            self.decode_errors += 1;
            return;
        };
        let tick = snap.universe_tick;
        // Anchor the render cursor only on an APPLIED frame (a held sub, fresh). The old
        // space-flip INFERENCE that lived here (forget the realm tracks when the avatar's
        // delivered frame changes) is GONE (§2.7): the composed level's epoch swap is the one
        // scene-change signal, stated by the server, never inferred from a feed.
        if self.view.on_snapshot(&self.held_subs, snap) == SnapshotVerdict::Apply {
            self.render_clock.observe(tick, now_s);
            self.snapshots_applied += 1;
            self.latest_universe_tick = Some(tick.0);
        }
    }

    /// Fold one delivered REALM frame in (FA-2c) — the render-plane twin of [`Self::on_snapshot`]. The
    /// realm observer feed is sub-agnostic (world observation), so it delegates the latest-wins gate to
    /// [`RealmView::on_realm_snapshot`] and reacts on `Apply`. It ALSO anchors the render cursor: a client
    /// spectating a moving realm with NO entity in view (or before its own avatar's first frame) must
    /// still get a cursor, else the moving box would FREEZE at boot (the very failure FA-2c prevents).
    /// Both feeds stamp the same shard `universe_tick`, so anchoring from either is consistent.
    fn on_realm_snapshot(&mut self, bytes: &[u8], now_s: f64) {
        let Ok(snap) = postcard::from_bytes::<RealmSnapshotDatagram>(bytes) else {
            // A malformed realm datagram is a decode fault like any other — reuse the shared
            // `decode_errors` counter (a decode fault is a decode fault, DRY + already surfaced).
            self.decode_errors += 1;
            return;
        };
        let tick = snap.universe_tick;
        // ★ THE SKY ANCHOR APPLIES ON THE EPOCH ALONE (owner decision 2, 2026-09-02). The rows'
        // verdict asks whether a row folded; a datagram carrying an anchor and NO applicable row —
        // a pilot alone in a hull, nothing else in range — would be dropped with its anchor, and
        // the sky would never place. So the anchor is taken from every datagram at the CURRENT
        // epoch (a stale epoch names a realm the picture no longer stands in; a newer one is held
        // and replayed by the scene swap below). Latest-wins by presence: none leaves the last.
        if snap.origin_epoch == self.realm_view.epoch()
            && let Some(anchor) = snap.sky_anchor
        {
            self.note_sky_anchor(anchor);
        }
        // THE ONE-SPACE RULE (crossing-render slice): rows fold only when stated in the space the
        // avatar stands in — the old home's still-draining feed is skipped, never mixed in.
        let standing_in = self.view.own_location_frame();
        if self.realm_view.on_realm_snapshot(standing_in, snap) == RealmVerdict::Apply {
            self.render_clock.observe(tick, now_s);
            self.latest_universe_tick = Some(tick.0);
        }
    }

    /// Queue "I hold this sky" for the next send (S11).
    ///
    /// ★ WHY THE CLIENT MUST SAY THIS AT ALL. The server no longer remembers who it has served — that
    /// memory was wrong in both directions and could not be fixed, because a gateway serves many
    /// clients and a shard sees none of them. The client is the only party that knows what it holds,
    /// so the exchange settles only when the client answers. A client that never states this is served
    /// the catalogue again on every beat, which is safe but wasteful; a client that states a sky it
    /// does not have would be denied one, which is why this is called only where a whole sky is proven.
    fn state_sky_held(&mut self, generation: u64) {
        self.pending_sky_held = Some(generation);
        // ★ THE SKY THE RENDERER ALREADY HOLDS IS NOT REBUILT (2026-08-29). A confirming beat arrives
        // 20 times a second and says the SAME generation; rebuilding then copies every star in the
        // galaxy to produce a drawable identical to the one on screen. MEASURED on THE world: 233 220
        // rows, 11.2 MB per copy, 20 times a second. The player sees it as the position readout
        // breaking up — and only once the catalogue is large, which is why a small sky never showed it.
        if self
            .sky_draw
            .as_ref()
            .is_some_and(|held| held.generation == generation)
        {
            return;
        }
        // THE ONE PLACE THE DRAWABLE SKY IS BUILT. This runs exactly where a whole sky is proven — on
        // the last part, on a confirming beat, and on a cache adoption — so the renderer's copy and the
        // statement to the server can never disagree about which sky is held.
        self.sky_draw = self
            .sky
            .complete()
            .map(|rows| std::sync::Arc::new(crate::render_snapshot::SkyDraw { rows, generation }));
    }

    /// THE KEY THE SKY'S CACHE LIVES UNDER (S11). One row: a client holds one galaxy.
    ///
    /// A prefix rather than a bare key so the read is a `scan`, which is the trait's own listing
    /// operation — a `get` does not exist on [`Store`], and inventing one for a single row would be a
    /// second way to read a store.
    pub const SKY_CACHE_KEY: &'static [u8] = b"sky/catalogue";

    /// READ the sky from the store and adopt it, if a good one is there (S11).
    ///
    /// Every proof is the one [`Self::adopt_sky_cache`] already runs — the stamp, the claimed count,
    /// the rows, and the fold. A store is not more trusted than a file: it IS the file.
    ///
    /// # Errors
    /// [`crate::star_sky::CacheRefusal`] if nothing is stored, or if what is stored failed a proof.
    /// The held sky is unchanged either way, so a bad cache costs bytes and never correctness.
    pub fn load_sky_cache(
        &mut self,
        store: &dyn vd_sim::io::Store,
        expected: &vd_core::store_stamp::StoreStamp,
    ) -> Result<u64, crate::star_sky::CacheRefusal> {
        let rows = store.scan(Self::SKY_CACHE_KEY);
        let Some((_, bytes)) = rows.first() else {
            // NOTHING STORED is a refusal like any other, and counted like any other. A first run and
            // a wiped disk are the same case, and both simply mean the server will serve a sky.
            self.sky.cache_refused += 1;
            return Err(crate::star_sky::CacheRefusal::Undecodable);
        };
        self.adopt_sky_cache(bytes, expected)
    }

    /// WRITE the sky to the store, at most once per generation (S11).
    ///
    /// ★ THE GUARD IS WHY THE BIN MAY CALL THIS EVERY STEP. Without it, a caller with no way to know
    /// when the sky changed would re-encode and re-commit 7.0 MB twenty times a second for a galaxy
    /// that never moves — the same shape of waste this whole slice removes from the wire, moved onto
    /// the disk instead.
    ///
    /// Returns whether anything was written.
    pub fn save_sky_cache(
        &mut self,
        store: &mut dyn vd_sim::io::Store,
        stamp: vd_core::store_stamp::StoreStamp,
    ) -> bool {
        let Some(generation) = self.sky.generation() else {
            return false; // no sky
        };
        if self.sky_cached == Some(generation) {
            return false; // this sky is already on disk
        }
        let Some(rows) = self.sky.complete() else {
            return false; // never write half a galaxy: the holes look like missing stars
        };
        let bytes = crate::star_sky::encode_cache(stamp, generation, &rows);
        store.put(Self::SKY_CACHE_KEY, &vd_sim::io::bytes(bytes));
        store.commit();
        self.sky_cached = Some(generation);
        true
    }

    /// Adopt a sky from the on-disk cache, and state it (S11).
    ///
    /// This is the whole point of having a cache: a returning player draws the galaxy from their own
    /// disk, states the generation, and the server sends none of it. The statement is what closes the
    /// loop — without it the server would serve a catalogue the client already had.
    ///
    /// # Errors
    /// [`crate::star_sky::CacheRefusal`] if the file failed any of its proofs. The held sky is
    /// unchanged, and the server simply serves one, so a bad cache costs bytes and never correctness.
    pub fn adopt_sky_cache(
        &mut self,
        bytes: &[u8],
        expected: &vd_core::store_stamp::StoreStamp,
    ) -> Result<u64, crate::star_sky::CacheRefusal> {
        let generation = self.sky.adopt_cache(bytes, expected)?;
        self.state_sky_held(generation);
        Ok(generation)
    }

    fn send_outbound(&mut self, transport: &mut dyn Transport) -> usize {
        if self.closing && self.phase != ClientPhase::Closed {
            let buf =
                postcard::to_allocvec(&ClientControlMsg::Bye).expect("closed wire enums serialize");
            let _ = self.send_bytes(transport, MsgClass::Control, buf);
            self.phase = ClientPhase::Closed;
            return 1;
        }
        // THE HELD-SKY STATEMENT (S11), ahead of the phase machine: it is not part of the login
        // handshake and must not wait on one. Taken, so a statement is sent once.
        if let Some(generation) = self.pending_sky_held.take() {
            let buf = postcard::to_allocvec(&ClientControlMsg::SkyHeld { generation })
                .expect("closed wire enums serialize");
            let _ = self.send_bytes(transport, MsgClass::Control, buf);
        }
        match self.phase {
            ClientPhase::Connecting => {
                let hello = ClientControlMsg::Hello {
                    version: ProtoVersion::CURRENT,
                    login: self.ticket.clone(),
                };
                let buf = postcard::to_allocvec(&hello).expect("closed wire enums serialize");
                if self.send_bytes(transport, MsgClass::Control, buf) {
                    self.phase = ClientPhase::AwaitingWelcome;
                    1
                } else {
                    0 // refused send is back-pressure: retry next step
                }
            }
            ClientPhase::Active => {
                let session = self.session.expect("active implies welcomed");
                self.next_input_seq += 1;
                let seq = self.next_input_seq;
                let (movement, look, action_bits) = self.input.take_frame();
                let input = InputDatagram {
                    seq,
                    // S3/S6: the cut is SERVER-TIMED — the client stamps NOTHING. A pure renderer
                    // just forwards its inputs; it never learns (nor marks) a transfer boundary.
                    is_cut_marker: false,
                    client_tick: self.tick,
                    movement,
                    look,
                    action_bits,
                };
                let buf = postcard::to_allocvec(&input).expect("closed wire enums serialize");
                if self.send_bytes(transport, MsgClass::Input, buf) {
                    self.sent_input_count += 1;
                    self.last_sent_input = Some((session, seq));
                    1
                } else {
                    0 // refused input is dropped (latest-wins tolerates the gap)
                }
            }
            ClientPhase::AwaitingWelcome
            | ClientPhase::AwaitingSubscription
            | ClientPhase::Closed => 0,
        }
    }

    /// Learn the cluster's universe-tick rate from the wire (R1) — applied to the
    /// render cursor ONCE (the first `UniverseRate`), clamped to >= 1 Hz so a bad/zero
    /// rate cannot freeze the cursor. A duplicate is idempotent (no cursor reset).
    /// Hold the newest sky anchor and fold it into the anchor's track (one track, latest-wins per
    /// tick — the same primitive every realm's placement rides).
    fn note_sky_anchor(&mut self, anchor: vd_core::pose::StampedPose) {
        self.sky_anchor = Some(anchor);
        match &mut self.sky_anchor_track {
            Some(track) => track.observe(anchor),
            None => self.sky_anchor_track = Some(crate::interp::EntityTrack::new(anchor)),
        }
    }

    fn set_tick_hz_from_wire(&mut self, tick_hz: u32) {
        if self.tick_hz_learned {
            return;
        }
        self.tick_hz_learned = true;
        let hz = f64::from(tick_hz).max(1.0);
        self.render_clock.set_tick_hz(hz);
        // The resurrect guard converts its wall-clock window at the same wire-taught rate the
        // render cursor advances at — one rate, learned once, used by both.
        self.view.set_tick_hz(hz);
    }

    fn send_bytes(&self, transport: &mut dyn Transport, class: MsgClass, buf: Vec<u8>) -> bool {
        transport
            .send(self.gateway, class, vd_sim::io::bytes(buf))
            .is_ok()
    }

    /// Request a graceful close; the next step sends `Bye` and enters `Closed`.
    pub fn request_close(&mut self) {
        self.closing = true;
    }

    // ---- input injection seam (Slice 2 dev-control + Slice 3 keyboard call these) --
    pub fn set_movement(&mut self, movement: [f32; 3]) {
        self.input.set_movement(movement);
    }
    pub fn add_look(&mut self, delta: [f32; 2]) {
        self.input.add_look(delta);
    }
    pub fn set_action_bit(&mut self, bit: u32, pressed: bool) {
        self.input.set_action_bit(bit, pressed);
    }

    /// Apply ONE decoded dev-control [`InputAction`] (the pure seam from `vd-devproto`)
    /// onto the SAME input setters the keyboard drives — agent input is byte-identical
    /// to real input (HR6). `ResetInput` clears all held input (the one privileged arm).
    pub fn apply_input_action(&mut self, action: InputAction) {
        match action {
            InputAction::Move(movement) => self.set_movement(movement),
            InputAction::Look(delta) => self.add_look(delta),
            InputAction::Action { bit, pressed } => self.set_action_bit(bit, pressed),
            InputAction::Close => self.request_close(),
            InputAction::ResetInput => self.input = InputState::default(),
        }
    }

    // ---- read-only views ----------------------------------------------------------
    #[must_use]
    pub fn phase(&self) -> ClientPhase {
        self.phase
    }
    #[must_use]
    pub fn session(&self) -> Option<SessionId> {
        self.session
    }
    #[must_use]
    pub fn own_entity(&self) -> Option<EntityId> {
        self.view.own_entity()
    }
    #[must_use]
    pub fn view(&self) -> &DeliveredView {
        &self.view
    }
    /// The SET of subscriptions the client currently holds (empty before the first
    /// `SubscriptionOpened`; holds BOTH subs during a transfer overlap — Track R / 1d.2d).
    #[must_use]
    pub fn held_subs(&self) -> &std::collections::BTreeSet<vd_wire::channels::SubId> {
        &self.held_subs
    }
    /// The render cursor at wall-time `now_s` (`None` until the first snapshot).
    #[must_use]
    pub fn cursor(&self, now_s: f64) -> Option<f64> {
        self.render_clock.cursor(now_s)
    }
    /// How many input datagrams this client has successfully sent (leak-free; the
    /// per-input history is not retained — `last_sent_input` carries the latest).
    #[must_use]
    pub fn sent_input_count(&self) -> u64 {
        self.sent_input_count
    }
    #[must_use]
    pub fn last_sent_input(&self) -> Option<(SessionId, u64)> {
        self.last_sent_input
    }
    #[must_use]
    pub fn dropped_counts(&self) -> (u64, u64, u64) {
        (self.decode_errors, self.ignored, self.foreign_peer_drops)
    }
    #[must_use]
    pub fn snapshots_applied(&self) -> u64 {
        self.snapshots_applied
    }

    /// An immutable [`RenderSnapshot`] for the windowed renderer (Slice-3 T4): the
    /// delivered view + the render clock + the lifecycle phase. The core thread publishes
    /// one per step via `ArcSwap`; the Bevy app reads it wait-free and samples at its own
    /// display cursor (sim cadence decoupled from frame rate). Cheap: the view is
    /// `BTreeMap`s of `Copy` tracks.
    #[must_use]
    pub fn render_snapshot(&self) -> RenderSnapshot {
        // SLICE 6 S4 — publish the boot scene and the live placements UNRESOLVED. This used to BAKE an
        // overlaid scene here, once per 20 Hz core step, from whatever had most recently arrived — no
        // render cursor anywhere in it. The renderer then drew those baked centres alongside entities it
        // sampled at its display cursor, so the ground and the player standing on it came from two
        // different instants. `RenderSnapshot::scene_at` now resolves both at ONE cursor, per drawn
        // frame, and this per-step allocation disappears.
        RenderSnapshot::with_realms(
            self.view.clone(),
            self.render_clock,
            self.phase,
            Arc::clone(&self.scene),
            self.realm_view.clone(),
        )
        // A pointer bump, whatever the census (S11).
        .with_sky(self.sky_draw.clone())
        .with_origin(self.origin)
        // Where the galaxy is, as last stated — the one thing that places the sky (2026-09-02).
        .with_sky_anchor(self.sky_anchor)
        .with_sky_anchor_track(self.sky_anchor_track)
    }

    /// Build the [`DevState`] diagnosis surface (HR6) from the DECODED DELIVERED view
    /// at wall-time `now_s` — wire truth, never internal hope. `counters` are the bin's
    /// own numbers (the mailbox the lib does not own, and the render thread's star cloud
    /// the lib never sees), so the bin passes them in. Every emitted float is sanitized
    /// finite, so the `encode_response` codec is infallible.
    #[must_use]
    pub fn devstate(&self, now_s: f64, counters: DevCounters) -> DevState {
        let DevCounters {
            dev_commands_applied,
            dev_commands_dropped,
            stars_drawn,
        } = counters;
        let render_cursor = self.cursor(now_s).map(sanitize_f64);
        // Entities are sampled at the cursor only once it is anchored (a snapshot has
        // applied); before that there is nothing to render.
        let entities = render_cursor
            .map(|cursor| {
                self.view
                    .rendered(cursor)
                    .into_iter()
                    .map(|(entity, sub, pose)| DevEntityRow {
                        entity: entity.to_string(),
                        // The RENDER pose, through the SAME chokepoint the renderer draws with, so the HR6
                        // diagnosis surface reports exactly what is on screen.
                        pos: sanitize_vec3(self.view.world_pos(&pose)),
                        orient: sanitize_quat(pose.orient),
                        authoritative_sub: sub.0,
                    })
                    .collect()
            })
            .unwrap_or_default();
        // The DRAWN realm boxes (VU diagnosis) — resolved at the SAME cursor this state reports and the
        // renderer draws at (slice 6 S4), so the diagnosis surface and the pixels agree by construction.
        // A `Planet` row proves its `RealmSceneDelta` landed; a center that moves across ticks proves
        // the realm-pose feed is being applied (the frozen-planet gate).
        let drawn_scene = self.render_snapshot().scene_now(now_s);
        let realm_boxes = drawn_scene
            .iter()
            .map(|(realm, b)| DevRealmBox {
                realm: format!("{realm:?}"),
                // The delivered hierarchy parent, verbatim off the composed row (G-PARENT-TRUE's
                // process read — look_horizon.md slice 0). Same rendering as `realm`.
                parent: b.parent.map(|p| format!("{p:?}")),
                // The STREAMED extent (§2.11): straight off the composed row's look bag — the
                // camera reconstruction and the harness verdicts read THIS, never a file.
                extent_m: box_extent_m(b),
                // THE DRAW LAW's arm, by data presence (owner decision 10) — the gates' assert.
                body_kind: match b.body {
                    crate::realm_scene::BodyKind::Look => "look".to_owned(),
                    crate::realm_scene::BodyKind::Marker => "marker".to_owned(),
                },
                // The parent's photometric datum, verbatim — what a pixel gate sizes the point
                // sprite's rectangle from, through the SAME Tier-A pair the renderer scales by.
                luma: b.luma,
                // THE SAME reduction the renderer draws with, through the ONE chokepoint — not a
                // hand-rolled subtraction. This line used to spell `b.center_offset - origin.offset()`,
                // which dropped the origin's COARSE half while `DevEntityRow.pos` twenty lines above
                // went through `world_pos` correctly. Two arithmetics for one quantity, in the surface
                // a test asserts against: the "player rides its realm" gate reads THIS value, so it was
                // effectively comparing the client to itself.
                center: sanitize_vec3(b.draw_center()),
                // SHAKE DIAGNOSIS: which moment THIS box is being drawn from. A box whose tick tracks
                // `entity_feed_newest_tick` shares the player's moment; one that drifts is authored by a
                // shard whose clock is running independently. `None` = never streamed (boot placement).
                newest_tick: self.realm_view.realm_newest_tick(realm).map(|t| t.0),
            })
            .collect();
        DevState {
            phase: dev_phase(self.phase),
            session: self.session.map(|s| s.to_string()),
            own_entity: self.view.own_entity().map(|e| e.to_string()),
            // The player's location (realm label) from the own entity's authoritative
            // frame — the player-stats HUD source, also surfaced to `vdctl state`.
            location: self
                .view
                .own_location_frame()
                .map(vd_core::pose::FrameRef::label),
            render_cursor,
            universe_tick: self.latest_universe_tick,
            // SHAKE DIAGNOSIS — the two feeds' freshest ticks, reported side by side so their drift is
            // directly observable next to `render_cursor`. Read the field docs on `DevState` for why.
            entity_feed_newest_tick: self.view.newest_tick().map(|t| t.0),
            realm_feed_newest_tick: self.realm_view.newest_tick().map(|t| t.0),
            // SLICE 6 S5 — how each feed classified AT THE REPORTED CURSOR. Both censuses are taken at
            // the SAME cursor, so they are directly comparable; a feed sitting on `clamped_old` is the
            // condition that WAS the shake, and it now shows up as a number instead of as a wobble the
            // user has to notice.
            entity_windows: window_census(render_cursor.map(|c| self.view.window_census(c))),
            realm_windows: window_census(render_cursor.map(|c| self.realm_view.window_census(c))),
            feed_skew_ticks: feed_skew(
                self.view.newest_tick().map(|t| t.0),
                self.realm_view.newest_tick().map(|t| t.0),
            ),
            entities,
            realm_boxes,
            // THE ORIGIN MARKER (§2.7/§2.11): (origin, epoch) as the last composed level stated
            // them — the pixel gates' "origin == home realm" + "epoch bumps once per crossing".
            origin: self
                .origin
                .map(|o| (format!("{o:?}"), self.realm_view.epoch())),
            // THE SKY (S11) — the whole catalogue this client holds, or nothing while it is partial.
            sky: self
                .sky_draw
                .as_ref()
                .map(|s| (s.generation, s.rows.len() as u64)),
            // The origin realm in the galaxy's frame, in metres — where the sky is placed from.
            sky_anchor: self.sky_anchor.map(|a| {
                sanitize_vec3(
                    a.pos
                        .delta_m(vd_core::pose::LatticePos::ORIGIN, a.frame.tier()),
                )
            }),
            // ★ IS ANYONE STILL SPEAKING FOR THE SKY (S11)? The stars stay on screen either way — a
            // stale sky is not a wrong sky, because stars do not move. This is the "says so" half of
            // SL1 clause 6, and it is the ONLY half that applies to a picture which cannot go wrong.
            sky_watch: format!(
                "{:?}",
                crate::star_sky::sky_watch(
                    self.sky.last_beat_tick,
                    self.latest_universe_tick.unwrap_or(0),
                    self.sky_beat_cadence_ticks,
                )
            ),
            stale_epoch_rows: self.realm_view.stale_epoch_rows(),
            snapshots_applied: self.snapshots_applied,
            realm_frames_applied: self.realm_view.frames_applied(),
            // SUM both feeds' faults — the realm feed computes+exposes its OWN stale/NaN counts, so a
            // corrupt realm feed must not be invisible behind the entity view's (both 0 at walk scale).
            stale_frames_dropped: self.view.stale_frames_dropped()
                + self.realm_view.stale_frames_dropped(),
            sent_input_count: self.sent_input_count,
            decode_errors: self.decode_errors,
            ignored: self.ignored,
            foreign_peer_drops: self.foreign_peer_drops,
            nonfinite_poses: self.view.nonfinite_poses() + self.realm_view.nonfinite_poses(),
            // The three row-drop honesty counters (audit :304): row refusal is a NORMAL client
            // behaviour since Step 5, so the diagnosis surface must distinguish "the shard stopped
            // emitting this entity" from "the client refused every row it sent" — above all for a
            // wrongly-armed resurrect guard (the permanently-undrawable-figure class).
            resurrect_rows_dropped: self.view.resurrect_rows_dropped(),
            // One-space skips happen on BOTH feeds; summed like the other two-feed fault counters.
            foreign_space_rows: self.view.foreign_space_rows()
                + self.realm_view.foreign_space_rows(),
            echo_rows_dropped: self.view.echo_rows_dropped(),
            dev_commands_applied,
            dev_commands_dropped,
            stars_drawn,
            transfer: DevTransferView::None,
        }
    }
}

/// THE BIN'S OWN NUMBERS for the diagnosis surface — counters the lib cannot see because it does
/// not own the thing counted: the dev-control mailbox (applied / shed) and the render thread's star
/// cloud (drawn). One struct, so a new counter is one field and not one more positional argument at
/// every call site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DevCounters {
    /// Dev actions drained and applied this session.
    pub dev_commands_applied: u64,
    /// Dev actions shed by the bounded mailbox (back-pressure is never silent).
    pub dev_commands_dropped: u64,
    /// Points of light in the renderer's star cloud right now; 0 with no renderer.
    pub stars_drawn: u64,
}

/// One drawn box's extent in metres — a sphere's radius, a box's half-diagonal length, and 0
/// exactly for a MARKER point (its drawn footprint IS sub-pixel until Slice D's sprites).
fn box_extent_m(b: &crate::realm_scene::RealmBox) -> f64 {
    crate::realm_scene::shape_extent_m(b.shape)
}

/// Map the client lifecycle phase to its serde-able twin (the dev surface mirrors the
/// phases without re-exporting `ClientPhase`'s internals).
fn dev_phase(phase: ClientPhase) -> DevPhase {
    match phase {
        ClientPhase::Connecting => DevPhase::Connecting,
        ClientPhase::AwaitingWelcome => DevPhase::AwaitingWelcome,
        ClientPhase::AwaitingSubscription => DevPhase::AwaitingSubscription,
        ClientPhase::Active => DevPhase::Active,
        ClientPhase::Closed => DevPhase::Closed,
    }
}

/// Force a float finite (NaN/±∞ → 0.0) so the `DevState` always JSON-encodes — a
/// non-finite float is the only value that makes `serde_json` fail on these types.
fn sanitize_f64(x: f64) -> f64 {
    if x.is_finite() { x } else { 0.0 }
}

fn sanitize_vec3(v: glam::DVec3) -> [f64; 3] {
    [sanitize_f64(v.x), sanitize_f64(v.y), sanitize_f64(v.z)]
}

/// Map an optional `(blended, clamped_old, clamped_new)` tally to its wire twin (SLICE 6 S5). Before
/// the clock is anchored there is no cursor and hence no classification — an all-zero census.
fn window_census(counts: Option<(u32, u32, u32)>) -> DevWindowCensus {
    match counts {
        Some((blended, clamped_old, clamped_new)) => DevWindowCensus {
            blended,
            clamped_old,
            clamped_new,
        },
        None => DevWindowCensus::default(),
    }
}

/// The two feeds' newest ticks differenced (entity minus realm), `None` until BOTH have delivered —
/// the arrival skew. Post-slice-6 this is harmless (both feeds are read at one cursor regardless), but
/// a growing value still means one feed's authoring path is falling behind the other's.
fn feed_skew(entity: Option<u64>, realm: Option<u64>) -> Option<i64> {
    let e = i64::try_from(entity?).ok()?;
    let r = i64::try_from(realm?).ok()?;
    Some(e - r)
}

/// Force a quaternion finite while PRESERVING a valid rotation: a non-finite quat (a
/// corrupt/diverged sender) maps to IDENTITY — NOT a component-wise zero, which is a
/// zero-length non-rotation that would poison `orient * -Z` / `orient.conjugate()` in the
/// look-at loop. Mirrors [`vd_core::pose::StampedPose::sanitized`]'s orient guarantee.
fn sanitize_quat(q: glam::DQuat) -> [f64; 4] {
    if q.is_finite() {
        [q.x, q.y, q.z, q.w]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// The branchless shim: owns a concrete transport and delegates every step to the
/// monomorphic [`ClientState`]. The bin instantiates `ClientCore<MeshTransport>`;
/// tests instantiate it over a mock — the logic is covered once, in `ClientState`.
pub struct ClientCore<T: Transport> {
    transport: T,
    state: ClientState,
}

impl<T: Transport> ClientCore<T> {
    pub fn new(
        transport: T,
        gateway: NodeId,
        ticket: LoginTicket,
        tuning: ClientInterpTuning,
    ) -> ClientCore<T> {
        ClientCore {
            transport,
            state: ClientState::new(gateway, ticket, tuning),
        }
    }

    pub fn step(&mut self, now_s: f64) -> ClientStepReport {
        self.state.step(&mut self.transport, now_s)
    }

    /// Drain + decode inbound (every render frame in the windowed client). See
    /// [`ClientState::pump_inbound`].
    pub fn pump_inbound(&mut self, now_s: f64) -> usize {
        self.state.pump_inbound(&mut self.transport, now_s)
    }

    /// Emit one outbound frame at the 20 Hz input cadence. See
    /// [`ClientState::assemble_input`].
    pub fn assemble_input(&mut self) -> usize {
        self.state.assemble_input(&mut self.transport)
    }

    /// This client's own node identity (from its transport).
    #[must_use]
    pub fn local_id(&self) -> NodeId {
        self.transport.local_id()
    }

    #[must_use]
    pub fn state(&self) -> &ClientState {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut ClientState {
        &mut self.state
    }
}

#[cfg(test)]
mod tests {
    /// Flatten a RenderPose to world metres (normalized lattice since the cell activation).
    fn rpw(p: &crate::interp::RenderPose) -> DVec3 {
        vd_core::pose::LatticePos::at(p.cell, p.pos)
            .delta_m(vd_core::pose::LatticePos::default(), p.tier)
    }

    use super::*;
    use glam::DVec3;
    use vd_core::entity_kind::EntityKind;
    use vd_core::pose::{FrameRef, StampedPose};
    use vd_core::{AccountId, EpochId, Fence, MsgId, TransferId, UniverseTick};
    use vd_sim::io::{Bytes, SendError};
    use vd_wire::channels::{EntitySnap, SubId};

    const GATEWAY: NodeId = NodeId(2);
    const OTHER: NodeId = NodeId(99);

    /// A minimal in-memory transport: a delivered-inbound queue + a sent log, with a
    /// switch to refuse sends (back-pressure).
    #[derive(Default)]
    struct MockTransport {
        inbound: Vec<Inbound>,
        sent: Vec<(MsgClass, Vec<u8>)>,
        refuse: bool,
        next_id: u64,
    }
    impl MockTransport {
        fn deliver(&mut self, from: NodeId, class: MsgClass, buf: Vec<u8>) {
            self.inbound.push(Inbound::Wire {
                from,
                class,
                bytes: vd_sim::io::bytes(buf),
            });
        }
    }
    impl Transport for MockTransport {
        fn send_durable(
            &mut self,
            _to: NodeId,
            class: MsgClass,
            bytes: Bytes,
            _durability: vd_sim::io::Durability,
        ) -> Result<MsgId, SendError> {
            if self.refuse {
                return Err(SendError::QueueFull(bytes));
            }
            self.next_id += 1;
            self.sent.push((class, bytes.to_vec()));
            Ok(MsgId(self.next_id))
        }
        fn drain_inbound(&mut self) -> Vec<Inbound> {
            std::mem::take(&mut self.inbound)
        }
        fn local_id(&self) -> NodeId {
            NodeId(100) // the client's own id; the core never reads it
        }
    }

    fn ticket() -> LoginTicket {
        // The mock transport never validates the ticket; any well-formed one works.
        LoginTicket {
            account: AccountId(1),
            epoch: EpochId(1),
            nonce: 1,
            signature: Vec::new(),
        }
    }

    fn core() -> ClientCore<MockTransport> {
        ClientCore::new(
            MockTransport::default(),
            GATEWAY,
            ticket(),
            ClientInterpTuning::DEFAULT,
        )
    }

    fn welcome() -> Vec<u8> {
        postcard::to_allocvec(&ServerControlMsg::Welcome {
            version: ProtoVersion::CURRENT,
            session: SessionId(9),
            session_fence: Fence(1),
            epoch: EpochId(1),
        })
        .expect("test fixture")
    }
    fn sub_opened() -> Vec<u8> {
        postcard::to_allocvec(&ServerControlMsg::SubscriptionOpened {
            sub: SubId(0),
            frame: FrameRef::SystemSpace { system_seed: 1 },
        })
        .expect("test fixture")
    }
    fn ent() -> EntityId {
        EntityId::pack(EntityKind::Player, 1, 1, 1)
    }
    fn snapshot(frame_id: u64, tick: u64, x: f64) -> Vec<u8> {
        postcard::to_allocvec(&SnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            entities: vec![EntitySnap {
                entity: ent(),
                pose: StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 1 },
                    DVec3::new(x, 0.0, 0.0),
                    UniverseTick(tick),
                ),
            }],
        })
        .expect("test fixture")
    }

    /// A `RealmSnapshotDatagram` moving `realm` to frame-local x (FA-2c) — the realm twin of [`snapshot`].
    fn realm_snapshot(frame_id: u64, tick: u64, realm: vd_core::pose::RealmId, x: f64) -> Vec<u8> {
        realm_snapshot_at_epoch(0, frame_id, tick, realm, x)
    }

    /// The epoch-carrying twin (§2.7): the client's scene epoch boots at 0, so the plain fixture
    /// above stays applicable without a level; the epoch tests drive this directly.
    fn realm_snapshot_at_epoch(
        origin_epoch: u64,
        frame_id: u64,
        tick: u64,
        realm: vd_core::pose::RealmId,
        x: f64,
    ) -> Vec<u8> {
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram};
        postcard::to_allocvec(&RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            origin_epoch,
            sky_anchor: None,
            realms: vec![RealmSnap {
                realm,
                // The edge HEAD (proto_minor 8): the CHILD realm's own frame, beside the TAIL
                // (`pose.frame`, the authoring parent's).
                frame: vd_core::pose::frame_for_realm(realm, None)
                    .expect("a seeded realm resolves"),
                pose: StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 7 },
                    DVec3::new(x, 0.0, 0.0),
                    UniverseTick(tick),
                ),
            }],
        })
        .expect("test fixture")
    }

    /// Drive Connecting → Active.
    fn activate(c: &mut ClientCore<MockTransport>) {
        let r = c.step(0.0); // sends Hello
        assert_eq!(r.sent, 1);
        assert_eq!(c.state().phase(), ClientPhase::AwaitingWelcome);
        c.transport.deliver(GATEWAY, MsgClass::Control, welcome());
        c.step(0.0);
        assert_eq!(c.state().phase(), ClientPhase::AwaitingSubscription);
        c.transport
            .deliver(GATEWAY, MsgClass::Control, sub_opened());
        c.step(0.0);
        assert_eq!(c.state().phase(), ClientPhase::Active);
        assert_eq!(c.state().session(), Some(SessionId(9)));
    }

    #[test]
    fn full_lifecycle_logs_input_and_advances_seq() {
        let mut c = core();
        activate(&mut c);
        c.state_mut().set_movement([1.0, 0.0, 0.0]);
        let r = c.step(0.0);
        assert_eq!(r.sent, 1, "an InputDatagram went out while Active");
        // The latest Input frame carries the held movement and a fresh seq.
        let (class, buf) = c.transport.sent.last().cloned().expect("test fixture");
        assert_eq!(class, MsgClass::Input);
        let input: InputDatagram = postcard::from_bytes(&buf).expect("test fixture");
        assert_eq!(input.movement, [1.0, 0.0, 0.0]);
        // Two inputs sent (activate's last step while Active, then this one); seq advanced.
        assert_eq!(c.state().sent_input_count(), 2);
        assert_eq!(c.state().last_sent_input(), Some((SessionId(9), 2)));
    }

    #[test]
    fn the_client_never_stamps_a_cut_marker() {
        // S6: the cut is SERVER-TIMED — a pure-renderer client stamps NOTHING. Even a legacy
        // `RequestCut` (still sent to old clients) is IGNORED (benign), never acted on: the next
        // input carries `is_cut_marker = false` like every input.
        let mut c = core();
        activate(&mut c);
        c.step(0.0);
        let before: InputDatagram =
            postcard::from_bytes(&c.transport.sent.last().expect("input sent").1).expect("decode");
        assert!(!before.is_cut_marker, "the client never marks an input");

        // Deliver a legacy RequestCut: it is IGNORED (a minor-2 client no longer acts on it).
        let req = postcard::to_allocvec(&ServerControlMsg::RequestCut {
            transfer: TransferId(1),
        })
        .expect("test fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, req);
        c.step(0.0);
        let after: InputDatagram =
            postcard::from_bytes(&c.transport.sent.last().expect("input sent").1).expect("decode");
        assert!(
            !after.is_cut_marker,
            "still no CUT_MARKER after a legacy RequestCut (the cut is server-timed)"
        );
        assert_eq!(
            c.state().dropped_counts().1,
            1,
            "the legacy RequestCut is counted as one benign ignored variant"
        );
    }

    #[test]
    fn own_entity_via_own_entity() {
        // The node-agnostic OwnEntity signal names this client's avatar by EntityId alone.
        let mut c = core();
        activate(&mut c);
        let msg =
            postcard::to_allocvec(&ServerControlMsg::OwnEntity { entity: ent() }).expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, msg);
        c.step(0.0);
        assert_eq!(c.state().own_entity(), Some(ent()));
        // A legacy AuthorityChanged (still sent to old clients) is IGNORED by a pure renderer.
        let auth = postcard::to_allocvec(&ServerControlMsg::AuthorityChanged {
            entity: EntityId::pack(EntityKind::Player, 1, 2, 2),
            sub: SubId(0),
        })
        .expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, auth);
        c.step(0.0);
        assert_eq!(
            c.state().own_entity(),
            Some(ent()),
            "AuthorityChanged does not re-point the own entity on a pure-renderer client"
        );
    }

    /// THE REMOVE MESSAGE arrives ON THE WIRE (proto_minor 14): a Control frame carrying
    /// ★ THE STAR CATALOGUE ARRIVES, AND THE GATE COUNTS WHAT ARRIVED (S11).
    ///
    /// **THIS TEST EXISTS BECAUSE THE OBVIOUS GATE IS A TRAP, and S11 says so in its own text.** Both
    /// earlier plans proposed to gate on *"per-tick sky bytes are ZERO after the first tick"* — and
    /// that is ALSO exactly what a dropped oversize message produces. The gate would pass on the
    /// failure it exists to prevent, and it would pass most convincingly at the census, where the
    /// message becomes big enough to be dropped.
    ///
    /// So this counts the stars the client actually HOLDS, never the absence of bytes. A sky that
    /// never arrived reads zero here and cannot be mistaken for a sky that did not need re-sending.
    /// ★ THE CLIENT STATES THE SKY IT HOLDS, OR THE EXCHANGE NEVER SETTLES (S11).
    ///
    /// The server keeps no memory of who it has served — that memory was wrong in both directions and
    /// could not be repaired, because a gateway serves many clients and a shard sees none of them. The
    /// client is the only party that knows what it holds. So if the client never answers, the server
    /// goes on serving the catalogue for ever, and the saving this whole slice exists for is lost.
    #[test]
    fn the_client_states_the_sky_it_holds_when_the_sky_becomes_whole() {
        let mut c = core();
        activate(&mut c);
        let star = |n: u64| vd_core::look::StarRow {
            realm: vd_core::pose::RealmId::System(n),
            cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        };
        let part = |part: u32, parts: u32, rows: Vec<vd_core::look::StarRow>| {
            postcard::to_allocvec(&ServerControlMsg::StarCatalogue {
                generation: 0x1234,
                part,
                parts,
                rows,
            })
            .expect("fixture")
        };
        let stated = |c: &ClientCore<MockTransport>| {
            c.transport
                .sent
                .iter()
                .filter(|(_, b)| {
                    postcard::from_bytes::<ClientControlMsg>(b)
                        == Ok(ClientControlMsg::SkyHeld { generation: 0x1234 })
                })
                .count()
        };

        // HALF A SKY SAYS NOTHING. Stating a sky the client cannot draw would tell the server to stop
        // sending the rest of it.
        c.transport
            .deliver(GATEWAY, MsgClass::Control, part(0, 2, vec![star(1)]));
        c.step(10.0);
        assert_eq!(stated(&c), 0, "a partial sky is not a held sky");

        // THE LAST PART: the sky is whole, and the client says so.
        c.transport
            .deliver(GATEWAY, MsgClass::Control, part(1, 2, vec![star(2)]));
        c.step(10.0);
        assert_eq!(stated(&c), 1, "the client answered, so the server can stop");

        // A CONFIRMING BEAT RE-STATES IT. A gateway that lost the statement — a reconnect, a session
        // that moved — would otherwise ask for a sky this client already has, for ever.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            postcard::to_allocvec(&ServerControlMsg::SkyAlive { generation: 0x1234 })
                .expect("fixture"),
        );
        c.step(10.0);
        assert_eq!(stated(&c), 2, "a confirming beat re-states what we hold");
    }

    /// ★ A SKY READ FROM DISK IS STATED TOO, AND CROSSES NOTHING (S11).
    ///
    /// This is the whole point of the cache: a returning player draws the galaxy from their own disk,
    /// says which one it is, and the server sends none of it.
    #[test]
    fn a_sky_adopted_from_the_cache_is_stated_without_the_catalogue_crossing() {
        let mut c = core();
        activate(&mut c);
        let rows = vec![vd_core::look::StarRow {
            realm: vd_core::pose::RealmId::System(1),
            cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        }];
        let generation =
            vd_core::look::catalogue_generation(&postcard::to_allocvec(&rows).expect("encodes"));
        let stamp = vd_core::store_stamp::StoreStamp::new(
            vd_core::store_stamp::StoreRole::ClientCatalogue,
            0x5EED,
            vd_core::ids::EpochId(1),
            &[1.0, 2.0],
        );
        let file = crate::star_sky::encode_cache(stamp, generation, &rows);

        assert_eq!(c.state_mut().adopt_sky_cache(&file, &stamp), Ok(generation));
        c.step(10.0);
        assert_eq!(
            c.state().sky.complete(),
            Some(rows),
            "the galaxy is drawable, and not one byte of it crossed the wire"
        );
        assert!(
            c.transport.sent.iter().any(|(_, b)| {
                postcard::from_bytes::<ClientControlMsg>(b)
                    == Ok(ClientControlMsg::SkyHeld { generation })
            }),
            "and the server is told, so it never serves what we already read from disk"
        );
    }

    /// ★ THE SKY SURVIVES A RESTART, AND COSTS THE WIRE NOTHING THE SECOND TIME (S11).
    ///
    /// The whole point of the cache: a returning player draws the galaxy from their own disk, states
    /// the generation, and the server sends none of it.
    #[test]
    fn a_saved_sky_is_read_back_after_a_restart_and_written_only_once() {
        let mut store = vd_sim::io::mem::MemStore::default();
        let stamp = vd_core::store_stamp::StoreStamp::new(
            vd_core::store_stamp::StoreRole::ClientCatalogue,
            0x5EED,
            vd_core::ids::EpochId(1),
            &[1.0, 2.0],
        );
        let star = |n: u64| vd_core::look::StarRow {
            realm: vd_core::pose::RealmId::System(n),
            cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        };
        let rows = vec![star(1), star(2), star(3)];
        let generation =
            vd_core::look::catalogue_generation(&postcard::to_allocvec(&rows).expect("encodes"));

        // THE FIRST SESSION: the sky arrives over the wire, and is written to disk.
        let mut a = core();
        activate(&mut a);
        a.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            postcard::to_allocvec(&ServerControlMsg::StarCatalogue {
                generation,
                part: 0,
                parts: 1,
                rows: rows.clone(),
            })
            .expect("fixture"),
        );
        a.step(10.0);
        assert!(
            a.state_mut().save_sky_cache(&mut store, stamp),
            "the first save writes"
        );
        // ...and asking again writes NOTHING. A caller with no way to know when the sky changed may
        // call this every step; without the guard it would re-encode 7.0 MB twenty times a second.
        assert!(
            !a.state_mut().save_sky_cache(&mut store, stamp),
            "the same sky is not written twice"
        );

        // THE SECOND SESSION — a restart. Nothing has crossed the wire.
        let mut b = core();
        activate(&mut b);
        assert_eq!(b.state().sky.complete(), None, "it starts with no sky");
        assert_eq!(b.state_mut().load_sky_cache(&store, &stamp), Ok(generation));
        assert_eq!(
            b.state().sky.complete(),
            Some(rows),
            "the galaxy came back from disk, and not one byte of it crossed the wire"
        );
        assert_eq!(b.state().sky.cache_adopted, 1);
        assert_eq!(b.state().sky.cache_refused, 0);

        // ★ SAVING WITH NO SKY WRITES NOTHING, and is not an error. A client that has not been served
        // yet has nothing to persist, and a caller that saves every step must not be made to check.
        let mut empty_client = core();
        activate(&mut empty_client);
        let mut untouched = vd_sim::io::mem::MemStore::default();
        assert!(
            !empty_client
                .state_mut()
                .save_sky_cache(&mut untouched, stamp),
            "no sky, nothing written"
        );
        assert!(
            untouched
                .scan(crate::net::ClientState::SKY_CACHE_KEY)
                .is_empty(),
            "and the store really is untouched"
        );

        // ★ HALF A SKY IS NEVER WRITTEN. A partial catalogue has a generation, so the guard above it
        // passes; only this arm stops a galaxy with holes reaching the disk, where it would be adopted
        // on the next run as if it were whole.
        let mut partial = core();
        activate(&mut partial);
        partial.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            postcard::to_allocvec(&ServerControlMsg::StarCatalogue {
                generation,
                part: 0,
                parts: 2,
                rows: vec![star(1)],
            })
            .expect("fixture"),
        );
        partial.step(10.0);
        assert_eq!(
            partial.state().sky.generation(),
            Some(generation),
            "the partial sky DOES have a generation — so the guard above cannot be what stops it"
        );
        let mut half = vd_sim::io::mem::MemStore::default();
        assert!(
            !partial.state_mut().save_sky_cache(&mut half, stamp),
            "half a galaxy is not written"
        );
        assert!(half.scan(crate::net::ClientState::SKY_CACHE_KEY).is_empty());

        // AN EMPTY STORE IS A COUNTED REFUSAL, not a crash and not a silent empty sky. A first run and
        // a wiped disk are the same case: the server simply serves one.
        let empty = vd_sim::io::mem::MemStore::default();
        let mut c = core();
        activate(&mut c);
        assert!(c.state_mut().load_sky_cache(&empty, &stamp).is_err());
        assert_eq!(c.state().sky.cache_refused, 1, "counted, never silent");
        assert_eq!(c.state().sky.complete(), None);

        // A SKY FROM ANOTHER WORLD is refused by the stamp, and leaves this client holding nothing.
        let other = vd_core::store_stamp::StoreStamp::new(
            vd_core::store_stamp::StoreRole::ClientCatalogue,
            0x5EED + 1,
            vd_core::ids::EpochId(1),
            &[1.0, 2.0],
        );
        let mut d = core();
        activate(&mut d);
        assert_eq!(
            d.state_mut().load_sky_cache(&store, &other),
            Err(crate::star_sky::CacheRefusal::ForeignWorld)
        );
    }

    /// ★ THE SKY'S LIVENESS BEAT, END TO END OVER THE WIRE (S11).
    ///
    /// Proves the arm decodes and reaches the assembler on the real ingest path — not just that the
    /// assembler's own method works. The beat is the client's only defence against a silence that
    /// means two things, so a beat that decoded but never landed would restore the ambiguity while
    /// every unit test still passed.
    #[test]
    fn a_wire_delivered_liveness_beat_reaches_the_sky_and_is_counted() {
        let mut c = core();
        activate(&mut c);
        let beat = |generation: u64| {
            postcard::to_allocvec(&ServerControlMsg::SkyAlive { generation }).expect("fixture")
        };
        let ignored_before = c.state().ignored;

        // No sky held yet: the beat proves the lane is alive and confirms nothing more.
        c.transport
            .deliver(GATEWAY, MsgClass::Control, beat(0x1234));
        c.step(10.0);
        assert_eq!(c.state().sky.beats_unheld, 1);
        assert_eq!(
            c.state().ignored,
            ignored_before,
            "a known arm is acted on, never counted as an ignored legacy message"
        );

        // Serve the whole sky, then beat again — now it CONFIRMS, which silence could never do.
        let star = vd_core::look::StarRow {
            realm: vd_core::pose::RealmId::System(1),
            cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        };
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            postcard::to_allocvec(&ServerControlMsg::StarCatalogue {
                generation: 0x1234,
                part: 0,
                parts: 1,
                rows: vec![star],
            })
            .expect("fixture"),
        );
        c.step(10.0);
        c.transport
            .deliver(GATEWAY, MsgClass::Control, beat(0x1234));
        c.step(10.0);
        assert_eq!(c.state().sky.beats_current, 1);

        // A beat naming a different sky says the held one is out of date.
        c.transport
            .deliver(GATEWAY, MsgClass::Control, beat(0x9999));
        c.step(10.0);
        assert_eq!(c.state().sky.beats_stale, 1);
        assert_eq!(
            c.state().sky.generation(),
            Some(0x1234),
            "and it changed nothing — the beat reads, the exchange that replaces a sky is owed"
        );
    }

    #[test]
    fn a_wire_delivered_star_catalogue_assembles_and_the_client_holds_the_sky() {
        let mut c = core();
        activate(&mut c);
        let star = |n: u64| vd_core::look::StarRow {
            realm: vd_core::pose::RealmId::System(n),
            cell: vd_core::glam::I64Vec3::new(1_313_684_865_644_610_304 + n as i64, 7, -3),
            class_code: 6,
            luma_lsun: 0.25,
        };
        let part = |part: u32, parts: u32, rows: Vec<vd_core::look::StarRow>| {
            postcard::to_allocvec(&ServerControlMsg::StarCatalogue {
                generation: 0x1234,
                part,
                parts,
                rows,
            })
            .expect("fixture")
        };

        // ONE PART OF TWO: nothing is drawable. A galaxy with a hole in it looks exactly like a
        // galaxy whose missing star does not exist, so half a sky must be invisible, not partial.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            part(0, 2, vec![star(1), star(2)]),
        );
        c.step(10.0);
        assert!(
            c.state().sky.complete().is_none(),
            "half a catalogue draws nothing"
        );

        // THE LAST PART: the whole sky becomes drawable in one step.
        c.transport
            .deliver(GATEWAY, MsgClass::Control, part(1, 2, vec![star(3)]));
        c.step(10.0);
        let sky = c.state().sky.complete().expect("the sky is whole");

        // ★ THE GATE ITSELF: the count of stars the client HOLDS. At the census this reads 150,000 or
        // the sky did not arrive — and a dropped message can no longer look like a quiet one.
        assert_eq!(sky.len(), 3, "every star in the galaxy arrived");
        assert_eq!(
            sky,
            vec![star(1), star(2), star(3)],
            "and unaltered, in order"
        );
        assert_eq!(c.state().sky.generation(), Some(0x1234));
        assert_eq!(
            c.state().sky.refused,
            0,
            "nothing was refused on a clean sky"
        );
    }

    /// `ServerControlMsg::Event` routes through `on_control` to the ONE `apply_event` handler and
    /// the track is evicted — the routing arm itself, not just the handler.
    #[test]
    fn a_wire_delivered_event_routes_through_on_control_and_evicts() {
        let mut c = core();
        activate(&mut c);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 100, 0.0));
        c.step(10.0);
        assert_eq!(c.state().view().render(100.0).len(), 1);
        let ev = postcard::to_allocvec(&ServerControlMsg::Event(EventMsg::EntityRemoved {
            entity: ent(),
            at: UniverseTick(100),
        }))
        .expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, ev);
        c.step(10.0);
        assert!(
            c.state().view().render(100.0).is_empty(),
            "the wire-delivered removal evicted the track through on_control"
        );
    }

    #[test]
    fn entity_removed_evicts_a_de_owned_entity_copy() {
        // The reliable per-entity eviction (EventMsg::EntityRemoved) via the client's
        // apply_event router. A de-owned copy is dropped; the own identity marker survives.
        let mut c = core();
        activate(&mut c);
        let own =
            postcard::to_allocvec(&ServerControlMsg::OwnEntity { entity: ent() }).expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, own);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 100, 0.0));
        c.step(10.0);
        assert_eq!(
            c.state().view().render(100.0).len(),
            1,
            "the avatar is delivered"
        );
        assert_eq!(c.state().own_entity(), Some(ent()));

        // A Notice event is benign (counted ignored). Then EntityRemoved evicts the avatar.
        c.state_mut()
            .apply_event(EventMsg::Notice { text: "hi".into() });
        c.state_mut().apply_event(EventMsg::EntityRemoved {
            entity: ent(),
            at: UniverseTick(100),
        });
        assert!(
            c.state().view().render(100.0).is_empty(),
            "the removed entity's track is evicted"
        );
        assert_eq!(
            c.state().own_entity(),
            Some(ent()),
            "the identity marker survives its own removal (see DeliveredView::remove_entity)"
        );
    }

    /// ★ THE SKY ANCHOR RIDES THE REALM LANE AND DIES WITH THE ORIGIN (owner ruling 2026-09-02 R1).
    #[test]
    fn the_sky_anchor_is_held_from_an_applied_datagram_and_cleared_by_a_scene_swap() {
        use vd_core::pose::RealmId;
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram};
        let mut c = core();
        activate(&mut c);
        let anchor = StampedPose::at_rest(
            FrameRef::GalaxySpace { galaxy_seed: 1 },
            DVec3::new(6.0, 0.0, -8.0),
            UniverseTick(10),
        );
        let with_anchor = |frame_id: u64, tick: u64, anchor: Option<StampedPose>| {
            postcard::to_allocvec(&RealmSnapshotDatagram {
                sub: SubId(0),
                frame_id,
                source_tick: TickId(1),
                universe_tick: UniverseTick(tick),
                origin_epoch: 0,
                sky_anchor: anchor,
                realms: vec![RealmSnap {
                    realm: RealmId::Planet(7),
                    frame: vd_core::pose::frame_for_realm(RealmId::Planet(7), None)
                        .expect("a seeded realm resolves"),
                    pose: StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::new(1.0e9, 0.0, 0.0),
                        UniverseTick(tick),
                    ),
                }],
            })
            .expect("test fixture")
        };
        assert!(
            c.state().render_snapshot().sky_anchor().is_none(),
            "nothing stated yet"
        );
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            with_anchor(1, 10, Some(anchor)),
        );
        c.step(0.0);
        assert_eq!(c.state().render_snapshot().sky_anchor(), Some(anchor));
        // The diagnosis surface says where the galaxy is, in metres of the galaxy's own frame.
        assert_eq!(
            c.state().devstate(0.0, DevCounters::default()).sky_anchor,
            Some([6.0, 0.0, -8.0])
        );
        // A later datagram WITHOUT an anchor leaves the last one standing (latest-wins by presence).
        c.transport
            .deliver(GATEWAY, MsgClass::RealmSnapshot, with_anchor(2, 11, None));
        c.step(0.0);
        assert_eq!(c.state().render_snapshot().sky_anchor(), Some(anchor));
        // A datagram with an anchor and NO row — a pilot alone in a hull — still delivers it: the
        // anchor applies on the epoch, never on the rows' verdict.
        let alone = StampedPose::at_rest(
            FrameRef::GalaxySpace { galaxy_seed: 1 },
            DVec3::new(9.0, 9.0, 9.0),
            UniverseTick(12),
        );
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            postcard::to_allocvec(&RealmSnapshotDatagram {
                sub: SubId(0),
                frame_id: 3,
                source_tick: TickId(1),
                universe_tick: UniverseTick(12),
                origin_epoch: 0,
                sky_anchor: Some(alone),
                realms: Vec::new(),
            })
            .expect("test fixture"),
        );
        c.step(0.0);
        assert_eq!(c.state().render_snapshot().sky_anchor(), Some(alone));
        // A scene swap — a new origin — drops it: the old anchor placed a realm the picture no
        // longer stands in.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            postcard::to_allocvec(&ServerControlMsg::RealmRegistry {
                origin: RealmId::System(7),
                origin_epoch: 1,
                rows: Vec::new(),
            })
            .expect("encode"),
        );
        c.step(0.0);
        assert!(c.state().render_snapshot().sky_anchor().is_none());
        assert_eq!(
            c.state().devstate(0.0, DevCounters::default()).sky_anchor,
            None
        );
    }

    /// ★ THE EARLY DATAGRAM BRINGS ITS SKY WITH IT. A realm datagram of the NEXT scene can beat the
    /// level that opens that scene. The client holds it for one beat and replays it when the level
    /// lands. Its sky anchor must ride that replay: it belongs to the scene just adopted, so the
    /// star cloud places at once instead of waiting for the next beat.
    ///
    /// The pilot's hull is handed from its star system to the galaxy. The hull's first datagram at
    /// the new epoch arrives first, and it carries where the galaxy is. The level follows.
    #[test]
    fn a_held_early_datagrams_sky_anchor_is_taken_when_its_level_lands() {
        use vd_core::pose::RealmId;
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram};
        let mut c = core();
        activate(&mut c);
        // The scene the client stands in now: origin System(7), epoch 0.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            postcard::to_allocvec(&ServerControlMsg::RealmRegistry {
                origin: RealmId::System(7),
                origin_epoch: 0,
                rows: Vec::new(),
            })
            .expect("encode"),
        );
        c.step(0.0);

        // A datagram of the NEXT epoch, carrying the anchor. It is held, not applied.
        let anchor = StampedPose::at_rest(
            FrameRef::GalaxySpace { galaxy_seed: 1 },
            DVec3::new(3.0, -4.0, 5.0),
            UniverseTick(20),
        );
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            postcard::to_allocvec(&RealmSnapshotDatagram {
                sub: SubId(0),
                frame_id: 9,
                source_tick: TickId(1),
                universe_tick: UniverseTick(20),
                origin_epoch: 1,
                sky_anchor: Some(anchor),
                realms: vec![RealmSnap {
                    realm: RealmId::Planet(7),
                    frame: vd_core::pose::frame_for_realm(RealmId::Planet(7), None)
                        .expect("a seeded realm resolves"),
                    pose: StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::new(2.0e9, 0.0, 0.0),
                        UniverseTick(20),
                    ),
                }],
            })
            .expect("test fixture"),
        );
        c.step(0.0);
        assert_eq!(
            c.state().render_snapshot().sky_anchor(),
            None,
            "a datagram of a scene the client does not hold yet states nothing"
        );

        // The level of that epoch lands, under the SAME origin: the held datagram replays and its
        // anchor is taken.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            postcard::to_allocvec(&ServerControlMsg::RealmRegistry {
                origin: RealmId::System(7),
                origin_epoch: 1,
                rows: Vec::new(),
            })
            .expect("encode"),
        );
        c.step(0.0);
        assert_eq!(
            c.state().render_snapshot().sky_anchor(),
            Some(anchor),
            "the replayed datagram placed the sky on the beat the level landed"
        );
        assert_eq!(
            c.state()
                .realm_view
                .realm_pose(RealmId::Planet(7), f64::INFINITY)
                .map(|p| rpw(&p)),
            Some(DVec3::new(2.0e9, 0.0, 0.0)),
            "and its row replayed too"
        );
    }

    /// ★ A SAME-EPOCH LEVEL RESTATES THE SCENE IN PLACE (2026-09-04): the gateway restates the
    /// level on its keep-alive beat so a client that refused one can catch up. A client already
    /// current replaces its scene with the restated rows and keeps every track — a same-epoch
    /// level is a restatement, never a swap.
    #[test]
    fn a_same_epoch_level_restates_the_scene_and_keeps_the_tracks() {
        use vd_core::pose::RealmId;
        let mut c = core();
        activate(&mut c);
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(RealmId::System(7), 0, Vec::new()),
        );
        c.step(0.0);
        assert!(c.state().render_snapshot().scene().is_empty());
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot(1, 10, RealmId::Planet(7), 1.0e9),
        );
        c.step(0.0);
        assert!(
            c.state()
                .realm_view
                .realm_pose(RealmId::Planet(7), f64::INFINITY)
                .is_some(),
            "the track is held"
        );
        // The restated level at the SAME epoch, now naming the planet's row.
        let row = scene_row(RealmId::Planet(7), Some(RealmId::System(7)), 10.0);
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(RealmId::System(7), 0, vec![row]),
        );
        c.step(0.0);
        assert_eq!(
            c.state().render_snapshot().scene().len(),
            1,
            "the scene is the restated level"
        );
        assert!(
            c.state()
                .realm_view
                .realm_pose(RealmId::Planet(7), f64::INFINITY)
                .is_some(),
            "and the track survived the restatement"
        );
        assert_eq!(c.state().decode_errors, 0);
    }

    #[test]
    fn a_realm_snapshot_routes_streams_anchors_the_cursor_and_is_not_counted_ignored() {
        use vd_core::pose::RealmId;
        let mut c = core();
        activate(&mut c);
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot(1, 10, RealmId::Planet(7), 1.0e9),
        );
        c.step(0.0);
        // The realm view holds the streamed placement (the route reached on_realm_snapshot).
        assert_eq!(
            c.state()
                .realm_view
                .realm_pose(RealmId::Planet(7), f64::INFINITY)
                .map(|p| rpw(&p)),
            Some(DVec3::new(1.0e9, 0.0, 0.0)),
        );
        // The realm feed ANCHORED the render cursor (must-fix ii — a spectator's moving box never
        // freezes): latest_universe_tick advanced even with NO entity snapshot delivered.
        assert_eq!(c.state().latest_universe_tick, Some(10));
        // A RealmSnapshot is NOT counted in `ignored` — it routes to on_realm_snapshot, not the wildcard.
        assert_eq!(c.state().dropped_counts().1, 0);
        // render_snapshot() takes the OVERLAY branch when the realm view is non-empty (the boot scene is
        // empty by default, so the published scene is the empty overlay — the is_empty()==false arm).
        let rs = c.state().render_snapshot();
        assert!(rs.scene().is_empty());
    }

    /// A `RealmSnapshotDatagram` whose row is STATED in `space` — the one-space rule's fixture (the
    /// plain [`realm_snapshot`] hardcodes `SystemSpace{7}`).
    fn realm_snapshot_in(
        frame_id: u64,
        tick: u64,
        realm: vd_core::pose::RealmId,
        space: FrameRef,
        x: f64,
    ) -> Vec<u8> {
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram};
        postcard::to_allocvec(&RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            origin_epoch: 0,
            sky_anchor: None,
            realms: vec![RealmSnap {
                realm,
                frame: vd_core::pose::frame_for_realm(realm, None)
                    .expect("a seeded realm resolves"),
                pose: StampedPose::at_rest(space, DVec3::new(x, 0.0, 0.0), UniverseTick(tick)),
            }],
        })
        .expect("test fixture")
    }

    /// The epoch-AND-space fixture (the one-space-at-current-epoch arm).
    fn realm_snapshot_at_epoch_in(
        origin_epoch: u64,
        frame_id: u64,
        tick: u64,
        realm: vd_core::pose::RealmId,
        space: FrameRef,
        x: f64,
    ) -> Vec<u8> {
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram};
        postcard::to_allocvec(&RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            origin_epoch,
            sky_anchor: None,
            realms: vec![RealmSnap {
                realm,
                frame: vd_core::pose::frame_for_realm(realm, None)
                    .expect("a seeded realm resolves"),
                pose: StampedPose::at_rest(space, DVec3::new(x, 0.0, 0.0), UniverseTick(tick)),
            }],
        })
        .expect("test fixture")
    }

    /// An entity snapshot that puts the OWN avatar in `space` at `x` — the crossing fixture.
    fn own_snapshot_in(frame_id: u64, tick: u64, space: FrameRef, x: f64) -> Vec<u8> {
        postcard::to_allocvec(&SnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            entities: vec![EntitySnap {
                entity: ent(),
                pose: StampedPose::at_rest(space, DVec3::new(x, 0.0, 0.0), UniverseTick(tick)),
            }],
        })
        .expect("test fixture")
    }

    /// THE EPOCH SWAP, end to end at the net layer (§2.7 — the flag day's replacement of the old
    /// space-flip inference): the crossing is STATED by the server as a new composed LEVEL with a
    /// bumped epoch. Adopting it swaps the scene atomically, forgets the old epoch's tracks
    /// (every stored placement was a position in the OLD origin's frame), drops old-epoch
    /// stragglers counted, holds an early NEXT-epoch datagram one beat, and replays it the moment
    /// its level lands — no gap, no double-draw. The one-space row filter stays alive through C1
    /// (§4.5 Topic 4) and keeps old-SPACE rows out even at the current epoch.
    /// ★ THE SAME-ORIGIN SWAP (owner 2026-09-04, *"still blinks and stops for a second on
    /// transitions"*): the hull the pilot stands in is handed from System 7 to the galaxy. The
    /// origin is still the hull, so the epoch bumps but every track and the sky anchor are still
    /// positions in the hull's frame: they are KEPT, and only a realm the new level no longer draws
    /// loses its track. A level with a NEW origin still forgets everything.
    /// ★ THE ANCHOR RIDES A TRACK (owner 2026-09-04, "galaxy, sun and all other objects are not
    /// moving together"): two anchors stated at ticks 10 and 12 give, at cursor 11, the pose halfway
    /// between them — the same interpolation every body gets, at the same cursor.
    #[test]
    fn the_sky_anchor_is_sampled_on_a_track_at_the_render_cursor() {
        use vd_core::pose::RealmId;
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram};
        let mut c = core();
        activate(&mut c);
        let anchor_at = |tick: u64, x: f64| {
            StampedPose::at_rest(
                FrameRef::GalaxySpace { galaxy_seed: 1 },
                DVec3::new(x, 0.0, 0.0),
                UniverseTick(tick),
            )
        };
        let datagram = |frame_id: u64, tick: u64, x: f64| {
            postcard::to_allocvec(&RealmSnapshotDatagram {
                sub: SubId(0),
                frame_id,
                source_tick: TickId(1),
                universe_tick: UniverseTick(tick),
                origin_epoch: 0,
                sky_anchor: Some(anchor_at(tick, x)),
                realms: vec![RealmSnap {
                    realm: RealmId::Planet(7),
                    frame: vd_core::pose::frame_for_realm(RealmId::Planet(7), None)
                        .expect("a seeded realm resolves"),
                    pose: StampedPose::at_rest(
                        FrameRef::SystemSpace { system_seed: 7 },
                        DVec3::new(1.0e9, 0.0, 0.0),
                        UniverseTick(tick),
                    ),
                }],
            })
            .expect("test fixture")
        };
        c.transport
            .deliver(GATEWAY, MsgClass::RealmSnapshot, datagram(1, 10, 6.0));
        c.step(0.0);
        c.transport
            .deliver(GATEWAY, MsgClass::RealmSnapshot, datagram(2, 12, 10.0));
        c.step(0.0);
        let snap = c.state().render_snapshot();
        assert_eq!(
            snap.sky_anchor(),
            Some(anchor_at(12, 10.0)),
            "the newest, as stated"
        );
        let mid = snap
            .sky_anchor_at_cursor(11.0)
            .expect("an anchor on a track");
        let x = mid
            .pos
            .delta_m(
                vd_core::pose::LatticePos::ORIGIN,
                FrameRef::GalaxySpace { galaxy_seed: 1 }.tier(),
            )
            .x;
        assert!(
            (x - 8.0).abs() < 1e-6,
            "halfway between the two statements: {x}"
        );
        assert_eq!(mid.frame, FrameRef::GalaxySpace { galaxy_seed: 1 });
        // At or past the newest: frozen at the newest, never coasted.
        let late = snap.sky_anchor_at_cursor(20.0).expect("an anchor");
        assert_eq!(late.pos, anchor_at(12, 10.0).pos);
    }

    #[test]
    fn a_same_origin_epoch_bump_keeps_the_tracks_and_the_sky_anchor() {
        use vd_core::pose::RealmId;
        use vd_wire::channels::RealmSnapshotDatagram;
        let hull = RealmId::Ship(EntityId::pack(
            vd_core::entity_kind::EntityKind::Ship,
            1,
            1,
            0,
        ));
        let planet = RealmId::Planet(9);
        let star = RealmId::Star(5);
        let mut c = core();
        activate(&mut c);
        // The first level: the origin is the hull, the planet and the star are drawn.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(
                hull,
                1,
                vec![
                    scene_row(planet, Some(hull), 5.0),
                    scene_row(star, Some(hull), 9.0),
                ],
            ),
        );
        c.step(0.0);
        let anchor = StampedPose::at_rest(
            FrameRef::GalaxySpace { galaxy_seed: 1 },
            DVec3::new(6.0, 0.0, -8.0),
            UniverseTick(12),
        );
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            postcard::to_allocvec(&RealmSnapshotDatagram {
                sub: SubId(0),
                frame_id: 7,
                source_tick: TickId(1),
                universe_tick: UniverseTick(12),
                origin_epoch: 1,
                sky_anchor: Some(anchor),
                realms: [(planet, -3.0), (star, 4.0)]
                    .into_iter()
                    .map(|(realm, x)| vd_wire::channels::RealmSnap {
                        realm,
                        frame: vd_core::pose::frame_for_realm(realm, None)
                            .expect("a seeded realm resolves"),
                        pose: StampedPose::at_rest(
                            FrameRef::SystemSpace { system_seed: 7 },
                            DVec3::new(x, 0.0, 0.0),
                            UniverseTick(12),
                        ),
                    })
                    .collect(),
            })
            .expect("test fixture"),
        );
        c.step(0.0);
        assert_eq!(c.state().render_snapshot().sky_anchor(), Some(anchor));
        assert!(c.state().realm_view.realm_pose(planet, 12.0).is_some());

        // THE HAND-OVER: the chain changed, the epoch bumped, the origin is STILL the hull. The
        // swap level names only the planet — the new hop's first level is still in flight — and
        // the star's track is KEPT: a realm missing from a swap level is not gone.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(hull, 2, vec![scene_row(planet, Some(hull), 5.0)]),
        );
        c.step(0.0);
        assert_eq!(c.state().realm_view.epoch(), 2);
        assert_eq!(
            c.state()
                .realm_view
                .realm_pose(planet, 12.0)
                .map(|p| rpw(&p).x),
            Some(-3.0),
            "the planet's track survives a same-origin swap — no refill, no stop"
        );
        assert!(
            c.state().realm_view.realm_pose(star, 12.0).is_some(),
            "a realm missing from the swap level keeps its track — it is coming back"
        );
        // A DELTA says the star left the drawn set: its track goes with it.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            postcard::to_allocvec(&ServerControlMsg::RealmSceneDelta {
                origin: hull,
                origin_epoch: 2,
                added: Vec::new(),
                removed: vec![star],
            })
            .expect("test fixture"),
        );
        c.step(0.0);
        assert_eq!(
            c.state().realm_view.realm_pose(star, 12.0),
            None,
            "a realm a delta removed loses its track"
        );
        assert_eq!(
            c.state().render_snapshot().sky_anchor(),
            Some(anchor),
            "the sky anchor is still the hull's own place in the galaxy"
        );

        // A NEW ORIGIN (the pilot stepped out onto the planet): everything is forgotten.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(planet, 3, vec![scene_row(hull, Some(planet), 1.0)]),
        );
        c.step(0.0);
        assert_eq!(c.state().realm_view.realm_pose(planet, 12.0), None);
        assert!(c.state().render_snapshot().sky_anchor().is_none());
    }

    #[test]
    fn a_new_epoch_level_swaps_the_scene_and_the_old_epochs_placements_are_forgotten() {
        use vd_core::pose::RealmId;
        let sys = FrameRef::SystemSpace { system_seed: 7 };
        let planet = RealmId::Planet(7);
        let planet_space = FrameRef::PlanetCentered { planet_seed: 7 };
        let mut c = core();
        activate(&mut c);
        // Epoch 0 (boot): the planet's row folds normally.
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot_in(1, 10, planet, sys, 17.9),
        );
        c.step(0.0);
        assert_eq!(
            c.state()
                .realm_view
                .realm_pose(planet, f64::INFINITY)
                .map(|p| rpw(&p).x),
            Some(17.9),
            "pre-crossing: the planet's placement folds at the boot epoch"
        );

        // An EARLY next-epoch datagram races its level (the unreliable lane outran the reliable
        // one): HELD one beat, not applied, not dropped.
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot_at_epoch(1, 1, 12, RealmId::Planet(9), -3.0),
        );
        c.step(0.0);
        assert_eq!(
            c.state()
                .realm_view
                .realm_pose(RealmId::Planet(9), f64::INFINITY),
            None,
            "the early new-epoch datagram is held, not applied"
        );

        // THE CROSSING'S LEVEL lands (epoch 1, new origin): the scene swaps atomically, the old
        // epoch's tracks are forgotten, and the held datagram REPLAYS into the fresh view.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(
                planet,
                1,
                vec![scene_row(RealmId::Planet(9), Some(planet), 5.0)],
            ),
        );
        c.step(0.0);
        assert_eq!(
            c.state().realm_view.realm_pose(planet, f64::INFINITY),
            None,
            "the swap forgot the old epoch's placements — the stale pre-crossing track is gone"
        );
        assert_eq!(
            c.state()
                .realm_view
                .realm_pose(RealmId::Planet(9), f64::INFINITY)
                .map(|p| rpw(&p).x),
            Some(-3.0),
            "the held one-beat datagram replayed the moment its level landed"
        );
        assert_eq!(
            c.state().devstate(0.0, DevCounters::default()).origin,
            Some((format!("{planet:?}"), 1)),
            "the origin marker + epoch ride the diagnosis surface"
        );

        // An OLD-epoch straggler (the swapped-away scene's feed still draining) is dropped +
        // counted — never folded into the new picture.
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot_at_epoch(0, 2, 13, planet, 18.4),
        );
        c.step(0.0);
        assert_eq!(
            c.state().realm_view.realm_pose(planet, f64::INFINITY),
            None,
            "an old-epoch row never re-creates the stale track"
        );
        assert_eq!(c.state().realm_view.stale_epoch_rows(), 1);

        // The ONE-SPACE row filter is still alive at the current epoch (it dies in C2 with its
        // cause): give the avatar a delivered frame in the planet's space, then a current-epoch
        // row stated in another SPACE is skipped + counted.
        let own =
            postcard::to_allocvec(&ServerControlMsg::OwnEntity { entity: ent() }).expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, own);
        c.transport.deliver(
            GATEWAY,
            MsgClass::Snapshot,
            own_snapshot_in(1, 14, planet_space, 0.1),
        );
        c.step(0.0);
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot_at_epoch_in(1, 2, 15, planet, sys, 19.0),
        );
        c.step(0.0);
        assert_eq!(
            c.state().realm_view.realm_pose(planet, f64::INFINITY),
            None,
            "a current-epoch row in another SPACE is still skipped by the one-space rule"
        );
        assert!(c.state().realm_view.foreign_space_rows() >= 1);
    }

    #[test]
    fn a_stale_realm_frame_is_dropped_and_a_malformed_one_counts_a_decode_error() {
        use vd_core::pose::RealmId;
        let mut c = core();
        activate(&mut c);
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot(5, 10, RealmId::Planet(7), 1.0),
        );
        c.step(0.0);
        // A STRICTLY-older realm frame is dropped — the placement does not regress.
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot(4, 10, RealmId::Planet(7), 999.0),
        );
        c.step(0.0);
        assert_eq!(
            c.state()
                .realm_view
                .realm_pose(RealmId::Planet(7), f64::INFINITY)
                .map(|p| rpw(&p).x),
            Some(1.0),
        );
        // A malformed realm datagram bumps the SHARED decode_errors counter (DRY).
        c.transport
            .deliver(GATEWAY, MsgClass::RealmSnapshot, vec![0xff, 0xff]);
        c.step(0.0);
        assert_eq!(c.state().dropped_counts().0, 1);
    }

    #[test]
    fn devstate_reports_each_feeds_newest_tick_separately_so_their_drift_is_visible() {
        // SHAKE DIAGNOSIS (slice 3): the ground under a player is placed by the realm feed while the
        // player's own body comes from the entity feed, and the two are authored by DIFFERENT shards
        // whose sense of universe time advances only when a clock sync arrives. If those two numbers
        // drift, the horizon wobbles. Reporting ONE combined tick would hide exactly that, so the two
        // are surfaced separately — and this test feeds them DELIBERATELY DIFFERENT ticks, so a change
        // that collapsed them into one value (or transposed them) fails here.
        use vd_core::pose::RealmId;
        let mut c = core();
        activate(&mut c);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 40, 1.0));
        c.step(0.0);
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot(1, 37, RealmId::Planet(7), 1.0),
        );
        c.step(0.0);

        let st = c.state().devstate(0.0, DevCounters::default());
        assert_eq!(st.entity_feed_newest_tick, Some(40));
        assert_eq!(st.realm_feed_newest_tick, Some(37));
    }

    #[test]
    fn devstate_reports_no_feed_ticks_before_anything_is_delivered() {
        // ABSENT IS NOT ZERO: before the first frame the honest answer is "nothing delivered", not
        // tick 0 — a reader must be able to tell a silent feed from one sitting at the epoch.
        let mut c = core();
        activate(&mut c);
        let st = c.state().devstate(0.0, DevCounters::default());
        assert_eq!(st.entity_feed_newest_tick, None);
        assert_eq!(st.realm_feed_newest_tick, None);
    }

    #[test]
    fn a_streamed_realm_box_reports_the_tick_it_was_drawn_from_and_a_level_box_reports_none() {
        // The per-box half of the same measurement, and the FROZEN-box discriminator: a box the
        // pose feed has streamed carries the tick it was last moved at (so a box that stops
        // advancing is visible as such, and cannot satisfy a smoothness gate by simply not
        // moving); a box that only ever came from the composed LEVEL reports nothing, because no
        // per-tick pose was ever delivered for it — not tick 0. Epoch-0 delta so the epoch-0
        // datagram beside it stays applicable.
        use vd_core::pose::RealmId;
        let mut c = core();
        activate(&mut c);
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(vec![scene_row(RealmId::System(7), None, 40.0)], Vec::new()),
        );
        // Stream a pose for a DIFFERENT realm than the level box, so both arms are exercised.
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot(1, 55, RealmId::Planet(7), 3.0),
        );
        c.step(0.0);

        let dev = c.state().devstate(0.0, DevCounters::default());
        let level_box = dev
            .realm_boxes
            .iter()
            .find(|b| b.realm == format!("{:?}", RealmId::System(7)))
            .expect("the level box is drawn");
        assert_eq!(level_box.newest_tick, None);
        assert_eq!(
            level_box.body_kind, "look",
            "a self-authored outline is a LOOK body on the diagnosis surface"
        );
        assert_eq!(
            level_box.extent_m, 40.0,
            "the STREAMED extent, never a file's"
        );
        assert_eq!(dev.realm_feed_newest_tick, Some(55));
    }

    #[test]
    fn devstate_surfaces_the_realm_feeds_own_fault_counters_not_just_the_entity_views() {
        // The holistic /goal audit MEDIUM (wf_c9444997): devstate() surfaced only the ENTITY view's
        // stale/NaN counters, so a corrupt realm feed was invisible on the HR6 diagnosis surface. The
        // realm feed computes+exposes its OWN counts — they are now SUMMED with the entity view's.
        use vd_core::pose::RealmId;
        let mut c = core();
        activate(&mut c);
        // A realm frame with a non-finite pose (a diverged shard) — a FAULT the realm feed sanitizes.
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot(2, 10, RealmId::Planet(7), f64::NAN),
        );
        c.step(0.0);
        // A strictly-older realm frame for the same realm — a stale drop the realm feed counts.
        c.transport.deliver(
            GATEWAY,
            MsgClass::RealmSnapshot,
            realm_snapshot(1, 10, RealmId::Planet(7), 1.0),
        );
        c.step(0.0);
        let st = c.state().devstate(0.0, DevCounters::default());
        // The entity view contributes 0 here; both realm-feed faults reach the surface (SUMMED).
        assert_eq!(
            st.nonfinite_poses, 1,
            "the realm feed's NaN sanitize is surfaced"
        );
        assert_eq!(
            st.stale_frames_dropped, 1,
            "the realm feed's stale drop is surfaced"
        );
    }

    #[test]
    fn render_snapshot_publishes_the_boot_scene_when_no_realm_streamed() {
        // The byte-identity arm (is_empty()==true): with NO realm frame, render_snapshot() publishes the
        // boot scene unchanged (empty by default) — walk scale is unmoved.
        let mut c = core();
        activate(&mut c);
        let rs = c.state().render_snapshot();
        assert!(rs.scene().is_empty());
    }

    #[test]
    fn snapshots_drive_the_view_and_anchor_a_continuous_cursor() {
        let mut c = core();
        activate(&mut c);
        assert_eq!(c.state().cursor(1.0), None, "no cursor before any snapshot");
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 100, 0.0));
        c.step(10.0);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(2, 102, 10.0));
        c.step(10.0);
        // Anchored at (102, 10.0): cursor advances with wall-time.
        let cursor = c.state().cursor(10.0).expect("anchored");
        assert!((cursor - (102.0 - 2.4)).abs() < 1e-9);
        // The entity renders, interpolated within its window.
        let rendered = c.state().view().render(101.0);
        assert_eq!(rpw(&rendered[&ent()]), DVec3::new(5.0, 0.0, 0.0));
    }

    #[test]
    fn the_client_never_panics_on_unexpected_input() {
        let mut c = core();
        activate(&mut c);
        // Wrong peer, wrong class, undecodable control, undecodable snapshot, an
        // ignored P2 control variant, and a NodeUnreachable — all tolerated.
        c.transport.deliver(OTHER, MsgClass::Control, welcome()); // foreign peer
        c.transport
            .deliver(GATEWAY, MsgClass::Membership, vec![1, 2, 3]); // wrong class
        c.transport
            .deliver(GATEWAY, MsgClass::Control, vec![0xff, 0xff]); // bad control
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, vec![0xff, 0xff]); // bad snapshot
        let ping = postcard::to_allocvec(&ServerControlMsg::Ping { nonce: 1 }).expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, ping); // P2 control, ignored
        c.transport.inbound.push(Inbound::NodeUnreachable {
            to: GATEWAY,
            class: MsgClass::Input,
            undelivered: MsgId(1),
        });
        let r = c.step(0.0);
        assert_eq!(r.received, 6);
        let (decode_errors, ignored, foreign) = c.state().dropped_counts();
        assert_eq!(decode_errors, 2, "the two undecodable payloads");
        assert_eq!(ignored, 2, "the wrong class + the P2 control variant");
        assert_eq!(foreign, 1, "the non-gateway peer");
        assert_eq!(c.state().phase(), ClientPhase::Active, "still alive");
    }

    #[test]
    fn subscription_closing_clears_the_held_sub_but_keeps_entity_tracks() {
        // The reliable per-sub teardown on a PURE-RENDERER client: closing a sub clears it from the
        // held SET (its datagrams stop being admitted) and forgets that sub's staleness high-water,
        // but it does NOT evict entity tracks — those are EntityId-keyed (per-entity eviction is the
        // reliable EventMsg::EntityRemoved). This is the intended pure-renderer split.
        let mut c = core();
        activate(&mut c);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 10, 5.0));
        c.step(10.0);
        assert_eq!(c.state().view().render(10.0).len(), 1, "entity delivered");

        // Closing a sub we do NOT hold leaves the held set + our tracks intact.
        let foreign =
            postcard::to_allocvec(&ServerControlMsg::SubscriptionClosing { sub: SubId(5) })
                .expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, foreign);
        c.step(10.0);
        assert_eq!(
            c.state().held_subs(),
            &std::collections::BTreeSet::from([SubId(0)]),
            "held set unchanged"
        );
        assert_eq!(
            c.state().view().render(10.0).len(),
            1,
            "our entity still present"
        );

        // Close the HELD sub: it leaves the set, but the entity track SURVIVES (EntityId-keyed).
        let closing =
            postcard::to_allocvec(&ServerControlMsg::SubscriptionClosing { sub: SubId(0) })
                .expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, closing);
        c.step(10.0);
        assert_eq!(
            c.state().view().render(10.0).len(),
            1,
            "the entity track survives a SubscriptionClosing (only EntityRemoved evicts it)"
        );
        assert!(c.state().held_subs().is_empty(), "held set cleared");
        // It is NOT counted as an ignored drop — it was handled.
        assert_eq!(c.state().dropped_counts().1, 0, "closing is not 'ignored'");
    }

    #[test]
    fn a_refused_send_is_back_pressure_not_loss() {
        let mut c = core();
        c.transport.refuse = true;
        let r = c.step(0.0);
        assert_eq!(r.sent, 0);
        assert_eq!(
            c.state().phase(),
            ClientPhase::Connecting,
            "Hello retries next step"
        );
        // Heal: Hello leaves, lifecycle proceeds.
        c.transport.refuse = false;
        activate(&mut c);
        // Refuse again while Active: the input is simply not counted (latest-wins).
        c.transport.refuse = true;
        c.state_mut().set_movement([1.0, 0.0, 0.0]);
        let before = c.state().sent_input_count();
        c.step(0.0);
        assert_eq!(
            c.state().sent_input_count(),
            before,
            "refused input not counted"
        );
    }

    #[test]
    fn a_duplicate_welcome_while_active_is_idempotent() {
        let mut c = core();
        activate(&mut c);
        c.transport.deliver(GATEWAY, MsgClass::Control, welcome());
        c.step(0.0);
        assert_eq!(c.state().phase(), ClientPhase::Active, "stays Active");
    }

    #[test]
    fn request_close_sends_bye_and_closes() {
        let mut c = core();
        activate(&mut c);
        c.state_mut().request_close();
        let r = c.step(0.0);
        assert_eq!(r.sent, 1);
        assert_eq!(c.state().phase(), ClientPhase::Closed);
        let (class, buf) = c.transport.sent.last().cloned().expect("test fixture");
        assert_eq!(class, MsgClass::Control);
        assert_eq!(
            postcard::from_bytes::<ClientControlMsg>(&buf).expect("decode bye"),
            ClientControlMsg::Bye
        );
        // Closed: nothing more is sent.
        assert_eq!(c.step(0.0).sent, 0);
    }

    #[test]
    fn a_gateway_close_ends_the_session() {
        let mut c = core();
        activate(&mut c);
        let close = postcard::to_allocvec(&ServerControlMsg::Close {
            reason: "go away".to_owned(),
        })
        .expect("test fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, close);
        c.step(0.0);
        assert_eq!(c.state().phase(), ClientPhase::Closed);
        assert_eq!(c.step(0.0).sent, 0, "nothing sent after a gateway Close");
    }

    #[test]
    fn injected_look_and_action_ride_the_next_input_datagram() {
        let mut c = core();
        activate(&mut c);
        // The dev-control / keyboard injection seam: look + action through the SAME
        // lib setters the assembler reads.
        c.state_mut().add_look([0.25, -0.5]);
        c.state_mut().set_action_bit(0b100, true);
        c.step(0.0);
        let (class, buf) = c.transport.sent.last().cloned().expect("test fixture");
        assert_eq!(class, MsgClass::Input);
        let input: InputDatagram = postcard::from_bytes(&buf).expect("decode input");
        assert_eq!(input.look, [0.25, -0.5]);
        assert_eq!(input.action_bits, 0b100);
    }

    #[test]
    fn the_client_exposes_its_own_node_id() {
        let c = core();
        assert_eq!(c.local_id(), NodeId(100));
    }

    #[test]
    fn the_windowed_pump_assemble_cadence_sends_identical_input_to_step() {
        // T2: a windowed client pumps inbound EVERY render frame but assembles input only
        // on the 20 Hz tick. That must produce the SAME InputDatagram as one `step()` —
        // input is byte-identical regardless of render rate.
        let mut stepped = core();
        let mut windowed = core();
        activate(&mut stepped);
        activate(&mut windowed); // identical state: same seq, same tick
        stepped.state_mut().set_movement([1.0, 0.0, 0.0]);
        windowed.state_mut().set_movement([1.0, 0.0, 0.0]);
        // stepped: one full step (= pump + assemble) -> one input.
        stepped.step(0.0);
        // windowed: three render-frame pumps (no send), then one 20 Hz assemble -> one input.
        windowed.pump_inbound(0.0);
        windowed.pump_inbound(0.0);
        windowed.pump_inbound(0.0);
        let sent = windowed.assemble_input();
        assert_eq!(sent, 1, "exactly one input on the 20 Hz tick, not per pump");
        let from_step = stepped
            .transport
            .sent
            .last()
            .cloned()
            .expect("stepped input");
        let from_windowed = windowed
            .transport
            .sent
            .last()
            .cloned()
            .expect("windowed input");
        assert_eq!(
            from_step, from_windowed,
            "pump×N + assemble == step (byte-identical input)"
        );
    }

    fn universe_rate(tick_hz: u32) -> Vec<u8> {
        postcard::to_allocvec(&ServerControlMsg::UniverseRate { tick_hz }).expect("test fixture")
    }

    #[test]
    fn the_client_learns_the_tick_rate_from_the_wire_once() {
        let mut c = core();
        activate(&mut c);
        // The default lib rate is 20 Hz; the cluster runs 50. Deliver UniverseRate{50}.
        c.transport
            .deliver(GATEWAY, MsgClass::Control, universe_rate(50));
        c.step(0.0);
        // Anchor the cursor and confirm it advances at 50 ticks/sec, not 20.
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 100, 0.0));
        c.step(10.0);
        let c0 = c.state().cursor(10.0).expect("anchored");
        let c1 = c.state().cursor(11.0).expect("anchored");
        assert!(
            (c1 - c0 - 50.0).abs() < 1e-9,
            "1 s advances 50 ticks (learned rate)"
        );
        // A DUPLICATE UniverseRate (even a different value) is idempotent — the rate
        // is learned once and must not reset the cursor.
        c.transport
            .deliver(GATEWAY, MsgClass::Control, universe_rate(200));
        c.step(11.0);
        let c2 = c.state().cursor(12.0).expect("anchored");
        assert!(
            (c2 - c1 - 50.0).abs() < 1e-9,
            "still 50 ticks/sec after a duplicate"
        );
    }

    #[test]
    fn a_zero_tick_rate_is_clamped_so_the_cursor_never_freezes() {
        let mut c = core();
        activate(&mut c);
        c.transport
            .deliver(GATEWAY, MsgClass::Control, universe_rate(0));
        c.step(0.0);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 100, 0.0));
        c.step(10.0);
        // Clamped to 1 Hz: the cursor still ADVANCES (1 tick/sec), never frozen at 0.
        let c0 = c.state().cursor(10.0).expect("anchored");
        let c1 = c.state().cursor(11.0).expect("anchored");
        assert!((c1 - c0 - 1.0).abs() < 1e-9, "clamped to 1 tick/sec");
    }

    #[test]
    fn a_duplicate_subscription_opened_while_active_keeps_the_phase() {
        let mut c = core();
        activate(&mut c); // Active, holding SubId(0)
        // A second SubscriptionOpened while already Active updates the sub but must
        // NOT bounce the phase (covers the not-AwaitingSubscription arm).
        c.transport
            .deliver(GATEWAY, MsgClass::Control, sub_opened());
        c.step(0.0);
        assert_eq!(c.state().phase(), ClientPhase::Active);
    }

    #[test]
    fn a_stale_snapshot_while_active_neither_applies_nor_re_anchors() {
        let mut c = core();
        activate(&mut c);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(5, 100, 0.0));
        c.step(10.0);
        let cursor_before = c.state().cursor(10.0);
        // An older frame on the held sub: the §6.3 gate drops it, so the render
        // cursor is NOT re-anchored to the older tick (covers the not-Apply arm).
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(4, 98, 99.0));
        c.step(10.0);
        assert_eq!(
            c.state().cursor(10.0),
            cursor_before,
            "a stale frame must not re-anchor the cursor"
        );
        // The dropped frame's pose (x=99) never landed.
        assert_eq!(rpw(&c.state().view().render(100.0)[&ent()]).x, 0.0);
    }

    // ---- dev-control glue (Slice 2 T3) --------------------------------------------

    #[test]
    fn injected_input_actions_are_byte_identical_to_keyboard_input() {
        // THE honesty invariant (HR6): a dev-control client driven through the pure
        // `InputAction` seam emits an `InputDatagram` byte-for-byte identical to a
        // client driven through the raw keyboard setters. Two clients stepped in
        // lockstep ⇒ identical seq/tick ⇒ identical bytes.
        let mut dev = core();
        let mut kbd = core();
        activate(&mut dev);
        activate(&mut kbd);
        dev.state_mut()
            .apply_input_action(InputAction::Move([0.5, -0.5, 1.0]));
        dev.state_mut()
            .apply_input_action(InputAction::Look([0.1, 0.2]));
        dev.state_mut().apply_input_action(InputAction::Action {
            bit: 0b10,
            pressed: true,
        });
        kbd.state_mut().set_movement([0.5, -0.5, 1.0]);
        kbd.state_mut().add_look([0.1, 0.2]);
        kbd.state_mut().set_action_bit(0b10, true);
        dev.step(0.0);
        kbd.step(0.0);
        let (dev_class, dev_buf) = dev.transport.sent.last().cloned().expect("dev input");
        let (kbd_class, kbd_buf) = kbd.transport.sent.last().cloned().expect("kbd input");
        assert_eq!(dev_class, MsgClass::Input);
        assert_eq!(kbd_class, MsgClass::Input);
        assert_eq!(
            dev_buf, kbd_buf,
            "dev-injected input is byte-identical to keyboard input"
        );
    }

    #[test]
    fn apply_input_action_close_requests_a_graceful_close() {
        let mut c = core();
        activate(&mut c);
        c.state_mut().apply_input_action(InputAction::Close);
        let r = c.step(0.0);
        assert_eq!(r.sent, 1, "the Bye went out");
        assert_eq!(c.state().phase(), ClientPhase::Closed);
    }

    #[test]
    fn apply_input_action_reset_clears_all_held_input() {
        let mut c = core();
        activate(&mut c);
        c.state_mut()
            .apply_input_action(InputAction::Move([1.0, 1.0, 1.0]));
        c.state_mut().apply_input_action(InputAction::Action {
            bit: 0b1,
            pressed: true,
        });
        c.state_mut().apply_input_action(InputAction::ResetInput);
        c.step(0.0);
        let (_class, buf) = c.transport.sent.last().cloned().expect("input");
        let input: InputDatagram = postcard::from_bytes(&buf).expect("decode input");
        assert_eq!(input.movement, [0.0, 0.0, 0.0], "reset cleared movement");
        assert_eq!(input.action_bits, 0, "reset cleared actions");
    }

    #[test]
    fn every_client_phase_maps_to_its_dev_phase() {
        assert_eq!(dev_phase(ClientPhase::Connecting), DevPhase::Connecting);
        assert_eq!(
            dev_phase(ClientPhase::AwaitingWelcome),
            DevPhase::AwaitingWelcome
        );
        assert_eq!(
            dev_phase(ClientPhase::AwaitingSubscription),
            DevPhase::AwaitingSubscription
        );
        assert_eq!(dev_phase(ClientPhase::Active), DevPhase::Active);
        assert_eq!(dev_phase(ClientPhase::Closed), DevPhase::Closed);
    }

    #[test]
    fn sanitize_forces_floats_finite_so_devstate_always_encodes() {
        assert_eq!(sanitize_f64(1.5), 1.5);
        assert_eq!(sanitize_f64(f64::NAN), 0.0);
        assert_eq!(sanitize_f64(f64::INFINITY), 0.0);
        assert_eq!(sanitize_f64(f64::NEG_INFINITY), 0.0);
        assert_eq!(
            sanitize_vec3(DVec3::new(1.0, f64::NAN, -2.0)),
            [1.0, 0.0, -2.0]
        );
    }

    #[test]
    fn sanitize_quat_keeps_a_finite_rotation_and_maps_a_corrupt_one_to_identity() {
        // A finite quat passes through in x,y,z,w order — the delivered rotation the
        // look-at loop reads as the own facing.
        let q = glam::DQuat::from_xyzw(0.1, 0.2, 0.3, 0.9);
        assert_eq!(sanitize_quat(q), [0.1, 0.2, 0.3, 0.9]);
        // A non-finite quat (a corrupt/diverged sender) maps to IDENTITY — a real
        // rotation, never a zero-length non-rotation that would poison `orient * -Z`.
        let corrupt = glam::DQuat::from_xyzw(f64::NAN, 0.0, 0.0, 1.0);
        assert_eq!(sanitize_quat(corrupt), [0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn devstate_before_welcome_is_honest_about_nothing_delivered() {
        let c = core();
        let s = c.state().devstate(0.0, DevCounters::default());
        assert_eq!(s.phase, DevPhase::Connecting);
        assert_eq!(s.session, None);
        assert_eq!(s.own_entity, None);
        assert_eq!(s.location, None, "no own entity ⇒ no location");
        assert_eq!(s.render_cursor, None);
        assert_eq!(s.universe_tick, None, "no snapshot ⇒ no universe tick");
        assert!(s.entities.is_empty(), "no cursor ⇒ nothing sampled");
        assert_eq!(s.snapshots_applied, 0);
        assert_eq!(s.transfer, DevTransferView::None);
    }

    #[test]
    fn devstate_when_live_reports_the_decoded_delivered_view() {
        let mut c = core();
        activate(&mut c);
        let own =
            postcard::to_allocvec(&ServerControlMsg::OwnEntity { entity: ent() }).expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, own);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 100, 0.0));
        c.step(10.0);
        // dev-command counters are the bin's mailbox totals, passed through verbatim.
        let s = c.state().devstate(
            10.0,
            DevCounters {
                dev_commands_applied: 3,
                dev_commands_dropped: 1,
                stars_drawn: 7,
            },
        );
        assert_eq!(s.phase, DevPhase::Active);
        assert_eq!(s.session, Some(SessionId(9).to_string()));
        assert_eq!(s.own_entity, Some(ent().to_string()));
        assert_eq!(
            s.location,
            Some("System 1".to_owned()),
            "the player's location = the own entity's delivered realm frame"
        );
        assert!(s.render_cursor.is_some(), "anchored after a snapshot");
        assert_eq!(
            s.universe_tick,
            Some(100),
            "the applied snapshot's universe tick"
        );
        assert_eq!(s.entities.len(), 1, "one composited entity");
        assert_eq!(s.entities[0].entity, ent().to_string());
        // The delivered orientation rides each row (x,y,z,w) — the rest pose is identity,
        // proving the orient field flows from the composited pose into the DevState.
        assert_eq!(s.entities[0].orient, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(s.entities[0].authoritative_sub, 0);
        assert_eq!(s.snapshots_applied, 1);
        assert_eq!(c.state().snapshots_applied(), 1);
        assert_eq!(s.dev_commands_applied, 3);
        assert_eq!(s.dev_commands_dropped, 1);
        // The bin's star count rides the same struct: the render thread's number, passed through.
        assert_eq!(s.stars_drawn, 7);
        // The honesty contract end-to-end: the built state JSON-encodes (finite floats).
        let line =
            vd_devproto::encode_response(&vd_devproto::DevResponse::State { state: s.clone() });
        assert!(!line.contains('\n'));
    }

    #[test]
    fn render_snapshot_carries_the_live_view_clock_and_phase() {
        // The Slice-3 T4 hand-off: render_snapshot() reflects the same delivered world
        // the headless DevState path sees, ready for the windowed renderer to sample at
        // its own display cursor.
        let mut c = core();
        activate(&mut c);
        let own =
            postcard::to_allocvec(&ServerControlMsg::OwnEntity { entity: ent() }).expect("fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, own);
        c.transport
            .deliver(GATEWAY, MsgClass::Snapshot, snapshot(1, 100, 0.0));
        c.step(10.0);
        let snap = c.state().render_snapshot();
        assert_eq!(snap.phase(), ClientPhase::Active);
        assert_eq!(snap.own_entity(), Some(ent()));
        assert_eq!(snap.location(), Some("System 1".to_owned()));
        // Anchored after a snapshot → a display cursor, and the entity renders.
        assert!(snap.cursor(10.0).is_some());
        let rendered = snap.rendered(10.0);
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].0, ent());
    }

    #[test]
    fn a_streamed_level_rides_the_render_snapshot_and_defaults_empty() {
        // ONE SOURCE — the stream (D-LANE-6 🟩, owner decision 10): there is no boot file and no
        // load path any more. Default: no level yet ⇒ the render snapshot carries an empty box
        // set; the first composed level then rides EVERY render_snapshot() onto the seam.
        use vd_core::pose::RealmId;
        let mut c = core();
        assert!(
            c.state().render_snapshot().scene().is_empty(),
            "no boxes until a level streams"
        );
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(
                RealmId::System(7),
                1,
                vec![scene_row(RealmId::System(7), None, 40.0)],
            ),
        );
        c.step(0.0);
        let snap = c.state().render_snapshot();
        assert_eq!(snap.scene().len(), 1, "the streamed box is on the seam");
        assert!(snap.scene().get(RealmId::System(7)).is_some());
    }

    #[test]
    fn devstate_exposes_the_overlaid_realm_boxes_for_the_agent_harness() {
        // The `realm_boxes` diagnostic (HR6): `devstate` maps the SAME overlaid scene
        // `render_snapshot` publishes into one `DevRealmBox` per drawn realm — with the STREAMED
        // extent and body kind (§2.11) — so a `vdctl` run can assert a realm landed and its
        // center moves. A MARKER row surfaces as a zero-extent "marker" box (the draw law's other
        // arm on the diagnosis surface).
        use vd_core::pose::RealmId;
        let mut c = core();
        let marker_row = vd_wire::channels::SceneRow {
            realm: RealmId::Planet(9),
            parent: Some(RealmId::System(7)),
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                DVec3::new(30.0, 0.0, 0.0),
                UniverseTick(100),
            ),
            bag: vd_core::look::luma_bag(3, 0.5),
        };
        // A BOX-shaped look too, so the extent surface covers both shapes (a box reports its
        // half-diagonal, a sphere its radius, a marker zero).
        let box_row = vd_wire::channels::SceneRow {
            realm: RealmId::Station(4),
            parent: Some(RealmId::System(7)),
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                DVec3::new(-20.0, 0.0, 0.0),
                UniverseTick(100),
            ),
            bag: vd_core::look::look_bag(&vd_core::geometry::Boundary::Aabb {
                half: DVec3::new(3.0, 4.0, 0.0),
            }),
        };
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(
                RealmId::System(7),
                1,
                vec![
                    scene_row(RealmId::System(7), None, 40.0),
                    marker_row,
                    box_row,
                ],
            ),
        );
        c.step(0.0);
        let dev = c.state().devstate(0.0, DevCounters::default());
        assert_eq!(dev.realm_boxes.len(), 3);
        let station = dev
            .realm_boxes
            .iter()
            .find(|b| b.realm == format!("{:?}", RealmId::Station(4)))
            .expect("the box body is surfaced");
        assert_eq!(
            station.extent_m, 5.0,
            "a box reports its half-diagonal (3-4-5)"
        );
        assert_eq!(station.body_kind, "look");
        let marker = dev
            .realm_boxes
            .iter()
            .find(|b| b.realm == format!("{:?}", RealmId::Planet(9)))
            .expect("the marker row is surfaced");
        assert_eq!(marker.body_kind, "marker");
        assert_eq!(
            marker.extent_m, 0.0,
            "a marker is a point until Slice D's sprites"
        );
        assert_eq!(
            dev.origin,
            Some((format!("{:?}", RealmId::System(7)), 1)),
            "the origin marker rides the surface"
        );
    }

    /// One composed row (proto_minor 18): a realm at the origin frame with a self-authored
    /// `Shell{r}` look — `r` large ⇒ an ambient (skipped) shell, `r` small ⇒ a finite drawn body.
    fn scene_row(
        realm: vd_core::pose::RealmId,
        parent: Option<vd_core::pose::RealmId>,
        r: f64,
    ) -> vd_wire::channels::SceneRow {
        vd_wire::channels::SceneRow {
            realm,
            parent,
            pose: StampedPose::at_rest(
                FrameRef::SystemSpace { system_seed: 0 },
                DVec3::ZERO,
                UniverseTick(100),
            ),
            bag: vd_core::look::look_bag(&vd_core::geometry::Boundary::Shell { r }),
        }
    }

    /// A composed-LEVEL control-message fixture (`RealmRegistry`, §2.4) at `origin`/`epoch`.
    fn scene_level(
        origin: vd_core::pose::RealmId,
        origin_epoch: u64,
        rows: Vec<vd_wire::channels::SceneRow>,
    ) -> Vec<u8> {
        postcard::to_allocvec(&ServerControlMsg::RealmRegistry {
            origin,
            origin_epoch,
            rows,
        })
        .expect("test fixture")
    }

    #[test]
    fn a_streamed_realm_registry_replaces_the_boot_scene_and_skips_ambient_shells() {
        use vd_core::pose::RealmId;
        let mut c = core();
        assert!(
            c.state().render_snapshot().scene().is_empty(),
            "no boxes before any stream arrives"
        );
        // The composed level the connection plane ships: a finite renderable planet under an
        // ~unbounded ambient shell. A fully-agnostic client draws its world from THIS alone.
        let mut ambient = scene_row(RealmId::System(0), None, 1.0);
        // The ambient root STATES NO LOOK (the bound/look split: the source never frames a
        // containment shell) — an empty bag: the row is tracked, never drawn.
        ambient.bag = vd_core::tlv::TlvWriter::new(vd_core::look::WINDOW_BODY_SCHEMA).finish();
        let rows = vec![
            ambient,
            scene_row(RealmId::Planet(7), Some(RealmId::System(0)), 10.0), // finite body — drawn
        ];
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(RealmId::System(0), 1, rows),
        );
        c.step(0.0);
        let snap = c.state().render_snapshot();
        assert_eq!(
            snap.scene().len(),
            1,
            "only the finite leaf is framed (the ambient shell is skipped)"
        );
        assert!(
            snap.scene().get(RealmId::Planet(7)).is_some(),
            "the streamed planet box is on the render seam"
        );
        assert_eq!(
            c.state().dropped_counts().0,
            0,
            "a well-formed stream is no decode error"
        );
    }

    #[test]
    fn a_malformed_streamed_registry_is_counted_and_keeps_the_previous_scene() {
        use vd_core::pose::RealmId;
        let mut c = core();
        // A valid level first establishes a scene.
        let good = vec![scene_row(RealmId::Planet(7), None, 10.0)];
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(RealmId::System(0), 1, good),
        );
        c.step(0.0);
        assert_eq!(c.state().render_snapshot().scene().len(), 1);
        // A DUPLICATE-realm level (a server bug) is rejected LOUD by `from_scene_rows`: counted on
        // the shared decode counter, and the previous good scene is KEPT — never a crash, never an
        // empty flash (and the epoch is NOT adopted from a level the client refused).
        let dup = vec![
            scene_row(RealmId::Planet(7), None, 10.0),
            scene_row(RealmId::Planet(7), None, 10.0),
        ];
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            scene_level(RealmId::System(0), 2, dup),
        );
        c.step(0.0);
        assert_eq!(
            c.state().dropped_counts().0,
            1,
            "the duplicate-realm stream is one decode error"
        );
        let snap = c.state().render_snapshot();
        assert_eq!(
            snap.scene().len(),
            1,
            "the previous good scene is kept when a stream is malformed"
        );
        assert!(snap.scene().get(RealmId::Planet(7)).is_some());
    }

    /// A composed-DELTA control-message fixture (§2.4) at `epoch`: rows that entered (`added`) +
    /// realms that left (`removed`).
    fn realm_scene_delta_at(
        origin_epoch: u64,
        added: Vec<vd_wire::channels::SceneRow>,
        removed: Vec<vd_core::pose::RealmId>,
    ) -> Vec<u8> {
        postcard::to_allocvec(&ServerControlMsg::RealmSceneDelta {
            origin: vd_core::pose::RealmId::System(0),
            origin_epoch,
            added,
            removed,
        })
        .expect("fixture")
    }

    /// The common no-level case: the client's scene epoch boots at 0, so an epoch-0 delta applies
    /// without a preceding level.
    fn realm_scene_delta(
        added: Vec<vd_wire::channels::SceneRow>,
        removed: Vec<vd_core::pose::RealmId>,
    ) -> Vec<u8> {
        realm_scene_delta_at(0, added, removed)
    }

    #[test]
    fn a_streamed_scene_delta_adds_then_removes_a_box() {
        use vd_core::pose::RealmId;
        let mut c = core();
        // A delta ADDING one realm ⇒ the scene gains that box (the view came into range).
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(vec![scene_row(RealmId::Planet(7), None, 10.0)], Vec::new()),
        );
        c.step(0.0);
        assert!(
            c.state()
                .render_snapshot()
                .scene()
                .get(RealmId::Planet(7))
                .is_some(),
            "the entered realm streams onto the render seam"
        );
        assert_eq!(
            c.state().dropped_counts().0,
            0,
            "a well-formed delta is no error"
        );
        // A follow-up delta REMOVING it ⇒ the box leaves the seam (the view moved on).
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(Vec::new(), vec![RealmId::Planet(7)]),
        );
        c.step(0.0);
        assert!(
            c.state()
                .render_snapshot()
                .scene()
                .get(RealmId::Planet(7))
                .is_none(),
            "the departed realm is removed from the seam"
        );
    }

    #[test]
    fn an_off_epoch_scene_delta_is_ignored_counted_and_keeps_the_scene() {
        use vd_core::pose::RealmId;
        let mut c = core();
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(vec![scene_row(RealmId::Planet(7), None, 10.0)], Vec::new()),
        );
        c.step(0.0);
        let ignored_before = c.state().dropped_counts().1;
        // A delta stamped for an epoch this client does not stand in (§2.7: a straggler from a
        // swapped-away scene, or one racing its own level): IGNORED + counted, the scene keeps.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta_at(7, Vec::new(), vec![RealmId::Planet(7)]),
        );
        c.step(0.0);
        assert_eq!(
            c.state().dropped_counts().1,
            ignored_before + 1,
            "the off-epoch delta is counted, never applied"
        );
        assert!(
            c.state()
                .render_snapshot()
                .scene()
                .get(RealmId::Planet(7))
                .is_some(),
            "the standing scene is untouched by the straggler"
        );
    }

    #[test]
    fn a_malformed_scene_delta_is_counted_and_keeps_the_scene() {
        use vd_core::pose::RealmId;
        let mut c = core();
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(vec![scene_row(RealmId::Planet(7), None, 10.0)], Vec::new()),
        );
        c.step(0.0);
        assert_eq!(c.state().render_snapshot().scene().len(), 1);
        // A CYCLIC delta (a server bug) is rejected: counted, previous scene kept, never a partial mutation.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(
                vec![
                    scene_row(RealmId::System(7), Some(RealmId::System(8)), 40.0),
                    scene_row(RealmId::System(8), Some(RealmId::System(7)), 40.0),
                ],
                Vec::new(),
            ),
        );
        c.step(0.0);
        assert_eq!(
            c.state().dropped_counts().0,
            1,
            "the cyclic delta is one decode error"
        );
        assert_eq!(
            c.state().render_snapshot().scene().len(),
            1,
            "the previous scene is kept on a malformed delta"
        );
        assert!(
            c.state()
                .render_snapshot()
                .scene()
                .get(RealmId::Planet(7))
                .is_some()
        );
    }
}
