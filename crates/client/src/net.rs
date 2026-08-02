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
use vd_devproto::{DevEntityRow, DevPhase, DevRealmBox, DevState, DevTransferView, InputAction};
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
    /// The boot-loaded realm-box scene (Visual Crossing Playground V2), shared onto the render
    /// seam via `Arc` so the per-step `render_snapshot()` clone is a pointer bump. The BOOT-STATIC
    /// config (never mutated); a MOVING realm's live pose rides `realm_view` and is OVERLAID at
    /// publish time (FA-2c-3.3), keeping this the pure config source-of-truth.
    scene: Arc<RealmScene>,
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
            realm_view: RealmView::default(),
        }
    }

    /// Boot-load the realm-box render scene (V2). Called ONCE at start-up with the dev-config
    /// `boxes.json` (single-sourced with the shard's boundary plant); the scene then rides every
    /// [`ClientState::render_snapshot`] onto the render seam. It is config, not delivered state, so
    /// this is a non-mutating one-shot set, distinct from the wire ingest path.
    pub fn load_scene(&mut self, scene: RealmScene) {
        self.scene = Arc::new(scene);
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
            // THE streamed realm render-scene (VU proto_minor 5): a fully-agnostic client draws its world
            // from the STREAM ALONE, so REPLACE the boot box scene with the AoI-scoped realm neighbourhood the
            // gateway shipped. The live per-realm poses keep riding `realm_view` and OVERLAY onto these boxes
            // at publish time — exactly as they did over the boot-file scene. `root` (the ambient container)
            // is unused until camera framing (VU-6). A malformed neighbourhood (a duplicate/cyclic realm — a
            // server bug) is counted on the shared decode counter and the previous scene kept, never a crash.
            ServerControlMsg::RealmRegistry {
                regions, pin_abs, ..
            } => {
                // A5 — adopt the SERVER-TOLD render origin (the client subtracts it from every absolute pose it
                // draws). Carry-last-pin (never reset to identity), so a warp re-anchor never teleports the scene.
                self.view.set_render_origin(pin_abs);
                match RealmScene::from_shapes(&regions) {
                    Ok(scene) => self.scene = Arc::new(scene),
                    Err(_) => self.decode_errors += 1,
                }
            }
            // THE incremental render-scene update (VU AoI, proto_minor 6): apply the realms that ENTERED this
            // client's view (`added`) and those that LEFT (`removed`) onto the current scene, so the world
            // follows the view continuously. Atomic — a malformed delta (a cyclic parent chain, a server bug)
            // is counted and the previous scene kept, never a partial mutation. The live per-realm poses keep
            // riding `realm_view` and overlay onto whatever boxes are present.
            ServerControlMsg::RealmSceneDelta {
                added,
                removed,
                pin_abs,
                ..
            } => {
                // A5 — every scene delta re-carries the render origin (a warp re-anchor refreshes it on the
                // reliable scene lane); adopt it the same carry-last-pin way as the registry.
                self.view.set_render_origin(pin_abs);
                match self.scene.with_delta(&added, &removed) {
                    Ok(scene) => self.scene = Arc::new(scene),
                    Err(_) => self.decode_errors += 1,
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

    /// Evict one entity's delivered copy (the reliable `EventMsg::EntityRemoved`, S6): the
    /// server-authoritative "this entity left your view" signal a pure-renderer client needs
    /// because it keys tracks by `EntityId` (a `SubscriptionClosing` can no longer evict them).
    /// Node-agnostic: it names only the entity, never a node.
    ///
    /// ROUTING: `EventMsg` shares the reliable Control carrier but is a DISTINCT enum, so the
    /// gateway will emit it on a dedicated reliable class (owed WITH the send-once server filter,
    /// S0 — until the server stops streaming a de-owned copy there is nothing for this to evict).
    /// This method is the client half, driven by the bin's inbound router and exercised directly in
    /// the unit tests; it decodes an `EventMsg` and applies the eviction.
    pub fn apply_event(&mut self, event: EventMsg) {
        match event {
            EventMsg::EntityRemoved { entity } => self.view.remove_entity(entity),
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
        // Anchor the render cursor only on an APPLIED frame (a held sub, fresh).
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
        if self.realm_view.on_realm_snapshot(snap) == RealmVerdict::Apply {
            self.render_clock.observe(tick, now_s);
            self.latest_universe_tick = Some(tick.0);
        }
    }

    fn send_outbound(&mut self, transport: &mut dyn Transport) -> usize {
        if self.closing && self.phase != ClientPhase::Closed {
            let buf =
                postcard::to_allocvec(&ClientControlMsg::Bye).expect("closed wire enums serialize");
            let _ = self.send_bytes(transport, MsgClass::Control, buf);
            self.phase = ClientPhase::Closed;
            return 1;
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
    fn set_tick_hz_from_wire(&mut self, tick_hz: u32) {
        if self.tick_hz_learned {
            return;
        }
        self.tick_hz_learned = true;
        self.render_clock.set_tick_hz(f64::from(tick_hz).max(1.0));
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
        // FA-2c-3.3: OVERLAY the streamed live realm placements onto the boot scene. When the realm feed
        // is EMPTY (walk/static scale — the server ships no moving realm) publish the boot `Arc` by
        // pointer-bump, so the published scene is byte-identical to the pre-FA-2c wire; otherwise build a
        // fresh immutable overlaid scene (the boot `Arc` is never mutated — no `make_mut` on the shared
        // boot config; the render thread reads an immutable snapshot wait-free, as it already does).
        let scene = if self.realm_view.is_empty() {
            Arc::clone(&self.scene)
        } else {
            Arc::new(self.scene.overlaid(&self.realm_view))
        };
        RenderSnapshot::with_scene(self.view.clone(), self.render_clock, self.phase, scene)
    }

    /// Build the [`DevState`] diagnosis surface (HR6) from the DECODED DELIVERED view
    /// at wall-time `now_s` — wire truth, never internal hope. `dev_commands_applied`/
    /// `dropped` are the bin's mailbox counters (the lib does not own that mailbox), so
    /// the bin passes them in. Every emitted float is sanitized finite, so the
    /// `encode_response` codec is infallible.
    #[must_use]
    pub fn devstate(
        &self,
        now_s: f64,
        dev_commands_applied: u64,
        dev_commands_dropped: u64,
    ) -> DevState {
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
                        // A5 — the RENDER pose: the server ships this absolute; reduce it against the server-told
                        // render origin (the SAME subtraction the renderer draws it with), so the HR6 diagnosis
                        // surface reports exactly what is drawn.
                        pos: sanitize_vec3(self.view.world_pos(&pose, self.view.render_origin())),
                        orient: sanitize_quat(pose.orient),
                        authoritative_sub: sub.0,
                    })
                    .collect()
            })
            .unwrap_or_default();
        // The DRAWN realm boxes (VU diagnosis) — the SAME overlaid scene `render_snapshot`
        // publishes to the renderer (boot/streamed boxes with each streamed realm's live
        // pose overlaid). A `Planet` row proves its `RealmSceneDelta` landed; a center that
        // moves across ticks proves the realm-pose feed is overlaying (the frozen-planet gate).
        let realm_boxes = self
            .render_snapshot()
            .scene()
            .iter()
            .map(|(realm, b)| DevRealmBox {
                realm: format!("{realm:?}"),
                // A5 — the box center is ABSOLUTE (the server ships absolute realm centers); route it through
                // the SAME render-origin subtraction as DevEntityRow.pos, so the diagnosis surface reports both
                // in ONE frame (a raw absolute here would silently disagree the moment the pin is non-identity).
                center: sanitize_vec3(b.center_offset - self.view.render_origin().offset()),
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
            entities,
            realm_boxes,
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
            dev_commands_applied,
            dev_commands_dropped,
            transfer: DevTransferView::None,
        }
    }
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
    use super::*;
    use glam::DVec3;
    use vd_core::entity_kind::EntityKind;
    use vd_core::pose::{FrameRef, StampedPose};
    use vd_core::{AccountId, EpochId, Fence, MsgId, TransferId, UniverseTick};
    use vd_sim::io::{Bytes, SendError};
    use vd_wire::channels::{EntitySnap, RealmShape, SubId};

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
        use vd_wire::channels::{RealmSnap, RealmSnapshotDatagram};
        postcard::to_allocvec(&RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(tick),
            realms: vec![RealmSnap {
                realm,
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

    #[test]
    fn entity_removed_evicts_a_de_owned_entity_copy() {
        // S6: the reliable per-entity eviction (EventMsg::EntityRemoved) via the client's
        // apply_event router. A de-owned copy is dropped; the own avatar's removal clears own_entity.
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
        c.state_mut()
            .apply_event(EventMsg::EntityRemoved { entity: ent() });
        assert!(
            c.state().view().render(100.0).is_empty(),
            "the removed entity's track is evicted"
        );
        assert_eq!(
            c.state().own_entity(),
            None,
            "removing the avatar clears own_entity"
        );
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
                .realm_latest(RealmId::Planet(7))
                .map(|p| p.pos),
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
                .realm_latest(RealmId::Planet(7))
                .map(|p| p.pos.x),
            Some(1.0),
        );
        // A malformed realm datagram bumps the SHARED decode_errors counter (DRY).
        c.transport
            .deliver(GATEWAY, MsgClass::RealmSnapshot, vec![0xff, 0xff]);
        c.step(0.0);
        assert_eq!(c.state().dropped_counts().0, 1);
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
        let st = c.state().devstate(0.0, 0, 0);
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
        assert_eq!(rendered[&ent()].pos, DVec3::new(5.0, 0.0, 0.0));
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
        assert_eq!(c.state().view().render(100.0)[&ent()].pos.x, 0.0);
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
        let s = c.state().devstate(0.0, 0, 0);
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
        let s = c.state().devstate(10.0, 3, 1);
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
    fn a_boot_loaded_scene_rides_the_render_snapshot_and_defaults_empty() {
        use vd_core::geometry::{CrossEffect, RealmBoundary};
        use vd_core::pose::{LatticePos, RealmId};
        let mut c = core();
        // Default: no scene loaded ⇒ the render snapshot carries an empty box set.
        assert!(
            c.state().render_snapshot().scene().is_empty(),
            "no boxes until a scene is loaded"
        );
        // Boot-load a one-box scene; it then rides EVERY render_snapshot() onto the seam.
        let scene = RealmScene::from_boundaries(&[RealmBoundary::shell(
            RealmId::System(7),
            LatticePos::local(DVec3::ZERO),
            1000.0,
            1.15,
            1.30,
            0.0,
            0.05,
            0.5,
            1.0,
            None,
            RealmId::System(7),
            CrossEffect::Authority,
        )])
        .expect("scene");
        c.state_mut().load_scene(scene);
        let snap = c.state().render_snapshot();
        assert_eq!(snap.scene().len(), 1, "the loaded box is on the seam");
        assert!(snap.scene().get(RealmId::System(7)).is_some());
    }

    #[test]
    fn devstate_exposes_the_overlaid_realm_boxes_for_the_agent_harness() {
        // The `realm_boxes` diagnostic (HR6): `devstate` maps the SAME overlaid scene `render_snapshot`
        // publishes into one `DevRealmBox` per drawn realm, so a `vdctl` run can assert a realm landed
        // and its center moves. Load a one-box scene and confirm the row shows up labelled by its realm.
        use vd_core::geometry::{CrossEffect, RealmBoundary};
        use vd_core::pose::{LatticePos, RealmId};
        let mut c = core();
        let scene = RealmScene::from_boundaries(&[RealmBoundary::shell(
            RealmId::System(7),
            LatticePos::local(DVec3::ZERO),
            1000.0,
            1.15,
            1.30,
            0.0,
            0.05,
            0.5,
            1.0,
            None,
            RealmId::System(7),
            CrossEffect::Authority,
        )])
        .expect("scene");
        c.state_mut().load_scene(scene);
        let dev = c.state().devstate(0.0, 0, 0);
        assert_eq!(dev.realm_boxes.len(), 1);
        assert_eq!(
            dev.realm_boxes[0].realm,
            format!("{:?}", RealmId::System(7))
        );
    }

    /// One streamed render shape (VU proto_minor 5): a realm at origin with a `Shell{r}` boundary — `r`
    /// large ⇒ an ambient (skipped) shell, `r` small ⇒ a finite drawn leaf.
    fn realm_shape(
        realm: vd_core::pose::RealmId,
        parent: Option<vd_core::pose::RealmId>,
        r: f64,
    ) -> RealmShape {
        RealmShape {
            realm,
            frame: FrameRef::SystemSpace { system_seed: 0 },
            center: vd_core::pose::LatticePos::local(DVec3::ZERO),
            shape: vd_core::geometry::Boundary::Shell { r },
            parent,
        }
    }

    /// A `RealmRegistry` control-message fixture carrying `regions` (root = the ambient `System(0)`).
    fn realm_registry(regions: Vec<RealmShape>) -> Vec<u8> {
        postcard::to_allocvec(&ServerControlMsg::RealmRegistry {
            regions,
            root: vd_core::pose::RealmId::System(0),
            pin: vd_core::pose::RealmId::System(0),
            pin_abs: vd_core::pose::LatticePos::default(),
            anchor_epoch: 0,
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
        // The AoI neighbourhood the gateway ships: a finite renderable planet under an ~unbounded ambient
        // shell. A fully-agnostic client draws its world from THIS alone — no --realm-boxes file.
        let regions = vec![
            realm_shape(RealmId::System(0), None, 1_000_000_000.0), // ambient — felt, not framed
            realm_shape(RealmId::Planet(7), Some(RealmId::System(0)), 10.0), // finite leaf — drawn
        ];
        c.transport
            .deliver(GATEWAY, MsgClass::Control, realm_registry(regions));
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
        // A valid stream first establishes a scene.
        let good = vec![realm_shape(RealmId::Planet(7), None, 10.0)];
        c.transport
            .deliver(GATEWAY, MsgClass::Control, realm_registry(good));
        c.step(0.0);
        assert_eq!(c.state().render_snapshot().scene().len(), 1);
        // A DUPLICATE-realm neighbourhood (a server bug) is rejected LOUD by `from_shapes`: counted on the
        // shared decode counter, and the previous good scene is KEPT — never a crash, never an empty flash.
        let dup = vec![
            realm_shape(RealmId::Planet(7), None, 10.0),
            realm_shape(RealmId::Planet(7), None, 10.0),
        ];
        c.transport
            .deliver(GATEWAY, MsgClass::Control, realm_registry(dup));
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

    /// A `RealmSceneDelta` control-message fixture (VU AoI, minor 6): realms that entered (`added`) + left
    /// (`removed`).
    fn realm_scene_delta(added: Vec<RealmShape>, removed: Vec<vd_core::pose::RealmId>) -> Vec<u8> {
        postcard::to_allocvec(&ServerControlMsg::RealmSceneDelta {
            added,
            removed,
            pin: vd_core::pose::RealmId::System(0),
            pin_abs: vd_core::pose::LatticePos::default(),
            anchor_epoch: 0,
        })
        .expect("fixture")
    }

    #[test]
    fn a_streamed_scene_delta_adds_then_removes_a_box() {
        use vd_core::pose::RealmId;
        let mut c = core();
        // A delta ADDING one realm ⇒ the scene gains that box (the view came into range).
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(
                vec![realm_shape(RealmId::Planet(7), None, 10.0)],
                Vec::new(),
            ),
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
    fn a_malformed_scene_delta_is_counted_and_keeps_the_scene() {
        use vd_core::pose::RealmId;
        let mut c = core();
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(
                vec![realm_shape(RealmId::Planet(7), None, 10.0)],
                Vec::new(),
            ),
        );
        c.step(0.0);
        assert_eq!(c.state().render_snapshot().scene().len(), 1);
        // A CYCLIC delta (a server bug) is rejected: counted, previous scene kept, never a partial mutation.
        c.transport.deliver(
            GATEWAY,
            MsgClass::Control,
            realm_scene_delta(
                vec![
                    realm_shape(RealmId::System(7), Some(RealmId::System(8)), 40.0),
                    realm_shape(RealmId::System(8), Some(RealmId::System(7)), 40.0),
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
