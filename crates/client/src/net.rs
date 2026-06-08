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
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::channels::{
    ClientControlMsg, InputDatagram, ServerControlMsg, SnapshotDatagram, SnapshotVerdict,
};
use vd_wire::seams::tickets::LoginTicket;
use vd_wire::version::ProtoVersion;

use crate::input::InputState;
use crate::render_clock::RenderClock;
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
    sub: Option<vd_wire::channels::SubId>,
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
    decode_errors: u64,
    ignored: u64,
    foreign_peer_drops: u64,
}

impl ClientState {
    #[must_use]
    pub fn new(gateway: NodeId, ticket: LoginTicket, tuning: ClientInterpTuning) -> ClientState {
        ClientState {
            gateway,
            ticket,
            phase: ClientPhase::Connecting,
            session: None,
            sub: None,
            view: DeliveredView::default(),
            render_clock: RenderClock::new(tuning),
            input: InputState::default(),
            tick: TickId(0),
            next_input_seq: 0,
            sent_input_count: 0,
            last_sent_input: None,
            closing: false,
            decode_errors: 0,
            ignored: 0,
            foreign_peer_drops: 0,
        }
    }

    /// One client step: drain delivered messages, then send (Hello while connecting,
    /// a 20 Hz `InputDatagram` while active, `Bye` when closing). `now_s` is wall-time
    /// fed IN by the render loop (the lib reads no clock); it anchors the render cursor.
    pub fn step(&mut self, transport: &mut dyn Transport, now_s: f64) -> ClientStepReport {
        self.tick = self.tick.next();
        let inbound = transport.drain_inbound();
        let received = inbound.len();
        for msg in inbound {
            self.ingest(msg, now_s);
        }
        let sent = self.send_outbound(transport);
        ClientStepReport { received, sent }
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
                self.sub = Some(sub);
                if self.phase == ClientPhase::AwaitingSubscription {
                    self.phase = ClientPhase::Active;
                }
            }
            ServerControlMsg::AuthorityChanged { entity, sub } => {
                self.view.set_authority(entity, sub);
            }
            ServerControlMsg::Close { .. } => {
                self.phase = ClientPhase::Closed;
            }
            // Transfer/cut/ping control is P2+; a P1.5 client ignores it (no crash).
            _ => self.ignored += 1,
        }
    }

    fn on_snapshot(&mut self, bytes: &[u8], now_s: f64) {
        let Ok(snap) = postcard::from_bytes::<SnapshotDatagram>(bytes) else {
            self.decode_errors += 1;
            return;
        };
        let tick = snap.universe_tick;
        // Anchor the render cursor only on an APPLIED frame (our sub, fresh).
        if self.view.on_snapshot(self.sub, snap) == SnapshotVerdict::Apply {
            self.render_clock.observe(tick, now_s);
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
    use vd_core::{AccountId, EpochId, Fence, MsgId, UniverseTick};
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
        fn send(&mut self, _to: NodeId, class: MsgClass, bytes: Bytes) -> Result<MsgId, SendError> {
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
    fn authority_changed_marks_own_entity() {
        let mut c = core();
        activate(&mut c);
        let msg = postcard::to_allocvec(&ServerControlMsg::AuthorityChanged {
            entity: ent(),
            sub: SubId(0),
        })
        .expect("test fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, msg);
        c.step(0.0);
        assert_eq!(c.state().own_entity(), Some(ent()));
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
        let closing =
            postcard::to_allocvec(&ServerControlMsg::SubscriptionClosing { sub: SubId(0) })
                .expect("test fixture");
        c.transport.deliver(GATEWAY, MsgClass::Control, closing); // P2 control, ignored
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
}
