//! `ScriptedClient` — THE P1 test driver (`docs/design/test_harness.md` §scripted
//! client). A real client over the real wire protocol: it logs in with a real
//! ticket, walks via 20 Hz `InputDatagram`s, and observes ONLY its DELIVERED world
//! view (post-fabric: possibly stale, dropped, reordered) — never another node's
//! internals. Instantiated N times, it IS the load driver (no separate tool).
//!
//! Binding assertions built in:
//! - ONE connection: every send targets the construction-time gateway `NodeId`, and
//!   every received byte must come FROM it — any other peer is a hard panic (the
//!   connection-target-constant P1 DoD assertion, structural + checked).
//! - Stale frames are dropped by (sub, frame_id): a STRICTLY older frame_id is
//!   stale, but sibling chunks of the SAME tick share one frame_id
//!   (connection_plane.md §6.3) and are all merged latest-wins, so a reordered
//!   same-tick chunk is never lost.
//! - Every sent input lands in the sent-log the INPUT-CONSERVATION oracle audits.

use std::collections::{BTreeMap, BTreeSet};

use vd_client::view::DeliveredView;
use vd_core::pose::StampedPose;
use vd_core::{EntityId, NodeId, SessionId, TickId};
use vd_node::TickReport;
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::channels::{
    ClientControlMsg, InputDatagram, ServerControlMsg, SnapshotDatagram, SnapshotVerdict, SubId,
    classify_snapshot,
};
use vd_wire::seams::tickets::LoginTicket;
use vd_wire::version::ProtoVersion;

use crate::fabric::FabricTransport;
use crate::topology::{InspectReport, SteppableNode};

/// What the script decides each tick once the session is live.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InputCmd {
    /// Movement axes in [-1, 1] (forward, strafe, vertical).
    pub movement: [f32; 3],
    /// Look delta (yaw, pitch) radians.
    pub look: [f32; 2],
}

/// The client's DELIVERED world view — decoded snapshots only, wire truth.
#[derive(Clone, Debug, Default)]
pub struct DeliveredWorldView {
    /// Latest delivered pose per entity.
    pub poses: BTreeMap<EntityId, StampedPose>,
    /// The entity this client renders authoritatively (from `AuthorityChanged`).
    pub own_entity: Option<EntityId>,
    /// Highest frame id applied per sub (stale frames are dropped by data).
    pub last_frame: BTreeMap<SubId, u64>,
    /// Frames that arrived below the high-water mark (counted, never applied).
    pub stale_frames_dropped: u64,
}

/// Where the client is in its session lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClientPhase {
    /// Will send `Hello` on its first step.
    Connecting,
    /// Hello sent; awaiting `Welcome`.
    AwaitingWelcome,
    /// Welcomed; awaiting `SubscriptionOpened`.
    AwaitingSubscription,
    /// Live: the script drives inputs.
    Active,
    /// The gateway closed the session.
    Closed,
}

/// A boxed per-tick script: sees the delivered view, may emit one input.
pub type ClientScript = Box<dyn FnMut(&DeliveredWorldView) -> Option<InputCmd> + Send>;

/// The scripted client. One logical connection, one script, full logs.
pub struct ScriptedClient {
    transport: FabricTransport,
    /// THE one connection target — never changes, structurally.
    gateway: NodeId,
    ticket: LoginTicket,
    script: ClientScript,
    phase: ClientPhase,
    session: Option<SessionId>,
    /// The SET of subscriptions the client currently holds (Track R / 1d.2d): grown on
    /// `SubscriptionOpened`, shrunk on `SubscriptionClosing`. During a cross-shard transfer
    /// overlap it holds BOTH the source and dest subs, so a frame on EITHER is admitted (the
    /// avatar is then composited to ONE sub via `AuthorityChanged` on the render layer).
    held_subs: BTreeSet<SubId>,
    pub view: DeliveredWorldView,
    /// The REAL composited render view (1d.3c) — the SAME `vd_client::view::DeliveredView` the
    /// production client renders. Fed from the SINGLE shared decode below (two sinks, never two
    /// decodes, so the flat `view` and this cannot diverge in what they admit). Per-`(sub, entity)`
    /// tracks survive the cross-shard overlap, so the capstone can prove the avatar renders EXACTLY
    /// ONCE from the dest sub during the two-holder window — which the flat last-writer-wins `view`
    /// physically cannot represent. The flat `view` stays for the D-28 conservation gate.
    pub delivered_view: DeliveredView,
    next_input_seq: u64,
    sent_inputs: Vec<(SessionId, u64)>,
    close_reason: Option<String>,
    paused: bool,
    /// Set when the gateway asks for the cut (`ServerControlMsg::RequestCut`): the NEXT input
    /// the client sends carries `is_cut_marker = true`. The client emits the marker in RESPONSE
    /// to the gateway (1c); the real net.rs autonomous emit is 1e (D-5).
    emit_marker_next: bool,
    /// The seq the client stamped its CUT_MARKER on — the one value the conservation gate
    /// threads end-to-end (client marker → gateway emit → dest seed). `None` until it emits one.
    marker_seq: Option<u64>,
    tick: TickId,
}

impl ScriptedClient {
    /// `script` is called once per tick while Active; returning `None` skips the
    /// tick (the input flow is latest-wins — silence is legal).
    pub fn new(
        transport: FabricTransport,
        gateway: NodeId,
        ticket: LoginTicket,
        script: impl FnMut(&DeliveredWorldView) -> Option<InputCmd> + Send + 'static,
    ) -> ScriptedClient {
        ScriptedClient {
            transport,
            gateway,
            ticket,
            script: Box::new(script),
            phase: ClientPhase::Connecting,
            session: None,
            held_subs: BTreeSet::new(),
            view: DeliveredWorldView::default(),
            delivered_view: DeliveredView::default(),
            next_input_seq: 0,
            sent_inputs: Vec::new(),
            close_reason: None,
            paused: false,
            emit_marker_next: false,
            marker_seq: None,
            tick: TickId(0),
        }
    }

    #[must_use]
    pub fn phase(&self) -> ClientPhase {
        self.phase
    }

    #[must_use]
    pub fn session(&self) -> Option<SessionId> {
        self.session
    }

    #[must_use]
    pub fn close_reason(&self) -> Option<&str> {
        self.close_reason.as_deref()
    }

    /// Stop driving the script (the session stays live; in-flight traffic
    /// drains) — the quiesce primitive for end-of-run conservation checks.
    pub fn pause_input(&mut self) {
        self.paused = true;
    }

    /// Resume the script after a pause — the cut-window primitive: the client pauses the tick it
    /// stamps the CUT_MARKER (so no `seq > marker` leaks to the gateway before the cut installs),
    /// then resumes once the scenario has stepped past `FreezeSource` so the next inputs buffer.
    pub fn resume_input(&mut self) {
        self.paused = false;
    }

    /// The seq the client stamped its CUT_MARKER on, if it has emitted one — the value the
    /// conservation gate threads end-to-end (never a hardcoded literal).
    #[must_use]
    pub fn marker_seq(&self) -> Option<u64> {
        self.marker_seq
    }

    /// Politely end the session on the next step.
    pub fn send_bye(&mut self) {
        let bytes =
            postcard::to_allocvec(&ClientControlMsg::Bye).expect("closed wire enums serialize");
        let _ = self
            .transport
            .send(self.gateway, MsgClass::Control, vd_sim::io::bytes(bytes));
        self.phase = ClientPhase::Closed;
    }

    fn on_control(&mut self, bytes: &[u8]) {
        let Ok(msg) = postcard::from_bytes::<ServerControlMsg>(bytes) else {
            panic!("client received undecodable control bytes from its gateway");
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
            // The gateway RELIABLY closed a subscription (a transfer's source-sub release, Track
            // R / 1d.2). Remove it from the held SET so its now-foreign datagrams stop being
            // admitted; any OTHER held sub (e.g. the dest sub of the overlap) keeps routing. Drop
            // the closed sub's tracks from the REAL view too — the SOUND per-sub eviction.
            ServerControlMsg::SubscriptionClosing { sub } => {
                self.held_subs.remove(&sub);
                self.delivered_view.drop_sub(sub);
            }
            // STOP discarding the sub (1d.3c): the REAL view records render authority per entity, so
            // the avatar composites to the named (dest) sub during the overlap — the flat view only
            // ever learns the entity id.
            ServerControlMsg::AuthorityChanged { entity, sub } => {
                self.view.own_entity = Some(entity);
                self.delivered_view.set_authority(entity, sub);
            }
            ServerControlMsg::Close { reason } => {
                self.close_reason = Some(reason);
                self.phase = ClientPhase::Closed;
            }
            // The cluster tick rate (minor 1): this scripted test client does not
            // interpolate, so it has nothing to apply — ignore it (the real client
            // learns its render rate from this; vd-client net.rs).
            ServerControlMsg::UniverseRate { .. } => {}
            // The gateway asks the client to cut its input flow: the NEXT input carries the
            // in-band CUT_MARKER (the seq becomes the transfer's marker_seq, threaded end-to-end).
            ServerControlMsg::RequestCut { .. } => self.emit_marker_next = true,
            // No pings or sub teardown reach a P1/P2 client; arriving here means a protocol
            // regression worth failing loudly.
            other => panic!("unexpected control message: {other:?}"),
        }
    }

    fn on_snapshot(&mut self, bytes: &[u8]) {
        let Ok(snap) = postcard::from_bytes::<SnapshotDatagram>(bytes) else {
            panic!("client received undecodable snapshot bytes from its gateway");
        };
        // SINGLE shared decode, TWO sinks (1d.3c): feed the SAME decoded `snap` into the REAL
        // composited `DeliveredView` AND the flat conservation `view`, so they cannot diverge in
        // what they admit (both run the SAME §6.3 `classify_snapshot` gate over the SAME held set).
        // The real view keeps per-(sub, entity) tracks (the cross-shard overlap shape); the flat
        // view is last-writer-wins (the D-28 conservation surface).
        self.delivered_view
            .on_snapshot(&self.held_subs, snap.clone());

        // THE §6.3 gate (shared with the real client via vd_wire, so they cannot
        // drift): a strictly-older frame_id is stale; an EQUAL frame_id is a sibling
        // chunk of the current tick and is applied — each chunk self-contained
        // latest-wins, so reordered same-tick chunks all land.
        let high_water = self.view.last_frame.get(&snap.sub).copied();
        match classify_snapshot(&self.held_subs, high_water, snap.sub, snap.frame_id) {
            SnapshotVerdict::Apply => {
                self.view.last_frame.insert(snap.sub, snap.frame_id);
                for entity in snap.entities {
                    self.view.poses.insert(entity.entity, entity.pose);
                }
            }
            SnapshotVerdict::DropForeignSub | SnapshotVerdict::DropStale => {
                self.view.stale_frames_dropped += 1;
            }
        }
    }

    fn send_input(&mut self, cmd: InputCmd) {
        let session = self.session.expect("active implies welcomed");
        self.next_input_seq += 1;
        let seq = self.next_input_seq;
        // Stamp the CUT_MARKER on the input that follows a RequestCut (PEEK, not take — a refused
        // send must leave the request armed so it retries on the next tick's higher seq).
        let is_cut_marker = self.emit_marker_next;
        let input = InputDatagram {
            seq,
            is_cut_marker,
            client_tick: self.tick,
            movement: cmd.movement,
            look: cmd.look,
            action_bits: 0,
        };
        let bytes = postcard::to_allocvec(&input).expect("closed wire enums serialize");
        if self
            .transport
            .send(self.gateway, MsgClass::Input, vd_sim::io::bytes(bytes))
            .is_ok()
        {
            self.sent_inputs.push((session, seq));
            if is_cut_marker {
                // Recorded (above — the SOLE sent-log emitter, honesty preserved), the request
                // consumed, and the client PAUSES: no `seq > marker` reaches the gateway until the
                // scenario resumes it past `FreezeSource`, so the source/dest input partition is
                // clean by construction.
                self.emit_marker_next = false;
                self.marker_seq = Some(seq);
                self.paused = true;
            }
        }
        // A refused send is back-pressure: the input is simply not sent this tick (latest-wins
        // input tolerates gaps; an un-acked marker request stays armed for the next tick).
    }
}

impl SteppableNode for ScriptedClient {
    fn node_id(&self) -> NodeId {
        self.transport.local_id()
    }

    fn step(&mut self) -> TickReport {
        self.tick = self.tick.next();
        let inbound = self.transport.drain_inbound();
        let drained = inbound.len();
        let mut unreachable = 0usize;
        for msg in inbound {
            match msg {
                Inbound::Wire { from, class, bytes } => {
                    // THE connection-target-constant assertion: the ONLY peer a
                    // client ever hears is its gateway.
                    assert_eq!(
                        from, self.gateway,
                        "client heard from {from} — the one-connection invariant is broken"
                    );
                    match class {
                        MsgClass::Control => self.on_control(&bytes),
                        MsgClass::Snapshot => self.on_snapshot(&bytes),
                        other => panic!("unexpected class toward a client: {other:?}"),
                    }
                }
                // The ScriptedClient is an in-process test double over FabricTransport, which has no
                // retry buffer and so never sheds (R-4d M3 mem/fabric parity) — a SendShed cannot
                // arise here. Folded with NodeUnreachable (one arm — a new Inbound variant still
                // forces reconsideration) as a delivery-failure notice; `shed` on the report it builds
                // stays structurally 0. The REAL client (vd-client `net.rs`) ignores both via let-else.
                Inbound::NodeUnreachable { .. } | Inbound::SendShed { .. } => unreachable += 1,
            }
        }

        let mut sent = 0usize;
        match self.phase {
            ClientPhase::Connecting => {
                let hello = ClientControlMsg::Hello {
                    version: ProtoVersion::CURRENT,
                    login: self.ticket.clone(),
                };
                let bytes = postcard::to_allocvec(&hello).expect("closed wire enums serialize");
                if self
                    .transport
                    .send(self.gateway, MsgClass::Control, vd_sim::io::bytes(bytes))
                    .is_ok()
                {
                    self.phase = ClientPhase::AwaitingWelcome;
                    sent += 1;
                }
            }
            ClientPhase::Active => {
                if !self.paused
                    && let Some(cmd) = (self.script)(&self.view)
                {
                    self.send_input(cmd);
                    sent += 1;
                }
            }
            ClientPhase::AwaitingWelcome
            | ClientPhase::AwaitingSubscription
            | ClientPhase::Closed => {}
        }

        TickReport {
            tick: self.tick,
            drained,
            sent,
            backpressured: 0,
            staging_shed: 0,
            reliable_shed: 0,
            unreachable,
            // A ScriptedClient's FabricTransport never sheds (R-4d M3) — structurally 0.
            shed: 0,
        }
    }

    fn inspect(&mut self) -> InspectReport {
        InspectReport {
            sent_inputs: self.sent_inputs.clone(),
            ..InspectReport::default()
        }
    }

    fn as_any_mut(&mut self) -> Option<&mut dyn std::any::Any> {
        Some(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fabric::{FaultFabric, LinkPolicy};
    use vd_core::glam::DVec3;
    use vd_core::pose::FrameRef;
    use vd_core::{AccountId, EpochId, Fence, UniverseTick};
    use vd_wire::channels::{EntitySnap, SnapshotDatagram};

    const GW: NodeId = NodeId(1);
    const CLIENT: NodeId = NodeId(100);

    fn ticket() -> LoginTicket {
        LoginTicket {
            account: AccountId(5),
            epoch: EpochId(1),
            nonce: 1,
            signature: vec![0; 64],
        }
    }

    fn rig(
        script: impl FnMut(&DeliveredWorldView) -> Option<InputCmd> + Send + 'static,
    ) -> (FaultFabric, crate::fabric::FabricTransport, ScriptedClient) {
        let fabric = FaultFabric::new(5, 2);
        let gw = fabric.register(GW);
        let client = ScriptedClient::new(fabric.register(CLIENT), GW, ticket(), script);
        (fabric, gw, client)
    }

    fn send_control(gw: &mut crate::fabric::FabricTransport, msg: &ServerControlMsg) {
        let bytes = postcard::to_allocvec(msg).expect("encode");
        gw.send(CLIENT, MsgClass::Control, bytes.into())
            .expect("sent");
    }

    fn snapshot_of(sub: SubId, frame_id: u64, entity: EntityId, x: f64) -> SnapshotDatagram {
        SnapshotDatagram {
            sub,
            frame_id,
            source_tick: TickId(1),
            universe_tick: UniverseTick(10),
            entities: vec![EntitySnap {
                entity,
                pose: StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 1 },
                    DVec3::new(x, 0.0, 0.0),
                    UniverseTick(10),
                ),
            }],
        }
    }

    fn snapshot(sub: SubId, frame_id: u64, x: f64) -> SnapshotDatagram {
        snapshot_of(sub, frame_id, EntityId(7), x)
    }

    fn send_snapshot(gw: &mut crate::fabric::FabricTransport, snap: &SnapshotDatagram) {
        let bytes = postcard::to_allocvec(snap).expect("encode");
        gw.send(CLIENT, MsgClass::Snapshot, bytes.into())
            .expect("sent");
    }

    /// Drive the full lifecycle to Active.
    fn activate(
        fabric: &FaultFabric,
        gw: &mut crate::fabric::FabricTransport,
        client: &mut ScriptedClient,
    ) {
        let _ = client.step(); // sends Hello
        assert_eq!(client.phase(), ClientPhase::AwaitingWelcome);
        send_control(
            gw,
            &ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: SessionId(9),
                session_fence: Fence(1),
                epoch: EpochId(1),
            },
        );
        fabric.pump(TickId(1));
        let _ = client.step();
        assert_eq!(client.phase(), ClientPhase::AwaitingSubscription);
        send_control(
            gw,
            &ServerControlMsg::SubscriptionOpened {
                sub: SubId(0),
                frame: FrameRef::SystemSpace { system_seed: 1 },
            },
        );
        send_control(
            gw,
            &ServerControlMsg::AuthorityChanged {
                entity: EntityId(7),
                sub: SubId(0),
            },
        );
        fabric.pump(TickId(2));
        let _ = client.step();
        assert_eq!(client.phase(), ClientPhase::Active);
        assert_eq!(client.session(), Some(SessionId(9)));
        assert_eq!(client.view.own_entity, Some(EntityId(7)));
    }

    #[test]
    fn lifecycle_reaches_active_and_inputs_flow_with_sent_log() {
        let (fabric, mut gw, mut client) = rig(|_| {
            Some(InputCmd {
                movement: [1.0, 0.0, 0.0],
                look: [0.0, 0.0],
            })
        });
        // The activation step itself ran the script once (seq 1).
        activate(&fabric, &mut gw, &mut client);
        let report = client.step();
        assert_eq!(report.sent, 1, "the script drove another input");
        assert_eq!(
            client.inspect().sent_inputs,
            vec![(SessionId(9), 1), (SessionId(9), 2)],
            "the sent-log feeds INPUT-CONSERVATION"
        );
        // Duplicate Welcome/SubscriptionOpened while Active: idempotent no-ops.
        send_control(
            &mut gw,
            &ServerControlMsg::Welcome {
                version: ProtoVersion::CURRENT,
                session: SessionId(9),
                session_fence: Fence(1),
                epoch: EpochId(1),
            },
        );
        send_control(
            &mut gw,
            &ServerControlMsg::SubscriptionOpened {
                sub: SubId(0),
                frame: FrameRef::SystemSpace { system_seed: 1 },
            },
        );
        fabric.pump(TickId(5));
        let _ = client.step();
        assert_eq!(client.phase(), ClientPhase::Active);
    }

    #[test]
    fn the_universe_rate_control_is_tolerated_by_the_scripted_client() {
        // The gateway relays UniverseRate (minor 1) after Welcome; this scripted
        // test client does not interpolate, so it must IGNORE it (never panic on the
        // legitimate variant), staying Active.
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client);
        send_control(&mut gw, &ServerControlMsg::UniverseRate { tick_hz: 50 });
        fabric.pump(TickId(9));
        let _ = client.step();
        assert_eq!(client.phase(), ClientPhase::Active);
    }

    #[test]
    fn snapshots_apply_in_order_stale_drops_and_same_tick_chunks_all_land() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client);
        send_snapshot(&mut gw, &snapshot(SubId(0), 5, 1.0));
        fabric.pump(TickId(3));
        let _ = client.step();
        assert_eq!(client.view.poses[&EntityId(7)].pos.offset().x, 1.0);
        // A STRICTLY older frame and a foreign sub both drop. A SECOND chunk of the
        // SAME tick (frame 5) carrying a DIFFERENT entity is a partitioned sibling
        // (§6.3) and MUST land — even arriving after frame 5's first chunk — or a
        // multi-datagram snapshot would silently lose the entities only that chunk
        // carried. A genuinely newer frame still applies.
        send_snapshot(&mut gw, &snapshot(SubId(0), 4, 9.0)); // strictly older -> drop
        send_snapshot(&mut gw, &snapshot_of(SubId(0), 5, EntityId(8), 7.0)); // sibling chunk -> land
        send_snapshot(&mut gw, &snapshot(SubId(3), 6, 9.0)); // foreign sub -> drop
        send_snapshot(&mut gw, &snapshot(SubId(0), 6, 2.0)); // newer -> apply
        fabric.pump(TickId(4));
        let _ = client.step();
        assert_eq!(
            client.view.poses[&EntityId(7)].pos.offset().x,
            2.0,
            "the newest frame for entity 7 wins"
        );
        assert_eq!(
            client.view.poses[&EntityId(8)].pos.offset().x,
            7.0,
            "the same-tick sibling chunk's entity landed (no MTU-partition loss)"
        );
        assert_eq!(
            client.view.stale_frames_dropped, 2,
            "only the strictly-older frame and the foreign sub dropped"
        );
    }

    #[test]
    fn pause_stops_the_script_and_bye_closes() {
        let (fabric, mut gw, mut client) = rig(|_| {
            Some(InputCmd {
                movement: [1.0, 0.0, 0.0],
                look: [0.0, 0.0],
            })
        });
        activate(&fabric, &mut gw, &mut client);
        client.pause_input();
        let report = client.step();
        assert_eq!(report.sent, 0, "paused: no input");
        client.send_bye();
        assert_eq!(client.phase(), ClientPhase::Closed);
        let report = client.step();
        assert_eq!(report.sent, 0, "closed: nothing more");
        assert_eq!(client.close_reason(), None, "bye is client-initiated");
    }

    #[test]
    fn request_cut_stamps_the_marker_then_pauses_until_resumed() {
        let (fabric, mut gw, mut client) = rig(|_| {
            Some(InputCmd {
                movement: [1.0, 0.0, 0.0],
                look: [0.0, 0.0],
            })
        });
        activate(&fabric, &mut gw, &mut client); // ran the script once (seq 1), no marker yet
        assert_eq!(client.marker_seq(), None, "no marker before RequestCut");

        // The gateway asks the client to cut its input flow.
        send_control(
            &mut gw,
            &ServerControlMsg::RequestCut {
                transfer: vd_core::TransferId(1),
            },
        );
        fabric.pump(TickId(5));

        // In ONE step the client drains RequestCut (drain precedes the script) and the next
        // input it sends carries the marker at its own seq; then it PAUSES so no seq > marker
        // can leak to the gateway before the cut installs.
        let report = client.step();
        assert_eq!(report.sent, 1, "the marker input was sent");
        let m = client.marker_seq().expect("the client stamped a marker");
        let report = client.step();
        assert_eq!(
            report.sent, 0,
            "paused after the marker — no post-marker leak"
        );
        assert_eq!(
            client.marker_seq(),
            Some(m),
            "the marker seq is final while paused"
        );

        // Resume past the cut window: inputs flow again (every seq > the marker).
        client.resume_input();
        let report = client.step();
        assert_eq!(report.sent, 1, "resumed after FreezeSource");
        assert_eq!(
            client.marker_seq(),
            Some(m),
            "the marker seq does not change on resume"
        );
    }

    #[test]
    fn subscription_closing_drops_the_held_sub_and_ignores_a_foreign_one() {
        // 1d.2: SubscriptionClosing removes the sub from the held SET (its datagrams then stop
        // being admitted); a Closing for a sub the client does NOT hold is a no-op.
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client); // holds SubId(0)
        assert_eq!(client.held_subs, BTreeSet::from([SubId(0)]));
        // A Closing for a DIFFERENT sub: the held set is untouched.
        send_control(
            &mut gw,
            &ServerControlMsg::SubscriptionClosing { sub: SubId(9) },
        );
        fabric.pump(TickId(3));
        let _ = client.step();
        assert_eq!(
            client.held_subs,
            BTreeSet::from([SubId(0)]),
            "a foreign Closing is a no-op"
        );
        // A subsequent snapshot on the held sub still applies (still admitted).
        send_snapshot(&mut gw, &snapshot(SubId(0), 1, 1.0));
        fabric.pump(TickId(4));
        let _ = client.step();
        assert_eq!(client.view.poses[&EntityId(7)].pos.offset().x, 1.0);
        // Now close the HELD sub: it leaves the set, and a later same-sub datagram is no longer
        // admitted (a foreign sub now).
        send_control(
            &mut gw,
            &ServerControlMsg::SubscriptionClosing { sub: SubId(0) },
        );
        fabric.pump(TickId(5));
        let _ = client.step();
        assert!(
            client.held_subs.is_empty(),
            "the held sub was dropped from the set"
        );
        send_snapshot(&mut gw, &snapshot(SubId(0), 2, 9.0));
        fabric.pump(TickId(6));
        let _ = client.step();
        assert_eq!(
            client.view.poses[&EntityId(7)].pos.offset().x,
            1.0,
            "a datagram on the closed sub is dropped (no longer admitted)"
        );
        assert_eq!(
            client.view.stale_frames_dropped, 1,
            "the closed-sub datagram dropped foreign"
        );
    }

    #[test]
    fn the_client_admits_frames_on_both_held_subs_during_a_transfer_overlap() {
        // 1d.2d: after a second SubscriptionOpened (the dest sub of a transfer overlap) the
        // client holds BOTH subs, so a frame on EITHER is admitted (neither dropped as foreign).
        // AuthorityChanged re-points the avatar to the dest sub.
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client); // SubId(0) held
        send_control(
            &mut gw,
            &ServerControlMsg::SubscriptionOpened {
                sub: SubId(1),
                frame: FrameRef::SystemSpace { system_seed: 8 },
            },
        );
        send_control(
            &mut gw,
            &ServerControlMsg::AuthorityChanged {
                entity: EntityId(7),
                sub: SubId(1),
            },
        );
        fabric.pump(TickId(3));
        let _ = client.step();
        assert_eq!(client.held_subs, BTreeSet::from([SubId(0), SubId(1)]));
        // A frame on the SOURCE sub (0) AND a frame on the DEST sub (1) are BOTH admitted.
        send_snapshot(&mut gw, &snapshot_of(SubId(0), 1, EntityId(7), 1.0));
        send_snapshot(&mut gw, &snapshot_of(SubId(1), 1, EntityId(7), 2.0));
        fabric.pump(TickId(4));
        let _ = client.step();
        assert_eq!(
            client.view.stale_frames_dropped, 0,
            "both held subs admitted — neither frame dropped as foreign"
        );
        // The avatar's authoritative sub is the dest (the AuthorityChanged re-point).
        assert_eq!(client.view.own_entity, Some(EntityId(7)));

        // 1d.3c: the REAL composited view received the SAME bytes (single shared decode). Both
        // subs hold a track for the avatar, but it renders EXACTLY ONCE — from the dest sub (1),
        // the AuthorityChanged re-point — at the dest's x=2.0. The flat view physically cannot
        // show this (last-writer-wins), which is why the capstone reads the REAL view.
        let rendered = client.delivered_view.rendered(10.0);
        assert_eq!(
            rendered.len(),
            1,
            "the avatar composites to exactly one copy"
        );
        let (rid, rsub, rpose) = rendered[0];
        assert_eq!(rid, EntityId(7));
        assert_eq!(
            rsub,
            SubId(1),
            "rendered from the dest sub (the AuthorityChanged re-point)"
        );
        assert_eq!(rpose.pos.x, 2.0, "the dest-sub pose, not the source copy");
    }

    #[test]
    fn closing_a_sub_evicts_its_tracks_from_the_real_view_too() {
        // 1d.3c: a reliable SubscriptionClosing drops the closed sub's tracks from the REAL
        // composited view (the SOUND per-sub eviction), not just the held SET — so the avatar's
        // dest-sub copy is the one that survives a source-sub release during a transfer.
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client); // holds SubId(0), authority on SubId(0)
        send_snapshot(&mut gw, &snapshot_of(SubId(0), 1, EntityId(7), 5.0));
        fabric.pump(TickId(3));
        let _ = client.step();
        assert_eq!(
            client.delivered_view.rendered(10.0).len(),
            1,
            "rendered on sub 0"
        );
        // Close sub 0: its track is evicted from the real view, so the avatar renders nowhere.
        send_control(
            &mut gw,
            &ServerControlMsg::SubscriptionClosing { sub: SubId(0) },
        );
        fabric.pump(TickId(4));
        let _ = client.step();
        assert!(
            client.delivered_view.rendered(10.0).is_empty(),
            "the closed sub's track was evicted from the real composited view",
        );
    }

    #[test]
    fn identity_and_downcast_hooks() {
        let (_fabric, _gw, mut client) = rig(|_| None);
        assert_eq!(SteppableNode::node_id(&client), CLIENT);
        let any = client.as_any_mut().expect("clients opt into downcasting");
        assert!(any.downcast_mut::<ScriptedClient>().is_some());
    }

    #[test]
    fn close_records_the_reason() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        let _ = client.step();
        send_control(
            &mut gw,
            &ServerControlMsg::Close {
                reason: "go away".to_owned(),
            },
        );
        fabric.pump(TickId(1));
        let _ = client.step();
        assert_eq!(client.phase(), ClientPhase::Closed);
        assert_eq!(client.close_reason(), Some("go away"));
    }

    #[test]
    fn refused_sends_are_backpressure_not_loss() {
        // (Never reaches Active, so no script body is needed.)
        let (fabric, _gw, mut client) = rig(|_| None);
        // Every send refused: Hello cannot leave; the client keeps trying.
        fabric.set_policy(
            CLIENT,
            GW,
            LinkPolicy {
                send_reject_p: 1.0,
                ..LinkPolicy::default()
            },
        );
        let report = client.step();
        assert_eq!(report.sent, 0);
        assert_eq!(
            client.phase(),
            ClientPhase::Connecting,
            "hello retries next tick"
        );
        // Heal: hello leaves on the next step.
        fabric.set_policy(CLIENT, GW, LinkPolicy::default());
        let report = client.step();
        assert_eq!(report.sent, 1);
        assert_eq!(client.phase(), ClientPhase::AwaitingWelcome);
    }

    #[test]
    fn refused_input_sends_are_not_logged_as_sent() {
        let (fabric, mut gw, mut client) = rig(|_| {
            Some(InputCmd {
                movement: [1.0, 0.0, 0.0],
                look: [0.0, 0.0],
            })
        });
        // Activation already sent seq 1; then every further send is refused.
        activate(&fabric, &mut gw, &mut client);
        fabric.set_policy(
            CLIENT,
            GW,
            LinkPolicy {
                send_reject_p: 1.0,
                ..LinkPolicy::default()
            },
        );
        let _ = client.step();
        assert_eq!(
            client.inspect().sent_inputs,
            vec![(SessionId(9), 1)],
            "the refused input never entered the sent-log (conservation honesty)"
        );
    }

    #[test]
    fn unreachable_notices_are_counted() {
        let (fabric, _gw, mut client) = rig(|_| None);
        fabric.kill(GW);
        let _ = client.step(); // hello accepted into the queue
        fabric.pump(TickId(1)); // bounces as NodeUnreachable
        let report = client.step();
        assert_eq!(report.unreachable, 1);
    }

    #[test]
    #[should_panic(expected = "one-connection invariant is broken")]
    fn a_foreign_peer_breaks_the_one_connection_invariant() {
        let (fabric, _gw, mut client) = rig(|_| None);
        let mut intruder = fabric.register(NodeId(66));
        intruder
            .send(CLIENT, MsgClass::Control, vec![1].into())
            .expect("sent");
        fabric.pump(TickId(1));
        let _ = client.step();
    }

    #[test]
    #[should_panic(expected = "undecodable control bytes")]
    fn garbage_control_bytes_panic_loudly() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        gw.send(CLIENT, MsgClass::Control, vec![0xFF].into())
            .expect("sent");
        fabric.pump(TickId(1));
        let _ = client.step();
    }

    #[test]
    #[should_panic(expected = "undecodable snapshot bytes")]
    fn garbage_snapshot_bytes_panic_loudly() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        gw.send(CLIENT, MsgClass::Snapshot, vec![0xFF].into())
            .expect("sent");
        fabric.pump(TickId(1));
        let _ = client.step();
    }

    #[test]
    #[should_panic(expected = "unexpected control message")]
    fn out_of_phase_protocol_messages_panic_loudly() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        send_control(&mut gw, &ServerControlMsg::Ping { nonce: 1 });
        fabric.pump(TickId(1));
        let _ = client.step();
    }

    #[test]
    #[should_panic(expected = "unexpected class toward a client")]
    fn wrong_message_classes_panic_loudly() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        gw.send(CLIENT, MsgClass::Saga, vec![1].into())
            .expect("sent");
        fabric.pump(TickId(1));
        let _ = client.step();
    }
}
