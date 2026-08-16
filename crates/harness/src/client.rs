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
// Re-exported so the vd-tests render-capture sites can name a `RealmView` without a direct
// vd-client dependency. The client below now FEEDS one (see `realm_view`).
pub use vd_client::realm_view::RealmView;
use vd_core::pose::StampedPose;
use vd_core::{EntityId, NodeId, SessionId, TickId};
use vd_node::TickReport;
use vd_sim::io::{Inbound, MsgClass, Transport};
use vd_wire::channels::{
    ClientControlMsg, InputDatagram, RealmSnapshotDatagram, ServerControlMsg, SnapshotDatagram,
    SnapshotVerdict, SubId, classify_snapshot,
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
    /// How many entity rows each DELIVERED snapshot datagram carried — `rows -> how many datagrams`.
    ///
    /// Wire truth, counted before the staleness gate: this is what the transport actually moved, not what
    /// the view chose to apply. It exists because the per-datagram row count is the number that decides
    /// whether a per-datagram header would be cheaper than per-row data, and that had never been measured
    /// on a real run — only assumed.
    pub snapshot_rows: BTreeMap<usize, u64>,
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
    /// The REAL realm-placement view (`vd_client::realm_view::RealmView`), fed from the SAME realm
    /// datagrams the production client consumes.
    ///
    /// It used to be absent, and the class was simply refused at the dispatch below ("unexpected class
    /// toward a client") — which meant no harness scenario could ever have a MOVING realm in it, because
    /// the first realm frame would panic the client. That is exactly the case the frame-conversion work
    /// exists for, so the gate proving a realm box and the occupant standing in it draw at ONE point had
    /// nowhere to read the box from. Now both lanes land in the two production consumers, and the drawn
    /// point of each is `DeliveredView::world_pos` over a `RenderPose` sampled at ONE cursor.
    pub realm_view: RealmView,
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
            realm_view: RealmView::default(),
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
            // admitted; any OTHER held sub keeps routing. Forget that sub's staleness high-water in
            // the REAL view (per-entity eviction is EventMsg::EntityRemoved now — S6).
            ServerControlMsg::SubscriptionClosing { sub } => {
                self.held_subs.remove(&sub);
                self.delivered_view.drop_sub(sub);
            }
            // THE pure-renderer own-entity signal (minor 2): name the avatar by EntityId alone — no
            // sub, no owning node. A node-agnostic client renders every entity latest-wins by
            // EntityId; this just marks which one is itself.
            ServerControlMsg::OwnEntity { entity } => {
                self.view.own_entity = Some(entity);
                self.delivered_view.set_own_entity(entity);
            }
            ServerControlMsg::Close { reason } => {
                self.close_reason = Some(reason);
                self.phase = ClientPhase::Closed;
            }
            // The cluster tick rate (minor 1): this scripted test client does not
            // interpolate, so it has nothing to apply — ignore it (the real client
            // learns its render rate from this; vd-client net.rs).
            ServerControlMsg::UniverseRate { .. } => {}
            // Node-AWARE legacy `AuthorityChanged` (superseded by `OwnEntity` + EntityId-keyed
            // latest-wins render): the gateway still emits it to OLD (minor<2) clients, so a
            // pure-renderer test client IGNORES it (inert, not a protocol regression).
            ServerControlMsg::AuthorityChanged { .. } => {}
            // The gateway asks the client to cut its input flow. The cut is SERVER-TIMED now (S3),
            // so this marker is INERT server-side — but the harness still stamps it (harmless) to
            // exercise the pause/resume input-partition primitive the input-conservation gates use.
            ServerControlMsg::RequestCut { .. } => self.emit_marker_next = true,
            // The composed scene lane (proto_minor 18): this conservation/render test client
            // draws its world from the ENTITY-snapshot lane and holds no box scene, but it MUST
            // track the scene EPOCH — the composed realm datagrams carry it, and a client that
            // ignored the level would hold every post-swap datagram forever (§2.7). Adopt the
            // level's epoch (the same swap the shipped net.rs runs) and replay the one-beat hold.
            ServerControlMsg::RealmRegistry { origin_epoch, .. } => {
                if let Some(held) = self.realm_view.swap_epoch(origin_epoch) {
                    let standing = self.delivered_view.own_location_frame();
                    let _ = self.realm_view.on_realm_snapshot(standing, held);
                }
            }
            // The composed delta carries no pose feed this client reads — tolerated (the real
            // vd-client applies it to its box scene). Explicit arm — NOT folded into `other` —
            // so a genuinely unexpected control message still fails loudly.
            ServerControlMsg::RealmSceneDelta { .. } => {}
            // THE REMOVE MESSAGE (proto_minor 14, D-4(a)): the reliable per-entity eviction. Both
            // stores this scripted client holds evict — the raw pose map (conservation probes) and
            // the production `DeliveredView` (whose resurrect guard also arms, exactly as in the
            // real client). Gameplay-event variants beyond the removal are ignored here.
            ServerControlMsg::Event(event) => {
                if let vd_wire::channels::EventMsg::EntityRemoved { entity, at } = event {
                    self.view.poses.remove(&entity);
                    self.delivered_view.remove_entity(entity, at);
                }
            }
            // No pings reach a P1/P2 client; arriving here means a protocol regression worth failing loudly.
            other => panic!("unexpected control message: {other:?}"),
        }
    }

    fn on_snapshot(&mut self, bytes: &[u8]) {
        let Ok(snap) = postcard::from_bytes::<SnapshotDatagram>(bytes) else {
            panic!("client received undecodable snapshot bytes from its gateway");
        };
        *self
            .view
            .snapshot_rows
            .entry(snap.entities.len())
            .or_default() += 1;
        // SINGLE shared decode, TWO sinks (1d.3c): feed the SAME decoded `snap` into the REAL
        // composited `DeliveredView` AND the flat conservation `view`, so they cannot diverge in
        // what they admit (both run the SAME §6.3 `classify_snapshot` gate over the SAME held set).
        // The real view keeps per-(sub, entity) tracks (the cross-shard overlap shape); the flat
        // view is last-writer-wins (the D-28 conservation surface).
        // (The old space-flip INFERENCE that ran here is GONE — §2.7: the composed level's epoch
        // swap, consumed in `on_control`, is the one scene-change signal, exactly as in the
        // shipped net.rs. The one-space row filter inside `RealmView` stays alive through C1.)
        let _delivered_verdict = self
            .delivered_view
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

    /// Fold one delivered REALM datagram into the production [`RealmView`] — the ambient world-observation
    /// lane that carries where each MOVING realm's own centre sits, already measured from the centre of
    /// the realm this session is standing in. The GATEWAY does not do that: it routes the shard's bytes
    /// without opening them. Each parent shard on the chain restates the value from its child's centre,
    /// subtracting the one placement it authored, so what lands here is finished.
    ///
    /// Sub-agnostic and un-fenced by design (see `RealmView`): a realm box is world observation, not
    /// per-session authority. The gate is the per-`RealmId` strictly-older check inside `RealmView`, the
    /// same one the shipped client runs, so the harness cannot admit a frame the real client would drop.
    fn on_realm_snapshot(&mut self, bytes: &[u8]) {
        let Ok(snap) = postcard::from_bytes::<RealmSnapshotDatagram>(bytes) else {
            panic!("client received undecodable realm snapshot bytes from its gateway");
        };
        // The one-space ingress rule, identical to the shipped client's net.rs (see RealmView).
        self.realm_view
            .on_realm_snapshot(self.delivered_view.own_location_frame(), snap);
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
                        MsgClass::RealmSnapshot => self.on_realm_snapshot(&bytes),
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
            &ServerControlMsg::OwnEntity {
                entity: EntityId(7),
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
    fn the_legacy_authority_changed_control_is_inertly_ignored_by_the_node_agnostic_client() {
        // `AuthorityChanged` is the pre-S6 NODE-AWARE own-entity signal, superseded by `OwnEntity`.
        // A node-agnostic pure-renderer client tolerates it as a NO-OP: the variant still ships for
        // minor<2 peers, so the exhaustive `on_control` match must handle it — it never panics and
        // does NOT change the own-entity (only `OwnEntity` sets that now). Stays Active.
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client);
        let own_before = client.view.own_entity;
        send_control(
            &mut gw,
            &ServerControlMsg::AuthorityChanged {
                entity: EntityId(7),
                sub: SubId(4),
            },
        );
        fabric.pump(TickId(9));
        let _ = client.step();
        assert_eq!(client.phase(), ClientPhase::Active);
        assert_eq!(
            client.view.own_entity, own_before,
            "legacy AuthorityChanged does NOT change the own entity (only OwnEntity does)"
        );
    }

    #[test]
    fn the_composed_scene_controls_adopt_the_epoch_and_tolerate_the_delta() {
        // The composed scene lane (proto_minor 18): this conservation/render test client draws
        // its world from the ENTITY-snapshot lane and holds no box scene, but it MUST track the
        // scene EPOCH off the level (§2.7 — a client that ignored it would hold every post-swap
        // composed datagram forever), and the delta stays a tolerated no-op. Guards both arms:
        // never a panic, staying Active, the epoch adopted.
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client);
        send_control(
            &mut gw,
            &ServerControlMsg::RealmRegistry {
                origin: vd_core::pose::RealmId::System(7),
                origin_epoch: 3,
                rows: vec![],
            },
        );
        send_control(
            &mut gw,
            &ServerControlMsg::RealmSceneDelta {
                origin: vd_core::pose::RealmId::System(7),
                origin_epoch: 3,
                added: vec![],
                removed: vec![],
            },
        );
        fabric.pump(TickId(9));
        let _ = client.step();
        assert_eq!(client.phase(), ClientPhase::Active);
        assert_eq!(
            client.realm_view.epoch(),
            3,
            "the level's epoch was adopted — post-swap composed datagrams stay applicable"
        );
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
    fn the_client_admits_frames_on_both_held_subs_and_renders_latest_wins() {
        // Multi-realm AoI: after a second SubscriptionOpened (the dest sub of a re-home overlap) the
        // client holds BOTH subs, so a frame on EITHER is admitted (neither dropped as foreign). A
        // pure-renderer client keys tracks by EntityId, so the SAME avatar arriving on both subs
        // folds into ONE track (latest-wins) — it never learns which node owns it.
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client); // SubId(0) held
        send_control(
            &mut gw,
            &ServerControlMsg::SubscriptionOpened {
                sub: SubId(1),
                frame: FrameRef::SystemSpace { system_seed: 8 },
            },
        );
        // The node-agnostic re-confirm of the avatar (the gateway's dest re-point emits OwnEntity).
        send_control(
            &mut gw,
            &ServerControlMsg::OwnEntity {
                entity: EntityId(7),
            },
        );
        fabric.pump(TickId(3));
        let _ = client.step();
        assert_eq!(client.held_subs, BTreeSet::from([SubId(0), SubId(1)]));
        // A frame on the SOURCE sub (0) AND a LATER frame on the DEST sub (1) are BOTH admitted.
        send_snapshot(&mut gw, &snapshot_of(SubId(0), 1, EntityId(7), 1.0));
        send_snapshot(&mut gw, &snapshot_of(SubId(1), 1, EntityId(7), 2.0));
        fabric.pump(TickId(4));
        let _ = client.step();
        assert_eq!(
            client.view.stale_frames_dropped, 0,
            "both held subs admitted — neither frame dropped as foreign"
        );
        assert_eq!(client.view.own_entity, Some(EntityId(7)));

        // The REAL view received the SAME bytes (single shared decode). Per-entity latest-wins: the
        // avatar renders EXACTLY ONCE, at the dest-sub pose x=2.0 (the last-delivered) — node-agnostic.
        let rendered = client.delivered_view.rendered(10.0);
        assert_eq!(rendered.len(), 1, "one track per entity");
        let (rid, _rsub, rpose) = rendered[0];
        assert_eq!(rid, EntityId(7));
        assert_eq!(
            rpose.pos.x, 2.0,
            "latest-wins: the dest-sub pose, not the source copy"
        );
    }

    #[test]
    fn closing_a_sub_keeps_the_entity_track_in_the_real_view() {
        // Pure-renderer split (S6): a reliable SubscriptionClosing forgets the sub's high-water but
        // does NOT evict the EntityId-keyed track (per-entity eviction is EventMsg::EntityRemoved).
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client); // holds SubId(0)
        send_snapshot(&mut gw, &snapshot_of(SubId(0), 1, EntityId(7), 5.0));
        fabric.pump(TickId(3));
        let _ = client.step();
        assert_eq!(
            client.delivered_view.rendered(10.0).len(),
            1,
            "rendered from its per-entity track"
        );
        // Close sub 0: the entity track SURVIVES (only EntityRemoved evicts it).
        send_control(
            &mut gw,
            &ServerControlMsg::SubscriptionClosing { sub: SubId(0) },
        );
        fabric.pump(TickId(4));
        let _ = client.step();
        assert_eq!(
            client.delivered_view.rendered(10.0).len(),
            1,
            "the entity track survives a SubscriptionClosing (EntityId-keyed)",
        );
    }

    /// THE REMOVE MESSAGE at the scripted client (proto_minor 14, D-4(a)): a wire-delivered
    /// `ServerControlMsg::Event(EntityRemoved)` evicts BOTH stores — the raw pose map the
    /// conservation probes read and the production `DeliveredView` — and a Notice event (the
    /// non-removal variant) is a clean no-op, never a panic.
    #[test]
    fn a_wire_delivered_entity_removed_evicts_both_stores_and_a_notice_is_inert() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client);
        send_snapshot(&mut gw, &snapshot_of(SubId(0), 1, EntityId(7), 5.0));
        fabric.pump(TickId(3));
        let _ = client.step();
        assert_eq!(client.delivered_view.rendered(10.0).len(), 1);
        assert!(client.view.poses.contains_key(&EntityId(7)));
        // A Notice is ignored (the non-removal Event variant) — both stores untouched.
        send_control(
            &mut gw,
            &ServerControlMsg::Event(vd_wire::channels::EventMsg::Notice { text: "hi".into() }),
        );
        fabric.pump(TickId(4));
        let _ = client.step();
        assert_eq!(client.delivered_view.rendered(10.0).len(), 1);
        // The removal evicts both stores.
        send_control(
            &mut gw,
            &ServerControlMsg::Event(vd_wire::channels::EventMsg::EntityRemoved {
                entity: EntityId(7),
                at: UniverseTick(100),
            }),
        );
        fabric.pump(TickId(5));
        let _ = client.step();
        assert!(
            client.delivered_view.rendered(10.0).is_empty(),
            "the production view evicted the track"
        );
        assert!(
            !client.view.poses.contains_key(&EntityId(7)),
            "the raw pose map evicted too — the conservation probes see the same world"
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
    #[should_panic(expected = "undecodable realm snapshot bytes")]
    fn garbage_realm_snapshot_bytes_panic_loudly() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        gw.send(CLIENT, MsgClass::RealmSnapshot, vec![0xFF].into())
            .expect("sent");
        fabric.pump(TickId(1));
        let _ = client.step();
    }

    /// The REALM lane lands in the production `RealmView`. It used to panic the client outright
    /// ("unexpected class toward a client"), which meant no harness scenario could contain a moving
    /// realm — the one case the frame-conversion work exists for.
    #[test]
    fn an_epoch_swap_forgets_the_realm_scene_exactly_like_the_shipped_client() {
        // THE FLAG DAY's rewrite of the old frame-flip inference (Slice C1, §2.7): the crossing
        // is STATED by the server as a composed LEVEL with a bumped epoch — never inferred from
        // the entity feed. The harness client mirrors the shipped net.rs: adopting the level's
        // epoch forgets every stored placement (positions in the OLD origin's frame), an
        // old-epoch straggler is dropped counted, and the feed refills at the new epoch.
        let (fabric, mut gw, mut client) = rig(|_| None);
        activate(&fabric, &mut gw, &mut client);
        send_snapshot(&mut gw, &snapshot(SubId(0), 5, 1.0));
        fabric.pump(TickId(3));
        let _ = client.step();
        // A realm placement at the boot epoch (0), held by the production realm view.
        let dg = RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id: 0,
            source_tick: TickId(1),
            universe_tick: UniverseTick(10),
            origin_epoch: 0,
            realms: vec![vd_wire::channels::RealmSnap {
                realm: vd_core::pose::RealmId::Planet(7),
                frame: FrameRef::PlanetCentered { planet_seed: 7 },
                pose: StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 1 },
                    DVec3::new(20.0, 0.0, 0.0),
                    UniverseTick(10),
                ),
            }],
        };
        let bytes = postcard::to_allocvec(&dg).expect("encode");
        gw.send(CLIENT, MsgClass::RealmSnapshot, bytes.into())
            .expect("sent");
        fabric.pump(TickId(4));
        let _ = client.step();
        assert!(
            client
                .realm_view
                .realm_pose(vd_core::pose::RealmId::Planet(7), 10.0)
                .is_some(),
            "the placement is held before the swap"
        );
        // An EARLY next-epoch datagram (racing its own level on the reliable lane): held one
        // beat, invisible now, REPLAYED by the swap below (§2.7's one-beat hold).
        let early = RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id: 1,
            source_tick: TickId(2),
            universe_tick: UniverseTick(11),
            origin_epoch: 1,
            realms: vec![vd_wire::channels::RealmSnap {
                realm: vd_core::pose::RealmId::Station(9),
                frame: FrameRef::StationLocal { station_seed: 9 },
                pose: StampedPose::at_rest(
                    // Stated in the avatar's standing space (the one-space ingress rule the
                    // replay rides through — a foreign-space row would be skipped, not held).
                    FrameRef::SystemSpace { system_seed: 1 },
                    DVec3::new(3.0, 0.0, 0.0),
                    UniverseTick(11),
                ),
            }],
        };
        let early_bytes = postcard::to_allocvec(&early).expect("encode");
        gw.send(CLIENT, MsgClass::RealmSnapshot, early_bytes.into())
            .expect("sent");
        fabric.pump(TickId(5));
        let _ = client.step();
        assert!(
            client
                .realm_view
                .realm_pose(vd_core::pose::RealmId::Station(9), 11.0)
                .is_none(),
            "a next-epoch datagram is HELD, not applied"
        );
        // THE SWAP: the composed LEVEL lands with a bumped epoch (the crossing's one signal) —
        // and the held datagram REPLAYS through the same production ingest.
        send_control(
            &mut gw,
            &ServerControlMsg::RealmRegistry {
                origin: vd_core::pose::RealmId::Planet(7),
                origin_epoch: 1,
                rows: vec![],
            },
        );
        fabric.pump(TickId(6));
        let _ = client.step();
        assert!(
            client
                .realm_view
                .realm_pose(vd_core::pose::RealmId::Planet(7), 12.0)
                .is_none(),
            "the swap forgot every stored placement — the old epoch's space is unrepresentable"
        );
        assert_eq!(client.realm_view.epoch(), 1);
        assert!(
            client
                .realm_view
                .realm_pose(vd_core::pose::RealmId::Station(9), 11.0)
                .is_some(),
            "the held next-epoch datagram REPLAYED at the swap (the one-beat hold, §2.7)"
        );

        // An OLD-epoch straggler (the swapped-away scene's feed still draining) drops counted.
        let stale = postcard::to_allocvec(&dg).expect("encode");
        gw.send(CLIENT, MsgClass::RealmSnapshot, stale.into())
            .expect("sent");
        fabric.pump(TickId(7));
        let _ = client.step();
        assert!(
            client
                .realm_view
                .realm_pose(vd_core::pose::RealmId::Planet(7), 12.0)
                .is_none(),
            "an old-epoch row never re-creates the stale track"
        );
        // ≥ 1, not == 1: the fabric is at-least-once, and this pump's retry window also
        // redelivers the PRE-swap epoch-0 datagram — every copy is an old-epoch straggler and
        // every one must be counted (the measurement fails at 0: a swallowed straggler).
        assert!(client.realm_view.stale_epoch_rows() >= 1);
    }

    #[test]
    fn a_realm_datagram_lands_in_the_production_realm_view() {
        let (fabric, mut gw, mut client) = rig(|_| None);
        let realm = vd_core::pose::RealmId::Planet(7);
        let dg = RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id: 0,
            source_tick: TickId(1),
            universe_tick: UniverseTick(10),
            origin_epoch: 0,
            realms: vec![vd_wire::channels::RealmSnap {
                realm,
                frame: FrameRef::PlanetCentered { planet_seed: 7 },
                pose: StampedPose::at_rest(
                    FrameRef::SystemSpace { system_seed: 7 },
                    DVec3::new(20.0, 0.0, 0.0),
                    UniverseTick(10),
                ),
            }],
        };
        let bytes = postcard::to_allocvec(&dg).expect("encode");
        gw.send(CLIENT, MsgClass::RealmSnapshot, bytes.into())
            .expect("sent");
        fabric.pump(TickId(1));
        let _ = client.step();
        assert_eq!(client.realm_view.frames_applied(), 1);
        let drawn = client
            .realm_view
            .realm_pose(realm, 10.0)
            .expect("the realm the feed just streamed");
        assert_eq!(
            client.delivered_view.world_pos(&drawn),
            DVec3::new(20.0, 0.0, 0.0),
        );
    }

    /// THE ASSERTION THAT CAN TELL "TOLD" FROM "GUESSED" APART — which the number-only one above cannot.
    ///
    /// `a_realm_datagram_lands_in_the_production_realm_view` asserts a VALUE, so it passes whoever did
    /// the arithmetic: it would have passed when the router composed, and it passes now that the shard
    /// chain does. A test whose subject is "the client converts nothing" has to be able to fail when the
    /// client converts something.
    ///
    /// This one ships the SAME integer cell twice, differing only in the UNIT the sender stated for it
    /// (the frame label's tier), and requires the two drawn points to differ by exactly that ratio. A
    /// client that picked metres-per-cell for itself — from a constant, or from any frame other than the
    /// one attached to this value — draws both at the same place and fails here by a factor of ~10^19.
    ///
    /// The other half of the property (that no shard in the chain ever held its own address) is not
    /// observable from a client: a client sees poses, never who computed them. It is asserted where the
    /// shards are, in `vd-tests`' `frame_conversion_e2e`, by asking every level of a real three-node
    /// chain about every realm in the forest and requiring each to answer only for itself and its own
    /// direct children.
    #[test]
    fn the_client_draws_a_cell_in_the_unit_the_sender_stated_for_it() {
        use vd_core::pose::{COARSE_CELL_EDGE_M, FINE_CELL_EDGE_M};
        let (fabric, mut gw, mut client) = rig(|_| None);
        // Two realms, ONE integer cell each, identical value — but authored on the two different
        // lattices the coordinate base supports.
        let fine = vd_core::pose::RealmId::Planet(7);
        let coarse = vd_core::pose::RealmId::System(9);
        let row = |realm, frame| vd_wire::channels::RealmSnap {
            realm,
            frame: FrameRef::PlanetCentered { planet_seed: 7 },
            pose: StampedPose {
                frame,
                pos: vd_core::pose::LatticePos::at(
                    vd_core::glam::I64Vec3::new(1, 0, 0),
                    DVec3::ZERO,
                ),
                vel: DVec3::ZERO,
                orient: vd_core::glam::DQuat::IDENTITY,
                universe_tick: UniverseTick(10),
            },
        };
        let dg = RealmSnapshotDatagram {
            sub: SubId(0),
            frame_id: 0,
            source_tick: TickId(1),
            universe_tick: UniverseTick(10),
            origin_epoch: 0,
            realms: vec![
                row(fine, FrameRef::SystemSpace { system_seed: 7 }),
                row(coarse, FrameRef::GalaxySpace),
            ],
        };
        let bytes = postcard::to_allocvec(&dg).expect("encode");
        gw.send(CLIENT, MsgClass::RealmSnapshot, bytes.into())
            .expect("sent");
        fabric.pump(TickId(1));
        let _ = client.step();

        let drawn = |realm| {
            let p = client
                .realm_view
                .realm_pose(realm, 10.0)
                .expect("the feed streamed this realm");
            client.delivered_view.world_pos(&p)
        };
        assert_eq!(drawn(fine), DVec3::new(FINE_CELL_EDGE_M, 0.0, 0.0));
        assert_eq!(drawn(coarse), DVec3::new(COARSE_CELL_EDGE_M, 0.0, 0.0));
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
