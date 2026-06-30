//! P2 transfer definition-of-done gates (PLAN.md P2). PERMANENT — no later phase may
//! regress them.
//!
//! The D-28 cross-cut INPUT-CONSERVATION gate: a player's input survives a REAL cross-shard
//! authority handoff exactly once, in order, with the CUT_MARKER threaded end-to-end (client →
//! saga → gateway → dest). It is the FIRST proof of the core transfer thesis, and the first to
//! JOIN the two crate halves — the gateway's buffer/drain (connection-plane) and the dest's
//! `OpenInputSlot`/`apply_input` (sim) — in one scenario over the real fabric, driven by the
//! REAL saga producer (never hand-fed acks).
//!
//! SCOPE (D-28 + 1d.5b.1): the transfer commits authority to the dest (directory CAS lands) AND the
//! ORDERED demote/release tail now CLOSES — the saga drives Swapping → Demoting → Promoting →
//! Releasing → Done: RouteSwapped pushes the saga `Demote` to the source; `DemoteAck` advances to
//! Promoting + pushes the `Promote` to the dest; `PromoteAck` AND `DeliveredToObservers` (the
//! Promoting gate) emit `ReleaseSubscribe` → `Released` → Done. So this gate proves BOTH input
//! conservation across the cut AND the final authority settle: `verify_authority_unique` +
//! `verify_authority_settled` after a full transfer (the D-28(b) graduation). FIDELITY
//! (pose/render/ghost) still defers to 1d (D-27 back half).

use vd_core::entity_kind::DurabilityClass;
use vd_core::glam::DVec3;
use vd_core::pose::{FrameRef, RealmId};
use vd_core::{AccountId, EntityId, NodeId, SessionId, TickId, TransferId};
use vd_harness::client::ScriptedClient;
use vd_harness::fabric::{FaultFabric, LinkPolicy};
use vd_harness::oracle::{
    RenderSample, RenderTolerances, RenderTrace, verify_authority_settled, verify_authority_unique,
    verify_input_conservation, verify_no_vanish, verify_pose_continuity,
};
use vd_harness::topology::{InspectReport, Topology};
use vd_sim::saga::SagaCtx;
use vd_tests::{
    DEST, GATEWAY, ORCH, SHARD, gateway_buffered_count, live_sagas, p1_client, p2_cluster,
    read_subject, saga_states, stub_config, trigger_transfer, walk_forward,
};
use vd_wire::channels::SubId;
use vd_wire::seams::directory::{AuthorityRef, DirectoryKey};

const CLIENT: NodeId = NodeId(100);

/// The render-trace cursor: a large finite value so `DeliveredView::rendered` clamps to (freezes
/// at) the FRESHEST delivered pose every tick — the actual last-delivered crossing pose, sampled
/// deterministically (no interpolation ambiguity in the capture).
const TRACE_CURSOR: f64 = 1.0e9;

/// One captured tick of the subject's composited render: its sample (`None` ⇒ rendered nowhere)
/// and the SUBS that hold a track for it (the anti-vacuity overlap probe — both source + dest live).
#[derive(Clone, Debug, PartialEq)]
struct CapturedTick {
    tick: TickId,
    sample: Option<RenderSample>,
    subs_holding: Vec<SubId>,
}

/// Capture the subject's composited render sample + held-subs from the client's REAL `DeliveredView`
/// this tick. The subject is the client's OWN entity (set by the login `AuthorityChanged`). Built
/// from delivered bytes only — never node internals.
fn capture_subject(topo: &mut Topology) -> CapturedTick {
    let tick = topo.tick();
    with_client(topo, |c| {
        let view = &c.delivered_view;
        let sample = view.own_entity().and_then(|own| {
            view.rendered(TRACE_CURSOR)
                .into_iter()
                .find(|(e, _, _)| *e == own)
                .map(|(_, sub, pose)| RenderSample {
                    sub,
                    frame: pose.frame,
                    raw_pos: pose.pos,
                    world_pos: view.world_pos(&pose, TRACE_CURSOR),
                    orient: pose.orient,
                })
        });
        let subs_holding = view
            .own_entity()
            .map(|own| view.subs_holding(own))
            .unwrap_or_default();
        CapturedTick {
            tick,
            sample,
            subs_holding,
        }
    })
}

fn report(reports: &[(NodeId, InspectReport)], id: NodeId) -> &InspectReport {
    &reports
        .iter()
        .find(|(n, _)| *n == id)
        .expect("node present")
        .1
}

fn with_client<R>(topo: &mut Topology, f: impl FnOnce(&mut ScriptedClient) -> R) -> R {
    let node = topo.node_mut(CLIENT).expect("client present");
    let client = node
        .as_any_mut()
        .expect("clients opt into downcasting")
        .downcast_mut::<ScriptedClient>()
        .expect("the node is a ScriptedClient");
    f(client)
}

/// Step (at most `max` ticks) until `cond` holds — a BOUNDED, deterministic poll on
/// deterministic state. Robust to unrelated timing shifts (unlike a hardcoded absolute tick);
/// the determinism sibling proves two runs poll byte-identically. Panics if never reached.
/// `observe` runs after EVERY step (the per-tick render-trace sampler — `&mut |_| {}` for the
/// gates that capture no trace), so the capstone samples DURING the overlap, not post-quiesce.
fn step_until(
    topo: &mut Topology,
    max: u64,
    observe: &mut dyn FnMut(&mut Topology),
    mut cond: impl FnMut(&mut Topology) -> bool,
) {
    for _ in 0..max {
        topo.step();
        observe(topo);
        if cond(topo) {
            return;
        }
    }
    panic!("condition not reached within {max} ticks");
}

/// Drive ONE full cross-shard transfer of a logged-in player from [`SHARD`] to [`DEST`] over
/// the real fabric, returning the quiesced topology plus the session, transferred entity, and
/// the marker seq the run threaded (all observed LIVE, never hardcoded).
///
/// `observe` is invoked after EVERY `topo.step()` — the per-tick observer (skeptic D2): the
/// function deliberately quiesces to single-holder steady state before returning, so a trace
/// sampled from the RETURNED topology renders "exactly once from the dest" TRIVIALLY (the source is
/// already gone). The capstone passes an observer that captures the subject's composited render
/// sample each tick, so it can assert exactly-once DURING the two-holder overlap. Gates that need no
/// trace pass `&mut |_| {}` (via [`run_cut_transfer_quiesced`]).
fn run_cut_transfer(
    fabric: &FaultFabric,
    observe: &mut dyn FnMut(&mut Topology),
) -> (Topology, SessionId, EntityId, u64) {
    let mut topo = p2_cluster(fabric, 8);
    topo.add_node(Box::new(p1_client(
        fabric,
        CLIENT,
        AccountId(1000),
        walk_forward(),
    )));

    // WARMUP: the player logs in and walks; the SOURCE grants its avatar and the DEST wins its
    // own realm lease (so `OpenInputSlot` will not be deferred). The client is emitting a
    // contiguous seq stream that the source applies.
    step_until(&mut topo, 80, observe, |t| {
        let r = t.inspect_all();
        !report(&r, SHARD).held_entities.is_empty()
            && report(&r, DEST)
                .held_realms
                .iter()
                .any(|(realm, _)| *realm == RealmId::System(8))
    });

    // TRIGGER the REAL transfer of the avatar. The CAS expectation is the DIRECTORY record's
    // fence (read live), the subject the directory `Entity` key.
    let (session, entity, fence) = read_subject(&mut topo);
    trigger_transfer(
        &mut topo,
        SagaCtx {
            transfer: TransferId(1),
            session,
            subject: DirectoryKey::Entity(entity),
            expected_fence: fence,
            source: SHARD,
            dest: DEST,
            class: DurabilityClass::Durable,
            needs_provision: false,
            // The realms the player crosses BETWEEN (source SHARD = System(7), dest = System(8)) —
            // stamped onto the StubCrossing the saga emits at commit (1d.1).
            from_realm: RealmId::System(7),
            to_realm: RealmId::System(8),
        },
    );

    // The saga reaches `Freezing`: `FreezeSource` is in flight and the cut installs on the
    // gateway next tick. The client has already stamped the CUT_MARKER (in response to the
    // gateway's `RequestCut`) and PAUSED, so no seq > marker has leaked to the source.
    step_until(&mut topo, 40, observe, |t| {
        saga_states(t).iter().any(|s| s.starts_with("Freezing"))
    });

    // RESUME now: the next inputs (seq > marker) reach the gateway AFTER the cut is installed,
    // so they BUFFER and are drained to the dest at commit — the buffer/drain path under test.
    with_client(&mut topo, ScriptedClient::resume_input);

    // The saga commits (directory CAS to DEST) and the ORDERED demote-before-promote tail (1d.5b.1)
    // drives it the rest of the way: Demote→DemoteAck advances Demoting→Promoting + pushes Promote;
    // PromoteAck AND DeliveredToObservers (the Promoting gate) emit ReleaseSubscribe, the gateway
    // acks Released → Done → Tombstone, so the saga reaches `live() == 0` (no longer parks).
    step_until(&mut topo, 40, observe, |t| live_sagas(t) == 0);

    // QUIESCE: stop emitting and let every in-flight post-marker input settle at the dest AND the
    // source's saga-driven Entity self-fence (`on_saga_demote`) land — the source dot demotes to a
    // retained Ghost (no longer simulating).
    with_client(&mut topo, ScriptedClient::pause_input);
    for _ in 0..8 {
        topo.step();
        observe(&mut topo);
    }

    // STEADY-STATE GUARD: wait until the transient transfer windows are resolved — the SOURCE
    // reports NOTHING for the subject (its dot self-fenced away via the saga `Demote`) AND the DEST
    // holds it — BEFORE the authority oracle (`verify_authority_unique`) is sampled. The ordered
    // demote-before-promote leaves a transient post-CAS window (the source holds at the old fence
    // until its `Demote`, then a brief zero-Owned handoff gap before the dest's adopt-promote — both
    // LEGAL mid-flight); the strict `== 1` oracle samples only after steady state, exactly as
    // input-conservation is asserted post-quiesce. (The per-tick mid-flight uniqueness gate, with
    // the transfer-window excuses, lands in 1d.5b.3 — see DEFERRED.md D-2.)
    step_until(&mut topo, 40, observe, |t| {
        let r = t.inspect_all();
        let src = report(&r, SHARD);
        let dst = report(&r, DEST);
        let source_clear = !src.held_entities.iter().any(|(e, _)| *e == entity)
            && !src.pending_entities.contains(&entity)
            && !src.departing_entities.contains(&entity);
        let dest_holds = dst.held_entities.iter().any(|(e, _)| *e == entity);
        source_clear && dest_holds
    });

    let marker = with_client(&mut topo, |c| c.marker_seq()).expect("the client stamped a marker");
    (topo, session, entity, marker)
}

/// `run_cut_transfer` with no per-tick observer — for the gates that read only the quiesced end
/// state (conservation, settle, stability).
fn run_cut_transfer_quiesced(fabric: &FaultFabric) -> (Topology, SessionId, EntityId, u64) {
    run_cut_transfer(fabric, &mut |_| {})
}

#[test]
fn p2_dod_cross_cut_input_is_conserved_exactly_once() {
    let fabric = FaultFabric::new(909, 2);
    let (mut topo, session, entity, m) = run_cut_transfer_quiesced(&fabric);

    // `inspect_all` yields nodes in NodeId order, so SOURCE (NodeId 3) precedes DEST (NodeId 4)
    // — a BINDING fixture invariant: the conservation oracle concatenates per-session applied
    // order in report-slice order WITHOUT sorting, so a clean cut stays strictly increasing
    // only if the source's `1..=M` precedes the dest's `M+1..`.
    let reports = topo.inspect_all();
    let src = report(&reports, SHARD);
    let dst = report(&reports, DEST);
    let orch = report(&reports, ORCH);

    // (2) EXACTLY-ONCE ACROSS THE CUT — the joined-halves proof. A lost drained input ⇒
    // Unaccounted; a double-apply across the seam ⇒ AppliedTwice; an out-of-order/replay apply
    // ⇒ NonMonotonicApply; a phantom ⇒ Phantom.
    verify_input_conservation(&reports).expect("cross-cut INPUT-CONSERVATION holds");

    // (3) THE PARTITION IS REAL (cut-specific, anti-vacuous): source applied only `<= marker`,
    // dest applied a NON-EMPTY batch all `> marker`, resuming exactly at `marker + 1`. (Holds by
    // construction because the client paused across the freeze window — no post-marker leak.)
    assert!(
        src.applied_inputs.iter().all(|(_, s)| *s <= m),
        "source applied only seqs <= the marker {m}: {:?}",
        src.applied_inputs,
    );
    assert!(
        dst.applied_inputs.iter().all(|(_, s)| *s > m),
        "dest applied only seqs > the marker {m}: {:?}",
        dst.applied_inputs,
    );
    assert!(
        !dst.applied_inputs.is_empty(),
        "the dest applied a NON-EMPTY post-marker batch",
    );
    assert_eq!(
        dst.applied_inputs.iter().map(|(_, s)| *s).min(),
        Some(m + 1),
        "the dest resumes exactly at marker+1 (no gap, no off-by-one)",
    );
    // The BUFFER/DRAIN path — the headline 1c.7 deliverable — was actually exercised: at least
    // one seq > marker hit the gateway AFTER the cut installed and was buffered (not merely
    // direct-forwarded to the dest after the route swapped). Pinned to a real observable so a
    // future tick-ordering shift that silently routes M+1 direct turns this RED, not green.
    assert!(
        gateway_buffered_count(&mut topo) >= 1,
        "the cut buffer held >= 1 frame — the gateway buffer/drain path was exercised",
    );

    // (4) THE MARKER ITSELF is applied at the SOURCE (the every-sent-accounted anchor: at marker
    // arrival no cut is installed yet, so it is forwarded-and-applied at source).
    assert!(
        src.applied_inputs.contains(&(session, m)),
        "the marker input (seq {m}) is applied at the source",
    );

    // (5) THE MARKER THREADED end-to-end: the resume watermark the gateway EMITTED in
    // `OpenInputSlot` (latched on the dest) equals the client's OWN marker seq — one live value,
    // no hardcoded literal on either side.
    assert_eq!(
        dst.dest_resume_seq,
        Some(m),
        "the dest seeded its resume watermark to the client's own marker seq",
    );

    // (6) NO SILENT WINDOW LOSS: the input logs saw the whole run (else conservation is a lie).
    assert_eq!(src.input_window_evictions, 0, "source log lost nothing");
    assert_eq!(dst.input_window_evictions, 0, "dest log lost nothing");

    // (7) THE FULL TRANSFER SETTLED (1c.8 — the demote/release tail closed). The directory holds
    // exactly ONE Entity(SUBJECT) record, at DEST, at the CAS new_fence; the saga reached Done.
    let entity_record = orch
        .directory
        .iter()
        .find_map(|(k, r)| (*k == DirectoryKey::Entity(entity)).then_some(*r))
        .expect("the transferred avatar is recorded");
    assert_eq!(
        entity_record.authority,
        AuthorityRef::Shard(DEST),
        "authority committed to the dest shard",
    );
    let new_fence = entity_record.fence;
    let entity_records = orch
        .directory
        .iter()
        .filter(|(k, _)| *k == DirectoryKey::Entity(entity))
        .count();
    assert_eq!(
        entity_records, 1,
        "exactly one Entity record for the subject"
    );
    assert_eq!(
        live_sagas(&mut topo),
        0,
        "the demote/release tail reached Done (no parked saga)",
    );

    // (8) THE HR2 AUTHORITY-CONSERVATION ORACLES, now assertable after a FULL transfer (the
    // D-28(b) graduation): exactly one holder fence-matching the directory, and NOTHING pending
    // or departing anywhere. Kind-generic — the same backstop settles any TransferableKind.
    verify_authority_unique(&reports)
        .expect("exactly one holder of the transferred entity, fence-matching the directory");
    verify_authority_settled(&reports)
        .expect("no lingering pending/departing anywhere after the tail");

    // (9) THE EXACT END STATE: DEST holds (SUBJECT, new_fence) once, nothing pending/departing
    // for it; SOURCE reports nothing for the subject (its dot self-fenced away).
    assert_eq!(
        dst.held_entities
            .iter()
            .filter(|(e, _)| *e == entity)
            .copied()
            .collect::<Vec<_>>(),
        vec![(entity, new_fence)],
        "DEST holds the subject exactly once at the CAS new_fence",
    );
    assert!(
        !dst.pending_entities.contains(&entity),
        "DEST has nothing pending for the subject",
    );
    assert!(
        !dst.departing_entities.contains(&entity),
        "DEST has nothing departing for the subject",
    );
    assert!(
        !src.held_entities.iter().any(|(e, _)| *e == entity),
        "SOURCE holds nothing for the subject (self-fenced)",
    );
    assert!(
        !src.pending_entities.contains(&entity),
        "SOURCE has nothing pending for the subject",
    );
    assert!(
        !src.departing_entities.contains(&entity),
        "SOURCE has nothing departing for the subject",
    );

    // (10) THE ENTITY STATE CROSSED (1d.1, the pose-only crossing): the DEST holds the transferred
    // entity at the source's flushed pose. This is the first proof that entity STATE (not just
    // authority) survives the handoff. (The VISIBLE in-client render of that crossed pose is proven
    // by the 1d.3 capstone `p2_dod_visible_crossing_*` below, on the REAL DeliveredView.)
    //
    // The LOAD-BEARING discriminator is the pose's FRAME, NOT its position. The dest's OWN input
    // integration (`integrate`) moves `pos` but NEVER changes `frame`, and a dropped crossing would
    // leave the dest's adopt-default dot in the DEST realm's frame (`SystemSpace{system_seed: 8}` —
    // `dest_stub_config`). The crossing carries the SOURCE realm's frame (`system_seed: 7`) and
    // overwrites the dot's pose with it. So a dest frame of seed-7 can ONLY come from a landed
    // crossing — making this assertion go RED if the crossing is dropped (a `pos != ZERO` check
    // could NOT: the dest's own drained post-marker input also moves it off origin). This also pins
    // the 1d.1 interim that the source frame is stored VERBATIM (frame-rebinding into the dest realm
    // is OWED for 1d.2/1d.3 — DEFERRED D-27); the assertion flips to seed-8 when rebinding lands.
    let dest_pose = dst
        .held_poses
        .iter()
        .find(|(e, _)| *e == entity)
        .map(|(_, p)| *p)
        .expect("the dest holds a pose for the transferred entity");
    assert_eq!(
        dest_pose.frame,
        FrameRef::SystemSpace { system_seed: 7 },
        "the dest holds the SOURCE realm's frame (seed 7), proving the crossing overwrote the dot's \
         adopt-default DEST frame (seed 8) — a dropped crossing would leave seed 8: {dest_pose:?}",
    );
    assert!(
        dest_pose.pos.offset().is_finite(),
        "the crossed pose is finite (sanitized at the dest ingress): {dest_pose:?}",
    );
    assert_ne!(
        dest_pose.pos.offset(),
        DVec3::ZERO,
        "the crossed pose is the walked source pose, not the origin-adopt default",
    );
}

/// 1d.5b.2 ROBUSTNESS: the settled end state is STABLE under continued operation. After the tail
/// closes, the source's dot is a RETAINED Ghost (1d.4b — `simulates()==false`, EXCLUDED from the
/// oracle held-set) and the dest holds the subject; many more ticks must NOT re-promote the source
/// Ghost, double-grant, or unsettle the oracles. With the 1c.8 granted-key poll TORN OUT (1d.5b.2)
/// there is no per-entity directory poll to (re)discover anything — the source self-fenced ONCE via
/// the saga-pushed `Demote`, and nothing re-promotes it. (The REDELIVERY idempotency — a re-acked
/// `Released`, and a re-delivered `Demote` finding the dot already a Ghost → `self_fence_skipped` —
/// is pinned at the unit level in `vd_connection_plane::gateway::release_subscribe_acks_released`
/// and `vd_sim::stub::the_saga_demote_flips_the_source_to_ghost_and_acks_unconditionally`.)
#[test]
fn p2_dod_settled_transfer_is_stable_under_continued_operation() {
    let fabric = FaultFabric::new(909, 2);
    let (mut topo, _session, entity, _m) = run_cut_transfer_quiesced(&fabric);

    // The tail closed: assert it once, then keep the cluster running.
    assert_eq!(
        live_sagas(&mut topo),
        0,
        "the tail reached Done before the stability soak"
    );
    for _ in 0..40 {
        topo.step();
    }

    let reports = topo.inspect_all();
    let src = report(&reports, SHARD);
    let dst = report(&reports, DEST);
    // The source never re-acquires the transferred dot: it is RETAINED as a Ghost (1d.4b) but
    // EXCLUDED from the held-set via the `simulates()` oracle filter. With the poll torn out
    // (1d.5b.2), nothing re-discovers or re-promotes it.
    assert!(
        !src.held_entities.iter().any(|(e, _)| *e == entity),
        "the retained source Ghost stays excluded from the held-set (does not simulate) under continued operation",
    );
    assert!(
        !src.pending_entities.contains(&entity),
        "the source never re-pends the transferred subject",
    );
    // The dest keeps its single hold; the directory still names exactly one owner.
    assert!(
        dst.held_entities.iter().any(|(e, _)| *e == entity),
        "the dest keeps holding the transferred subject",
    );
    // The HR2 oracles still hold after the soak — no split-brain, no orphan, nothing unsettled.
    verify_authority_unique(&reports).expect("still exactly one holder after the soak");
    verify_authority_settled(&reports).expect("still settled after the soak");
    // And the cross-cut input conservation never regressed.
    verify_input_conservation(&reports).expect("input conservation holds after the soak");
}

/// 1d.5b.3d DoD — PER-TICK MID-FLIGHT AUTHORITY-UNIQUE (the final D-2 piece). verify_authority_unique
/// holds EVERY tick across the full transfer (not only post-quiesce), with the W1 excuse covering the
/// legal post-CAS, pre-demote window (the source still holds the subject Owned at the old fence while
/// the directory already records the dest) and the existing in-flight-to-owner path covering the
/// zero-Owned handoff gap. ANTI-VACUITY: the W1 window actually OCCURS on ≥1 sampled tick — so the
/// per-tick green is the excuse working, not a window that never happened. A split-brain would still
/// RED (the `len == 1` guard precedes the excuse — unit-proven in `oracle::tests`).
#[test]
fn p2_dod_authority_is_unique_every_mid_flight_tick() {
    let fabric = FaultFabric::new(909, 2);
    let mut per_tick: Vec<Vec<(NodeId, InspectReport)>> = Vec::new();
    let (_topo, _session, entity, _m) = run_cut_transfer(&fabric, &mut |t| {
        let reports = t.inspect_all();
        verify_authority_unique(&reports).expect(
            "AUTHORITY-UNIQUE holds EVERY mid-flight tick (W1 + in-flight-to-owner excuses)",
        );
        per_tick.push(reports);
    });

    // ANTI-VACUITY: the W1 window actually occurred — at >= 1 sampled tick the SOURCE held the subject
    // Owned WHILE the directory already recorded the DEST. Without it, "unique every tick" could pass
    // with the excuse never exercised (e.g. if the window sub-tick-collapsed).
    let saw_w1 = per_tick.iter().any(|reports| {
        let src_owns = report(reports, SHARD)
            .held_entities
            .iter()
            .any(|(e, _)| *e == entity);
        let dir_dest = report(reports, ORCH).directory.iter().any(|(k, rec)| {
            (*k == DirectoryKey::Entity(entity)) & (rec.authority == AuthorityRef::Shard(DEST))
        });
        src_owns & dir_dest
    });
    assert!(
        saw_w1,
        "the W1 mid-flight window (source-Owned + directory-DEST) actually occurred — the excuse was exercised, not vacuous",
    );
}

/// Whether any saga passed through (or reached) an ABORT state during the run — sampled each tick
/// (a terminal Aborted tombstones fast, so it is caught in-flight as `Aborting`). The Slice 2a
/// false-timeout guard: a HEALTHY transfer must NEVER abort.
fn observed_an_abort(topo: &mut Topology) -> bool {
    saga_states(topo)
        .iter()
        .any(|s| s.starts_with("Aborting") || s.starts_with("Aborted"))
}

/// Slice 2a DoD — FALSE-TIMEOUT NEGATIVE CONTROL #1 (the catastrophic-risk guard). The deadline
/// PRODUCER is LIVE in the capstone cluster, yet a HEALTHY lockstep transfer reaches Done with NO
/// saga EVER entering Aborting (no false destructive abort) and COMMITS to the dest. Goes RED if a
/// deadline drops below the worst healthy phase dwell (an abort storm on the happy path).
#[test]
fn p2_dod_a_healthy_transfer_never_false_aborts_with_the_producer_live() {
    let fabric = FaultFabric::new(909, 2);
    let mut saw_abort = false;
    let (mut topo, _s, entity, _m) = run_cut_transfer(&fabric, &mut |t| {
        saw_abort |= observed_an_abort(t);
    });
    assert!(
        !saw_abort,
        "a healthy transfer NEVER aborts — the producer did not false-fire the destructive abort",
    );
    assert_eq!(
        live_sagas(&mut topo),
        0,
        "the transfer reached a terminal state"
    );
    let reports = topo.inspect_all();
    assert!(
        report(&reports, DEST)
            .held_entities
            .iter()
            .any(|(e, _)| *e == entity),
        "the transfer COMMITTED to the dest (not aborted back to the source) — terminal success, not failure",
    );
}

/// Slice 2a DoD — FALSE-TIMEOUT NEGATIVE CONTROL #2 (slow-but-alive). With the saga-control links
/// DELAYED (all nodes ALIVE, no drops/kills), the transfer is slower — the producer may harmlessly
/// re-drive a post-commit step (cheap, idempotent) — but NO saga ever aborts (the destructive
/// `abort_deadline_ticks=24` dominates the slow pre-freeze round-trip) and the transfer still
/// COMMITS. This is the control the lockstep one cannot be: it pins the destructive threshold
/// against a slow-but-healthy link. Goes RED if `abort_deadline_ticks` is tuned below the slow
/// pre-freeze dwell.
#[test]
fn p2_dod_a_slow_but_alive_transfer_never_aborts() {
    let fabric = FaultFabric::new(909, 2);
    // Delay the saga-control plane (orchestrator ↔ gateway/source/dest), both directions. All nodes
    // stay ALIVE — a slow link, not a crash. max_extra=4 keeps the worst pre-freeze dwell well under
    // abort_deadline_ticks=24 (no flakiness) while exceeding redrive=8 (so a post-commit re-drive may
    // harmlessly fire — proving the cheap arm is non-destructive).
    let slow = LinkPolicy {
        max_extra_delay_ticks: 4,
        ..LinkPolicy::default()
    };
    for (a, b) in [
        (ORCH, GATEWAY),
        (GATEWAY, ORCH),
        (ORCH, SHARD),
        (SHARD, ORCH),
        (ORCH, DEST),
        (DEST, ORCH),
    ] {
        fabric.set_policy(a, b, slow);
    }
    let mut saw_abort = false;
    let (mut topo, _s, entity, _m) = run_cut_transfer(&fabric, &mut |t| {
        saw_abort |= observed_an_abort(t);
    });
    assert!(
        !saw_abort,
        "a SLOW-but-alive transfer never aborts — abort_deadline_ticks dominates the slow pre-freeze round-trip",
    );
    assert_eq!(
        live_sagas(&mut topo),
        0,
        "the slow transfer still reached a terminal state"
    );
    assert!(
        report(&topo.inspect_all(), DEST)
            .held_entities
            .iter()
            .any(|(e, _)| *e == entity),
        "the slow transfer COMMITTED to the dest (slowness delays, never aborts)",
    );
}

/// 1d.5b.3c DoD — THE SOURCE-GHOST LIFECYCLE END (band-exit). After the transfer settles, the SOURCE
/// retains the avatar as a kinematic collider GHOST fed by the dest (`GhostFlow`). As the dest-owned
/// avatar keeps walking AWAY from the boundary it crossed, it leaves the overlap band: the dest emits
/// `GhostFlow::Despawn` + deregisters the feed, and the source TEARS THE GHOST DOWN (removes the dot).
/// The teardown is strictly POST-release (the destroy edge is many per-tick steps out), so the source
/// ghost is gone only once the dest is the sole render source — the avatar renders CONTINUOUSLY across
/// the band-exit (NO vanish). The mechanics are unit-proven (`vd_sim::stub`); THIS gate is the
/// integrated seamless proof over the real fabric.
#[test]
fn p2_dod_band_exit_tears_down_the_source_ghost_seamlessly() {
    let fabric = FaultFabric::new(909, 2);
    let mut caps: Vec<CapturedTick> = Vec::new();
    let (mut topo, _session, entity, _m) =
        run_cut_transfer(&fabric, &mut |t| caps.push(capture_subject(t)));

    // PRECONDITION (anti-vacuity): the transfer settled with the SOURCE hosting the avatar as a
    // retained collider ghost — the thing band-exit must tear down actually exists (else "torn down"
    // is vacuously true). Empirically the dest's post-marker drain leaves it ~0.5 m from the crossing
    // anchor, well inside the 2.0 m destroy edge, so the ghost survives the settle.
    assert!(
        report(&topo.inspect_all(), SHARD)
            .ghost_dots
            .contains(&entity),
        "precondition: the source hosts the avatar as a retained collider ghost before band-exit",
    );

    // RESUME walking + keep sampling the render until the SOURCE tears the ghost down (bounded). The
    // dest-owned avatar walks past the destroy edge, the dest Despawns + deregisters, the source
    // removes the ghost dot — driven over the real fabric, not hand-fed.
    with_client(&mut topo, ScriptedClient::resume_input);
    step_until(&mut topo, 80, &mut |t| caps.push(capture_subject(t)), |t| {
        !report(&t.inspect_all(), SHARD).ghost_dots.contains(&entity)
    });

    let reports = topo.inspect_all();
    assert!(
        !report(&reports, SHARD).ghost_dots.contains(&entity),
        "the source ghost is torn down on band-exit",
    );
    assert!(
        report(&reports, DEST)
            .held_entities
            .iter()
            .any(|(e, _)| *e == entity),
        "the dest still holds the avatar after band-exit",
    );
    assert_eq!(
        max_absent_run(&caps),
        0,
        "ZERO vanish across the band-exit teardown (the dest is the sole render source by then)",
    );
    verify_authority_unique(&reports)
        .expect("exactly one holder after band-exit (the source ghost is gone, the dest owns)");
}

/// The transfer run is fully deterministic under one seed — the standing in-process replay gate
/// (cross-process parity is P3 chaos scope). It compares `inspect_all` ground truth, the harness
/// TRACE (the purpose-built byte-comparable per-node step/sent/drain artifact — this is what
/// catches divergence in the transfer machinery PAST the commit point: the saga's post-CAS phases
/// and the gateway's route/buffer mutate NO directory state, so they are invisible to `inspect_all`
/// alone but show in the trace), the gateway buffer-fill count, and the threaded marker. (NOTE:
/// 1c.8 now drives to `live()==0`, so `saga`/`live` are the terminal sentinels — `[]`/`0` — not a
/// mid-flight phase; they pin "the run reached Done", while the TRACE carries the per-tick FSM
/// ordering across the whole run.)
#[test]
fn p2_dod_cross_cut_transfer_is_byte_identical_under_same_seed() {
    let run = || {
        // FOLD IN the per-tick RENDER trace (1d.3): the visible-crossing capture is also proven
        // deterministic — the composited sub flip + the crossed pose render byte-identically under
        // one seed, so the headline visibility proof cannot be flaky on delivery ordering.
        let mut caps: Vec<CapturedTick> = Vec::new();
        let (mut topo, _, _, marker) = run_cut_transfer(&FaultFabric::new(909, 2), &mut |t| {
            caps.push(capture_subject(t))
        });
        let saga = saga_states(&mut topo); // terminal sentinel ([] — the run drove to Done)
        let live = live_sagas(&mut topo); // terminal sentinel (0)
        // The gateway's cut-buffer fill count, compared DIRECTLY (not merely inferred from the
        // trace's per-tick send deferral) — so a divergence in the gateway's route/buffer
        // decision cannot hide behind an identical trace+reports+saga.
        let buffered = gateway_buffered_count(&mut topo);
        let reports = topo.inspect_all();
        let trace = topo.trace_bytes();
        (reports, trace, saga, live, buffered, marker, caps)
    };
    assert_eq!(
        run(),
        run(),
        "identical seed ⇒ identical ground truth, trace, terminal sentinels, buffer fill, marker, \
         AND the per-tick render trace (the visible-crossing capture is deterministic)",
    );
}

/// The longest run of consecutive ticks the subject rendered NOWHERE, BETWEEN its first and last
/// rendered tick (ignoring the leading pre-login absence). 1d.5a's delivery-gated source-sub release
/// closes the former interim gap to ZERO (the seamless transition); a render-lag regression that
/// re-opens a gap turns the capstone's `== 0` assertion (and `verify_no_vanish`) RED.
fn max_absent_run(caps: &[CapturedTick]) -> usize {
    let (Some(first), Some(last)) = (
        caps.iter().position(|c| c.sample.is_some()),
        caps.iter().rposition(|c| c.sample.is_some()),
    ) else {
        return 0;
    };
    let mut run = 0usize;
    let mut max = 0usize;
    for c in &caps[first..=last] {
        run = if c.sample.is_none() { run + 1 } else { 0 };
        max = max.max(run);
    }
    max
}

/// 1d.3/1d.5a DoD — THE SEAMLESS CROSS-SHARD CROSSING. The client's OWN avatar, captured EVERY tick
/// across the full transfer, FLIPS source→dest rendering the CROSSED pose (the SOURCE-realm seed-7
/// pose, NOT the origin-adopt default) with NO vanish — a true two-holder overlap the client de-dups
/// to ONE render via `AuthorityChanged`. Proven at the render-decision layer (the REAL `DeliveredView`
/// the production client uses, driven from the harness `ScriptedClient`) by the WireMonitor render
/// oracles `verify_no_vanish` + `verify_pose_continuity` (world_pos basis; motion-derived ε, tightest K=0). Mutation-RED-
/// verified at THIS e2e tier against drop-flip + suppress-dest. (render-origin — the render flip
/// MISplaced to the adopt grant arm — is enforced at the UNIT tier by
/// `stub.rs::the_adopt_grant_flip_holds_authority_announces_the_sub_without_render_or_attach`, NOT here:
/// this fixture buffers the crossing and drains+flips it in the SAME tick BEFORE `emit_frames`, so a
/// misplaced flip never emits a seed-8 origin frame — the e2e crossing-AFTER-adopt variant is the
/// follow-up. Documented in the negative controls.)
///
/// ✅ SEAMLESS LANDED IN 1d.5a (earlier than the D-2 plan predicted): delivery-gating holds the source
/// sub OPEN until the dest is delivered to its observer, so the FORK-0a two-sub overlap (designed in
/// 1d.2) finally manifests — `overlap_ticks >= 1`, `vanish_gap == 0`, the no-vanish/pose-continuity
/// oracles GREEN. This is the READ-plane seam; the WRITE-plane fence-enforced ordered
/// demote-before-promote + the `GhostFlow` collider feed + the band-exit `Despawn` are still the
/// D-2/1d.5b tear-out (authority-ordering robustness + cross-boundary collision, NOT the render).
#[test]
fn p2_dod_the_cross_shard_crossing_renders_at_the_dest_at_the_crossed_pose() {
    let fabric = FaultFabric::new(909, 2);
    let mut caps: Vec<CapturedTick> = Vec::new();
    let _ = run_cut_transfer(&fabric, &mut |t| caps.push(capture_subject(t)));

    // The subject's rendered samples in order (skipping the pre-login ticks; there is no vanish now).
    let rendered: Vec<RenderSample> = caps.iter().filter_map(|c| c.sample).collect();
    let source_sub = rendered
        .first()
        .expect("the subject renders at some tick")
        .sub;
    let dest_sample = *rendered.last().expect("the subject renders at some tick");

    // (1) THE VISIBLE CROSSING: the authoritative rendered sub flips source→dest EXACTLY ONCE (the
    // crossing becomes visible from the dest; the de-dup is structural — `chosen_subs` is one sub per
    // entity, so the captured sample is already at-most-one). suppress-dest leaves source==dest → RED.
    assert_ne!(
        source_sub, dest_sample.sub,
        "the rendered sub flipped source→dest — the crossing is visible from the dest",
    );
    let flips = rendered.windows(2).filter(|w| w[0].sub != w[1].sub).count();
    assert_eq!(
        flips, 1,
        "exactly one source→dest authority flip (no flapping)"
    );

    // (2) THE DEST RENDERS THE CROSSED POSE, not the origin-adopt default — the load-bearing
    // discriminator (drop-flip turns THIS red): the SOURCE realm's `SystemSpace{seed:7}` at a
    // non-origin, finite world position, NEVER the dest's own seed-8 origin default at ZERO.
    // (render-origin — the flip misplaced to the adopt grant arm — is UNIT-covered and moot here, per
    // the docstring; the e2e variant forcing crossing-after-adopt is the D-2 follow-up.)
    assert_eq!(
        dest_sample.frame,
        FrameRef::SystemSpace { system_seed: 7 },
        "the dest renders the crossed SOURCE-realm pose (seed 7), not the origin-adopt default (seed 8): {dest_sample:?}",
    );
    assert!(
        dest_sample.world_pos.is_finite(),
        "the rendered world pose is finite (sanitized): {dest_sample:?}",
    );
    assert_ne!(
        dest_sample.world_pos,
        DVec3::ZERO,
        "the rendered pose is the walked crossed pose, not the origin-adopt default",
    );

    // (3) SEAMLESS — the interim vanish is CLOSED (1d.5a). Delivery-gating now holds the source sub
    // OPEN until the dest is delivered to the observer, so the FORK-0a two-sub overlap finally
    // manifests: the source sub still holds its track AT the tick the dest sub goes live, the client
    // renders the dest (chosen_subs de-dups to ONE — no double-vision), and the avatar renders
    // CONTINUOUSLY across the flip — NO vanish. The WireMonitor render oracles (built 1d.3b, world_pos
    // basis) gate it, from the subject's FIRST render (the client draws nothing before its own login)
    // through quiesce. NO MAGIC: ε_pos/ε_rot are DERIVED from the shard's motion params
    // (move_speed·dt + slack); the no-vanish K is 0 — it follows from band_ticks=1 (the single-tick
    // FORK-0a overlap) and delay=0 (zero-fault lockstep), the intentionally tightest bound (present
    // EVERY tick), NOT from motion params. So verify_no_vanish(K=0) is the oracle form of the explicit
    // `max_absent_run == 0` below; both gate the seamless no-vanish.
    let first = caps
        .iter()
        .position(|c| c.sample.is_some())
        .expect("the subject renders at some tick");
    let trace: RenderTrace = caps[first..].iter().map(|c| (c.tick, c.sample)).collect();
    let cfg = stub_config();
    let tol = RenderTolerances::derive(cfg.move_speed_mps, cfg.tick_dt_s, 0, 0, 1, 1e-9, 0.0, 1e-9);
    verify_no_vanish(&trace, tol).expect("the avatar never vanishes across the seamless crossing");
    verify_pose_continuity(&trace, tol)
        .expect("the avatar's world pose is continuous across the source→dest flip (no teleport)");
    // ANTI-VACUITY: the two-holder OVERLAP actually happened — at ≥1 tick BOTH the source sub AND
    // the dest sub held a track for the subject (the seamless window the no-vanish rides on). Without
    // it, "no vanish" could pass with the source never dropping (a degenerate single-sub render).
    let overlap_ticks = caps.iter().filter(|c| c.subs_holding.len() >= 2).count();
    assert!(
        overlap_ticks >= 1,
        "the source + dest subs OVERLAP (both emit the avatar) — the seamless two-holder window",
    );
    assert_eq!(
        max_absent_run(&caps),
        0,
        "ZERO vanish: the source sub holds until the dest is delivered (the seamless transition)",
    );
}

// ---------------------------------------------------------------------------------------------
// NEGATIVE CONTROLS — verified by LOCAL mutation (run each, confirm RED, revert), kept here so
// the gate's sensitivity is auditable. The oracle alone is blind to a clean-but-wrong partition,
// so assertions (3)/(5) are load-bearing beside it.
//
// VERIFIED to turn a DIFFERENT named assertion RED in THIS e2e gate:
//   (a) LOST DRAINED INPUT  — `for input_bytes in buffered.into_iter().skip(1)` in apply_commit
//                              ⇒ that seq is sent-but-applied-nowhere ⇒ verify_input_conservation
//                                 returns Unaccounted (assertion 2). [confirmed]
//   (c) MARKER MIS-THREAD   — emit `OpenInputSlot{ resume_from_seq: marker_seq + 5 }`
//                              ⇒ dest seeds M+5, rejects the drained batch ⇒ dest empty
//                                 (assertion 3); dest_resume_seq would also be Some(M+5) ≠ Some(M)
//                                 (assertion 5). [confirmed]
//   (e) NO PAUSE (leak)     — drop the client's pause-after-marker ⇒ seq > M forwarded to the
//                              source before the cut installs ⇒ source applied > M (assertion 3,
//                              the cut-correctness check the oracle CANNOT make). [confirmed]
//
// COVERED AT THE UNIT LEVEL (moot in THIS e2e gate, by design — documented honestly):
//   (b) DOUBLE-APPLY        — the dest's `last_applied_seq` dedup structurally rejects a replayed
//                              drained frame, so a naive double-apply cannot be injected here; the
//                              oracle's `AppliedTwice` arm is unit-tested in `harness::oracle`.
//   (d) OUT-OF-ORDER DRAIN  — the e2e freeze→commit window buffers a SINGLE frame (later
//                              post-marker inputs forward directly to the dest after the route
//                              swaps), so reversing a 1-element buffer is a no-op. Multi-frame
//                              drain ORDER is gated by the connection-plane unit test
//                              `commit_opens_the_dest_slot_then_drains_the_buffer_in_seq_order`.
//   (f) RENDER-ORIGIN       — Promoting the dest Ghost→Owned in the adopt grant arm (`stub.rs`
//                              flip_grant Adopted) instead of in `apply_crossing` would emit a seed-8
//                              origin ZERO frame ONLY if the crossing arrives AFTER the adopt. THIS
//                              fixture buffers the crossing and drains+promotes it in the same tick
//                              BEFORE `emit_frames` (`stub.rs` schedule `(request_pending_grants,
//                              process_inbound, emit_frames).chain()`), so the misplaced Promote is
//                              structurally unobservable here. It is caught at the unit tier by
//                              `stub.rs::the_adopt_grant_flip_holds_authority_announces_the_sub_without_render_or_attach`.
//                              The e2e variant that DELAYS the crossing past the adopt (where this
//                              gate WOULD see the origin frame) lands with the D-2 reorder machinery
//                              + the `verify_no_vanish`/`verify_pose_continuity` wiring — recorded in
//                              DEFERRED.md D-2. [audit wf_c3e4f1b7 Finding 1]
// ---------------------------------------------------------------------------------------------
