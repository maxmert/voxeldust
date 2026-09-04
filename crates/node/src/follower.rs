//! The follower side of universe time: every non-orchestrator node observes the
//! orchestrator's `ClockSync` broadcasts through a monotonic clamp
//! (`FollowerClock`) — backward slew is REJECTED AS DATA and counted, never
//! applied (`docs/design/identity_persistence.md` §clock; R7).

use bevy_ecs::prelude::{Res, ResMut, Resource, Schedule, World};
use vd_core::NodeId;
use vd_sim::io::{Inbound, MsgClass};
use vd_sim::runtime::{ClockSample, InboundBox};
use vd_wire::intershard::InterShardFlow;
use vd_wire::seams::directory::DirectoryOp;

use crate::universe_clock::{FollowerClock, SyncOutcome};

/// The follower clamp (None until the first sync arrives) plus its honesty
/// counters — anomalies are counted on a metric, never silently swallowed.
#[derive(Resource, Debug, Default)]
pub struct FollowerState {
    pub clock: Option<FollowerClock>,
    /// ★ WHO DRIVES MY CLOCK — the orchestrator, learned from the first `ClockSync`'s sender. The peer
    /// book asks it where an unbooked node listens; nothing else names the orchestrator to a node.
    pub source: Option<NodeId>,
    /// Backward-slew observations rejected by the clamp.
    pub rejected_backward: u64,
    /// Syncs carrying a different epoch than the one first observed (a wiped or
    /// rolled universe) — fail-safe ignored, loudly counted.
    pub epoch_mismatches: u64,
    /// Membership-class frames that did not decode to a `ClockSync` (a malformed
    /// payload or an unexpected arm on the membership class). Ignored, but counted
    /// so the failure is observable rather than log-only (ROB-E2E-1). 0 in any
    /// healthy run.
    pub undecodable: u64,
}

/// Install the clock-follower system (shards and gateways).
pub fn register_clock_follower(world: &mut World, schedule: &mut Schedule) {
    world.insert_resource(FollowerState::default());
    schedule.add_systems(observe_clock_syncs);
}

fn observe_clock_syncs(
    inbox: Res<InboundBox>,
    mut state: ResMut<FollowerState>,
    mut sample: ResMut<ClockSample>,
) {
    for msg in &inbox.0 {
        let Inbound::Wire { from, class, bytes } = msg else {
            continue;
        };
        if *class != MsgClass::Membership {
            continue;
        }
        let Ok(InterShardFlow::Directory(DirectoryOp::ClockSync {
            universe_tick,
            epoch,
        })) = postcard::from_bytes::<InterShardFlow>(bytes)
        else {
            state.undecodable += 1;
            tracing::error!("undecodable membership-class message");
            continue;
        };
        state.source = Some(*from);
        match &mut state.clock {
            None => {
                state.clock = Some(FollowerClock::new(epoch, universe_tick));
            }
            Some(clock) => {
                if clock.epoch() != epoch {
                    state.epoch_mismatches += 1;
                    continue;
                }
                if let SyncOutcome::RejectedBackward { .. } = clock.observe(universe_tick) {
                    state.rejected_backward += 1;
                }
            }
        }
        let clock = state.clock.as_ref().expect("set above");
        sample.universe_tick = clock.now();
        sample.epoch = clock.epoch();
    }
    // RLM Step 2 (D-Finding-1 / L2): set the synced gate ONCE the clock exists, at fn EXIT — NOT inside
    // the loop (which fires only on delivery ticks → a sync-flap the run-condition would inherit). Once
    // `state.clock` is `Some` it never resets, so every subsequent tick reads `synced = true`.
    if !sample.synced && state.clock.is_some() {
        tracing::debug!("CLOCK SYNCED: first ClockSync applied — the gated authors run from here");
    }
    sample.synced = state.clock.is_some();
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy_ecs::prelude::World;
    use vd_core::{EpochId, MsgId, NodeId, UniverseTick};

    fn sync_bytes(tick: u64, epoch: u64) -> Vec<u8> {
        postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::ClockSync {
            universe_tick: UniverseTick(tick),
            epoch: EpochId(epoch),
        }))
        .expect("encode")
    }

    fn rig() -> (World, Schedule) {
        let mut world = World::new();
        world.insert_resource(InboundBox::default());
        world.insert_resource(ClockSample::default());
        let mut schedule = Schedule::default();
        register_clock_follower(&mut world, &mut schedule);
        (world, schedule)
    }

    fn observe(world: &mut World, schedule: &mut Schedule, inbound: Vec<Inbound>) {
        world.resource_mut::<InboundBox>().0 = inbound;
        schedule.run(world);
    }

    fn wire(class: MsgClass, bytes: Vec<u8>) -> Inbound {
        Inbound::Wire {
            from: NodeId(1),
            class,
            bytes: bytes.into(),
        }
    }

    #[test]
    fn the_clock_source_is_remembered_from_the_sync_that_drove_it() {
        // The peer book asks the node that drives the clock; nothing else names the orchestrator.
        let (mut world, mut schedule) = rig();
        assert_eq!(world.resource::<FollowerState>().source, None);
        observe(
            &mut world,
            &mut schedule,
            vec![wire(MsgClass::Membership, sync_bytes(10, 7))],
        );
        assert_eq!(world.resource::<FollowerState>().source, Some(NodeId(1)));
    }

    #[test]
    fn first_sync_initializes_then_advances_monotonically() {
        let (mut world, mut schedule) = rig();
        observe(
            &mut world,
            &mut schedule,
            vec![wire(MsgClass::Membership, sync_bytes(10, 7))],
        );
        let sample = *world.resource::<ClockSample>();
        assert_eq!(sample.universe_tick, UniverseTick(10));
        assert_eq!(sample.epoch, EpochId(7));

        // Forward and equal observations apply cleanly.
        observe(
            &mut world,
            &mut schedule,
            vec![
                wire(MsgClass::Membership, sync_bytes(12, 7)),
                wire(MsgClass::Membership, sync_bytes(12, 7)),
            ],
        );
        assert_eq!(
            world.resource::<ClockSample>().universe_tick,
            UniverseTick(12)
        );
        assert_eq!(world.resource::<FollowerState>().rejected_backward, 0);
    }

    #[test]
    fn backward_slew_is_rejected_and_counted_never_applied() {
        let (mut world, mut schedule) = rig();
        observe(
            &mut world,
            &mut schedule,
            vec![
                wire(MsgClass::Membership, sync_bytes(20, 7)),
                wire(MsgClass::Membership, sync_bytes(5, 7)),
            ],
        );
        assert_eq!(
            world.resource::<ClockSample>().universe_tick,
            UniverseTick(20),
            "time never rewinds"
        );
        assert_eq!(world.resource::<FollowerState>().rejected_backward, 1);
    }

    #[test]
    fn epoch_mismatch_is_failsafe_ignored_and_counted() {
        let (mut world, mut schedule) = rig();
        observe(
            &mut world,
            &mut schedule,
            vec![
                wire(MsgClass::Membership, sync_bytes(20, 7)),
                wire(MsgClass::Membership, sync_bytes(30, 8)),
            ],
        );
        let sample = *world.resource::<ClockSample>();
        assert_eq!(
            sample.universe_tick,
            UniverseTick(20),
            "foreign epoch ignored"
        );
        assert_eq!(sample.epoch, EpochId(7));
        assert_eq!(world.resource::<FollowerState>().epoch_mismatches, 1);
    }

    #[test]
    fn non_sync_traffic_is_ignored() {
        let (mut world, mut schedule) = rig();
        observe(
            &mut world,
            &mut schedule,
            vec![
                // Wrong class.
                wire(MsgClass::Control, sync_bytes(10, 7)),
                // Right class, undecodable.
                wire(MsgClass::Membership, vec![0xFF, 0x00]),
                // Right class, wrong directory op.
                wire(
                    MsgClass::Membership,
                    postcard::to_allocvec(&InterShardFlow::Directory(DirectoryOp::HeadRead {
                        key: vd_wire::seams::directory::DirectoryKey::Session(vd_core::SessionId(
                            1,
                        )),
                    }))
                    .expect("encode"),
                ),
                // Non-wire inbound.
                Inbound::NodeUnreachable {
                    to: NodeId(9),
                    class: MsgClass::Membership,
                    undelivered: MsgId(0),
                },
            ],
        );
        assert_eq!(world.resource::<FollowerState>().clock, None);
        assert_eq!(
            world.resource::<ClockSample>().universe_tick,
            UniverseTick(0)
        );
        // Both membership frames that did not yield a ClockSync — the undecodable
        // garbage AND the valid-but-wrong directory op — are COUNTED (ROB-E2E-1);
        // the wrong-class frame is skipped before decode, so it adds nothing.
        assert_eq!(world.resource::<FollowerState>().undecodable, 2);
    }
}
