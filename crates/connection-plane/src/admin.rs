//! RLM RG-4 — the gateway's `/admin/snapshot` view (the gateway half of the ledgered D-10 per-node
//! observability). The gateway analogue of `vd_node::orchestrator::admin_snapshot`: it projects the gateway's
//! in-World counters + gauges into the JSON-only [`AdminSnapshot`] (the frozen postcard `InterShardFlow` wire
//! is untouched). A gateway snapshot carries ONLY the `gateway` view — the directory/sagas/leases stay empty,
//! because the ORCHESTRATOR owns those; a gateway reporting a directory would mislead an operator.

use bevy_ecs::prelude::World;
use vd_sim::runtime::ClockSample;
use vd_wire::admin::{AdminSnapshot, GatewayView};

use crate::gateway::{GatewaySessions, GatewayStats};

/// Project the gateway's live counters + gauges into the admin contract. EXHAUSTIVELY destructures
/// [`GatewayStats`] (no `..`), so a future 23rd counter is a COMPILE ERROR here until it is surfaced on the
/// operator view — the same completeness guard the io-prod metrics renderer uses. Branchless (HR5).
#[must_use]
pub fn gateway_view(stats: &GatewayStats, sessions_open: u64, dynamic_shards: u64) -> GatewayView {
    let GatewayStats {
        logins_rejected,
        version_rejected,
        sessions_refused_capacity,
        session_mints_refused,
        resumes_refused,
        inputs_deduped,
        inputs_unroutable,
        inputs_malformed,
        stale_frames_dropped,
        stale_removals_dropped,
        undecodable,
        refused_unknown_sender,
        shard_rosters_applied,
        shard_roster_stale,
        frame_sub_desync,
        transfer_unroutable,
        transfer_control_parked,
        commit_without_cut,
        inputs_buffered_for_dest,
        dest_inputs_dropped,
        sessions_self_fenced_lapsed,
        home_bootstrap_timeouts,
        home_wait_desync,
        logins_held_pre_sync,
        sessions_self_fenced_revoked,
        presence_announces,
        sub_close_refused_authority,
        window_rows_unconsumed,
    } = *stats;
    GatewayView {
        logins_rejected,
        version_rejected,
        sessions_refused_capacity,
        session_mints_refused,
        resumes_refused,
        inputs_deduped,
        inputs_unroutable,
        inputs_malformed,
        stale_frames_dropped,
        stale_removals_dropped,
        undecodable,
        refused_unknown_sender,
        shard_rosters_applied,
        shard_roster_stale,
        frame_sub_desync,
        transfer_unroutable,
        transfer_control_parked,
        commit_without_cut,
        inputs_buffered_for_dest,
        dest_inputs_dropped,
        sessions_self_fenced_lapsed,
        home_bootstrap_timeouts,
        home_wait_desync,
        logins_held_pre_sync,
        sessions_self_fenced_revoked,
        presence_announces,
        sub_close_refused_authority,
        window_rows_unconsumed,
        sessions_open,
        dynamic_shards,
    }
}

/// Build the gateway's `/admin/snapshot` body from its World. The three resource reads are INFALLIBLE:
/// `build_app` inserts [`ClockSample`] and `register_gateway` inserts [`GatewayStats`]/[`GatewaySessions`] on
/// every gateway world, so a tolerant `get_resource` would only add an uncoverable panic-free arm (HR5). The
/// directory/sagas/leases stay EMPTY — a gateway does not own them (the orchestrator does), so a gateway
/// snapshot's `cluster_bootstrapped` is always false and never misleads an operator.
#[must_use]
pub fn gateway_admin_snapshot(world: &mut World) -> AdminSnapshot {
    let (universe_tick, epoch) = {
        let clock = world.resource::<ClockSample>();
        (clock.universe_tick, clock.epoch)
    };
    let (sessions_open, dynamic_shards) = {
        let sessions = world.resource::<GatewaySessions>();
        (sessions.len() as u64, sessions.dynamic_shard_count() as u64)
    };
    let stats = *world.resource::<GatewayStats>();
    let mut snapshot = AdminSnapshot::shaped_empty(universe_tick, epoch);
    snapshot.gateway = Some(gateway_view(&stats, sessions_open, dynamic_shards));
    snapshot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_view_pairs_every_counter_without_transposition() {
        // Every field a DISTINCT value: a copy-paste swap between two counters fails this.
        let stats = GatewayStats {
            logins_rejected: 1,
            version_rejected: 2,
            sessions_refused_capacity: 3,
            session_mints_refused: 4,
            resumes_refused: 5,
            inputs_deduped: 6,
            inputs_unroutable: 7,
            inputs_malformed: 8,
            stale_frames_dropped: 9,
            stale_removals_dropped: 29,
            undecodable: 10,
            // 26 rather than renumbering: the sequence is positional, and rewriting every value below to
            // slot this in is the transposition this fixture exists to catch.
            refused_unknown_sender: 26,
            shard_rosters_applied: 27,
            shard_roster_stale: 28,
            frame_sub_desync: 11,
            transfer_unroutable: 12,
            transfer_control_parked: 13,
            commit_without_cut: 14,
            inputs_buffered_for_dest: 15,
            dest_inputs_dropped: 16,
            sessions_self_fenced_lapsed: 17,
            home_bootstrap_timeouts: 18,
            home_wait_desync: 19,
            logins_held_pre_sync: 20,
            sessions_self_fenced_revoked: 21,
            presence_announces: 22,
            sub_close_refused_authority: 23,
            // 30, not 24: appended after the fixture reached 29 (the same positional-sequence
            // reasoning as refused_unknown_sender's 26 above).
            window_rows_unconsumed: 30,
        };
        let view = gateway_view(&stats, 24, 25);
        assert_eq!(view.logins_rejected, 1);
        assert_eq!(view.version_rejected, 2);
        assert_eq!(view.sessions_refused_capacity, 3);
        assert_eq!(view.session_mints_refused, 4);
        assert_eq!(view.resumes_refused, 5);
        assert_eq!(view.inputs_deduped, 6);
        assert_eq!(view.inputs_unroutable, 7);
        assert_eq!(view.inputs_malformed, 8);
        assert_eq!(view.stale_frames_dropped, 9);
        assert_eq!(view.stale_removals_dropped, 29);
        assert_eq!(view.undecodable, 10);
        assert_eq!(view.refused_unknown_sender, 26);
        assert_eq!(view.shard_rosters_applied, 27);
        assert_eq!(view.shard_roster_stale, 28);
        assert_eq!(view.frame_sub_desync, 11);
        assert_eq!(view.transfer_unroutable, 12);
        assert_eq!(view.transfer_control_parked, 13);
        assert_eq!(view.commit_without_cut, 14);
        assert_eq!(view.inputs_buffered_for_dest, 15);
        assert_eq!(view.dest_inputs_dropped, 16);
        assert_eq!(view.sessions_self_fenced_lapsed, 17);
        assert_eq!(view.home_bootstrap_timeouts, 18);
        assert_eq!(view.home_wait_desync, 19);
        assert_eq!(view.logins_held_pre_sync, 20);
        assert_eq!(view.sessions_self_fenced_revoked, 21);
        assert_eq!(view.presence_announces, 22);
        assert_eq!(view.sub_close_refused_authority, 23);
        assert_eq!(view.window_rows_unconsumed, 30);
        assert_eq!(view.sessions_open, 24);
        assert_eq!(view.dynamic_shards, 25);
    }
}
