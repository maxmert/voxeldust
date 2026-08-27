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
pub fn gateway_view(
    stats: &GatewayStats,
    sessions_open: u64,
    dynamic_shards: u64,
    windows_open: u64,
) -> GatewayView {
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
        window_rows_ingested,
        star_catalogue_parts_sent: _,
        sky_alive_beats_sent: _,
        sky_requests_sent: _,
        sky_held_stated: _,
        sky_parts_skipped: _,
        window_sender_mismatch,
        window_misauthored_body,
        window_unknown_row,
        window_open_sent,
        window_close_sent,
        window_keepalives_sent,
        window_level_refused,
        window_body_stale,
        window_body_preroster,
        window_folds,
        window_fold_hits,
        window_fold_divergence,
        window_full_chain_folds,
        window_chains_held,
        window_compose_hold_ticks,
        window_hop_dead,
        window_t_monotone_stalled,
        window_chain_cycle,
        window_instant_mismatch,
        window_rotated_refused,
        window_alien_rows,
        window_hop_invalid,
        window_unresolved_standing,
        window_dedup_disagree,
        window_dedup_max_dev_cells,
        window_head_reads_sent,
        window_composed_rows,
        old_realm_frames_dropped,
        old_scene_deltas_dropped,
        scene_levels_sent,
        scene_deltas_sent,
        scene_datagrams_sent,
        window_relays_ingested,
        window_relay_undecodable,
        window_relay_unvouched,
        window_relay_stale,
        window_relay_rows_composed,
        window_relay_descent_refused,
        window_relay_stamp_missing,
        window_relay_unrostered,
        window_relay_stamp_skew_ticks,
        window_relay_depth_max,
        window_looks_pruned,
        window_relay_levels_pruned,
        window_relay_interior_unvouched,
        window_relay_interior_filtered,
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
        window_rows_ingested,
        window_sender_mismatch,
        window_misauthored_body,
        window_unknown_row,
        window_open_sent,
        window_close_sent,
        window_keepalives_sent,
        window_level_refused,
        window_body_stale,
        window_body_preroster,
        window_folds,
        window_fold_hits,
        window_fold_divergence,
        window_full_chain_folds,
        window_chains_held,
        window_compose_hold_ticks,
        window_hop_dead,
        window_t_monotone_stalled,
        window_chain_cycle,
        window_instant_mismatch,
        window_rotated_refused,
        window_alien_rows,
        window_hop_invalid,
        window_unresolved_standing,
        window_dedup_disagree,
        window_dedup_max_dev_cells,
        window_head_reads_sent,
        window_composed_rows,
        old_realm_frames_dropped,
        old_scene_deltas_dropped,
        scene_levels_sent,
        scene_deltas_sent,
        scene_datagrams_sent,
        window_relays_ingested,
        window_relay_undecodable,
        window_relay_unvouched,
        window_relay_stale,
        window_relay_rows_composed,
        window_relay_descent_refused,
        window_relay_stamp_missing,
        window_relay_unrostered,
        window_relay_stamp_skew_ticks,
        window_relay_depth_max,
        window_looks_pruned,
        window_relay_levels_pruned,
        window_relay_interior_unvouched,
        window_relay_interior_filtered,
        sessions_open,
        dynamic_shards,
        windows_open,
    }
}

/// Build the gateway's `/admin/snapshot` body from its World. The three resource reads are INFALLIBLE:
/// `build_app` inserts [`ClockSample`] and `register_gateway` inserts [`GatewayStats`]/[`GatewaySessions`] on
/// every gateway world, so a tolerant `get_resource` would only add an uncoverable panic-free arm (HR5). The
/// directory/sagas/leases stay EMPTY — a gateway does not own them (the orchestrator does), so a gateway
/// snapshot's `cluster_bootstrapped` is always false and never misleads an operator.
#[must_use]
pub fn gateway_admin_snapshot(world: &mut World, world_generation: u64) -> AdminSnapshot {
    let (universe_tick, epoch) = {
        let clock = world.resource::<ClockSample>();
        (clock.universe_tick, clock.epoch)
    };
    let (sessions_open, dynamic_shards, windows_open) = {
        let sessions = world.resource::<GatewaySessions>();
        (
            sessions.len() as u64,
            sessions.dynamic_shard_count() as u64,
            sessions.windows_open_count() as u64,
        )
    };
    let stats = *world.resource::<GatewayStats>();
    let mut snapshot = AdminSnapshot::shaped_empty(universe_tick, epoch, world_generation);
    snapshot.gateway = Some(gateway_view(
        &stats,
        sessions_open,
        dynamic_shards,
        windows_open,
    ));
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
            window_rows_ingested: 30,
            // 31..36: the Slice-A window-lane counters, appended in declaration order.
            window_sender_mismatch: 31,
            window_misauthored_body: 32,
            window_unknown_row: 33,
            window_open_sent: 34,
            window_close_sent: 35,
            window_keepalives_sent: 36,
            // 38..58: the Slice-B composer counters, declaration order (37 is the windows_open
            // gauge below, minted before this block landed).
            window_level_refused: 38,
            window_body_stale: 39,
            window_body_preroster: 40,
            window_folds: 41,
            window_fold_hits: 42,
            window_fold_divergence: 43,
            window_full_chain_folds: 44,
            window_chains_held: 45,
            window_compose_hold_ticks: 46,
            window_hop_dead: 47,
            window_t_monotone_stalled: 48,
            window_chain_cycle: 49,
            window_instant_mismatch: 50,
            window_rotated_refused: 51,
            window_alien_rows: 52,
            window_hop_invalid: 53,
            window_unresolved_standing: 54,
            window_dedup_disagree: 55,
            window_dedup_max_dev_cells: 56,
            window_head_reads_sent: 57,
            window_composed_rows: 58,
            // 59..69: the tombstoned-lane drop counters + the composed egress + the Q2 relay,
            // in declaration order.
            old_realm_frames_dropped: 59,
            old_scene_deltas_dropped: 60,
            scene_levels_sent: 61,
            scene_deltas_sent: 62,
            scene_datagrams_sent: 63,
            window_relays_ingested: 64,
            window_relay_undecodable: 65,
            window_relay_unvouched: 66,
            window_relay_stale: 67,
            window_relay_rows_composed: 68,
            // 82..86: the C5 split of the retired window_relay_unplaceable (69) + its two
            // gauges (look_horizon.md slice 0), appended in declaration order.
            window_relay_descent_refused: 82,
            window_relay_stamp_missing: 83,
            window_relay_unrostered: 84,
            window_relay_stamp_skew_ticks: 85,
            window_relay_depth_max: 86,
            window_looks_pruned: 80,
            window_relay_levels_pruned: 81,
            // 87..88: the look-horizon slice-3 interior-forward pair, appended in
            // declaration order.
            window_relay_interior_unvouched: 87,
            window_relay_interior_filtered: 88,
            star_catalogue_parts_sent: 0,
            sky_alive_beats_sent: 0,
            sky_requests_sent: 0,
            sky_held_stated: 0,
            sky_parts_skipped: 0,
        };
        let view = gateway_view(&stats, 24, 25, 37);
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
        assert_eq!(view.window_rows_ingested, 30);
        assert_eq!(view.window_sender_mismatch, 31);
        assert_eq!(view.window_misauthored_body, 32);
        assert_eq!(view.window_unknown_row, 33);
        assert_eq!(view.window_open_sent, 34);
        assert_eq!(view.window_close_sent, 35);
        assert_eq!(view.window_keepalives_sent, 36);
        assert_eq!(view.window_level_refused, 38);
        assert_eq!(view.window_body_stale, 39);
        assert_eq!(view.window_body_preroster, 40);
        assert_eq!(view.window_folds, 41);
        assert_eq!(view.window_fold_hits, 42);
        assert_eq!(view.window_fold_divergence, 43);
        assert_eq!(view.window_full_chain_folds, 44);
        assert_eq!(view.window_chains_held, 45);
        assert_eq!(view.window_compose_hold_ticks, 46);
        assert_eq!(view.window_hop_dead, 47);
        assert_eq!(view.window_t_monotone_stalled, 48);
        assert_eq!(view.window_chain_cycle, 49);
        assert_eq!(view.window_instant_mismatch, 50);
        assert_eq!(view.window_rotated_refused, 51);
        assert_eq!(view.window_alien_rows, 52);
        assert_eq!(view.window_hop_invalid, 53);
        assert_eq!(view.window_unresolved_standing, 54);
        assert_eq!(view.window_dedup_disagree, 55);
        assert_eq!(view.window_dedup_max_dev_cells, 56);
        assert_eq!(view.window_head_reads_sent, 57);
        assert_eq!(view.window_composed_rows, 58);
        assert_eq!(view.old_realm_frames_dropped, 59);
        assert_eq!(view.old_scene_deltas_dropped, 60);
        assert_eq!(view.scene_levels_sent, 61);
        assert_eq!(view.scene_deltas_sent, 62);
        assert_eq!(view.scene_datagrams_sent, 63);
        assert_eq!(view.window_relays_ingested, 64);
        assert_eq!(view.window_relay_undecodable, 65);
        assert_eq!(view.window_relay_unvouched, 66);
        assert_eq!(view.window_relay_stale, 67);
        assert_eq!(view.window_relay_rows_composed, 68);
        assert_eq!(view.window_relay_descent_refused, 82);
        assert_eq!(view.window_relay_stamp_missing, 83);
        assert_eq!(view.window_relay_unrostered, 84);
        assert_eq!(view.window_relay_stamp_skew_ticks, 85);
        assert_eq!(view.window_relay_depth_max, 86);
        assert_eq!(view.window_looks_pruned, 80);
        assert_eq!(view.window_relay_levels_pruned, 81);
        assert_eq!(view.window_relay_interior_unvouched, 87);
        assert_eq!(view.window_relay_interior_filtered, 88);
        assert_eq!(view.sessions_open, 24);
        assert_eq!(view.dynamic_shards, 25);
        assert_eq!(view.windows_open, 37);
    }
}
