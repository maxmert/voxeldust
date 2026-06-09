//! The `--at-tick T` capture-alignment (pure). The deterministic artifact is the
//! captured STATE (wire truth) aligned to the shot; pixels are structurally-reproducible
//! only. `--at-tick T` reuses the wait-until predicate: block until the delivered
//! UNIVERSE tick reaches T (run-stable + join-independent — NOT the session-relative
//! `snapshots_applied` count, so two runs that drop different early frames still capture
//! the same world state), then capture and record the ACTUAL `(cursor, freshest_tick)`.

use vd_devproto::{DevState, WaitField, WaitOp, WaitPredicate};

use crate::manifest::{CaptureEntry, CaptureKind};

/// The wait-until predicate `--at-tick T` blocks on: the delivered UNIVERSE tick reached `T`.
#[must_use]
pub fn at_tick_predicate(target_tick: u64) -> WaitPredicate {
    WaitPredicate {
        field: WaitField::UniverseTick,
        op: WaitOp::Ge,
        value: target_tick,
    }
}

/// Whether the delivered state has reached the `--at-tick` target (capture now). A
/// `None` target captures immediately.
#[must_use]
pub fn ready_to_capture(target_tick: Option<u64>, state: &DevState) -> bool {
    match target_tick {
        None => true,
        Some(target) => at_tick_predicate(target).eval(state),
    }
}

/// Build the manifest entry for a capture, recording the ACTUAL delivered alignment —
/// the run-stable `freshest_tick` (the deterministic alignment quantity) plus the cursor
/// and the diagnostic session count — from the state at capture time.
#[must_use]
pub fn capture_entry(
    kind: CaptureKind,
    path: String,
    target_tick: Option<u64>,
    state: &DevState,
    state_path: Option<String>,
) -> CaptureEntry {
    CaptureEntry {
        kind,
        path,
        tick: target_tick,
        cursor: state.render_cursor,
        freshest_tick: state.universe_tick,
        snapshots_applied: state.snapshots_applied,
        state_path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_devproto::{DevPhase, DevTransferView};

    fn state(universe_tick: Option<u64>, snapshots_applied: u64, cursor: Option<f64>) -> DevState {
        DevState {
            phase: DevPhase::Active,
            session: None,
            own_entity: None,
            location: None,
            render_cursor: cursor,
            universe_tick,
            entities: Vec::new(),
            snapshots_applied,
            stale_frames_dropped: 0,
            sent_input_count: 0,
            decode_errors: 0,
            ignored: 0,
            foreign_peer_drops: 0,
            nonfinite_poses: 0,
            dev_commands_applied: 0,
            dev_commands_dropped: 0,
            transfer: DevTransferView::None,
        }
    }

    #[test]
    fn at_tick_predicate_is_universe_tick_ge_target() {
        let p = at_tick_predicate(50);
        assert_eq!(p.field, WaitField::UniverseTick);
        assert_eq!(p.op, WaitOp::Ge);
        assert_eq!(p.value, 50);
    }

    #[test]
    fn ready_immediately_with_no_target_else_waits_for_the_universe_tick() {
        assert!(ready_to_capture(None, &state(None, 0, None)));
        // A client that JOINED late (snapshots_applied small) but the universe tick is
        // already past T captures correctly — alignment is on the universe tick.
        assert!(!ready_to_capture(Some(50), &state(Some(49), 1, None)));
        assert!(ready_to_capture(Some(50), &state(Some(50), 1, None)));
        assert!(ready_to_capture(Some(50), &state(Some(500), 2, None)));
    }

    #[test]
    fn capture_entry_records_the_actual_delivered_alignment() {
        let e = capture_entry(
            CaptureKind::Screenshot,
            "shots/0001.png".to_owned(),
            Some(100),
            &state(Some(102), 7, Some(99.6)),
            Some("state/0001.json".to_owned()),
        );
        assert_eq!(e.tick, Some(100)); // the requested target
        assert_eq!(e.freshest_tick, Some(102)); // the ACTUAL universe tick (run-stable)
        assert_eq!(e.snapshots_applied, 7); // the session count (diagnostic)
        assert_eq!(e.cursor, Some(99.6)); // the ACTUAL cursor
        assert_eq!(e.kind, CaptureKind::Screenshot);
        assert_eq!(e.state_path.as_deref(), Some("state/0001.json"));
    }
}
