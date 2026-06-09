//! The `--at-tick T` capture-alignment (pure). The deterministic artifact is the
//! captured STATE (wire truth) aligned to the shot; pixels are structurally-reproducible
//! only. `--at-tick T` reuses the Slice-2 wait-until predicate: block until the delivered
//! snapshots reach T, then capture and record the ACTUAL `(cursor, snapshots_applied)`.

use vd_devproto::{DevState, WaitField, WaitOp, WaitPredicate};

use crate::manifest::{CaptureEntry, CaptureKind};

/// The wait-until predicate `--at-tick T` blocks on: delivered snapshots reached `T`.
#[must_use]
pub fn at_tick_predicate(target_tick: u64) -> WaitPredicate {
    WaitPredicate {
        field: WaitField::SnapshotsApplied,
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

/// Build the manifest entry for a capture, recording the ACTUAL delivered alignment
/// (cursor + applied-snapshot count) from the state at capture time.
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
        snapshots_applied: state.snapshots_applied,
        state_path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vd_devproto::{DevPhase, DevTransferView};

    fn state(snapshots_applied: u64, cursor: Option<f64>) -> DevState {
        DevState {
            phase: DevPhase::Active,
            session: None,
            own_entity: None,
            render_cursor: cursor,
            entities: Vec::new(),
            snapshots_applied,
            stale_frames_dropped: 0,
            sent_input_count: 0,
            decode_errors: 0,
            ignored: 0,
            foreign_peer_drops: 0,
            dev_commands_applied: 0,
            dev_commands_dropped: 0,
            transfer: DevTransferView::None,
        }
    }

    #[test]
    fn at_tick_predicate_is_snapshots_ge_target() {
        let p = at_tick_predicate(50);
        assert_eq!(p.field, WaitField::SnapshotsApplied);
        assert_eq!(p.op, WaitOp::Ge);
        assert_eq!(p.value, 50);
    }

    #[test]
    fn ready_immediately_with_no_target_else_waits_for_the_tick() {
        assert!(ready_to_capture(None, &state(0, None)));
        assert!(!ready_to_capture(Some(50), &state(49, None)));
        assert!(ready_to_capture(Some(50), &state(50, None)));
        assert!(ready_to_capture(Some(50), &state(51, None)));
    }

    #[test]
    fn capture_entry_records_the_actual_delivered_alignment() {
        let e = capture_entry(
            CaptureKind::Screenshot,
            "shots/0001.png".to_owned(),
            Some(100),
            &state(102, Some(99.6)),
            Some("state/0001.json".to_owned()),
        );
        assert_eq!(e.tick, Some(100)); // the requested target
        assert_eq!(e.snapshots_applied, 102); // the ACTUAL delivered count
        assert_eq!(e.cursor, Some(99.6)); // the ACTUAL cursor
        assert_eq!(e.kind, CaptureKind::Screenshot);
        assert_eq!(e.state_path.as_deref(), Some("state/0001.json"));
    }
}
