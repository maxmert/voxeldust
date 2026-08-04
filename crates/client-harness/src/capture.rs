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

/// Build the manifest entry for a capture. `freshest_tick` + `cursor` are the alignment
/// quantities sampled from the SAME render snapshot the PIXELS came from (NOT a post-capture
/// poll, which drifts forward by the render round-trip) — so the manifest identifies the
/// captured world. `snapshots_applied` is the session-relative diagnostic count.
#[must_use]
pub fn capture_entry(
    kind: CaptureKind,
    path: String,
    target_tick: Option<u64>,
    freshest_tick: Option<u64>,
    cursor: Option<f64>,
    snapshots_applied: u64,
    state_path: Option<String>,
) -> CaptureEntry {
    CaptureEntry {
        kind,
        path,
        tick: target_tick,
        cursor,
        freshest_tick,
        snapshots_applied,
        state_path,
    }
}

/// A planned `record` sequence: a clamped frame count + the per-frame interval. Pure (the
/// bin supplies the live clock + the safety bounds), so the cadence math is Tier-A tested
/// rather than buried in the dev-control bin shell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RecordPlan {
    pub fps: u32,
    pub frames: u64,
    pub interval_secs: f64,
}

/// Plan `record fps secs`: clamp `fps` to `[1, max_fps]` and the frame count to
/// `[1, max_frames]` (so a fat-fingered `--secs 1e9` cannot run unbounded), with
/// `interval = 1/fps`. `None` for a non-finite or non-positive duration — a loud reject,
/// never a silent zero-frame run.
#[must_use]
pub fn plan_record(fps: u32, secs: f64, max_fps: u32, max_frames: u64) -> Option<RecordPlan> {
    if !(secs.is_finite() && secs > 0.0) {
        return None;
    }
    let fps = fps.clamp(1, max_fps.max(1));
    // fps>=1 and secs>0 ⇒ want>0; the f64→u64 cast saturates, then clamps into the budget.
    let frames = ((f64::from(fps) * secs).round() as u64).clamp(1, max_frames.max(1));
    Some(RecordPlan {
        fps,
        frames,
        interval_secs: 1.0 / f64::from(fps),
    })
}

/// The run-relative state-dump path paired with a capture's PNG: `frames/foo.png` →
/// `state/foo.json` (so the aligned wire-truth dump sits beside its pixels). Pure string
/// mapping — handles a path with or without a directory and with or without the `.png` ext.
#[must_use]
pub fn state_rel_for(rel_path: &str) -> String {
    let after_slash = rel_path.rfind('/').map_or(rel_path, |i| &rel_path[i + 1..]);
    let stem = after_slash.strip_suffix(".png").unwrap_or(after_slash);
    format!("state/{stem}.json")
}

/// Sanitize an agent-supplied name into a single SAFE path component: `[A-Za-z0-9_-]`
/// kept, everything else (slashes, dots, spaces, …) mapped to `_`, and an empty result
/// becomes `"capture"`. The ONE filename rule for everything an agent names on disk
/// (capture labels, run names) — so a label like `../../x` can never escape the run dir
/// or desynchronize the PNG↔state pairing.
#[must_use]
pub fn sanitize_stem(raw: &str) -> String {
    let safe: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if safe.is_empty() {
        "capture".to_owned()
    } else {
        safe
    }
}

/// The run-relative PNG path for a capture: screenshots under `shots/`, record frames
/// under `frames/`, the stem sanitized through [`sanitize_stem`]. The ONE derivation of
/// a capture's on-disk identity (the render thread writes it; the manifest stores it;
/// [`state_rel_for`] pairs the state dump with it).
#[must_use]
pub fn capture_rel_path(kind: CaptureKind, stem: &str) -> String {
    let subdir = match kind {
        CaptureKind::Screenshot => "shots",
        CaptureKind::Frame => "frames",
    };
    format!("{subdir}/{}.png", sanitize_stem(stem))
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
            entity_feed_newest_tick: universe_tick,
            realm_feed_newest_tick: None,
            render_origin: vd_devproto::DevRenderOrigin {
                cell: [0, 0, 0],
                offset: [0.0, 0.0, 0.0],
            },
            entity_windows: Default::default(),
            realm_windows: Default::default(),
            feed_skew_ticks: None,
            entities: Vec::new(),
            realm_boxes: Vec::new(),
            snapshots_applied,
            realm_frames_applied: 0,
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
    fn capture_entry_records_the_render_sampled_alignment() {
        let e = capture_entry(
            CaptureKind::Screenshot,
            "shots/0001.png".to_owned(),
            Some(100),  // requested target
            Some(102),  // render-sampled freshest tick (the pixels' tick)
            Some(99.6), // render-sampled cursor
            7,          // session diagnostic
            Some("state/0001.json".to_owned()),
        );
        assert_eq!(e.tick, Some(100));
        assert_eq!(e.freshest_tick, Some(102));
        assert_eq!(e.snapshots_applied, 7);
        assert_eq!(e.cursor, Some(99.6));
        assert_eq!(e.kind, CaptureKind::Screenshot);
        assert_eq!(e.state_path.as_deref(), Some("state/0001.json"));
    }

    #[test]
    fn plan_record_clamps_fps_and_frames_and_rejects_bad_durations() {
        // Normal: 8 fps for 1 s = 8 frames, 0.125 s interval.
        let p = plan_record(8, 1.0, 120, 3600).expect("valid");
        assert_eq!(
            p,
            RecordPlan {
                fps: 8,
                frames: 8,
                interval_secs: 0.125
            }
        );
        // fps clamped up to 1; a sub-frame duration still yields at least 1 frame.
        assert_eq!(plan_record(0, 0.01, 120, 3600).expect("valid").frames, 1);
        // fps + frames clamped down to the ceilings.
        let c = plan_record(10_000, 1e9, 120, 3600).expect("valid");
        assert_eq!((c.fps, c.frames), (120, 3600));
        // Non-finite / non-positive durations are rejected loudly.
        assert_eq!(plan_record(30, 0.0, 120, 3600), None);
        assert_eq!(plan_record(30, -1.0, 120, 3600), None);
        assert_eq!(plan_record(30, f64::NAN, 120, 3600), None);
    }

    #[test]
    fn state_rel_for_pairs_the_state_dump_with_the_png() {
        assert_eq!(
            state_rel_for("frames/flyby-0007.png"),
            "state/flyby-0007.json"
        );
        assert_eq!(state_rel_for("shots/hero.png"), "state/hero.json");
        // No directory, and a non-.png path, both handled.
        assert_eq!(state_rel_for("hero.png"), "state/hero.json");
        assert_eq!(state_rel_for("weird"), "state/weird.json");
    }

    #[test]
    fn sanitize_stem_contains_traversal_and_never_returns_empty() {
        assert_eq!(sanitize_stem("hero-Shot_42"), "hero-Shot_42"); // safe chars kept
        // Slashes + dots become '_' — a traversal attempt is contained to ONE component.
        assert_eq!(sanitize_stem("../../etc/passwd"), "______etc_passwd");
        assert_eq!(sanitize_stem("a b.png"), "a_b_png");
        assert_eq!(sanitize_stem(""), "capture"); // empty → the fallback stem
        assert_eq!(sanitize_stem("///"), "___"); // non-empty unsafe input keeps its length
    }

    #[test]
    fn capture_rel_path_routes_by_kind_and_sanitizes_the_stem() {
        assert_eq!(
            capture_rel_path(CaptureKind::Screenshot, "hero"),
            "shots/hero.png"
        );
        assert_eq!(
            capture_rel_path(CaptureKind::Frame, "fly-0001"),
            "frames/fly-0001.png"
        );
        // A hostile label cannot escape the run dir or desync the state pairing.
        let rel = capture_rel_path(CaptureKind::Screenshot, "../../x");
        assert_eq!(rel, "shots/______x.png");
        assert_eq!(state_rel_for(&rel), "state/______x.json");
    }
}
