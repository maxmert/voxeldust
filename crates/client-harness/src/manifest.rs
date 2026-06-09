//! The `runs/<UTC>__<scenario>/manifest.json` serde — aligns every capture to its tick
//! AND its `state` dump so an agent's diagnosis is reproducible. Pure serde; the
//! timestamp is passed IN (the lib reads no clock).

use serde::{Deserialize, Serialize};

/// What a capture is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaptureKind {
    /// A single `vdctl screenshot`.
    Screenshot,
    /// One frame of a `vdctl record` sequence.
    Frame,
}

/// One capture, aligned to the delivered state at capture time.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaptureEntry {
    pub kind: CaptureKind,
    /// Path relative to the run dir (`shots/…` or `frames/…`).
    pub path: String,
    /// The `--at-tick` target, if one was requested.
    pub tick: Option<u64>,
    /// The ACTUAL render cursor at capture (`None` before the first snapshot).
    pub cursor: Option<f64>,
    /// The ACTUAL freshest UNIVERSE tick at capture (run-stable; the alignment quantity
    /// that lets two runs recover the same world state for a golden diff). `None` before
    /// the first snapshot.
    pub freshest_tick: Option<u64>,
    /// The ACTUAL applied-snapshot count at capture (session-relative; diagnostic).
    pub snapshots_applied: u64,
    /// The aligned `state/…` dump for this capture, if written.
    pub state_path: Option<String>,
}

/// The run manifest: one per `runs/<UTC>__<scenario>/`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunManifest {
    pub scenario: String,
    /// The run start time — passed IN by the bin (the lib never reads a clock).
    pub started_utc: String,
    pub captures: Vec<CaptureEntry>,
}

impl RunManifest {
    #[must_use]
    pub fn new(scenario: String, started_utc: String) -> RunManifest {
        RunManifest {
            scenario,
            started_utc,
            captures: Vec::new(),
        }
    }

    pub fn push(&mut self, entry: CaptureEntry) {
        self.captures.push(entry);
    }

    /// Pretty-JSON the manifest — infallible (all fields are plain serde types).
    #[must_use]
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("RunManifest is plain serde")
    }

    /// Parse a manifest from JSON.
    ///
    /// # Errors
    /// If the JSON does not match the manifest shape.
    pub fn from_json(json: &str) -> Result<RunManifest, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_roundtrips_through_json_with_both_capture_kinds() {
        let mut m = RunManifest::new("walk_to_dot".to_owned(), "2026-06-08T20:00:00Z".to_owned());
        m.push(CaptureEntry {
            kind: CaptureKind::Screenshot,
            path: "shots/0001.png".to_owned(),
            tick: Some(100),
            cursor: Some(97.6),
            freshest_tick: Some(100),
            snapshots_applied: 12,
            state_path: Some("state/0001.json".to_owned()),
        });
        m.push(CaptureEntry {
            kind: CaptureKind::Frame,
            path: "frames/0001.png".to_owned(),
            tick: None,
            cursor: None,
            freshest_tick: None,
            snapshots_applied: 0,
            state_path: None,
        });
        let json = m.to_json();
        assert!(json.contains("\"kind\": \"screenshot\""));
        assert!(json.contains("\"kind\": \"frame\""));
        assert_eq!(RunManifest::from_json(&json).expect("decode"), m);
    }

    #[test]
    fn bad_json_is_a_typed_error() {
        RunManifest::from_json("not json").expect_err("malformed manifest rejected");
    }
}
