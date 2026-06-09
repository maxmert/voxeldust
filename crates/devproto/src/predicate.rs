//! The `wait-until` predicate (HR6 closed-loop): a STRUCTURED `{field, op, value}`
//! over `DevState`, not a parsed mini-language — so it is trivially 100%-branch
//! coverable and cannot drift between the client (which evaluates it each tick) and
//! `vdctl` (which sends it). All fields are integer-valued, so a `u64` compare is
//! exact (no float-equality hazard).

use serde::{Deserialize, Serialize};

use crate::state::{DevPhase, DevState};

/// A `DevState` field a predicate can test (all integer-valued).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitField {
    /// 1 once the client is Active (welcomed + subscribed), else 0.
    Active,
    /// The number of composited render entities.
    EntityCount,
    /// 1 once `own_entity` is known (from `AuthorityChanged`), else 0.
    OwnEntitySet,
    /// The count of applied snapshots (session-relative; proves frames are landing).
    SnapshotsApplied,
    /// The freshest APPLIED universe tick (run-stable + join-independent) — what
    /// `screenshot --at-tick` aligns on for reproducible captures. Reads `0` BOTH before
    /// any frame has landed AND for a genuine applied tick 0, so the two are
    /// indistinguishable; wait for a specific frame with `universe_tick ge N` (`N >= 1`),
    /// not `eq 0` / `le 0`.
    UniverseTick,
}

impl WaitField {
    /// Whether this field is a 0/1 flag (vs an open-ended count) — lets the `vdctl`
    /// front-end reject a nonsense `wait active eq 2` instead of letting it silently
    /// never fire. Lives here (next to the field) so the client and `vdctl` can't drift.
    #[must_use]
    pub fn is_boolean(self) -> bool {
        match self {
            WaitField::Active | WaitField::OwnEntitySet => true,
            WaitField::EntityCount | WaitField::SnapshotsApplied | WaitField::UniverseTick => false,
        }
    }

    /// Every field — the SINGLE source for CLI help / validation, so a new variant can
    /// never leave `vdctl`'s help text or error message stale (the DRY drift the audit hit).
    pub const ALL: [WaitField; 5] = [
        WaitField::Active,
        WaitField::EntityCount,
        WaitField::OwnEntitySet,
        WaitField::SnapshotsApplied,
        WaitField::UniverseTick,
    ];

    /// The canonical snake_case name — matches the serde representation (pinned by a test),
    /// so callers can build the field list from `WaitField::ALL` without re-listing strings.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            WaitField::Active => "active",
            WaitField::EntityCount => "entity_count",
            WaitField::OwnEntitySet => "own_entity_set",
            WaitField::SnapshotsApplied => "snapshots_applied",
            WaitField::UniverseTick => "universe_tick",
        }
    }
}

/// The comparison operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WaitOp {
    Eq,
    Ne,
    Ge,
    Le,
    Gt,
    Lt,
}

/// A wait-until predicate: `field op value` over the delivered state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WaitPredicate {
    pub field: WaitField,
    pub op: WaitOp,
    pub value: u64,
}

impl WaitOp {
    /// The raw integer comparison (the monomorphic op-match shared by `eval` and the
    /// satisfiability check, so the two cannot diverge).
    #[must_use]
    fn matches(self, actual: u64, value: u64) -> bool {
        match self {
            WaitOp::Eq => actual == value,
            WaitOp::Ne => actual != value,
            WaitOp::Ge => actual >= value,
            WaitOp::Le => actual <= value,
            WaitOp::Gt => actual > value,
            WaitOp::Lt => actual < value,
        }
    }

    /// Every operator — the SINGLE source for CLI help / validation (mirrors
    /// [`WaitField::ALL`]), so a new operator can never leave `vdctl`'s usage text or
    /// `bad op` error stale.
    pub const ALL: [WaitOp; 6] = [
        WaitOp::Eq,
        WaitOp::Ne,
        WaitOp::Ge,
        WaitOp::Le,
        WaitOp::Gt,
        WaitOp::Lt,
    ];

    /// The canonical snake_case name — matches the serde representation (pinned by a
    /// test), so `vdctl` builds the operator list from `WaitOp::ALL` without re-listing.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            WaitOp::Eq => "eq",
            WaitOp::Ne => "ne",
            WaitOp::Ge => "ge",
            WaitOp::Le => "le",
            WaitOp::Gt => "gt",
            WaitOp::Lt => "lt",
        }
    }
}

impl WaitPredicate {
    /// Evaluate the predicate against a delivered `DevState` (pure).
    #[must_use]
    pub fn eval(self, state: &DevState) -> bool {
        self.op.matches(field_value(state, self.field), self.value)
    }

    /// Whether this predicate CAN ever hold — so `vdctl` rejects a wait that could only
    /// ever time out, for ANY field class (not just booleans). `field_is_boolean`
    /// selects the rule:
    /// - boolean (0/1) field: satisfiable iff it holds for actual 0 OR 1 (rejects
    ///   `active gt 1`, `active lt 0`, `active eq 2`, …);
    /// - open-ended `u64` count: constant-false only when the bound excludes the ENTIRE
    ///   `u64` range — `< 0` (nothing below the floor) OR `> u64::MAX` (nothing above the
    ///   ceiling). So `universe_tick lt 0` and `… gt u64::MAX` are rejected; every other
    ///   `(op, value)` is reachable by some count and is kept.
    ///
    /// Operator-aware (not a value-range check); the boolean arm evaluates both actuals.
    #[must_use]
    pub fn is_satisfiable(self, field_is_boolean: bool) -> bool {
        if field_is_boolean {
            self.op.matches(0, self.value) || self.op.matches(1, self.value)
        } else {
            // Symmetric empty-range guard: `< 0` (below the floor) and `> u64::MAX`
            // (above the ceiling) are the only forms no `u64` actual can satisfy.
            !matches!(
                (self.op, self.value),
                (WaitOp::Lt, 0) | (WaitOp::Gt, u64::MAX)
            )
        }
    }
}

fn field_value(state: &DevState, field: WaitField) -> u64 {
    match field {
        WaitField::Active => u64::from(state.phase == DevPhase::Active),
        WaitField::EntityCount => state.entities.len() as u64,
        WaitField::OwnEntitySet => u64::from(state.own_entity.is_some()),
        WaitField::SnapshotsApplied => state.snapshots_applied,
        WaitField::UniverseTick => state.universe_tick.unwrap_or(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::tests::sample;

    fn pred(field: WaitField, op: WaitOp, value: u64) -> WaitPredicate {
        WaitPredicate { field, op, value }
    }

    #[test]
    fn each_field_reads_the_expected_value() {
        let s = sample(); // Active, 1 entity, own set, 4 snapshots applied
        assert_eq!(field_value(&s, WaitField::Active), 1);
        assert_eq!(field_value(&s, WaitField::EntityCount), 1);
        assert_eq!(field_value(&s, WaitField::OwnEntitySet), 1);
        assert_eq!(field_value(&s, WaitField::SnapshotsApplied), 4);
        assert_eq!(field_value(&s, WaitField::UniverseTick), 101);
        // UniverseTick is None before the first snapshot -> reads 0.
        let mut fresh = sample();
        fresh.universe_tick = None;
        assert_eq!(field_value(&fresh, WaitField::UniverseTick), 0);
        // The "not yet" side of the boolean fields.
        let mut connecting = sample();
        connecting.phase = DevPhase::Connecting;
        connecting.own_entity = None;
        assert_eq!(field_value(&connecting, WaitField::Active), 0);
        assert_eq!(field_value(&connecting, WaitField::OwnEntitySet), 0);
    }

    #[test]
    fn every_operator_arm_is_exercised() {
        let s = sample(); // SnapshotsApplied == 4
        let f = WaitField::SnapshotsApplied;
        assert!(pred(f, WaitOp::Eq, 4).eval(&s));
        assert!(!pred(f, WaitOp::Eq, 5).eval(&s));
        assert!(pred(f, WaitOp::Ne, 5).eval(&s));
        assert!(!pred(f, WaitOp::Ne, 4).eval(&s));
        assert!(pred(f, WaitOp::Ge, 4).eval(&s));
        assert!(!pred(f, WaitOp::Ge, 5).eval(&s));
        assert!(pred(f, WaitOp::Le, 4).eval(&s));
        assert!(!pred(f, WaitOp::Le, 3).eval(&s));
        assert!(pred(f, WaitOp::Gt, 3).eval(&s));
        assert!(!pred(f, WaitOp::Gt, 4).eval(&s));
        assert!(pred(f, WaitOp::Lt, 5).eval(&s));
        assert!(!pred(f, WaitOp::Lt, 4).eval(&s));
    }

    #[test]
    fn field_names_match_the_serde_representation_and_all_is_complete() {
        // name() is the canonical source; pin it to serde so the two can't drift, and
        // confirm ALL carries every distinct field the CLI offers.
        for field in WaitField::ALL {
            let serde = serde_json::to_string(&field).expect("encode");
            assert_eq!(serde, format!("\"{}\"", field.name()), "name() == serde");
        }
        let mut names: Vec<&str> = WaitField::ALL.iter().map(|f| f.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), WaitField::ALL.len(), "ALL names are distinct");
    }

    #[test]
    fn boolean_fields_are_distinguished_from_counts() {
        assert!(WaitField::Active.is_boolean());
        assert!(WaitField::OwnEntitySet.is_boolean());
        assert!(!WaitField::EntityCount.is_boolean());
        assert!(!WaitField::SnapshotsApplied.is_boolean());
        assert!(!WaitField::UniverseTick.is_boolean());
    }

    #[test]
    fn operator_names_match_the_serde_representation_and_all_is_complete() {
        // name() is the canonical source for vdctl's op list; pin it to serde so the two
        // can't drift, and confirm ALL carries every distinct operator (mirrors the
        // WaitField guard — the same DRY discipline applied to its sibling enum).
        for op in WaitOp::ALL {
            let serde = serde_json::to_string(&op).expect("encode");
            assert_eq!(serde, format!("\"{}\"", op.name()), "name() == serde");
        }
        let mut names: Vec<&str> = WaitOp::ALL.iter().map(|o| o.name()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), WaitOp::ALL.len(), "ALL operators are distinct");
    }

    #[test]
    fn boolean_satisfiability_rejects_constant_false_predicates() {
        // Satisfiable: `eq 0` holds at actual 0 (LHS true); `eq 1` holds only at
        // actual 1 (LHS false, RHS true); `ge 0` / `ne 2` hold for any 0/1.
        assert!(pred(WaitField::Active, WaitOp::Eq, 0).is_satisfiable(true));
        assert!(pred(WaitField::Active, WaitOp::Eq, 1).is_satisfiable(true));
        assert!(pred(WaitField::Active, WaitOp::Ge, 0).is_satisfiable(true));
        assert!(pred(WaitField::OwnEntitySet, WaitOp::Ne, 2).is_satisfiable(true));
        // Constant-false for a 0/1 field — the forms the value-only guard missed.
        assert!(!pred(WaitField::Active, WaitOp::Gt, 1).is_satisfiable(true));
        assert!(!pred(WaitField::Active, WaitOp::Lt, 0).is_satisfiable(true));
        assert!(!pred(WaitField::Active, WaitOp::Eq, 2).is_satisfiable(true));
    }

    #[test]
    fn count_satisfiability_rejects_the_empty_range_forms() {
        // Open-ended u64 count: the two forms no actual can satisfy are `< 0` (below the
        // floor) and `> u64::MAX` (above the ceiling) — both rejected (symmetric guard).
        assert!(!pred(WaitField::UniverseTick, WaitOp::Lt, 0).is_satisfiable(false));
        assert!(!pred(WaitField::SnapshotsApplied, WaitOp::Lt, 0).is_satisfiable(false));
        assert!(!pred(WaitField::UniverseTick, WaitOp::Gt, u64::MAX).is_satisfiable(false));
        // Every reachable bound is kept: `lt` of a positive; `gt` BELOW the ceiling; the
        // ceiling itself reachable via ge/eq; and the other operators at any value.
        assert!(pred(WaitField::UniverseTick, WaitOp::Lt, 1).is_satisfiable(false));
        assert!(pred(WaitField::EntityCount, WaitOp::Gt, 9999).is_satisfiable(false));
        assert!(pred(WaitField::UniverseTick, WaitOp::Gt, u64::MAX - 1).is_satisfiable(false));
        assert!(pred(WaitField::UniverseTick, WaitOp::Ge, u64::MAX).is_satisfiable(false));
        assert!(pred(WaitField::SnapshotsApplied, WaitOp::Eq, 0).is_satisfiable(false));
        assert!(pred(WaitField::SnapshotsApplied, WaitOp::Ge, 0).is_satisfiable(false));
    }

    #[test]
    fn predicate_roundtrips_through_json() {
        let p = pred(WaitField::Active, WaitOp::Eq, 1);
        let json = serde_json::to_string(&p).expect("encode");
        assert_eq!(
            serde_json::from_str::<WaitPredicate>(&json).expect("decode"),
            p
        );
    }
}
