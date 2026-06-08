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
    /// The count of applied snapshots (proves frames are landing, not just connected).
    SnapshotsApplied,
}

impl WaitField {
    /// Whether this field is a 0/1 flag (vs an open-ended count) — lets the `vdctl`
    /// front-end reject a nonsense `wait active eq 2` instead of letting it silently
    /// never fire. Lives here (next to the field) so the client and `vdctl` can't drift.
    #[must_use]
    pub fn is_boolean(self) -> bool {
        match self {
            WaitField::Active | WaitField::OwnEntitySet => true,
            WaitField::EntityCount | WaitField::SnapshotsApplied => false,
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
}

impl WaitPredicate {
    /// Evaluate the predicate against a delivered `DevState` (pure).
    #[must_use]
    pub fn eval(self, state: &DevState) -> bool {
        self.op.matches(field_value(state, self.field), self.value)
    }

    /// Whether this predicate CAN ever hold for a 0/1 (boolean) field. `false` means
    /// it is constant-false (e.g. `active gt 1`, `active lt 0`, `active eq 2`) — `vdctl`
    /// rejects those so an agent never issues a wait that can only ever time out. The
    /// rule is operator-aware (not a value-range check), evaluated against both
    /// possible actuals.
    #[must_use]
    pub fn is_satisfiable_for_boolean(self) -> bool {
        self.op.matches(0, self.value) || self.op.matches(1, self.value)
    }
}

fn field_value(state: &DevState, field: WaitField) -> u64 {
    match field {
        WaitField::Active => u64::from(state.phase == DevPhase::Active),
        WaitField::EntityCount => state.entities.len() as u64,
        WaitField::OwnEntitySet => u64::from(state.own_entity.is_some()),
        WaitField::SnapshotsApplied => state.snapshots_applied,
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
    fn boolean_fields_are_distinguished_from_counts() {
        assert!(WaitField::Active.is_boolean());
        assert!(WaitField::OwnEntitySet.is_boolean());
        assert!(!WaitField::EntityCount.is_boolean());
        assert!(!WaitField::SnapshotsApplied.is_boolean());
    }

    #[test]
    fn boolean_satisfiability_rejects_constant_false_predicates() {
        // Satisfiable: `eq 0` holds at actual 0 (LHS true); `eq 1` holds only at
        // actual 1 (LHS false, RHS true); `ge 0` / `ne 2` hold for any 0/1.
        assert!(pred(WaitField::Active, WaitOp::Eq, 0).is_satisfiable_for_boolean());
        assert!(pred(WaitField::Active, WaitOp::Eq, 1).is_satisfiable_for_boolean());
        assert!(pred(WaitField::Active, WaitOp::Ge, 0).is_satisfiable_for_boolean());
        assert!(pred(WaitField::OwnEntitySet, WaitOp::Ne, 2).is_satisfiable_for_boolean());
        // Constant-false for a 0/1 field — the forms the value-only guard missed.
        assert!(!pred(WaitField::Active, WaitOp::Gt, 1).is_satisfiable_for_boolean());
        assert!(!pred(WaitField::Active, WaitOp::Lt, 0).is_satisfiable_for_boolean());
        assert!(!pred(WaitField::Active, WaitOp::Eq, 2).is_satisfiable_for_boolean());
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
