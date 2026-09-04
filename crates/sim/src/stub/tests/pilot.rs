//! THE PILOT'S BODY (HR5, 2026-09-04): a realm that flies on an occupant's stick parks that
//! occupant's body, and every other realm walks it. One key used to move two things — the pilot
//! walked out of a 40 m hull in a fifth of a second while the hull burned the other way — so the
//! stick is kept either way and the feet are the half that is withheld.

use super::*;
use crate::capability::profiles;
use crate::stub::drive::OwnBody;

/// The stored row a built hull is: what it weighs, how wide it is to the air, how hard it pushes.
fn a_built_hull() -> vd_core::built::BuiltBody {
    vd_core::built::BuiltBody {
        realm: config().realm,
        owner: AccountId(1000),
        blueprint: vd_core::built::BlueprintId(0),
        bound: Boundary::Shell { r: 20.0 },
        look: Boundary::Shell { r: 20.0 },
        facts: vd_core::built::BuiltFacts {
            mass_g: 50_000_000,
            cross_section_mm2: 12_000_000,
            drag_micro: 820_000,
            max_push_micro_mps2: 98_100_000,
            max_turn_micro_radps2: 800_000,
        },
        fence: Fence::GENESIS,
    }
}

#[test]
fn a_pilot_in_a_realm_that_flies_on_a_stick_keeps_the_stick_and_does_not_also_walk() {
    // A realm that pushes itself AND holds a built body flies on its occupant's stick. The same
    // input on a realm that does neither walks the body, which is the control this reads against.
    let mut flying = Rig::with_config_and_kind(
        config(),
        NodeKind::Shard(profiles::ship().expect("ship profile")),
    );
    flying.grant_realm();
    flying.world.insert_resource(OwnBody(Some(a_built_hull())));
    insert_owned_dot(&mut flying, SESSION, player(1), DVec3::ZERO);
    let parked = flying.world.resource::<Dots>().0[&SESSION].pose.pos;
    let _ = flying.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
    let dot = &flying.world.resource::<Dots>().0[&SESSION];
    assert_eq!(dot.last_applied_seq, Some(1), "the input was applied");
    assert!(
        dot.last_stick.is_some(),
        "the stick the hull flies on is kept"
    );
    assert_eq!(dot.pose.pos, parked, "and the body stays where it was");

    // The control: a realm that does not fly on a stick walks the very same input.
    let mut walker = Rig::new();
    walker.grant_realm();
    insert_owned_dot(&mut walker, SESSION, player(1), DVec3::ZERO);
    let start = walker.world.resource::<Dots>().0[&SESSION].pose.pos;
    let _ = walker.tick(vec![input_msg(1, Fence(1), [1.0, 0.0, 0.0], [0.0, 0.0])]);
    assert_ne!(
        walker.world.resource::<Dots>().0[&SESSION].pose.pos,
        start,
        "on foot the feet move"
    );
}
