//! THE TEMPORARY CONTROL SEAM — a stick becomes a push (D-MOVE-2, owner-approved 2026-08-31).
//!
//! Owns: the one conversion from a player's input into the six numbers the movement contract carries,
//! and the engine rating that scales it.
//!
//! Does NOT own: where anything is. Nothing here reads or writes a placement, and nothing here knows
//! what the push will do — the PARENT decides that (SL1: only a parent authors a placement).
//!
//! ★ WHY THIS EXISTS AT ALL, and what deletes it. There is no character to control, no signal bus and
//! no functional blocks, and we still want the whole movement path testable NOW. So the controls attach
//! to the SHIP REALM directly and make a force here, in the ship's own shard. The signal system and
//! functional blocks (P9) replace it: a thruster block will publish the force and this file goes. What
//! is interim is the SOURCE of the numbers, never their shape.
use vd_core::kinematics;
use vd_wire::intershard::{DRIVE_UNITS_PER_MPS2, TURN_UNITS_PER_RADPS2};

/// WHAT A HULL CAN DO — the interim engine rating (owner ruling M-C, 2026-08-27: *"a rated speed and a
/// rated acceleration are facts about what a ship IS"*).
///
/// **PER-SHIP DATA, NEVER A CONSTANT.** A rating belongs to one hull, exactly as its mass does. A
/// shared number here would make every ship in the world fly identically and would be the magic number
/// the project's own rule forbids.
///
/// ★ IT DOES NOT CROSS A BOUNDARY. The ship scales its own stick by its own rating and sends the
/// RESULT, so the parent never needs the rating and never receives it (SL6: data crosses only when the
/// receiver cannot do without it). What crosses is an acceleration, which is what the parent applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EngineRating {
    /// The hardest this hull can push itself, in metres per second, per second.
    pub max_push_mps2: f64,
    /// The hardest this hull can turn itself, in radians per second, per second.
    pub max_turn_radps2: f64,
}

/// A stick position and a rating become the six numbers of the movement contract.
///
/// ★ THE STICK IS A FRACTION OF WHAT THIS HULL CAN DO, never a speed. A player never states a speed —
/// a throttle makes a FORCE (owner ruling M-D, 2026-08-27) — so there is nothing here that could be
/// refused, clamped or governed. Full forward means "push me as hard as you can", and what speed
/// results is the engines' business.
///
/// **A DIAGONAL IS NOT FASTER.** Full forward and full right would otherwise push about 1.41 times
/// harder than full forward alone, so a hull's strongest direction would be a corner. The magnitude is
/// clamped to one and the DIRECTION is kept.
#[must_use]
pub fn drive_from_stick(
    push: glam::DVec3,
    turn_stick: [f32; 3],
    rating: &EngineRating,
) -> ([i64; 3], [i64; 3]) {
    let push = clamp_to_unit(push) * rating.max_push_mps2;
    let turn =
        clamp_to_unit(kinematics::local_axes_from_movement(turn_stick)) * rating.max_turn_radps2;
    (
        [
            on_grid(push.x, DRIVE_UNITS_PER_MPS2),
            on_grid(push.y, DRIVE_UNITS_PER_MPS2),
            on_grid(push.z, DRIVE_UNITS_PER_MPS2),
        ],
        [
            on_grid(turn.x, TURN_UNITS_PER_RADPS2),
            on_grid(turn.y, TURN_UNITS_PER_RADPS2),
            on_grid(turn.z, TURN_UNITS_PER_RADPS2),
        ],
    )
}

/// A stick vector, never longer than one, with its direction kept. Monomorphic: the branch is here
/// (HR5), and both arms are named by tests.
fn clamp_to_unit(v: glam::DVec3) -> glam::DVec3 {
    let len = v.length();
    if len > 1.0 { v / len } else { v }
}

/// One number onto the wire's integer grid, rounded half away from zero.
///
/// **A NON-FINITE INPUT BECOMES A STILL STICK, NOT A REFUSAL.** The input lane already refuses a
/// non-finite datagram before this runs, so reaching here means a rating went wrong — and a ship that
/// coasts is recoverable, while a ship that panics takes its crew with it. Saturating rather than
/// wrapping for the same reason: the strongest lawful push beats a push that flips direction.
fn on_grid(value: f64, units_per: f64) -> i64 {
    let scaled = value * units_per;
    if scaled.is_finite() {
        // `as` saturates at the type's bounds in Rust, and rounding first keeps the grid symmetric
        // about zero (so an equal push left and right encodes to equal magnitudes).
        scaled.round() as i64
    } else {
        0
    }
}

/// WHERE A DRIVEN CHILD IS AND HOW IT IS MOVING, in its PARENT'S frame — the state only a parent
/// holds, and the only party entitled to write it (SL1: a child never states its own placement).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrivenState {
    /// Position in the parent's frame, in metres.
    pub pos_m: glam::DVec3,
    /// Velocity in the parent's frame, in metres per second. **It lives HERE and never travels
    /// upward** — a velocity is half a placement, and only the parent writes those.
    pub vel_mps: glam::DVec3,
    /// Which way the child is facing, in the parent's frame. The parent authored it, which is why the
    /// parent can rotate the child's own-frame push into its own directions.
    pub orient: glam::DQuat,
    /// How fast the child is turning, in radians per second, in the parent's frame.
    pub spin_radps: glam::DVec3,
}

/// WHAT A REALM IS MADE OF, where its children fly — the ambient the parent adds to every child's own
/// push (D-MOVE-1 M2a: *"the realm holds its own medium; nothing about the medium crosses in either
/// direction"*).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ambient {
    /// The pull this realm applies, in its own frame, in metres per second, per second. A child's own
    /// mass CANCELS out of gravity, so this is an acceleration and the same for everything.
    pub pull_mps2: glam::DVec3,
    /// How thick the medium is, in kilograms per cubic metre. Zero in space, which makes the drag term
    /// below exactly zero without a special case.
    pub density_kgpm3: f64,
}

/// ★ ONE TICK OF A DRIVEN CHILD — the parent's whole job, and the step the movement contract exists
/// for (D-MOVE-2).
///
/// The order is the ruling's own worked example, in the ruling's own order:
/// 1. rotate the child's push into MY directions — the child said *"this hard, along my own nose"*,
///    and only I know where its nose points, because I authored its facing;
/// 2. add MY pull and MY medium's drag;
/// 3. advance it one tick and write down where it now is.
///
/// **THE PARENT NEVER CHOOSES A SPEED.** It sums what it was handed with its own ambient and
/// integrates. Whatever speed results is the child's engines' business — there is no cap here, no
/// clamp and no refusal (owner ruling M-D, 2026-08-27: the ceiling leaves the flight path completely).
///
/// **WHY THE CHILD'S MASS IS NEEDED, when gravity does not need it.** Mass cancels out of gravity, so
/// a heavy ship and a light drone fall identically and `pull_mps2` is shared. It does NOT cancel out of
/// drag, which is an outside push: identical hulls at identical speed slow at very different rates for
/// a ten-to-one mass difference. That is the whole reason mass crosses at all.
#[must_use]
pub fn advance_driven(
    state: &DrivenState,
    push_units: [i64; 3],
    turn_units: [i64; 3],
    facts: &BodyFacts,
    ambient: &Ambient,
    dt_s: f64,
) -> DrivenState {
    // 1. THE CHILD'S OWN-FRAME PUSH, ROTATED INTO MINE. The child cannot do this itself: it does not
    //    know where its nose points in my frame, and telling it would be telling it where it is.
    let own_push = glam::DVec3::new(
        off_grid(push_units[0], DRIVE_UNITS_PER_MPS2),
        off_grid(push_units[1], DRIVE_UNITS_PER_MPS2),
        off_grid(push_units[2], DRIVE_UNITS_PER_MPS2),
    );
    let pushed = state.orient * own_push;
    // 2. MY OWN AMBIENT: the pull everything here feels, plus the medium's drag on THIS hull.
    let accel =
        pushed + ambient.pull_mps2 + drag_accel(state.vel_mps, facts, ambient.density_kgpm3);
    // 3. ADVANCE. Velocity first, then position from the NEW velocity (semi-implicit): it is stable
    //    under a strong pull where the naive order quietly gains energy every tick.
    let vel = state.vel_mps + accel * dt_s;
    let pos = state.pos_m + vel * dt_s;
    // The turn, by the same route: the child states it about its own axes, so the parent rotates it.
    let own_turn = glam::DVec3::new(
        off_grid(turn_units[0], TURN_UNITS_PER_RADPS2),
        off_grid(turn_units[1], TURN_UNITS_PER_RADPS2),
        off_grid(turn_units[2], TURN_UNITS_PER_RADPS2),
    );
    let spin = state.spin_radps + (state.orient * own_turn) * dt_s;
    DrivenState {
        pos_m: pos,
        vel_mps: vel,
        orient: turned(state.orient, spin, dt_s),
        spin_radps: spin,
    }
}

/// WHAT A CHILD IS, as the parent holds it — the arriving statement, kept.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyFacts {
    /// Mass in kilograms.
    pub mass_kg: f64,
    /// The area the medium pushes against, in square metres.
    pub cross_section_m2: f64,
    /// The drag coefficient — a plain ratio.
    pub drag_coefficient: f64,
}

/// The medium's push on this hull, as an acceleration opposing its travel.
///
/// `a = ½·ρ·v²·Cd·A / m`, pointed against the velocity. In space the density is zero and this is
/// exactly zero without a special case — which is why the vacuum needs no arm of its own.
///
/// A hull with no mass would divide by zero, so a non-finite result becomes no drag: a body that
/// coasts is recoverable, and a `NaN` position is not.
fn drag_accel(vel: glam::DVec3, facts: &BodyFacts, density: f64) -> glam::DVec3 {
    let speed = vel.length();
    let magnitude = 0.5 * density * speed * speed * facts.drag_coefficient * facts.cross_section_m2
        / facts.mass_kg;
    let a = -vel.normalize_or_zero() * magnitude;
    if a.is_finite() { a } else { glam::DVec3::ZERO }
}

/// The facing after one tick of spin. A zero spin returns the facing unchanged rather than building a
/// rotation about an undefined axis.
fn turned(orient: glam::DQuat, spin: glam::DVec3, dt_s: f64) -> glam::DQuat {
    let angle = spin.length() * dt_s;
    if angle > 0.0 {
        (glam::DQuat::from_axis_angle(spin.normalize(), angle) * orient).normalize()
    } else {
        orient
    }
}

/// A whole grid number read back as metres per second, per second — the inverse of [`on_grid`].
fn off_grid(units: i64, units_per: f64) -> f64 {
    units as f64 / units_per
}

/// ★ HOW LONG A STATED PUSH STAYS THIS PILOT'S INTENT, in ticks (D-MOVE-2).
///
/// **NOT A PACKET-LOSS ALLOWANCE — A CLOCK-SKEW ONE.** Two shards do not tick in lockstep: a child
/// stamps its own tick and its parent applies on its own, so demanding an exact match would discard
/// nearly every lawful push. Five ticks is a tenth of a second at fifty ticks a second: long enough
/// that no honest push is ever thrown away, short enough that a ship whose shard died stops thrusting
/// before anybody could see it drift.
pub const DRIVE_STALE_AFTER_TICKS: u64 = 5;

/// ★ WHAT THIS REALM IS — read from its own file at boot (D-MOVE-2; owner rulings 2026-09-01).
///
/// `None` for every realm the seed made: a planet's mass is the generator's business, and a planet
/// states nothing to its parent because it moves on rails the parent already computes. This is what a
/// BUILT realm holds — a hull somebody made, whose facts no seed can produce.
///
/// **IT IS WHERE THE SHARED ENGINE CONSTANT DIES.** Until now one number in the source stated how hard
/// every ship in the world pushes, which is the magic number this project's own rule forbids: two hulls
/// could never differ. The rating is per-hull data on the row, and this is the realm reading its own.
#[derive(Debug, Default, bevy_ecs::prelude::Resource)]
pub struct OwnBody(pub Option<vd_core::built::BuiltBody>);

/// EVERY DRIVEN CHILD THIS REALM HOLDS — the parent's own book, and the only place a driven child's
/// velocity and facing exist (SL1: only a parent writes those).
///
/// **UNBOUNDED, like every other child set (SL9).** A realm may hold six ships or six hundred; nothing
/// here is a fixed width and nothing walks the whole set to find one.
#[derive(Debug, Default, bevy_ecs::prelude::Resource)]
pub struct DrivenChildren(pub std::collections::BTreeMap<vd_core::pose::RealmId, DrivenChild>);

/// One driven child, as its parent holds it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DrivenChild {
    /// Where it is and how it is moving — written only here.
    pub state: DrivenState,
    /// The freshest push and turn it stated, and the tick it stated them for.
    pub drive: ([i64; 3], [i64; 3]),
    /// The fence and instant of the freshest drive held, for latest-wins ordering.
    pub drive_at: (vd_core::fence::Fence, vd_core::ids::UniverseTick),
    /// What it declared it IS. `None` until it says so — a child that has not stated its facts is not
    /// integrated, because guessing a mass is worse than not moving it.
    pub facts: Option<BodyFacts>,
    /// The fence and instant of the freshest facts held.
    pub facts_at: (vd_core::fence::Fence, vd_core::ids::UniverseTick),
    /// ★ FROZEN FOR A HAND-OVER (the ruler switch, slice 2): the exterior was flushed to a new parent
    /// and the drive is no longer applied — the child coasts at the flushed velocity, exactly as the
    /// destination re-advances it, until the demote removes it or an abort thaws it.
    pub frozen: bool,
}

/// ★ A CHILD'S DRIVE ARRIVES — admitted, or refused and counted (D-MOVE-2).
///
/// The guards mirror every other up-lane's, in the same order, for the same reasons:
/// 1. **MISROUTE** — the sender must be one of MY OWN direct children. A realm integrates what it
///    holds and nothing else.
/// 2. **UNATTESTED** — the byte must come from the node the directory places at that child. A deposed
///    incarnation's push is not this child's push.
/// 3. **CAPABILITY** — I must do physics for what is inside me. Checked here at runtime rather than by
///    installing different systems, because the same systems are installed everywhere and the profile
///    decides what they DO (HR3).
/// 4. **STALE** — latest-wins by (fence, instant). An out-of-order datagram is old news; on this lane
///    the next tick already carries the whole intent again.
///
/// ★ **THERE IS NO VALUE GUARD, AND THAT IS THE INTEGER GRID PAYING FOR ITSELF.** The interest lane
/// beside this one must refuse a distance that is not a distance, because a float can arrive as `NaN`.
/// A whole number cannot: every bit pattern that decodes is a lawful push. The guard that lane needs
/// does not exist here, because there is nothing it could catch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_child_drive(
    cd: vd_wire::intershard::ChildDrive,
    from: vd_core::ids::NodeId,
    own_realm: vd_core::pose::RealmId,
    integrates_children: bool,
    child_nodes: &std::collections::BTreeMap<vd_core::pose::RealmId, vd_core::ids::NodeId>,
    held: &mut DrivenChildren,
    stats: &mut super::StubStats,
) {
    let child = cd.child.lowered();
    if cd.child.parent().map(|p| p.lowered()) != Some(own_realm) {
        stats.child_drive_misrouted += 1;
        return;
    }
    if child_nodes.get(&child) != Some(&from) {
        stats.child_drive_unattested += 1;
        return;
    }
    if !integrates_children {
        stats.child_drive_uncapable += 1;
        return;
    }
    let entry = held.0.entry(child).or_insert_with(|| DrivenChild {
        state: DrivenState {
            pos_m: glam::DVec3::ZERO,
            vel_mps: glam::DVec3::ZERO,
            orient: glam::DQuat::IDENTITY,
            spin_radps: glam::DVec3::ZERO,
        },
        drive: ([0; 3], [0; 3]),
        drive_at: (
            vd_core::fence::Fence::GENESIS,
            vd_core::ids::UniverseTick(0),
        ),
        facts: None,
        facts_at: (
            vd_core::fence::Fence::GENESIS,
            vd_core::ids::UniverseTick(0),
        ),
        frozen: false,
    });
    if (cd.child_fence, cd.at) < entry.drive_at {
        stats.child_drive_stale += 1;
        return;
    }
    stats.child_drive_received += 1;
    entry.drive = (cd.push, cd.turn);
    entry.drive_at = (cd.child_fence, cd.at);
}

/// ★ A CHILD'S FACTS ARRIVE — the same four guards, plus the one this lane needs and the drive lane
/// does not (D-MOVE-2).
///
/// **THE LAWFUL-BODY GUARD.** A mass of zero divides by zero in the drag term, and a negative area is
/// not an area. Refused WHOLE rather than clamped to something plausible: a hull quietly given a
/// made-up mass flies wrong forever and nobody ever finds out, while a refusal is counted and visible.
#[allow(clippy::too_many_arguments)]
pub(crate) fn on_child_facts(
    cf: vd_wire::intershard::ChildFacts,
    from: vd_core::ids::NodeId,
    own_realm: vd_core::pose::RealmId,
    integrates_children: bool,
    child_nodes: &std::collections::BTreeMap<vd_core::pose::RealmId, vd_core::ids::NodeId>,
    held: &mut DrivenChildren,
    stats: &mut super::StubStats,
) {
    let child = cf.child.lowered();
    if cf.child.parent().map(|p| p.lowered()) != Some(own_realm) {
        stats.child_facts_misrouted += 1;
        return;
    }
    if child_nodes.get(&child) != Some(&from) {
        stats.child_facts_unattested += 1;
        return;
    }
    if !integrates_children {
        stats.child_drive_uncapable += 1;
        return;
    }
    if (cf.mass_g == 0) | (cf.cross_section_mm2 == 0) {
        stats.child_facts_unlawful += 1;
        return;
    }
    let entry = held.0.entry(child).or_insert_with(|| DrivenChild {
        state: DrivenState {
            pos_m: glam::DVec3::ZERO,
            vel_mps: glam::DVec3::ZERO,
            orient: glam::DQuat::IDENTITY,
            spin_radps: glam::DVec3::ZERO,
        },
        drive: ([0; 3], [0; 3]),
        drive_at: (
            vd_core::fence::Fence::GENESIS,
            vd_core::ids::UniverseTick(0),
        ),
        facts: None,
        facts_at: (
            vd_core::fence::Fence::GENESIS,
            vd_core::ids::UniverseTick(0),
        ),
        frozen: false,
    });
    if (cf.child_fence, cf.at) < entry.facts_at {
        stats.child_facts_stale += 1;
        return;
    }
    stats.child_facts_received += 1;
    entry.facts = Some(BodyFacts {
        mass_kg: cf.mass_g as f64 / 1000.0,
        cross_section_m2: cf.cross_section_mm2 as f64 / 1.0e6,
        drag_coefficient: f64::from(cf.drag_micro) / 1.0e6,
    });
    entry.facts_at = (cf.child_fence, cf.at);
}

impl DrivenChildren {
    /// ★ ONE TICK FOR EVERY DRIVEN CHILD THIS REALM HOLDS — the parent's physics pass.
    ///
    /// **A CHILD THAT HAS NOT STATED ITS FACTS IS NOT MOVED.** Its mass decides its drag and, later,
    /// its impacts. Guessing one would fly the hull wrong forever and silently; leaving it still is
    /// visible the moment anybody looks. This is the same discipline as the ban on decoding a durable
    /// blob to a default.
    ///
    /// **NO WALK OF ANYTHING BUT THE DRIVEN SET (SL9).** A realm may hold a hundred and fifty thousand
    /// children; this touches only the ones that push themselves, which is the set that actually needs
    /// integrating.
    pub(crate) fn advance_all(
        &mut self,
        now: vd_core::ids::UniverseTick,
        stale_after_ticks: u64,
        ambient: &Ambient,
        dt_s: f64,
    ) {
        for child in self.0.values_mut() {
            let Some(facts) = child.facts else {
                continue;
            };
            // A frozen child coasts: its drive is withheld, never its motion (the ruler switch, slice 2).
            let (push, turn) = if child.frozen {
                ([0; 3], [0; 3])
            } else {
                fresh_drive(child, now, stale_after_ticks)
            };
            child.state = advance_driven(&child.state, push, turn, &facts, ambient, dt_s);
        }
    }

    /// ★ DOES THIS CHILD MOVE? The question the window lane's two-lane split asks, answered from the
    /// driven book (D-MOVE-2).
    ///
    /// A driven child moves the moment it is HELD, not only when its push is non-zero: a ship coasting
    /// at speed still changes place every tick, and a ship that stops pushing keeps its velocity. A
    /// zero-push test here would file a coasting hull as static and flood the reliable lane exactly as
    /// the missing test did.
    #[must_use]
    pub fn moves(&self, child: vd_core::pose::RealmId) -> bool {
        self.0.contains_key(&child)
    }

    /// Where this realm has authored a driven child to be, if it holds one. The placement path reads
    /// this through ONE lookup, exactly as it reads the opaque motion book beside it — it never asks
    /// what KIND of thing the child is, which SL4 forbids by name.
    #[must_use]
    pub fn state_of(&self, child: vd_core::pose::RealmId) -> Option<DrivenState> {
        self.0.get(&child).map(|c| c.state)
    }
}

/// ★ SILENCE MEANS NO FORCE (owner ruling 2026-08-31): *"when I exited the pilot chair I stop sending
/// the requests, so my forces will not come to the parent any more, so the parent stops applying them
/// — so acceleration stops, but speed is kept."*
///
/// A held push that never expired was WRONG, and dangerously so: a ship whose shard died, or whose
/// pilot simply let go, would have accelerated for ever. The per-tick lane states what a child is doing
/// THIS TICK, so a child that says nothing is doing nothing. Speed is untouched — nothing in space
/// removes it — and the hull coasts at whatever it reached.
///
/// **THE WINDOW CANNOT BE ZERO, and the reason is not packet loss.** Two shards do not tick in
/// lockstep: the child stamps its own tick and the parent applies on its own, so demanding an exact
/// match would discard nearly every lawful push. The window is a few ticks wide, derived from the tick
/// rate rather than chosen — at fifty ticks a second a lost message costs twenty milliseconds of
/// thrust, which nobody can see, while a dead shard stops thrusting within the window instead of never.
fn fresh_drive(
    child: &DrivenChild,
    now: vd_core::ids::UniverseTick,
    stale_after_ticks: u64,
) -> ([i64; 3], [i64; 3]) {
    let stated_at = child.drive_at.1.0;
    // Saturating: a drive stamped in the FUTURE (a child running ahead of its parent) is fresh, never
    // a huge negative age that would wrap into "ancient".
    let age = now.0.saturating_sub(stated_at);
    if age <= stale_after_ticks {
        child.drive
    } else {
        ([0; 3], [0; 3])
    }
}

/// ★ THE PUSH LEAVES THIS REALM — the producer the two lanes were built for (D-MOVE-2).
///
/// Runs on every shard, every tick, and does nothing on almost all of them: a realm that cannot push
/// itself has no stick to read and never speaks. That is the capability deciding what an installed
/// system DOES, rather than which systems exist (HR3).
///
/// **WHAT IT SENDS AND WHAT IT CANNOT.** Six whole numbers in this realm's OWN frame, its own name and
/// fence, and the tick they belong to. There is no field for a speed or a position, so this cannot
/// state half a placement however wrong the rest of it went (SL1 clause 3).
///
/// **IT SENDS EVERY TICK WHILE A STICK IS HELD, and stops the moment one is not.** The lane says what
/// this realm is doing THIS TICK, so silence is the honest way to say "nothing" — a pilot who leaves
/// the chair stops the acceleration and keeps the speed (owner, 2026-08-31).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_child_drive(
    self_driven: bool,
    own_coord: &vd_core::realm_coord::RealmCoord,
    realm_fence: Option<vd_core::fence::Fence>,
    at: vd_core::ids::UniverseTick,
    parent_node: Option<vd_core::ids::NodeId>,
    rating: &EngineRating,
    stick: Option<(glam::DVec3, [f32; 3])>,
    outbox: &mut crate::runtime::OutboundBox,
    stats: &mut super::StubStats,
) {
    // A realm with no engines, no parent to speak to, or nobody at the controls says nothing at all.
    // Three separate silences, and none of them is an error worth counting: most realms are all three.
    if !self_driven {
        return;
    }
    // Four silences, and none of them is an error worth counting: most realms are all four.
    //   - no engines, so nothing to state;
    //   - no parent resolved yet, so nobody to state it to;
    //   - nobody at the controls;
    //   - no realm fence yet, which means the directory has not granted this realm — a shard that
    //     cannot prove which incarnation it is must not speak for the realm, or a deposed one would.
    let (Some(parent), Some((push, turn_stick)), Some(fence)) = (parent_node, stick, realm_fence)
    else {
        return;
    };
    let (push, turn) = drive_from_stick(push, turn_stick, rating);
    outbox.push_flow(
        parent,
        // The UNRELIABLE carrier, deliberately: the next tick restates the whole intent, so a lost
        // datagram is corrected before anybody could read the gap. Reliability here would put a
        // retransmit of stale news in front of fresh news.
        crate::io::MsgClass::SignalDelta,
        &vd_wire::intershard::InterShardFlow::ChildDrive(vd_wire::intershard::ChildDrive {
            child: own_coord.clone(),
            child_fence: fence,
            at,
            push,
            turn,
        }),
    );
    stats.child_drive_sent += 1;
}

/// ★ THE PER-TICK SYSTEM that speaks for a self-driven realm (D-MOVE-2).
///
/// Installed on every shard and silent on almost all of them: a realm that cannot push itself has no
/// engines to state, so it returns before touching anything. That is the profile deciding what an
/// installed system DOES rather than which systems exist (HR3).
///
/// ⚠ **THE ENGINE RATING IS AN INTERIM CONSTANT, and it is the one number here that is not yet real.**
/// The owner's ruling lets a ship state its rated acceleration as a fact about itself, and a real hull
/// will derive it from the thrusters actually built into it (P6/P9). Until blocks exist there is
/// nothing to derive it FROM, so this is a stated rating for one fixture ship — recorded in the ledger
/// as interim, never a shipped default for every hull in the world.
#[allow(clippy::too_many_arguments)] // a Bevy system: all params are injected resources/queries
pub(crate) fn emit_own_drive(
    config: bevy_ecs::prelude::Res<super::StubConfig>,
    identity: bevy_ecs::prelude::Res<crate::runtime::NodeIdentity>,
    clock: bevy_ecs::prelude::Res<crate::runtime::ClockSample>,
    authority: bevy_ecs::prelude::Res<super::RealmAuthority>,
    parent_node: bevy_ecs::prelude::Res<super::ParentRealmNode>,
    dots: bevy_ecs::prelude::Res<super::Dots>,
    own_body: bevy_ecs::prelude::Res<OwnBody>,
    mut stated: bevy_ecs::prelude::ResMut<StatedFacts>,
    mut stats: bevy_ecs::prelude::ResMut<super::StubStats>,
    mut outbox: bevy_ecs::prelude::ResMut<crate::runtime::OutboundBox>,
) {
    let self_driven = match identity.kind {
        crate::capability::NodeKind::Shard(profile) => profile.self_driven(),
        _ => false,
    };
    // THE PILOT AT THE CONTROLS. Today: any pilot this realm holds, which is the interim seam the
    // ruling describes. Later a SEAT names one, and only this line changes.
    let stick = dots.0.values().find_map(|d| d.last_stick);
    // ★ THIS HULL'S OWN RATING, from its own row. A realm that has no body states no drive: a hull
    // whose facts nobody wrote is a hull nobody knows the mass of, and guessing one flies it wrong for
    // ever with nothing to say so. Silence is the honest answer, and it is the same silence a realm
    // with no engines gives.
    let Some(body) = own_body.0.as_ref() else {
        return;
    };
    let rating = rating_of(&body.facts);
    emit_child_drive(
        self_driven,
        &config.own_coord,
        authority.0,
        clock.universe_tick,
        parent_node.0,
        &rating,
        stick,
        &mut outbox,
        &mut stats,
    );
    // ★ WHAT I AM, on the reliable lane, only when it CHANGES. A parent needs this for the forces IT
    // applies — drag today, impacts later — and a change stated once and then lost would leave a parent
    // computing drag from a mass that is wrong for ever.
    emit_child_facts(
        self_driven,
        &config.own_coord,
        authority.0,
        clock.universe_tick,
        parent_node.0,
        body,
        &mut stated,
        &mut outbox,
        &mut stats,
    );
}

/// A stored hull's whole-number facts, read back as the rates the stick is scaled by.
///
/// The row stores whole units for the same reason the wire does — two processes must read one number
/// the same way — and this is the one place they become the rates a push is computed from.
fn rating_of(facts: &vd_core::built::BuiltFacts) -> EngineRating {
    EngineRating {
        max_push_mps2: off_grid(facts.max_push_micro_mps2, DRIVE_UNITS_PER_MPS2),
        max_turn_radps2: off_grid(facts.max_turn_micro_radps2, TURN_UNITS_PER_RADPS2),
    }
}

/// ★ WHAT THIS REALM LAST TOLD ITS PARENT IT IS (D-MOVE-2).
///
/// The facts lane is ON CHANGE ONLY, so something must remember what was said. Without this a realm
/// either repeats itself every tick — on the RELIABLE lane, which is the flood this design exists to
/// avoid — or says it once and never again, which loses it to a single dropped connection.
#[derive(Debug, Default, bevy_ecs::prelude::Resource)]
pub struct StatedFacts(pub(crate) Option<vd_core::built::BuiltFacts>);

/// ★ WHAT I AM — stated when it changes, and once per connection (D-MOVE-2).
///
/// **ONCE PER CONNECTION IS NOT BELT AND BRACES.** The carrier is at-least-once across a blip, but the
/// receiver's memory of what it has seen is RAM-only: a parent that restarts has forgotten everything.
/// So a child restates on a new parent route — redundancy applied exactly where it is needed, and
/// nowhere else.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_child_facts(
    self_driven: bool,
    own_coord: &vd_core::realm_coord::RealmCoord,
    realm_fence: Option<vd_core::fence::Fence>,
    at: vd_core::ids::UniverseTick,
    parent_node: Option<vd_core::ids::NodeId>,
    body: &vd_core::built::BuiltBody,
    stated: &mut StatedFacts,
    outbox: &mut crate::runtime::OutboundBox,
    stats: &mut super::StubStats,
) {
    if !self_driven {
        return;
    }
    let (Some(parent), Some(fence)) = (parent_node, realm_fence) else {
        return;
    };
    if stated.0 == Some(body.facts) {
        return; // unchanged — a declared property says nothing twice
    }
    // ★ RETAINED, AND THE GUARD CAUGHT ME FORGETTING IT. This arm is producer-less-reliable: it is
    // stated ONCE on a change and no timer re-drives it, so the carrier must keep it across a source
    // crash or the one-shot is silently lost — and a parent would compute drag from a mass that is
    // wrong for ever. The obligation was written into the wire when the arm was classified, and the
    // push site's own debug guard fired the first time this producer existed.
    outbox.push_flow_durable(
        parent,
        // The RELIABLE carrier: a change stated once and then lost would leave a parent computing drag
        // from a mass that is wrong for ever, with nothing to correct it.
        crate::io::MsgClass::Saga,
        &vd_wire::intershard::InterShardFlow::ChildFacts(vd_wire::intershard::ChildFacts {
            child: own_coord.clone(),
            child_fence: fence,
            at,
            mass_g: body.facts.mass_g,
            cross_section_mm2: body.facts.cross_section_mm2,
            drag_micro: body.facts.drag_micro,
            declared: vd_wire::intershard::DeclaredStates::default(),
        }),
        crate::io::Durability::Retained,
    );
    stated.0 = Some(body.facts);
    stats.child_facts_sent += 1;
}

#[cfg(test)]
mod tests {
    use super::{
        Ambient, BodyFacts, DrivenState, EngineRating, advance_driven, clamp_to_unit,
        drive_from_stick, on_grid,
    };
    use glam::{DQuat, DVec3};
    use vd_wire::intershard::DRIVE_UNITS_PER_MPS2;

    fn at_rest() -> DrivenState {
        DrivenState {
            pos_m: DVec3::ZERO,
            vel_mps: DVec3::ZERO,
            orient: DQuat::IDENTITY,
            spin_radps: DVec3::ZERO,
        }
    }
    fn hull() -> BodyFacts {
        BodyFacts {
            mass_kg: 50_000.0,
            cross_section_m2: 12.0,
            drag_coefficient: 0.82,
        }
    }
    fn vacuum() -> Ambient {
        Ambient {
            pull_mps2: DVec3::ZERO,
            density_kgpm3: 0.0,
        }
    }

    fn rating() -> EngineRating {
        EngineRating {
            max_push_mps2: 4.0,
            max_turn_radps2: 0.5,
        }
    }

    /// A push straight along the nose (facing identity), as the stick states it.
    fn fwd_push(fwd: f32) -> glam::DVec3 {
        vd_core::kinematics::local_axes_from_movement([fwd, 0.0, 0.0])
    }

    #[test]
    fn full_forward_pushes_the_whole_rating_along_the_nose() {
        let (push, turn) = drive_from_stick(fwd_push(1.0), [0.0; 3], &rating());
        // The shared axis map sends "forward" to −Z, so a full forward stick is the rating on −Z and
        // nothing anywhere else. This pins the CONVENTION, not just the arithmetic: were the map to
        // drift, a ship would fly sideways and every test above this would still pass.
        assert_eq!(push, [0, 0, -4_000_000]);
        assert_eq!(turn, [0, 0, 0]);
    }

    #[test]
    fn a_still_stick_pushes_nothing() {
        let (push, turn) = drive_from_stick(glam::DVec3::ZERO, [0.0; 3], &rating());
        assert_eq!(push, [0, 0, 0]);
        assert_eq!(turn, [0, 0, 0]);
    }

    #[test]
    fn reverse_is_the_same_push_pointing_the_other_way() {
        let (fwd, _) = drive_from_stick(fwd_push(1.0), [0.0; 3], &rating());
        let (back, _) = drive_from_stick(fwd_push(-1.0), [0.0; 3], &rating());
        // ★ THE LANE NAMES NO MANOEUVRE. Reverse is not a mode, a flag or a second arm — it is the
        // same three numbers with their sign flipped, which is the whole reason the contract can carry
        // every future way to fly without growing.
        assert_eq!(back, [-fwd[0], -fwd[1], -fwd[2]]);
    }

    #[test]
    fn a_diagonal_stick_is_not_stronger_than_a_straight_one() {
        let (straight, _) = drive_from_stick(fwd_push(1.0), [0.0; 3], &rating());
        let (corner, _) = drive_from_stick(
            vd_core::kinematics::local_axes_from_movement([1.0, 1.0, 1.0]),
            [0.0; 3],
            &rating(),
        );
        let mag = |v: [i64; 3]| ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64).sqrt();
        // Without the clamp a corner would push about 1.73 times harder than straight ahead, so a
        // hull's strongest direction would be a diagonal — and every pilot would fly at an angle.
        assert!(
            (mag(corner) - mag(straight)).abs() < 2.0,
            "corner {corner:?} vs straight {straight:?}"
        );
    }

    #[test]
    fn the_turn_stick_drives_the_turn_and_leaves_the_push_alone() {
        let (push, turn) = drive_from_stick(glam::DVec3::ZERO, [1.0, 0.0, 0.0], &rating());
        assert_eq!(push, [0, 0, 0]);
        assert_eq!(turn, [0, 0, -500_000]);
    }

    #[test]
    fn a_short_stick_is_kept_as_it_is() {
        // The false arm of the clamp: a half-pushed stick must stay half, or a light touch would fly
        // like a full one.
        let v = clamp_to_unit(glam::DVec3::new(0.5, 0.0, 0.0));
        assert_eq!(v, glam::DVec3::new(0.5, 0.0, 0.0));
    }

    #[test]
    fn a_long_stick_is_shortened_but_keeps_its_direction() {
        let v = clamp_to_unit(glam::DVec3::new(3.0, 4.0, 0.0));
        assert!((v.length() - 1.0).abs() < 1e-12);
        assert!((v.x - 0.6).abs() < 1e-12);
        assert!((v.y - 0.8).abs() < 1e-12);
    }

    #[test]
    fn a_number_lands_on_the_grid_rounded_away_from_zero() {
        assert_eq!(on_grid(1.0, DRIVE_UNITS_PER_MPS2), 1_000_000);
        assert_eq!(on_grid(-1.0, DRIVE_UNITS_PER_MPS2), -1_000_000);
        // Symmetric about zero: an equal push each way encodes to equal magnitudes.
        assert_eq!(on_grid(0.0000005, DRIVE_UNITS_PER_MPS2), 1);
        assert_eq!(on_grid(-0.0000005, DRIVE_UNITS_PER_MPS2), -1);
    }

    #[test]
    fn a_broken_rating_coasts_rather_than_panicking() {
        // The false arm of the finite test. A ship that coasts is recoverable; a ship that panics
        // takes its crew with it.
        assert_eq!(on_grid(f64::NAN, DRIVE_UNITS_PER_MPS2), 0);
        assert_eq!(on_grid(f64::INFINITY, DRIVE_UNITS_PER_MPS2), 0);
        let broken = EngineRating {
            max_push_mps2: f64::NAN,
            max_turn_radps2: 0.5,
        };
        let (push, _) = drive_from_stick(fwd_push(1.0), [0.0; 3], &broken);
        assert_eq!(push, [0, 0, 0]);
    }

    #[test]
    fn an_absurd_rating_saturates_rather_than_flipping_direction() {
        let huge = EngineRating {
            max_push_mps2: 1.0e30,
            max_turn_radps2: 0.5,
        };
        let (push, _) = drive_from_stick(fwd_push(1.0), [0.0; 3], &huge);
        // Saturating, never wrapping: the strongest lawful push beats a push that reverses.
        assert_eq!(push[2], i64::MIN);
    }

    #[test]
    fn a_push_becomes_speed_and_speed_becomes_distance() {
        // Four metres per second, per second, for one tick of a fiftieth of a second.
        let after = advance_driven(
            &at_rest(),
            [4_000_000, 0, 0],
            [0; 3],
            &hull(),
            &vacuum(),
            0.02,
        );
        assert!(
            (after.vel_mps.x - 0.08).abs() < 1e-12,
            "vel {:?}",
            after.vel_mps
        );
        // Position comes from the NEW velocity (semi-implicit), which is stable under a strong pull.
        assert!(
            (after.pos_m.x - 0.0016).abs() < 1e-12,
            "pos {:?}",
            after.pos_m
        );
    }

    #[test]
    fn the_parent_rotates_the_childs_own_frame_push_into_its_own() {
        // The child says "push me along MY x". It is facing a quarter turn about Z, so in the
        // PARENT's frame that is +y. The child never knows this — it cannot, because knowing where its
        // nose points in the parent IS knowing where it is (SL1).
        let facing = DrivenState {
            orient: DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2),
            ..at_rest()
        };
        let after = advance_driven(&facing, [1_000_000, 0, 0], [0; 3], &hull(), &vacuum(), 1.0);
        assert!(
            after.vel_mps.x.abs() < 1e-9,
            "x should be ~0: {:?}",
            after.vel_mps
        );
        assert!(
            (after.vel_mps.y - 1.0).abs() < 1e-9,
            "y should be 1: {:?}",
            after.vel_mps
        );
    }

    #[test]
    fn a_realms_pull_applies_with_no_push_at_all() {
        // Gravity needs no engine and no mass: an empty ship falls exactly like a full one.
        let pulled = Ambient {
            pull_mps2: DVec3::new(0.0, -9.81, 0.0),
            density_kgpm3: 0.0,
        };
        let after = advance_driven(&at_rest(), [0; 3], [0; 3], &hull(), &pulled, 1.0);
        assert!((after.vel_mps.y + 9.81).abs() < 1e-12);
    }

    #[test]
    fn a_heavy_hull_and_a_light_one_fall_identically_but_drag_differently() {
        // ★ THE WHOLE REASON MASS CROSSES. It cancels out of gravity and does NOT cancel out of drag.
        let pulled = Ambient {
            pull_mps2: DVec3::new(0.0, -9.81, 0.0),
            density_kgpm3: 0.0,
        };
        let light = BodyFacts {
            mass_kg: 5_000.0,
            ..hull()
        };
        let a = advance_driven(&at_rest(), [0; 3], [0; 3], &hull(), &pulled, 1.0);
        let b = advance_driven(&at_rest(), [0; 3], [0; 3], &light, &pulled, 1.0);
        assert_eq!(a.vel_mps, b.vel_mps, "gravity must not care about mass");

        let air = Ambient {
            pull_mps2: DVec3::ZERO,
            density_kgpm3: 1.225,
        };
        let moving = DrivenState {
            vel_mps: DVec3::new(100.0, 0.0, 0.0),
            ..at_rest()
        };
        let heavy_slow = advance_driven(&moving, [0; 3], [0; 3], &hull(), &air, 1.0);
        let light_slow = advance_driven(&moving, [0; 3], [0; 3], &light, &air, 1.0);
        assert!(
            light_slow.vel_mps.x < heavy_slow.vel_mps.x,
            "the lighter hull must slow MORE: light {} vs heavy {}",
            light_slow.vel_mps.x,
            heavy_slow.vel_mps.x
        );
    }

    #[test]
    fn space_has_no_drag_and_needs_no_special_case() {
        let moving = DrivenState {
            vel_mps: DVec3::new(100.0, 0.0, 0.0),
            ..at_rest()
        };
        let after = advance_driven(&moving, [0; 3], [0; 3], &hull(), &vacuum(), 1.0);
        assert_eq!(
            after.vel_mps, moving.vel_mps,
            "a coasting ship in vacuum keeps its speed exactly"
        );
    }

    #[test]
    fn drag_opposes_travel_and_never_reverses_it() {
        let air = Ambient {
            pull_mps2: DVec3::ZERO,
            density_kgpm3: 1.225,
        };
        let moving = DrivenState {
            vel_mps: DVec3::new(10.0, 0.0, 0.0),
            ..at_rest()
        };
        let after = advance_driven(&moving, [0; 3], [0; 3], &hull(), &air, 0.02);
        // HR5: two facts, two asserts. `a && b` short-circuits, so the false arm of the first
        // test is a region no input can reach.
        assert!(after.vel_mps.x < 10.0, "air slowed it: {:?}", after.vel_mps);
        assert!(
            after.vel_mps.x > 0.0,
            "and never pushed it backwards: {:?}",
            after.vel_mps
        );
    }

    #[test]
    fn a_massless_hull_coasts_rather_than_producing_a_broken_position() {
        // The false arm of the finite guard: dividing by a zero mass must not put a NaN into a
        // position, because a NaN position is unrecoverable and a coast is not.
        let air = Ambient {
            pull_mps2: DVec3::ZERO,
            density_kgpm3: 1.225,
        };
        let broken = BodyFacts {
            mass_kg: 0.0,
            ..hull()
        };
        let moving = DrivenState {
            vel_mps: DVec3::new(10.0, 0.0, 0.0),
            ..at_rest()
        };
        let after = advance_driven(&moving, [0; 3], [0; 3], &broken, &air, 0.02);
        // HR5: two facts, two asserts — `&&` short-circuits and hides the first test's false arm.
        assert!(
            after.pos_m.is_finite(),
            "the place stays a number: {:?}",
            after.pos_m
        );
        assert!(
            after.vel_mps.is_finite(),
            "and so does the speed: {:?}",
            after.vel_mps
        );
        assert_eq!(after.vel_mps, moving.vel_mps);
    }

    #[test]
    fn a_turn_spins_the_hull_and_a_still_stick_leaves_it_facing_the_same_way() {
        let after = advance_driven(&at_rest(), [0; 3], [500_000, 0, 0], &hull(), &vacuum(), 1.0);
        assert!((after.spin_radps.x - 0.5).abs() < 1e-12);
        assert!(
            after.orient.angle_between(DQuat::IDENTITY) > 0.4,
            "it must have turned"
        );
        // The false arm: no spin returns the facing untouched rather than rotating about an
        // undefined axis.
        let still = advance_driven(&at_rest(), [0; 3], [0; 3], &hull(), &vacuum(), 1.0);
        assert_eq!(still.orient, DQuat::IDENTITY);
    }

    #[test]
    fn nothing_here_caps_a_speed_however_fast_the_child_goes() {
        // ★ THE CEILING LEFT THE FLIGHT PATH (owner ruling M-D). A parent sums and integrates; it
        // never chooses a speed. Ten thousand ticks of full push must keep adding speed.
        let mut s = at_rest();
        for _ in 0..10_000 {
            s = advance_driven(&s, [4_000_000, 0, 0], [0; 3], &hull(), &vacuum(), 0.02);
        }
        assert!(
            (s.vel_mps.x - 800.0).abs() < 1e-6,
            "no clamp anywhere: {:?}",
            s.vel_mps
        );
    }

    use super::{DrivenChildren, on_child_drive, on_child_facts};
    use crate::stub::StubStats;
    use vd_core::fence::Fence;
    use vd_core::ids::{NodeId, UniverseTick};
    use vd_core::realm_coord::RealmCoord;
    use vd_core::realm_path::{RealmKindTag, RealmLevel, RealmPath};

    const SHIP_NODE: NodeId = NodeId(77);

    /// A driven child inside a star system — the child whose drives arrive.
    ///
    /// ⚠ **IT IS A STATION, NOT A SHIP, AND THAT IS A FINDING RATHER THAN A CHOICE (2026-08-31).**
    /// `RealmKindTag` holds seven kinds and NONE of them is a ship, so a ship realm cannot be named in
    /// a lineage at all today — even though `RealmId::Ship` and `FrameRef::ShipLocal` both exist. The
    /// tag list is frozen and APPEND-ONLY, so a `Ship` tag is a deliberate wire append, not something
    /// to slip in beside a test.
    ///
    /// A station is a real kind that a system really holds, and this lane is generic by construction,
    /// so it proves exactly what a ship would: a child realm states a push and its parent applies it.
    /// Using a station as a STAND-IN for a ship would be the lossy trick S9 had to undo; using a
    /// station because a station is a lawful driven child is not.
    fn ship_coord() -> RealmCoord {
        RealmCoord::from_path(RealmPath::from_levels(vec![
            RealmLevel::new(RealmKindTag::Universe, 0),
            RealmLevel::new(RealmKindTag::Galaxy, 2),
            RealmLevel::new(RealmKindTag::System, 7),
            RealmLevel::new(RealmKindTag::Station, 5),
        ]))
        .expect("4-level path has a leaf")
    }
    fn parent_realm() -> vd_core::pose::RealmId {
        ship_coord()
            .parent()
            .expect("a ship has a parent")
            .lowered()
    }
    fn nodes() -> std::collections::BTreeMap<vd_core::pose::RealmId, NodeId> {
        std::collections::BTreeMap::from([(ship_coord().lowered(), SHIP_NODE)])
    }
    fn a_drive(at: u64) -> vd_wire::intershard::ChildDrive {
        vd_wire::intershard::ChildDrive {
            child: ship_coord(),
            child_fence: Fence(3),
            at: UniverseTick(at),
            push: [4_000_000, 0, 0],
            turn: [0; 3],
        }
    }
    fn some_facts(at: u64) -> vd_wire::intershard::ChildFacts {
        vd_wire::intershard::ChildFacts {
            child: ship_coord(),
            child_fence: Fence(3),
            at: UniverseTick(at),
            mass_g: 50_000_000,
            cross_section_mm2: 12_000_000,
            drag_micro: 820_000,
            declared: vd_wire::intershard::DeclaredStates::default(),
        }
    }

    #[test]
    fn a_childs_drive_is_admitted_and_held() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_drive_received, 1);
        assert_eq!(held.0[&ship_coord().lowered()].drive.0, [4_000_000, 0, 0]);
    }

    #[test]
    fn a_drive_from_somebody_elses_child_is_refused() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        // A realm integrates what IT holds and nothing else — the misroute guard every up-lane carries.
        let stranger = vd_core::pose::RealmId::Planet(4242);
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            stranger,
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_drive_misrouted, 1);
        assert_eq!(stats.child_drive_received, 0);
        assert!(held.0.is_empty());
    }

    #[test]
    fn a_drive_from_the_wrong_node_is_refused() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        // A deposed incarnation's push is not this child's push.
        on_child_drive(
            a_drive(9),
            NodeId(999),
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_drive_unattested, 1);
        assert!(held.0.is_empty());
    }

    #[test]
    fn a_realm_that_does_no_physics_refuses_a_drive_and_counts_it() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        // The capability is checked HERE at runtime, not by installing different systems: the same
        // systems are installed everywhere and the profile decides what they DO (HR3).
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            parent_realm(),
            false,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_drive_uncapable, 1);
        assert!(held.0.is_empty());
    }

    #[test]
    fn an_out_of_order_drive_is_stale_news_and_never_a_correction() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_drive(
            a_drive(20),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        let mut older = a_drive(9);
        older.push = [1, 1, 1];
        on_child_drive(
            older,
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_drive_stale, 1);
        assert_eq!(
            held.0[&ship_coord().lowered()].drive.0,
            [4_000_000, 0, 0],
            "the fresher push stands"
        );
    }

    #[test]
    fn facts_are_admitted_and_converted_out_of_their_whole_units() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        let f = held.0[&ship_coord().lowered()].facts.expect("held");
        assert!((f.mass_kg - 50_000.0).abs() < 1e-9);
        assert!((f.cross_section_m2 - 12.0).abs() < 1e-9);
        assert!((f.drag_coefficient - 0.82).abs() < 1e-9);
    }

    #[test]
    fn a_body_that_cannot_exist_is_refused_whole_rather_than_repaired() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        // A zero mass divides by zero in the drag term. Refused rather than clamped: a hull quietly
        // given a made-up mass flies wrong forever and nobody finds out.
        let mut massless = some_facts(9);
        massless.mass_g = 0;
        on_child_facts(
            massless,
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_facts_unlawful, 1);
        let mut flat = some_facts(9);
        flat.cross_section_mm2 = 0;
        on_child_facts(
            flat,
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_facts_unlawful, 2);
        assert_eq!(stats.child_facts_received, 0);
    }

    #[test]
    fn facts_carry_the_same_four_guards_as_the_drive() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        let stranger = vd_core::pose::RealmId::Planet(4242);
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            stranger,
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_facts_misrouted, 1);
        on_child_facts(
            some_facts(9),
            NodeId(999),
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_facts_unattested, 1);
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            false,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_drive_uncapable, 1);
        on_child_facts(
            some_facts(20),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        assert_eq!(stats.child_facts_stale, 1);
    }

    #[test]
    fn a_child_that_never_stated_its_facts_is_not_moved_at_all() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        held.advance_all(UniverseTick(9), 5, &vacuum(), 1.0);
        // Its mass decides its drag and later its impacts. Guessing one flies it wrong forever and
        // silently; leaving it still is visible the moment anybody looks.
        assert_eq!(
            held.state_of(ship_coord().lowered()).expect("held").pos_m,
            DVec3::ZERO
        );
    }

    #[test]
    fn a_frozen_child_coasts_and_its_push_is_withheld_until_it_thaws() {
        // The ruler switch, slice 2: a child whose exterior was flushed to a new parent keeps the
        // velocity it had and gains none — both sides of the hand-over coast at the same speed.
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        let child = ship_coord().lowered();
        held.0.get_mut(&child).expect("held").state.vel_mps = DVec3::new(1.0, 0.0, 0.0);
        held.0.get_mut(&child).expect("held").frozen = true;
        held.advance_all(UniverseTick(9), 5, &vacuum(), 1.0);
        let s = held.state_of(child).expect("held");
        assert!(
            (s.vel_mps.x - 1.0).abs() < 1e-9,
            "coasting: {:?}",
            s.vel_mps
        );
        assert!(
            (s.pos_m.x - 1.0).abs() < 1e-9,
            "one metre of coast: {:?}",
            s.pos_m
        );
        // Thawed (an aborted hand-over): the same held drive applies again.
        held.0.get_mut(&child).expect("held").frozen = false;
        held.advance_all(UniverseTick(9), 5, &vacuum(), 1.0);
        let s = held.state_of(child).expect("held");
        assert!(
            (s.vel_mps.x - 5.0).abs() < 1e-9,
            "the push is back: {:?}",
            s.vel_mps
        );
    }

    #[test]
    fn a_child_with_facts_and_a_push_actually_moves() {
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        held.advance_all(UniverseTick(9), 5, &vacuum(), 1.0);
        let s = held.state_of(ship_coord().lowered()).expect("held");
        assert!(
            (s.vel_mps.x - 4.0).abs() < 1e-9,
            "four metres per second after one second: {:?}",
            s.vel_mps
        );
    }

    #[test]
    fn a_realm_holding_no_driven_child_advances_nothing() {
        let mut held = DrivenChildren::default();
        held.advance_all(UniverseTick(9), 5, &vacuum(), 1.0);
        assert!(held.state_of(ship_coord().lowered()).is_none());
    }

    #[test]
    fn leaving_the_chair_stops_the_acceleration_and_keeps_the_speed() {
        // ★ THE OWNER'S OWN WORDS (2026-08-31): "when I exited the pilot chair I stop sending the
        // requests, so my forces will not come to the parent any more, so the parent stops applying
        // them — so acceleration stops, but speed is kept."
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        // Two ticks with the pilot at the controls: the push is fresh, so it applies.
        held.advance_all(UniverseTick(9), 5, &vacuum(), 1.0);
        held.advance_all(UniverseTick(10), 5, &vacuum(), 1.0);
        let flying = held
            .state_of(ship_coord().lowered())
            .expect("held")
            .vel_mps
            .x;
        assert!(
            (flying - 8.0).abs() < 1e-9,
            "two seconds of four m/s²: {flying}"
        );

        // The pilot leaves. Nothing new arrives. Past the window the push decays to nothing.
        held.advance_all(UniverseTick(99), 5, &vacuum(), 1.0);
        held.advance_all(UniverseTick(100), 5, &vacuum(), 1.0);
        let coasting = held
            .state_of(ship_coord().lowered())
            .expect("held")
            .vel_mps
            .x;
        assert!(
            (coasting - flying).abs() < 1e-9,
            "speed must be KEPT, not lost: {coasting} vs {flying}"
        );
    }

    #[test]
    fn a_ship_whose_shard_died_stops_thrusting_rather_than_accelerating_for_ever() {
        // The safety half of the same rule. A held push that never expired would have run a dead
        // ship's engines until the heat death of the world.
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        for tick in 0..1_000 {
            held.advance_all(UniverseTick(9 + tick), 5, &vacuum(), 0.02);
        }
        let v = held
            .state_of(ship_coord().lowered())
            .expect("held")
            .vel_mps
            .x;
        // Six ticks of thrust at most (the window), never a thousand.
        assert!(v < 1.0, "a silent ship must stop thrusting: {v}");
    }

    #[test]
    fn a_lost_message_costs_nothing_a_pilot_can_feel() {
        // The window's other job: two shards do not tick in lockstep, and an unreliable lane drops a
        // datagram now and then. A push stated a couple of ticks ago is still this pilot's intent.
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        on_child_drive(
            a_drive(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        held.advance_all(UniverseTick(12), 5, &vacuum(), 1.0);
        let v = held
            .state_of(ship_coord().lowered())
            .expect("held")
            .vel_mps
            .x;
        assert!(
            (v - 4.0).abs() < 1e-9,
            "a three-tick-old push still flies the ship: {v}"
        );
    }

    #[test]
    fn a_drive_stamped_ahead_of_the_parent_is_fresh_not_ancient() {
        // A child running slightly ahead of its parent's clock must not have its push read as
        // enormously old — which is what an unsaturated subtraction would do.
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_facts(
            some_facts(50),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        on_child_drive(
            a_drive(50),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        held.advance_all(UniverseTick(48), 5, &vacuum(), 1.0);
        let v = held
            .state_of(ship_coord().lowered())
            .expect("held")
            .vel_mps
            .x;
        assert!(
            (v - 4.0).abs() < 1e-9,
            "a push from just ahead is fresh: {v}"
        );
    }

    #[test]
    fn a_held_driven_child_counts_as_moving_even_with_no_push() {
        // ★ THE LANE-SPLIT GUARD (2026-09-01). The window lane asks this to decide which of two lanes a
        // child rides. A driven child must answer YES the moment it is HELD, not only while it pushes.
        //
        // WHY IT MUST NOT TEST THE PUSH: a ship that stops thrusting KEEPS ITS SPEED — that is the
        // owner's own rule — so a coasting hull still changes place every tick. Filing it as static
        // sends it down the reliable lane, which sends on CHANGE and fingerprints the WHOLE static set.
        // One coasting ship would then re-send every planet, star and structure in the realm, to every
        // window, at tick rate. The lane's own comment measures that at 285 MB/s per subscriber.
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
        on_child_facts(
            some_facts(9),
            SHIP_NODE,
            parent_realm(),
            true,
            &nodes(),
            &mut held,
            &mut stats,
        );
        // Facts only — no push has ever arrived, and no drive is stated.
        assert!(
            held.moves(ship_coord().lowered()),
            "a held driven child MOVES, push or no push"
        );
        assert!(
            !held.moves(vd_core::pose::RealmId::Planet(4242)),
            "a realm this book does not hold does not move by this rule"
        );
    }

    #[test]
    fn a_hull_states_what_it_is_once_and_then_stays_quiet() {
        // ★ THE FACTS LANE HAD NO PRODUCER AT ALL until this. It is ON CHANGE ONLY, so something must
        // remember what was said — otherwise a realm either repeats itself every tick on the RELIABLE
        // lane, which is the flood this whole design avoids, or says it once and loses it to a dropped
        // connection.
        let mut outbox = crate::runtime::OutboundBox::default();
        let mut stats = StubStats::default();
        let mut stated = super::StatedFacts::default();
        let body = a_built_body();
        let coord = ship_coord();
        let parent = Some(vd_core::ids::NodeId(9));

        super::emit_child_facts(
            true,
            &coord,
            Some(Fence(1)),
            UniverseTick(1),
            parent,
            &body,
            &mut stated,
            &mut outbox,
            &mut stats,
        );
        assert_eq!(stats.child_facts_sent, 1, "it stated what it is");

        // Nothing changed, so nothing is said. A declared property says nothing twice.
        super::emit_child_facts(
            true,
            &coord,
            Some(Fence(1)),
            UniverseTick(2),
            parent,
            &body,
            &mut stated,
            &mut outbox,
            &mut stats,
        );
        assert_eq!(stats.child_facts_sent, 1, "and it did not repeat itself");

        // The hull drops cargo. That IS a change, and it must travel.
        let mut lighter = body;
        lighter.facts.mass_g = 30_000_000;
        super::emit_child_facts(
            true,
            &coord,
            Some(Fence(1)),
            UniverseTick(3),
            parent,
            &lighter,
            &mut stated,
            &mut outbox,
            &mut stats,
        );
        assert_eq!(stats.child_facts_sent, 2, "a change is stated");
    }

    #[test]
    fn a_realm_that_pushes_nothing_states_nothing() {
        // A star system holds hulls and does not steer. It has no facts to declare and must never
        // speak on this lane — the same silence a realm with no parent gives.
        let mut outbox = crate::runtime::OutboundBox::default();
        let mut stats = StubStats::default();
        let mut stated = super::StatedFacts::default();
        super::emit_child_facts(
            false,
            &ship_coord(),
            Some(Fence(1)),
            UniverseTick(1),
            Some(vd_core::ids::NodeId(9)),
            &a_built_body(),
            &mut stated,
            &mut outbox,
            &mut stats,
        );
        assert_eq!(stats.child_facts_sent, 0);
    }

    /// A hull with no parent, and a hull with no fence, both say nothing about what they ARE.
    ///
    /// The facts lane is RELIABLE and RETAINED, so a statement sent to nobody is not lost — it is
    /// KEPT, and it is delivered to whichever node the carrier books next. A hull that speaks before
    /// the directory grants it would therefore state a deposed incarnation's mass to its parent, and
    /// the parent would compute drag from it for ever. Silence is the only safe answer.
    #[test]
    fn a_hull_states_no_facts_before_it_has_a_parent_or_a_fence() {
        let mut outbox = crate::runtime::OutboundBox::default();
        let mut stats = StubStats::default();
        let mut stated = super::StatedFacts::default();
        // No parent resolved yet: the head read has not come back.
        super::emit_child_facts(
            true,
            &ship_coord(),
            Some(Fence(1)),
            UniverseTick(1),
            None,
            &a_built_body(),
            &mut stated,
            &mut outbox,
            &mut stats,
        );
        // A parent, but the directory has not granted this realm yet.
        super::emit_child_facts(
            true,
            &ship_coord(),
            None,
            UniverseTick(1),
            Some(vd_core::ids::NodeId(9)),
            &a_built_body(),
            &mut stated,
            &mut outbox,
            &mut stats,
        );
        assert_eq!(stats.child_facts_sent, 0, "neither hull spoke");
        assert_eq!(outbox.0.len(), 0, "and nothing left the shard");
        assert_eq!(stated.0, None, "and it remembers having said nothing");
    }

    /// ★ THE PUSH LEAVES THE HULL. A pilot holds the stick full forward, so the hull states six whole
    /// numbers in its OWN frame to the node that holds its parent realm — and nothing else.
    ///
    /// The message carries no speed and no place: a hull may say what it is DOING, never where it IS
    /// (SL1 clause 3). It rides the UNRELIABLE lane on purpose, because the next tick restates the
    /// whole intent.
    #[test]
    fn a_hull_at_full_stick_states_its_push_to_its_parents_node() {
        let mut outbox = crate::runtime::OutboundBox::default();
        let mut stats = StubStats::default();
        let parent = vd_core::ids::NodeId(9);
        super::emit_child_drive(
            true,
            &ship_coord(),
            Some(Fence(7)),
            UniverseTick(11),
            Some(parent),
            &rating(),
            Some((fwd_push(1.0), [0.0; 3])),
            &mut outbox,
            &mut stats,
        );
        assert_eq!(stats.child_drive_sent, 1, "the hull spoke once");
        assert_eq!(outbox.0.len(), 1, "and sent exactly one frame");
        let (to, class, bytes, _) = outbox.0.remove(0);
        assert_eq!(to, parent, "it went to the node holding the parent realm");
        assert_eq!(
            class,
            crate::io::MsgClass::SignalDelta,
            "the unreliable lane: next tick restates the whole intent"
        );
        assert_eq!(
            postcard::from_bytes::<vd_wire::intershard::InterShardFlow>(&bytes).expect("decode"),
            vd_wire::intershard::InterShardFlow::ChildDrive(vd_wire::intershard::ChildDrive {
                child: ship_coord(),
                child_fence: Fence(7),
                at: UniverseTick(11),
                // The shared axis map sends "forward" to −Z, at the whole rating.
                push: [0, 0, -4_000_000],
                turn: [0; 3],
            }),
            "six whole numbers in the hull's own frame, and no place among them"
        );
    }

    /// A realm with no engines never states a push, whatever the pilot inside it does.
    ///
    /// A star system holds ships and does not steer. The switch decides, never a test of what KIND of
    /// realm this is — written as "is this a ship?", a station with thrusters could not fly (HR3).
    #[test]
    fn a_realm_with_no_engines_states_no_push_even_with_a_stick_held() {
        let mut outbox = crate::runtime::OutboundBox::default();
        let mut stats = StubStats::default();
        super::emit_child_drive(
            false,
            &ship_coord(),
            Some(Fence(7)),
            UniverseTick(11),
            Some(vd_core::ids::NodeId(9)),
            &rating(),
            Some((fwd_push(1.0), [0.0; 3])),
            &mut outbox,
            &mut stats,
        );
        assert_eq!(stats.child_drive_sent, 0);
        assert_eq!(outbox.0.len(), 0);
    }

    /// THE THREE SILENCES of the drive lane: no parent, nobody at the controls, no fence.
    ///
    /// None of them is an error worth counting. A hull that just booted is all three at once: it has
    /// not resolved its parent, nobody has sat down, and the directory has not granted it yet.
    #[test]
    fn a_hull_states_no_push_without_a_parent_a_pilot_and_a_fence() {
        let mut outbox = crate::runtime::OutboundBox::default();
        let mut stats = StubStats::default();
        let held = Some((fwd_push(1.0), [0.0; 3]));
        // No parent resolved yet.
        super::emit_child_drive(
            true,
            &ship_coord(),
            Some(Fence(7)),
            UniverseTick(11),
            None,
            &rating(),
            held,
            &mut outbox,
            &mut stats,
        );
        // Nobody at the controls: the pilot left the chair, so the push stops and the speed stays.
        super::emit_child_drive(
            true,
            &ship_coord(),
            Some(Fence(7)),
            UniverseTick(11),
            Some(vd_core::ids::NodeId(9)),
            &rating(),
            None,
            &mut outbox,
            &mut stats,
        );
        // Granted by nobody: a shard that cannot prove its incarnation must not speak for the realm.
        super::emit_child_drive(
            true,
            &ship_coord(),
            None,
            UniverseTick(11),
            Some(vd_core::ids::NodeId(9)),
            &rating(),
            held,
            &mut outbox,
            &mut stats,
        );
        assert_eq!(stats.child_drive_sent, 0, "three silences, no message");
        assert_eq!(outbox.0.len(), 0);
    }

    #[test]
    fn a_hulls_own_row_decides_how_hard_it_pushes_and_not_a_shared_number() {
        // ★ THE SHARED CONSTANT IS GONE. One number in the source stated how hard EVERY ship in the
        // world pushes, which is the magic number this project's own rule forbids: two hulls could
        // never differ, however they were built.
        let heavy = a_built_body();
        let mut nimble = heavy;
        nimble.facts.max_push_micro_mps2 = 200_000_000;
        let a = super::rating_of(&heavy.facts);
        let b = super::rating_of(&nimble.facts);
        assert!(
            b.max_push_mps2 > a.max_push_mps2,
            "two hulls fly differently: {a:?} vs {b:?}"
        );
        // And the stored whole numbers read back as the rates the stick is scaled by.
        assert!(
            (a.max_push_mps2 - 98.1).abs() < 1e-9,
            "ten gravities: {}",
            a.max_push_mps2
        );
    }

    fn a_built_body() -> vd_core::built::BuiltBody {
        vd_core::built::BuiltBody {
            realm: ship_coord().lowered(),
            owner: vd_core::ids::AccountId(1000),
            blueprint: vd_core::built::BlueprintId(0),
            bound: vd_core::geometry::Boundary::Shell { r: 20.0 },
            look: vd_core::geometry::Boundary::Shell { r: 20.0 },
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

    // ===== ★ THE FLIGHT (D-MOVE-2; owner rulings 2026-08-31 and 2026-09-01) ====================
    //
    // A built hull states what it IS and what it is DOING; its parent admits both through the shipped
    // guards, adds its own ambient, and authors where the hull ends up. Nothing on that path asks what
    // KIND of thing it is looking at — which is what the two runs below prove.

    /// The hull's lineage under whichever parent holds it. ★ THE SAME LEAF BOTH TIMES: a hull's name
    /// does not change when its parent does.
    fn hull_under(parent: vd_core::pose::RealmId) -> RealmCoord {
        use vd_core::realm_path::RealmLevel;
        let id = vd_core::ids::EntityId::pack(vd_core::entity_kind::EntityKind::Ship, 1, 1, 0);
        RealmCoord::from_path(RealmPath::from_levels(vec![
            vd_core::worldgen::level_of(parent),
            RealmLevel::for_ship(id),
        ]))
        .expect("a two-level path has a leaf")
    }

    /// Fly one hull under one parent for a second, through the REAL admission path, and report how far
    /// it went and how fast it ended up going.
    fn fly_under(parent: vd_core::pose::RealmId) -> (f64, f64) {
        let body = a_built_body();
        let coord = hull_under(parent);
        let child = coord.lowered();
        let node = NodeId(77);
        let nodes = std::collections::BTreeMap::from([(child, node)]);
        let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());

        // WHAT I AM — on the reliable lane, from the hull's own row.
        on_child_facts(
            vd_wire::intershard::ChildFacts {
                child: coord.clone(),
                child_fence: Fence(1),
                at: UniverseTick(1),
                mass_g: body.facts.mass_g,
                cross_section_mm2: body.facts.cross_section_mm2,
                drag_micro: body.facts.drag_micro,
                declared: vd_wire::intershard::DeclaredStates::default(),
            },
            node,
            parent,
            true,
            &nodes,
            &mut held,
            &mut stats,
        );
        // WHAT I AM DOING — the stick full forward, scaled by THIS hull's own rating.
        let (push, turn) = drive_from_stick(
            vd_core::kinematics::local_axes_from_movement([1.0, 0.0, 0.0]),
            [0.0; 3],
            &super::rating_of(&body.facts),
        );
        on_child_drive(
            vd_wire::intershard::ChildDrive {
                child: coord,
                child_fence: Fence(1),
                at: UniverseTick(1),
                push,
                turn,
            },
            node,
            parent,
            true,
            &nodes,
            &mut held,
            &mut stats,
        );
        assert_eq!(
            stats.child_facts_received, 1,
            "the parent admitted what the hull IS"
        );
        assert_eq!(stats.child_drive_received, 1, "and what it is DOING");

        // ★ THE PILOT HOLDS THE STICK, which means the drive arrives EVERY TICK. The first version of
        // this test stated the push once and advanced fifty ticks, and measured 11.772 m/s instead of
        // 98.1 — exactly six ticks of thrust, which is the freshness window.
        //
        // That was the rule working, not a defect: silence means no force, so a hull whose pilot stops
        // asking stops accelerating and keeps its speed. The test had let go of the stick and then
        // complained the ship coasted.
        //
        // Open space: no pull, no medium. A realm's own ambient is owed with the physics phase, and
        // the stub says so rather than inventing a number.
        for tick in 0..50 {
            let at = UniverseTick(1 + tick);
            on_child_drive(
                vd_wire::intershard::ChildDrive {
                    child: hull_under(parent),
                    child_fence: Fence(1),
                    at,
                    push,
                    turn,
                },
                node,
                parent,
                true,
                &nodes,
                &mut held,
                &mut stats,
            );
            held.advance_all(at, super::DRIVE_STALE_AFTER_TICKS, &vacuum(), 0.02);
        }
        let s = held.state_of(child).expect("the parent holds the hull");
        (s.pos_m.length(), s.vel_mps.length())
    }

    #[test]
    fn a_built_hull_flies_and_does_not_know_which_parent_holds_it() {
        // ★ THE RULING'S OWN ACCEPTANCE LINE: "From the next tick the ship sends the identical six
        // numbers to the planet… It does not know it moved house."
        //
        // Same record, same stick, same six numbers, two different KINDS of parent. The hull must
        // travel identically, because nothing about the parent reaches the child. This is also HR4's
        // gate: the identical fixture on two realm kinds, or the feature does not land.
        let in_system = fly_under(vd_core::pose::RealmId::System(7));
        let on_planet = fly_under(vd_core::pose::RealmId::Planet(7));

        assert!(
            in_system.0 > 0.0,
            "the hull moved in a star system: {in_system:?}"
        );
        assert!(on_planet.0 > 0.0, "and under a planet: {on_planet:?}");
        assert!(
            (in_system.0 - on_planet.0).abs() < 1e-9,
            "IT DOES NOT KNOW IT MOVED HOUSE — same distance: {in_system:?} vs {on_planet:?}",
        );
        assert!(
            (in_system.1 - on_planet.1).abs() < 1e-9,
            "and the same speed: {in_system:?} vs {on_planet:?}",
        );
        // One second of this hull's OWN rated push — ten gravities, from its own row, not a constant.
        assert!(
            (in_system.1 - 98.1).abs() < 1e-6,
            "one second of its own rated push: {} m/s",
            in_system.1,
        );
    }

    #[test]
    fn a_heavier_hull_flies_the_same_in_vacuum_and_differently_in_air() {
        // ★ WHY MASS CROSSES AT ALL, proven on the flight rather than in the arithmetic. Mass cancels
        // out of a realm's pull and does NOT cancel out of its medium — so two hulls coast alike in
        // space and part company the moment there is air.
        let parent = vd_core::pose::RealmId::System(7);
        let coord = hull_under(parent);
        let child = coord.lowered();
        let node = NodeId(77);
        let nodes = std::collections::BTreeMap::from([(child, node)]);

        let fly = |mass_g: u64, density: f64| -> f64 {
            let (mut held, mut stats) = (DrivenChildren::default(), StubStats::default());
            let mut facts = a_built_body().facts;
            facts.mass_g = mass_g;
            on_child_facts(
                vd_wire::intershard::ChildFacts {
                    child: coord.clone(),
                    child_fence: Fence(1),
                    at: UniverseTick(1),
                    mass_g: facts.mass_g,
                    cross_section_mm2: facts.cross_section_mm2,
                    drag_micro: facts.drag_micro,
                    declared: vd_wire::intershard::DeclaredStates::default(),
                },
                node,
                parent,
                true,
                &nodes,
                &mut held,
                &mut stats,
            );
            let (push, turn) = drive_from_stick(
                vd_core::kinematics::local_axes_from_movement([1.0, 0.0, 0.0]),
                [0.0; 3],
                &super::rating_of(&facts),
            );
            on_child_drive(
                vd_wire::intershard::ChildDrive {
                    child: coord.clone(),
                    child_fence: Fence(1),
                    at: UniverseTick(1),
                    push,
                    turn,
                },
                node,
                parent,
                true,
                &nodes,
                &mut held,
                &mut stats,
            );
            let air = Ambient {
                pull_mps2: DVec3::ZERO,
                density_kgpm3: density,
            };
            for tick in 0..50 {
                let at = UniverseTick(1 + tick);
                on_child_drive(
                    vd_wire::intershard::ChildDrive {
                        child: coord.clone(),
                        child_fence: Fence(1),
                        at,
                        push,
                        turn,
                    },
                    node,
                    parent,
                    true,
                    &nodes,
                    &mut held,
                    &mut stats,
                );
                held.advance_all(at, super::DRIVE_STALE_AFTER_TICKS, &air, 0.02);
            }
            held.state_of(child).expect("held").vel_mps.length()
        };

        let (heavy, light) = (50_000_000, 5_000_000);
        assert!(
            (fly(heavy, 0.0) - fly(light, 0.0)).abs() < 1e-9,
            "in vacuum the hulls fly alike — a push is an acceleration",
        );
        assert!(
            fly(light, 1.225) < fly(heavy, 1.225),
            "in air the LIGHTER hull is slowed more — which is the whole reason mass crosses",
        );
    }
}
