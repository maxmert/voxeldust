//! Server-driven animation graph for character visuals.
//!
//! Phase D of the character plan. For every visual entity spawned by
//! [`super::render::spawn_remote_character_visuals`], drive the
//! attached [`AnimationPlayer`] from the server-broadcast
//! `Locomotion` state + `locomotion_speed` so observers see the
//! character idling / walking / running / jumping / sitting in
//! lockstep with the authoritative state machine.
//!
//! # Mapping
//!
//! Pure function [`clip_for_locomotion`]:
//!
//! | Server state           | Speed condition                    | Clip         |
//! |------------------------|------------------------------------|--------------|
//! | `Grounded`             | `> walk_run_threshold`             | `Run`        |
//! | `Grounded`             | `> stationary_speed_threshold`     | `Walk`       |
//! | `Grounded`             | otherwise                           | `Idle`       |
//! | `Airborne`             | (any)                               | `Falling`    |
//! | `Seated`               | (any)                               | `Sit`        |
//! | `Climbing` / `Ragdoll` | (any)                               | `Idle` †     |
//!
//! † `Climbing` has no clip yet; `Ragdoll` is overridden by Phase I's
//! bone-transform broadcast — `Idle` keeps the rig in a known pose
//! until the proper clip lands.
//!
//! # Crossfades
//!
//! All transitions go through [`bevy::animation::AnimationTransitions`]
//! so blends are automatic. Per-pair fade durations live in
//! [`clip_fade_duration`] and follow AAA conventions:
//!
//!  * Idle ↔ Walk / Run: 0.20 s — visible weight blend.
//!  * Walk ↔ Run: 0.15 s — tighter so the gait stays convincing.
//!  * Anything ↔ Falling / Jump: 0.10 s — snappy so air time feels
//!    responsive.
//!  * Anything ↔ Sit: 0.30 s — deliberate, matches the longer chair
//!    animation pacing.
//!
//! Class-tunable via [`voxeldust_core::character::CharacterClass`] —
//! the function reads class metadata so future races can override.
//!
//! # Repeat semantics
//!
//! Driven from [`voxeldust_core::character::ClipKind`]:
//!  * `Looping` (idle / walk / run / falling / sit): `set_repeat(Forever)`.
//!  * `OneShot` (jump / turn-in-place): `set_repeat(Never)`. Phase D
//!    does not currently route Jump or Turn-in-place via this driver —
//!    Jump waits on the server emitting an explicit Airborne→Jump
//!    sub-state, Turn-in-place is Phase F's exclusive responsibility.
//!  * `Additive` (reserved for future aim layers): unused here.
//!
//! # Where the driver runs
//!
//! In [`CharacterAnimSet`], chained after [`super::CharacterRenderSet`]
//! so it sees the latest spawn / despawn this frame and the bone
//! hierarchy is materialized. Two systems chained:
//!
//!  1. [`attach_anim_state_when_ready`] — once per visual, when
//!     [`super::render::BoneRegistry`] flips to `resolved`, insert
//!     [`LocomotionAnimState`] + [`AnimationTransitions`] so the
//!     driver has somewhere to record its last-played gait.
//!  2. [`drive_animation_graph`] — every frame, diff target gait
//!     against last-played; on change, fire a crossfade.

use std::time::Duration;

use bevy::animation::{
    transition::AnimationTransitions, AnimationPlayer, RepeatAnimation,
};
use bevy::prelude::*;

use voxeldust_core::character::{
    state_machine::wrap_pi, CharacterClass, ClipKind, ClipLabel, LocomotionState,
};

use crate::remote::{RemoteEntity, RemotePlayers};

use super::assets::CharacterAssetRegistry;
use super::render::{BoneRegistry, CharacterRenderSet, RemoteCharacterTag};

/// SystemSet for the animation driver. Runs after
/// [`CharacterRenderSet`] so the visual + bone hierarchy is
/// materialized first.
#[derive(SystemSet, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CharacterAnimSet;

/// Per-visual record of which clip the driver last asked the player
/// to crossfade to. Diffed against [`clip_for_locomotion`]'s output
/// each frame; mismatch fires a transition. `None` only on the very
/// first frame after attach, so the initial play happens with a
/// 0-duration "fade" (i.e. instant snap to the first pose).
#[derive(Component, Default, Debug)]
pub struct LocomotionAnimState {
    pub current_clip: Option<ClipLabel>,
}

pub struct CharacterAnimPlugin;

impl Plugin for CharacterAnimPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(Update, CharacterAnimSet.after(CharacterRenderSet))
            .add_systems(
                Update,
                (attach_anim_state_when_ready, drive_animation_graph)
                    .chain()
                    .in_set(CharacterAnimSet),
            );
    }
}

/// Attach the per-visual driver state once the visual's scene
/// hierarchy is materialized (i.e. when
/// [`BoneRegistry::resolved`] flips). Runs cheap each frame: skips
/// any visual that already has [`LocomotionAnimState`].
fn attach_anim_state_when_ready(
    mut commands: Commands,
    visuals: Query<
        (Entity, &BoneRegistry),
        (With<RemoteCharacterTag>, Without<LocomotionAnimState>),
    >,
) {
    for (entity, bones) in &visuals {
        if !bones.resolved {
            continue;
        }
        commands
            .entity(entity)
            .insert((LocomotionAnimState::default(), AnimationTransitions::new()));
    }
}

/// Per-frame: diff the target clip against the last-played one; on
/// change, crossfade via [`AnimationTransitions::play`].
///
/// Skips visuals where:
///  * [`RemotePlayers`] has no entry for this `player_id` (the
///    despawn system will reap the visual on the next frame).
///  * The class assets aren't `Ready` (transient — the spawn system
///    wouldn't have created the visual without `Ready`, but a future
///    asset hot-reload could flip the registry back to `Loading`).
///  * The class doesn't define this clip — logged once per visual at
///    `WARN` so a class authoring drift is loud.
fn drive_animation_graph(
    remote_players: Res<RemotePlayers>,
    asset_registry: Res<CharacterAssetRegistry>,
    mut characters: Query<(
        &RemoteCharacterTag,
        &mut AnimationPlayer,
        &mut AnimationTransitions,
        &mut LocomotionAnimState,
    )>,
) {
    for (tag, mut player, mut transitions, mut state) in &mut characters {
        let Some(remote) = remote_players.by_id.get(&tag.player_id) else {
            continue;
        };
        let Some(assets) = asset_registry.ready(tag.class_id) else {
            continue;
        };
        let class = assets.class;

        let target = clip_for_locomotion(remote, class);
        if state.current_clip == Some(target) {
            continue;
        }

        let Some(node) = assets.node_for(target) else {
            warn!(
                player_id = tag.player_id,
                clip = ?target,
                "drive_animation_graph: class '{}' has no clip for {:?}; visual stays at previous gait",
                class.name,
                target,
            );
            // Mark as current so we don't spam the warn each frame —
            // the rig will hold whatever pose was last set.
            state.current_clip = Some(target);
            continue;
        };

        let fade = clip_fade_duration(state.current_clip, target, class);
        let active = transitions.play(&mut player, node, fade);

        // Repeat semantics from class metadata — looping clips loop
        // forever, one-shots play once. `ActiveAnimation::default()`
        // is `Never`, so we only need to flip on Looping.
        match class
            .clip_kind(target)
            .unwrap_or(ClipKind::Looping)
        {
            ClipKind::Looping => {
                active.set_repeat(RepeatAnimation::Forever);
            }
            ClipKind::OneShot => {
                active.set_repeat(RepeatAnimation::Never);
            }
            // Additive layers compose differently — reserved for a
            // future aim-offset / reaction-layer phase. Treat as
            // looping so the rig stays animated.
            ClipKind::Additive => {
                active.set_repeat(RepeatAnimation::Forever);
            }
        }

        info!(
            player_id = tag.player_id,
            class = class.name,
            from = ?state.current_clip,
            to = ?target,
            fade_ms = fade.as_millis(),
            speed = remote.locomotion_speed,
            "character anim: transitioned"
        );
        state.current_clip = Some(target);
    }
}

/// Pure mapping from server-broadcast locomotion → clip label.
/// Pure function so it's testable in isolation. Reads thresholds
/// from [`CharacterClass`] so future races / vehicles can have
/// different walk-run boundaries without driver-code changes.
///
/// **Phase F: turn-in-place takes priority.** When the server
/// state machine marks `is_turning = true`, the body is mid-step
/// rotating toward `turn_target_yaw`. The OneShot Mixamo
/// "Standing Turn 90 Left/Right" clip plays the leg + arm step
/// pose on top; the actual body rotation is server-authoritative
/// via `body_yaw` and applied through `local_transform_for`. Sign
/// of `wrap_pi(target - current)` picks the direction:
///
///  * positive (CCW around +Y, from-above-looking-down) → `TurnL`
///  * negative (CW)                                     → `TurnR`
pub fn clip_for_locomotion(remote: &RemoteEntity, class: &CharacterClass) -> ClipLabel {
    if remote.is_turning {
        let delta = wrap_pi(remote.turn_target_yaw - remote.body_yaw);
        // `wrap_pi` keeps `delta` in `(-π, π]`, so pure sign
        // suffices. A delta of exactly zero (e.g. degenerate
        // single-tick turn that completed instantly) defaults to
        // TurnR; the choice is arbitrary and the clip terminates
        // immediately on the next state change.
        return if delta > 0.0 {
            ClipLabel::TurnL
        } else {
            ClipLabel::TurnR
        };
    }
    let state = LocomotionState::from_u8(remote.locomotion);
    match state {
        LocomotionState::Grounded => {
            if remote.locomotion_speed > class.walk_run_threshold {
                ClipLabel::Run
            } else if remote.locomotion_speed > class.stationary_speed_threshold {
                ClipLabel::Walk
            } else {
                ClipLabel::Idle
            }
        }
        // Phase D treats all Airborne as Falling. A future Phase D.5
        // can split takeoff vs. fall on `velocity.y` sign and use the
        // OneShot Jump clip on the rising edge.
        LocomotionState::Airborne => ClipLabel::Falling,
        LocomotionState::Seated => ClipLabel::Sit,
        // No climb clip yet — Idle keeps the rig in a known pose.
        LocomotionState::Climbing => ClipLabel::Idle,
        // Ragdoll path is overridden by Phase I's bone-transform
        // broadcast; Idle keeps the rig sane in the meantime.
        LocomotionState::Ragdoll => ClipLabel::Idle,
    }
}

/// Per-transition crossfade duration. Encodes AAA-typical pacing.
/// Pulled out of the driver so it's swappable per character class
/// without driver-code changes.
pub fn clip_fade_duration(
    from: Option<ClipLabel>,
    to: ClipLabel,
    _class: &CharacterClass,
) -> Duration {
    use ClipLabel::*;
    let secs = match (from, to) {
        // First clip after spawn — instant snap to bind-pose-aligned
        // first frame so the T-pose disappears as soon as Phase D's
        // first transition fires.
        (None, _) => 0.0,
        // Sitting transitions are deliberate and match the longer
        // sit-down animation pacing.
        (_, Sit) | (Some(Sit), _) => 0.30,
        // Air ↔ ground: snappy. The player expects jump / land
        // response within ~100 ms.
        (_, Falling) | (Some(Falling), _) | (_, Jump) | (Some(Jump), _) => 0.10,
        // Walk ↔ Run: tight gait blend; longer would smear the
        // footplant cadence.
        (Some(Walk), Run) | (Some(Run), Walk) => 0.15,
        // Turn-in-place: tight on entry so the leg-step starts in
        // sync with the server's body rotation (which begins the
        // same tick). Tight on exit so the rig returns to the
        // resulting locomotion (idle / walk) without lingering in
        // the turn pose.
        (_, TurnL) | (_, TurnR) | (Some(TurnL), _) | (Some(TurnR), _) => 0.10,
        // Default — idle ↔ walk / idle ↔ run / similar.
        _ => 0.20,
    };
    Duration::from_secs_f32(secs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use voxeldust_core::character::HUMAN_DEFAULT;
    use voxeldust_core::client_message::EntityKind;

    fn remote_with(loco: LocomotionState, speed: f32) -> RemoteEntity {
        RemoteEntity {
            entity_id: 1,
            kind: EntityKind::GroundedPlayer,
            position: glam::DVec3::ZERO,
            rotation: glam::DQuat::IDENTITY,
            velocity: glam::DVec3::ZERO,
            shard: crate::shard::ShardKey::new(2, 1),
            name: String::new(),
            health: 100.0,
            shield: 100.0,
            body_yaw: 0.0,
            head_yaw: 0.0,
            head_pitch: 0.0,
            locomotion: loco.as_u8(),
            locomotion_speed: speed,
            is_turning: false,
            turn_target_yaw: 0.0,
            turn_t: 0.0,
        }
    }

    #[test]
    fn grounded_zero_speed_is_idle() {
        let r = remote_with(LocomotionState::Grounded, 0.0);
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::Idle);
    }

    #[test]
    fn grounded_walking_speed_is_walk() {
        // HUMAN_DEFAULT thresholds: stationary=0.5, walk_run=4.0.
        let r = remote_with(LocomotionState::Grounded, 2.0);
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::Walk);
    }

    #[test]
    fn grounded_running_speed_is_run() {
        let r = remote_with(LocomotionState::Grounded, 5.0);
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::Run);
    }

    #[test]
    fn airborne_is_falling() {
        let r = remote_with(LocomotionState::Airborne, 3.0);
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::Falling);
    }

    #[test]
    fn seated_is_sit() {
        let r = remote_with(LocomotionState::Seated, 0.0);
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::Sit);
    }

    #[test]
    fn first_transition_is_instant() {
        assert_eq!(
            clip_fade_duration(None, ClipLabel::Idle, &HUMAN_DEFAULT),
            Duration::from_secs_f32(0.0)
        );
    }

    #[test]
    fn walk_run_fade_is_tight() {
        assert_eq!(
            clip_fade_duration(Some(ClipLabel::Walk), ClipLabel::Run, &HUMAN_DEFAULT),
            Duration::from_secs_f32(0.15)
        );
    }

    #[test]
    fn airborne_fade_is_snappy() {
        assert_eq!(
            clip_fade_duration(Some(ClipLabel::Idle), ClipLabel::Falling, &HUMAN_DEFAULT),
            Duration::from_secs_f32(0.10)
        );
    }

    #[test]
    fn sit_fade_is_deliberate() {
        assert_eq!(
            clip_fade_duration(Some(ClipLabel::Idle), ClipLabel::Sit, &HUMAN_DEFAULT),
            Duration::from_secs_f32(0.30)
        );
    }

    #[test]
    fn turning_left_yields_turn_l_clip() {
        // Turning CCW (positive yaw delta around +Y from above) is
        // the character's own LEFT.
        let mut r = remote_with(LocomotionState::Grounded, 0.0);
        r.is_turning = true;
        r.body_yaw = 0.0;
        r.turn_target_yaw = 1.2; // ~70° to the left
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::TurnL);
    }

    #[test]
    fn turning_right_yields_turn_r_clip() {
        let mut r = remote_with(LocomotionState::Grounded, 0.0);
        r.is_turning = true;
        r.body_yaw = 0.0;
        r.turn_target_yaw = -1.2;
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::TurnR);
    }

    #[test]
    fn turning_takes_priority_over_locomotion() {
        // Even if speed > 0 (e.g., the player just started moving
        // the same tick a turn fired), the turn clip wins. This
        // matches the server's state machine — `TurnInPlace` is
        // exclusive while it's the active component.
        let mut r = remote_with(LocomotionState::Grounded, 5.0);
        r.is_turning = true;
        r.body_yaw = 0.0;
        r.turn_target_yaw = 1.2;
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::TurnL);
    }

    #[test]
    fn turn_with_wrap_around_pi_picks_correct_direction() {
        // Body at +170°, target at -170°. Naïve `target - body =
        // -340°` would say "right turn", but the SHORT-PATH (after
        // wrap_pi) is +20° — a left turn. Verifies we use wrap_pi.
        let mut r = remote_with(LocomotionState::Grounded, 0.0);
        r.is_turning = true;
        r.body_yaw = std::f32::consts::PI - 0.17; // +170°
        r.turn_target_yaw = -std::f32::consts::PI + 0.17; // -170°
        assert_eq!(clip_for_locomotion(&r, &HUMAN_DEFAULT), ClipLabel::TurnL);
    }

    #[test]
    fn turn_clip_fade_is_snappy() {
        // Entry to the turn — body has already started rotating
        // server-side this tick, so the clip needs to overlap fast.
        assert_eq!(
            clip_fade_duration(Some(ClipLabel::Idle), ClipLabel::TurnL, &HUMAN_DEFAULT),
            Duration::from_secs_f32(0.10)
        );
        // Exit to a regular state.
        assert_eq!(
            clip_fade_duration(Some(ClipLabel::TurnR), ClipLabel::Idle, &HUMAN_DEFAULT),
            Duration::from_secs_f32(0.10)
        );
    }
}
