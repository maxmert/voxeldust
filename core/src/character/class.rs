//! Per-class character configuration.
//!
//! Drives the body/head decoupling state machine, the asset pipeline,
//! and the eventual ragdoll bone setup. Per the project's
//! `feedback_no_magic_numbers` rule, **every** tunable that varies by
//! character lives here — never bare constants in shard or render code.
//!
//! A future cosmetic system will let players pick a body type by
//! swapping the [`super::body_head::CharacterClassComp`] component on
//! their entity; the visible asset, IK limits, joint limits, and
//! audible footsteps follow automatically.

use super::components::CharacterCapsule;
use glam::Vec3;

/// Stable label for a clip slot in the animation graph. Phase D maps
/// `Locomotion.state` → `ClipLabel` → graph node weights, so adding a
/// new label is the only change needed when a new locomotion variant
/// lands. Discriminants are stable across releases — append-only;
/// never repurpose.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ClipLabel {
    Idle = 0,
    Walk = 1,
    Run = 2,
    Jump = 3,
    Falling = 4,
    Sit = 5,
    /// Step-turn-in-place 90° to the left (camera-relative).
    TurnL = 6,
    /// Step-turn-in-place 90° to the right.
    TurnR = 7,
}

impl ClipLabel {
    /// Total number of variants — used to size dense array maps in the
    /// loader / animation driver. Update on every new variant.
    pub const COUNT: usize = 8;

    /// All variants in declaration order — handy for dense iteration.
    pub const ALL: [ClipLabel; Self::COUNT] = [
        Self::Idle,
        Self::Walk,
        Self::Run,
        Self::Jump,
        Self::Falling,
        Self::Sit,
        Self::TurnL,
        Self::TurnR,
    ];

    /// Stable byte index for dense array indexing (`as usize` is the
    /// canonical way; this method is documentation).
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }
}

/// How the animation driver should play a clip.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClipKind {
    /// Loops indefinitely — idle, walk, run, sit, falling.
    Looping,
    /// Plays once and ends — jump, turn-in-place.
    OneShot,
    /// Adds onto whatever is currently playing — future aim offsets,
    /// reaction layer.
    Additive,
}

/// One clip's metadata: where it lives on disk, what it does, what
/// label drives it.
#[derive(Clone, Copy, Debug)]
pub struct ClipDef {
    pub label: ClipLabel,
    /// Asset path relative to `client/assets/`.
    pub path: &'static str,
    pub kind: ClipKind,
}

/// One immutable bundle of per-class parameters, copied by value into
/// the per-entity [`super::body_head::CharacterClassComp`] component.
///
/// Field grouping:
/// - **Identity**: stable wire id + diagnostic name.
/// - **Geometry**: capsule + eye offset (where the camera lives).
/// - **Look limits**: head yaw / pitch swing before the body steps.
/// - **Locomotion**: thresholds and time constants.
/// - **Assets**: skeleton scene + clip array — drives client load.
/// - **Bone names**: stable identifiers for IK + camera + cull.
#[derive(Clone, Copy, Debug)]
pub struct CharacterClass {
    /// Stable wire id (room for 65 535 races; 0 = `HUMAN_DEFAULT`).
    pub id: u16,
    /// Human-readable name for diagnostics + tools.
    pub name: &'static str,

    // -- Geometry ----------------------------------------------------
    /// KCC capsule. Total height ≈ `height + 2*radius` per Rapier convention.
    pub capsule: CharacterCapsule,
    /// Camera eye position in the **head bone's local frame**. Added
    /// in after the head bone's world transform so the camera follows
    /// the head animation rather than a fixed body offset.
    ///
    /// For Mixamo rigs the head bone sits at the *base of the skull*
    /// (around the C1 vertebra, jaw-line height) — eye level is
    /// ~12 cm up + ~8 cm forward from there. Tune per-class for
    /// non-Mixamo rigs.
    pub eye_offset_local: Vec3,
    /// Y-offset applied to the rendered visual relative to the
    /// server-broadcast capsule centre (`remote.position`). The
    /// server's capsule centre sits ~`height/2 + radius` above the
    /// feet (Rapier `capsule_y` convention), but Mixamo / glTF mesh
    /// origins are at the feet. So the visual gets shifted DOWN by
    /// the capsule half-extent so feet land on the floor.
    /// Pre-computed from the capsule, exposed as a field so future
    /// classes with non-Mixamo origins (mesh root at hips, head, …)
    /// can override.
    pub visual_y_offset: f32,
    /// Additive yaw applied to the visual's `Quat::from_rotation_y`
    /// so the *mesh's bind-pose forward* aligns with the *server's
    /// body-yaw convention* (which is shared with the KCC: yaw=0
    /// faces +X, positive yaw turns RIGHT — CW from above).
    ///
    /// For Mixamo Y-bot through FBX2glTF (`export_yup=True`) the
    /// bind pose faces **-Z**; rotating by `-π/2` around +Y lands
    /// the mesh facing +X = body_yaw=0. Future classes baked from
    /// other pipelines (Blender, Unreal-style FBX) may need 0,
    /// `+π/2`, or `+π`.
    pub mesh_yaw_offset: f32,
    /// Sign multiplier on `body_yaw` in the visual rotation formula.
    /// Camera yaw convention is +ve = turn right (CW from above);
    /// `Quat::from_rotation_y` follows the right-hand rule (+ve =
    /// CCW from above). Without flipping the sign the visual body
    /// rotates **opposite** to the camera (only invisible when
    /// body_yaw=0 happens to align). Mixamo Y-bot needs `-1`.
    pub body_yaw_sign: f32,

    // -- Look limits -------------------------------------------------
    /// Head yaw swing before the body must step (radians, half-cone).
    pub head_yaw_limit: f32,
    /// Head pitch swing (radians, half-cone). Common AAA value: 85°.
    pub head_pitch_limit: f32,
    /// After a turn fires, re-anchor head yaw to this fraction of the
    /// limit (`0.78` ≈ 70°/90°), so the new neutral leaves headroom.
    pub turn_reanchor_fraction: f32,
    /// Sign multiplier applied to `head_yaw` when post-multiplying the
    /// head bone's local rotation. `+1` for rigs where positive yaw
    /// rotates the head's face to the player's left (right-hand rule
    /// around the head bone's local Y); `-1` if the rig's head bone
    /// has its local frame flipped. Mixamo Y-bot via FBX2glTF + Y-up:
    /// `+1`.
    pub head_yaw_sign: f32,
    /// Sign multiplier applied to `head_pitch`. `+1` for rigs where
    /// positive pitch tilts the face up; `-1` for rigs whose head
    /// bone local X axis is flipped (positive rotation tilts the
    /// face DOWN). Mixamo Y-bot via FBX2glTF + Y-up needs `-1` —
    /// the head bone's bind X points opposite to character-right, so
    /// a positive Euler X rotation pitches the chin into the chest.
    pub head_pitch_sign: f32,

    // -- Locomotion --------------------------------------------------
    /// Duration of one turn-in-place clip (seconds). Should match the
    /// authored Mixamo "standing turn 90" clip length.
    pub turn_in_place_duration: f32,
    /// Exponential-decay time constant for body-yaw chasing the
    /// movement direction while walking (seconds; `0.18` is snappy
    /// but not jittery at 20 Hz).
    pub body_align_tau: f32,
    /// Below this horizontal speed the body stays still and the head
    /// can swing toward the turn limit (m/s).
    pub stationary_speed_threshold: f32,
    /// At or above this horizontal speed the animation graph blends
    /// from walk → run (m/s).
    pub walk_run_threshold: f32,

    // -- Assets (consumed by Phase B+ client only) -------------------
    /// Path to the skeleton + base mesh scene (glTF/GLB).
    pub asset_path: &'static str,
    /// Animation clips, keyed by stable label. Order doesn't matter —
    /// the loader builds a label→handle map. Adding a new clip is one
    /// line here plus a new `ClipLabel` variant.
    pub clips: &'static [ClipDef],

    // -- Bone names -------------------------------------------------
    /// Names of bones the runtime needs to find by `Name` lookup.
    /// Mismatches against the loaded skeleton are flagged on load with
    /// a clear error (asset / class drift detection).
    pub head_bone: &'static str,
    pub neck_bone: &'static str,
    /// Per-leg bone chain for foot-IK 2-bone solver.
    pub left_hip_bone: &'static str,
    pub left_knee_bone: &'static str,
    pub left_foot_bone: &'static str,
    pub right_hip_bone: &'static str,
    pub right_knee_bone: &'static str,
    pub right_foot_bone: &'static str,
    /// Top-of-skin bone whose XZ translation gets zeroed each frame
    /// after animation to strip "root motion" — the forward
    /// translation Mixamo bakes into walk / run / turn clips when
    /// the *In Place* checkbox is off (or its FBX2glTF re-export
    /// re-introduces it). Server position is the only source of
    /// world translation; the clip is allowed to drive the bone's
    /// **Y** for natural hip bob. For Mixamo: `mixamorig:Hips`.
    pub root_bone: &'static str,
    /// Bones the camera-bone-attachment system culls when in
    /// first-person (Phase E). Mostly head + clavicles + upper arms so
    /// the player doesn't see their own neck stub when looking down.
    pub fp_cull_bones: &'static [&'static str],

    // -- Foot IK (Phase G) ------------------------------------------
    /// Vertical distance from the ankle bone's pivot to the foot's
    /// sole, in metres. Used so we plant the SOLE of the foot on the
    /// ground, not the ankle bone (which sits ~10 cm above the
    /// floor in bind pose). Positive = sole below ankle.
    pub foot_ik_sole_offset: f32,
    /// Maximum descent below the animated ankle position the IK
    /// will search for ground. Capped to avoid catastrophic over-
    /// stretch when the player's feet briefly leave terrain (jump,
    /// edge of platform); above this distance the foot stays at the
    /// animated position.
    pub foot_ik_max_descent: f32,
    /// Maximum ascent above the animated ankle position the IK will
    /// raycast (covers small steps, slope crests). Above this we
    /// assume the floor is at the animated position (no lift).
    pub foot_ik_max_ascent: f32,

    // -- Look-at IK (Phase H) ---------------------------------------
    /// Forward axis of the head + neck bones in their LOCAL frame
    /// after bind. Mixamo Y-bot via FBX2glTF + Y-up: +Z. The look-at
    /// solver rotates the bone so this axis points at the target.
    pub look_forward_local: Vec3,
    /// Up axis of the head + neck bones in their LOCAL frame after
    /// bind — used as the YAW axis of the look-at decomposition.
    /// Mixamo Y-bot: +Y.
    pub look_up_local: Vec3,
    /// Fraction of the look-at rotation applied at the NECK bone
    /// (`0.0..1.0`). Industry convention: ~0.3 — the neck does
    /// roughly a third of the work, the head finishes. Higher values
    /// make the upper body lean into the look; lower values produce
    /// snappy "head only" tracking (uncanny on long turns).
    pub look_neck_share: f32,
    /// Hard limits on neck rotation, half-cones in radians.
    pub look_neck_yaw_limit: f32,
    pub look_neck_pitch_limit: f32,
    /// Hard limits on head rotation (applied after neck), half-cones
    /// in radians. Together with the neck limits this caps the
    /// reachable target cone (e.g. neck 30° + head 60° = 90°
    /// effective head turn before the body must rotate).
    pub look_head_yaw_limit: f32,
    pub look_head_pitch_limit: f32,
    /// Maximum range at which a character will pay attention to
    /// other characters for the "track nearby player" behaviour.
    /// Beyond this distance other players don't draw the head.
    pub look_attention_distance: f32,
    /// Half-angle of the attention FOV cone, in radians. A character
    /// only registers other players whose direction from this
    /// character is within this cone of its body-forward — a player
    /// directly behind is ignored, you have to body-turn to notice
    /// them. Common AAA value: 60° (π/3) — peripheral-vision wide.
    pub look_attention_fov_half: f32,
    /// Vertical offset added to the OBSERVED player's `Position`
    /// (which is the capsule centre) along shard up to land at
    /// roughly eye level — gives the look at the right point on the
    /// other character's head, not their hip.
    pub look_eye_world_offset: f32,
}

/// Default humanoid class. Mixamo Y-bot rig (~22 bones), targets the
/// asset layout described in the Phase B plan. The numeric tunables
/// below are AAA-conventional and aligned with [`CharacterCapsule`]'s
/// 1.2 m total height — a future taller race overrides via a new
/// `CharacterClass` const without touching shard code.
pub const HUMAN_DEFAULT: CharacterClass = CharacterClass {
    id: 0,
    name: "human_default",
    capsule: CharacterCapsule {
        height: 0.6,
        radius: 0.3,
    },
    // Eye sits at the bridge of the nose: from the Mixamo head bone
    // (located at the C1 vertebra ≈ jaw-line) we need ~12 cm up to
    // reach eye level, plus ~8 cm forward to land at the front of
    // the skull rather than inside it.
    eye_offset_local: Vec3::new(0.0, 0.12, 0.08),
    // capsule centre is `height * 0.5 + radius = 0.6` m above feet,
    // so we shift the rendered visual down by the same amount to put
    // feet on the floor instead of floating above it.
    visual_y_offset: -(0.6 * 0.5 + 0.3),
    // Mixamo Y-bot via FBX2glTF + Y-up faces **+Z** at identity
    // (post-conversion the Z-up FBX forward axis lands on glTF
    // Y-up's +Z). KCC yaw=0 means facing +X. Rotating the visual
    // by +π/2 around +Y takes +Z → +X so mesh facing matches
    // movement direction. (Empirically verified — earlier `-π/2`
    // accidentally matched at body_yaw=0 only because the sign-
    // flipped formula collapsed to the same value, but it left a
    // hidden 180° error that surfaced once `body_yaw_sign` was
    // wired up. The pair `mesh_yaw_offset = +π/2` and
    // `body_yaw_sign = -1` is the AAA-correct combination.)
    mesh_yaw_offset: std::f32::consts::FRAC_PI_2,
    // Camera yaw convention is +ve=right; Bevy quat is +ve=CCW (left).
    // Negate body_yaw so the body rotates the same way as the camera.
    body_yaw_sign: -1.0,

    head_yaw_limit: std::f32::consts::FRAC_PI_2,           // 90°
    head_pitch_limit: 1.4835298_f32,                       // 85° in radians
    turn_reanchor_fraction: 0.78,
    head_yaw_sign: 1.0,
    head_pitch_sign: -1.0, // Mixamo head bone local X is flipped — see field doc.

    turn_in_place_duration: 0.5,
    body_align_tau: 0.18,
    stationary_speed_threshold: 0.5,
    walk_run_threshold: 4.0,

    asset_path: "characters/human_default/y_bot.glb",
    clips: &HUMAN_DEFAULT_CLIPS,

    head_bone: "mixamorig:Head",
    neck_bone: "mixamorig:Neck",
    left_hip_bone: "mixamorig:LeftUpLeg",
    left_knee_bone: "mixamorig:LeftLeg",
    left_foot_bone: "mixamorig:LeftFoot",
    right_hip_bone: "mixamorig:RightUpLeg",
    right_knee_bone: "mixamorig:RightLeg",
    right_foot_bone: "mixamorig:RightFoot",
    root_bone: "mixamorig:Hips",
    fp_cull_bones: &[
        "mixamorig:Head",
        "mixamorig:LeftShoulder",
        "mixamorig:RightShoulder",
        "mixamorig:LeftArm",
        "mixamorig:RightArm",
    ],
    // Mixamo Y-bot ankle bone sits ~10 cm above the foot sole.
    foot_ik_sole_offset: 0.10,
    // 60 cm: covers a full leg's down-stretch on a deep step. Beyond
    // this we'd risk hyper-extending the knee through the back.
    foot_ik_max_descent: 0.60,
    // 30 cm: tallest step we expect the IK to fix transparently —
    // matches the KCC's `autostep_height` so the visual doesn't lag
    // the physics step.
    foot_ik_max_ascent: 0.30,

    // -- Look-at IK ---------------------------------------------------
    // Mixamo Y-bot via FBX2glTF + Y-up: head/neck bind faces +Z, up = +Y.
    look_forward_local: Vec3::Z,
    look_up_local: Vec3::Y,
    // 30/70 split — industry standard. Neck contributes a subtle lean
    // that sells the look without flipping the upper body.
    look_neck_share: 0.30,
    // Real cervical spine: ~25° yaw, ~15° pitch on the neck itself.
    // Slightly under-cap so the head finishes the look, even when the
    // neck has run out of room.
    look_neck_yaw_limit: 0.4363,   // 25°
    look_neck_pitch_limit: 0.2618, // 15°
    // Head joint: ~60° yaw, ~30° pitch — the rest of the cervical range.
    // Total reach with neck: 85° yaw / 45° pitch — enough for
    // peripheral-attention turns; deeper turns require body rotation
    // (the body/head decoupling state machine handles that).
    look_head_yaw_limit: 1.0472,   // 60°
    look_head_pitch_limit: 0.5236, // 30°
    // 30 m: comfortable indoor / surface attention range. Outside this
    // you barely register another player's facial detail anyway.
    look_attention_distance: 30.0,
    // 60° half-cone = 120° total peripheral vision — humans are ~110°
    // monocular each side; AAA games average ~120° (Detroit, RDR2).
    look_attention_fov_half: 1.0472,
    // Player capsule is 1.2 m total height with `Position` at the
    // centre, so eye level is roughly capsule centre + 0.5 m.
    look_eye_world_offset: 0.5,
};

const HUMAN_DEFAULT_CLIPS: [ClipDef; 8] = [
    ClipDef {
        label: ClipLabel::Idle,
        path: "characters/human_default/animations/idle.glb",
        kind: ClipKind::Looping,
    },
    ClipDef {
        label: ClipLabel::Walk,
        path: "characters/human_default/animations/walk.glb",
        kind: ClipKind::Looping,
    },
    ClipDef {
        label: ClipLabel::Run,
        path: "characters/human_default/animations/run.glb",
        kind: ClipKind::Looping,
    },
    ClipDef {
        label: ClipLabel::Jump,
        path: "characters/human_default/animations/jump.glb",
        kind: ClipKind::OneShot,
    },
    ClipDef {
        label: ClipLabel::Falling,
        path: "characters/human_default/animations/falling.glb",
        kind: ClipKind::Looping,
    },
    ClipDef {
        label: ClipLabel::Sit,
        path: "characters/human_default/animations/sit_chair.glb",
        kind: ClipKind::Looping,
    },
    ClipDef {
        label: ClipLabel::TurnL,
        path: "characters/human_default/animations/turn_l_90.glb",
        kind: ClipKind::OneShot,
    },
    ClipDef {
        label: ClipLabel::TurnR,
        path: "characters/human_default/animations/turn_r_90.glb",
        kind: ClipKind::OneShot,
    },
];

/// Master registry of all character classes available in the game.
/// Adding a race / body type = adding a `CharacterClass` const above
/// and an entry to this slice. The client's Phase B asset loader
/// iterates this to enqueue loads; the shard layer reads class
/// metadata by id via [`class_by_id`].
pub const ALL_CLASSES: &[&CharacterClass] = &[&HUMAN_DEFAULT];

/// Look up a class by its stable wire id. Returns `None` for unknown
/// ids — observers seeing a future-version client's class fall back
/// to rendering as the default.
pub fn class_by_id(id: u16) -> Option<&'static CharacterClass> {
    ALL_CLASSES.iter().copied().find(|c| c.id == id)
}

impl CharacterClass {
    /// Look up the path for a clip label, or `None` if the class does
    /// not define this clip (which is allowed — a vehicle "character"
    /// might not have walk).
    pub fn clip_path(&self, label: ClipLabel) -> Option<&'static str> {
        self.clips
            .iter()
            .find(|c| c.label == label)
            .map(|c| c.path)
    }

    /// Look up the kind for a clip label.
    pub fn clip_kind(&self, label: ClipLabel) -> Option<ClipKind> {
        self.clips
            .iter()
            .find(|c| c.label == label)
            .map(|c| c.kind)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn human_default_has_expected_clips() {
        for label in ClipLabel::ALL {
            assert!(
                HUMAN_DEFAULT.clip_path(label).is_some(),
                "HUMAN_DEFAULT missing clip for {:?}",
                label
            );
        }
    }

    #[test]
    fn class_by_id_finds_human_default() {
        let c = class_by_id(HUMAN_DEFAULT.id).expect("HUMAN_DEFAULT registered");
        assert_eq!(c.name, "human_default");
    }

    #[test]
    fn class_by_id_returns_none_for_unknown() {
        assert!(class_by_id(0xFFFF).is_none());
    }

    #[test]
    fn clip_label_count_matches_all_array() {
        assert_eq!(ClipLabel::ALL.len(), ClipLabel::COUNT);
    }

    #[test]
    fn human_default_clip_paths_are_unique() {
        use std::collections::HashSet;
        let paths: HashSet<&str> = HUMAN_DEFAULT.clips.iter().map(|c| c.path).collect();
        assert_eq!(
            paths.len(),
            HUMAN_DEFAULT.clips.len(),
            "duplicate clip paths in HUMAN_DEFAULT"
        );
    }
}
