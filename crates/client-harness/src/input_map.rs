//! Keyboard/mouse → [`InputAction`] mapping (pure). The render glue translates winit
//! events into these abstract inputs and feeds the result to the SAME mailbox the
//! dev-control listener uses, so a windowed keypress is byte-identical to an injected
//! command. CRITICAL (slice_3 plan §10): this maps PHYSICAL keys to RAW input only —
//! movement axes, a look delta, and a single-bit action MASK by index. It NEVER assigns
//! gameplay meaning (no "space → jump"); the bit→named-signal binding is SERVER-side
//! (the future Functional Panel).

use vd_devproto::InputAction;

/// Mouse-look sensitivity — radians of look per pixel of motion.
pub const LOOK_SENSITIVITY: f32 = 0.0025;

/// THROWAWAY (test instrument, owner-ordered 2026-08-20): the number of THROTTLE TIERS between a
/// standstill and the containing realm's own speed ceiling.
///
/// The server's throttle is GEOMETRIC — a commanded axis magnitude `t` in `[0, 1]` asks for
/// `v_foot · (v_cap/v_foot)^t`. So EVENLY spaced tiers give evenly spaced *ratios*, which is the only
/// spacing that can span walking pace and interstellar cruise on one control. Tier `n` commands a
/// magnitude of `n / TIERS`; tier 0 is a standstill and tier `TIERS` is the realm's ceiling — which is
/// its own width in three minutes, so full throttle always crosses whatever you are inside in the same
/// time, at every level.
///
/// This is a STAND-IN and it is scheduled for deletion: when the ship realm becomes the thing you fly,
/// the throttle stops being a keyboard tier and becomes a commanded force from a functional block. The
/// lesson it encodes is that ONE control must span both scales, not that a tier key exists.
pub const THROTTLE_TIERS: u8 = 10;

/// THROWAWAY (same instrument): how quickly the commanded magnitude eases toward the tier — the time
/// constant of the client-side ramp, in seconds.
///
/// The keyboard is a switch; a stick is not. Without this, a key press commands the ceiling instantly
/// and a release commands a standstill instantly, so the ship starts and stops dead. Easing the
/// COMMANDED magnitude turns the switch into a stick: the server sees a smoothly varying throttle and
/// needs no change at all. It is input shaping in the input device, never a fake in the world — and
/// the server's own ramp still bounds the acceleration underneath it.
pub const THROTTLE_EASE_TAU_S: f32 = 1.2;

/// The set of held movement keys this frame (idempotent — latest-wins on the wire).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MovementKeys {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    /// THROWAWAY: the eased commanded magnitude in `[0, 1]` — the selected tier, ramped by the render
    /// glue so a keypress reads like a stick rather than a switch. The axes already ride the wire as a
    /// magnitude and the server integrates `axes · speed` through its GEOMETRIC throttle, so this needs
    /// no new message and no server change: a smaller number simply asks for a slower ratio.
    pub throttle: f32,
}

impl MovementKeys {
    /// The `[forward, strafe, vertical]` axes in `[-1, 1]` — per-axis (no diagonal
    /// normalization; matches the server's per-axis clamp).
    #[must_use]
    pub fn axes(self) -> [f32; 3] {
        // The eased tier IS the magnitude the geometric throttle reads. Clamped here, once, so a
        // render-glue slip can never command more than the realm already allows.
        let scale = self.throttle.clamp(0.0, 1.0);
        [
            axis(self.forward, self.back) * scale,
            axis(self.right, self.left) * scale,
            axis(self.up, self.down) * scale,
        ]
    }

    /// The `Move` input for the currently held movement keys.
    #[must_use]
    pub fn move_action(self) -> InputAction {
        InputAction::Move(self.axes())
    }
}

/// `+1` if only the positive key is held, `-1` if only the negative, else `0` —
/// branchless (no short-circuit arms to leave uncovered).
fn axis(positive: bool, negative: bool) -> f32 {
    f32::from(positive) - f32::from(negative)
}

/// A raw mouse-motion delta (pixels) → a `Look` input, scaled once by [`LOOK_SENSITIVITY`].
#[must_use]
pub fn mouse_look(dx: f32, dy: f32) -> InputAction {
    InputAction::Look([dx * LOOK_SENSITIVITY, dy * LOOK_SENSITIVITY])
}

/// A physical action-key index + edge → an `Action` input, or `None` if `index` is
/// outside the `u32` action channel (`> vd_devproto::MAX_ACTION_INDEX`). The index→mask
/// transform is the SHARED [`vd_devproto::action_bit`] — the SAME bound `vdctl` applies,
/// so the two input-injection paths can't diverge. The bit→meaning binding is
/// server-side, never here.
#[must_use]
pub fn action(index: u32, pressed: bool) -> Option<InputAction> {
    vd_devproto::action_bit(index).map(|bit| InputAction::Action { bit, pressed })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axes_map_each_held_direction_per_axis() {
        // Full throttle throughout, so DIRECTION is asserted without the tier scaling every number; the
        // tier magnitude is pinned separately below.
        let held = |f: fn(&mut MovementKeys)| {
            let mut k = MovementKeys {
                throttle: 1.0,
                ..Default::default()
            };
            f(&mut k);
            k.axes()
        };
        assert_eq!(MovementKeys::default().axes(), [0.0, 0.0, 0.0]);
        assert_eq!(held(|k| k.forward = true), [1.0, 0.0, 0.0]);
        assert_eq!(held(|k| k.back = true), [-1.0, 0.0, 0.0]);
        assert_eq!(held(|k| k.left = true), [0.0, -1.0, 0.0]);
        assert_eq!(held(|k| k.down = true), [0.0, 0.0, -1.0]);
        // opposing keys cancel; strafe + vertical resolve on their own axes.
        let mixed = MovementKeys {
            forward: true,
            back: true,
            right: true,
            up: true,
            down: true,
            left: false,
            throttle: 1.0,
        };
        assert_eq!(mixed.axes(), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn the_tier_scales_every_axis_and_clamps_out_of_range() {
        // THE ONE MAGNITUDE. The axes ride the wire as a magnitude and the server integrates
        // `axes · speed` through its GEOMETRIC throttle, so a tier slows you without a new message, a
        // new field, or a server change. Pinned on every axis because a throttle that only applied to
        // forward would be a trap when manoeuvring.
        let all = |throttle: f32| {
            MovementKeys {
                forward: true,
                right: true,
                up: true,
                throttle,
                ..Default::default()
            }
            .axes()
        };
        assert_eq!(all(1.0), [1.0, 1.0, 1.0]);
        assert_eq!(all(0.5), [0.5, 0.5, 0.5]);
        assert_eq!(all(0.0), [0.0, 0.0, 0.0]);
        // Out of range is CLAMPED here, once, so a render-glue slip can never command more than the
        // realm allows — nor invert a direction with a negative.
        assert_eq!(all(4.0), [1.0, 1.0, 1.0]);
        assert_eq!(all(-1.0), [0.0, 0.0, 0.0]);
        // …and stationary is stationary at any tier — the throttle scales movement, never invents it.
        assert_eq!(
            MovementKeys {
                throttle: 1.0,
                ..Default::default()
            }
            .axes(),
            [0.0; 3]
        );
    }

    #[test]
    fn move_action_wraps_the_axes() {
        let keys = MovementKeys {
            forward: true,
            right: true,
            ..Default::default()
        };
        // A half tier is half the magnitude; the axes carry the throttle as their magnitude.
        let half = MovementKeys {
            throttle: 0.5,
            ..keys
        };
        assert_eq!(half.move_action(), InputAction::Move([0.5, 0.5, 0.0]));
        let fast = MovementKeys {
            throttle: 1.0,
            ..keys
        };
        assert_eq!(fast.move_action(), InputAction::Move([1.0, 1.0, 0.0]));
    }

    #[test]
    fn mouse_look_scales_by_sensitivity() {
        assert_eq!(
            mouse_look(4.0, -2.0),
            InputAction::Look([4.0 * LOOK_SENSITIVITY, -2.0 * LOOK_SENSITIVITY])
        );
    }

    #[test]
    fn action_is_a_single_bit_mask_by_index() {
        assert_eq!(
            action(3, true),
            Some(InputAction::Action {
                bit: 0b1000,
                pressed: true
            })
        );
        assert_eq!(
            action(0, false),
            Some(InputAction::Action {
                bit: 1,
                pressed: false
            })
        );
        // Out of the u32 action channel → None (same bound vdctl enforces).
        assert_eq!(action(vd_devproto::MAX_ACTION_INDEX + 1, true), None);
    }
}
