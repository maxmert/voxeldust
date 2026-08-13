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

/// THROWAWAY (tiny world): the fraction of full speed an un-boosted move commands.
///
/// Two scales cannot share one speed. Neighbouring stars sit kilometres apart while a star system is
/// ~150 m across, so a speed that crosses interstellar space in seconds crosses a whole system in two —
/// you arrive somewhere and blast out the far side before it resolves around you. Holding the boost key
/// gives full speed for the crossing; releasing it gives this fraction for manoeuvring once there.
///
/// It is a stand-in for the real thing, which is a THROTTLE the player controls continuously and a warp
/// that DECELERATES on approach. Nothing here belongs in the final game; the real lesson it encodes is
/// that arrival needs its own phase, not that a magic key exists.
const CRUISE_FRACTION: f32 = 0.03;

/// The set of held movement keys this frame (idempotent — latest-wins on the wire).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MovementKeys {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    /// THROWAWAY: full speed while held, [`CRUISE_FRACTION`] otherwise. The axes already ride the wire
    /// as a magnitude in `[-1, 1]` and the server integrates `axes · speed`, so throttling needs no new
    /// message and no server change — a smaller number simply moves you slower.
    pub boost: bool,
}

impl MovementKeys {
    /// The `[forward, strafe, vertical]` axes in `[-1, 1]` — per-axis (no diagonal
    /// normalization; matches the server's per-axis clamp).
    #[must_use]
    pub fn axes(self) -> [f32; 3] {
        // Branchless scale: `f32::from(bool)` picks the multiplier with no arm to leave uncovered.
        let scale = CRUISE_FRACTION + (1.0 - CRUISE_FRACTION) * f32::from(self.boost);
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
        // Boosted throughout, so DIRECTION is asserted without the throttle scaling every number; the
        // un-boosted magnitude is pinned separately below.
        let held = |f: fn(&mut MovementKeys)| {
            let mut k = MovementKeys {
                boost: true,
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
            boost: true,
        };
        assert_eq!(mixed.axes(), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn cruise_is_a_fraction_of_boosted_speed_on_every_axis() {
        // THE TWO SCALES. The axes ride the wire as a MAGNITUDE and the server integrates `axes · speed`,
        // so releasing boost slows you without a new message, a new field, or a server change. Pinned on
        // every axis because a throttle that only applied to forward would be a trap when manoeuvring.
        let all = |boost: bool| {
            MovementKeys {
                forward: true,
                right: true,
                up: true,
                boost,
                ..Default::default()
            }
            .axes()
        };
        assert_eq!(all(true), [1.0, 1.0, 1.0]);
        assert_eq!(all(false), [CRUISE_FRACTION; 3]);
        // …and stationary is stationary at either throttle — cruise scales movement, never invents it.
        assert_eq!(
            MovementKeys {
                boost: true,
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
        // Un-boosted is CRUISE speed; the axes carry the throttle as their magnitude.
        assert_eq!(
            keys.move_action(),
            InputAction::Move([CRUISE_FRACTION, CRUISE_FRACTION, 0.0])
        );
        let fast = MovementKeys {
            boost: true,
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
