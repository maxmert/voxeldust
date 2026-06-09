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

/// The set of held movement keys this frame (idempotent — latest-wins on the wire).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MovementKeys {
    pub forward: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
}

impl MovementKeys {
    /// The `[forward, strafe, vertical]` axes in `[-1, 1]` — per-axis (no diagonal
    /// normalization; matches the server's per-axis clamp).
    #[must_use]
    pub fn axes(self) -> [f32; 3] {
        [
            axis(self.forward, self.back),
            axis(self.right, self.left),
            axis(self.up, self.down),
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
        assert_eq!(MovementKeys::default().axes(), [0.0, 0.0, 0.0]);
        let fwd = MovementKeys {
            forward: true,
            ..Default::default()
        };
        assert_eq!(fwd.axes(), [1.0, 0.0, 0.0]);
        let back = MovementKeys {
            back: true,
            ..Default::default()
        };
        assert_eq!(back.axes(), [-1.0, 0.0, 0.0]);
        // opposing keys cancel; strafe + vertical resolve on their own axes.
        let mixed = MovementKeys {
            forward: true,
            back: true,
            right: true,
            up: true,
            down: true,
            left: false,
        };
        assert_eq!(mixed.axes(), [0.0, 1.0, 0.0]);
        let strafe_left = MovementKeys {
            left: true,
            ..Default::default()
        };
        assert_eq!(strafe_left.axes(), [0.0, -1.0, 0.0]);
        let down = MovementKeys {
            down: true,
            ..Default::default()
        };
        assert_eq!(down.axes(), [0.0, 0.0, -1.0]);
    }

    #[test]
    fn move_action_wraps_the_axes() {
        let keys = MovementKeys {
            forward: true,
            right: true,
            ..Default::default()
        };
        assert_eq!(keys.move_action(), InputAction::Move([1.0, 1.0, 0.0]));
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
