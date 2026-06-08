//! The client input state — THE injection seam (HR6). Both the dev-control
//! listener (Slice 2) and the real keyboard/mouse (Slice 3) write it through these
//! SAME lib setters, so injected input is byte-identical to real input: every
//! transform (clamp, look-accumulation) lives HERE, never at the call site.
//! The 20 Hz assembler reads it into an `InputDatagram`.

/// Held movement + accumulated look delta + action bits. Movement is a held axis
/// (persists across frames); look is a per-frame delta consumed at each assemble.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct InputState {
    movement: [f32; 3],
    look: [f32; 2],
    action_bits: u32,
}

impl InputState {
    /// Set the held movement axes, clamped to `[-1, 1]`. The clamp is the transform
    /// that makes injected and real input identical — it lives in the setter.
    pub fn set_movement(&mut self, movement: [f32; 3]) {
        self.movement = [
            movement[0].clamp(-1.0, 1.0),
            movement[1].clamp(-1.0, 1.0),
            movement[2].clamp(-1.0, 1.0),
        ];
    }

    /// Accumulate a look delta `(yaw, pitch)` this frame; consumed at the next assemble.
    pub fn add_look(&mut self, delta: [f32; 2]) {
        self.look[0] += delta[0];
        self.look[1] += delta[1];
    }

    /// Set (`pressed`) or clear an action bit (jump/interact/...).
    pub fn set_action_bit(&mut self, bit: u32, pressed: bool) {
        if pressed {
            self.action_bits |= bit;
        } else {
            self.action_bits &= !bit;
        }
    }

    #[must_use]
    pub fn movement(&self) -> [f32; 3] {
        self.movement
    }

    #[must_use]
    pub fn action_bits(&self) -> u32 {
        self.action_bits
    }

    /// Take this frame's `(movement, look, action_bits)`, consuming the look delta
    /// (movement and action bits are held; look is per-frame).
    pub(crate) fn take_frame(&mut self) -> ([f32; 3], [f32; 2], u32) {
        let look = std::mem::take(&mut self.look);
        (self.movement, look, self.action_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn movement_is_clamped_in_the_setter() {
        let mut input = InputState::default();
        input.set_movement([2.0, -3.0, 0.5]);
        assert_eq!(input.movement(), [1.0, -1.0, 0.5]);
    }

    #[test]
    fn look_accumulates_and_is_consumed_per_frame() {
        let mut input = InputState::default();
        input.add_look([0.1, 0.2]);
        input.add_look([0.3, -0.1]);
        let (movement, look, action) = input.take_frame();
        assert_eq!(movement, [0.0, 0.0, 0.0]);
        // Split (not `a && b`): a short-circuited `&&` leaves the false arm uncovered.
        assert!((look[0] - 0.4).abs() < 1e-6);
        assert!((look[1] - 0.1).abs() < 1e-6);
        assert_eq!(action, 0);
        // Consumed: the next frame's look starts fresh; movement persists.
        input.set_movement([1.0, 0.0, 0.0]);
        let (movement, look, _) = input.take_frame();
        assert_eq!(movement, [1.0, 0.0, 0.0]);
        assert_eq!(look, [0.0, 0.0]);
    }

    #[test]
    fn action_bits_set_and_clear() {
        let mut input = InputState::default();
        input.set_action_bit(0b01, true);
        input.set_action_bit(0b10, true);
        assert_eq!(input.action_bits(), 0b11);
        input.set_action_bit(0b01, false);
        assert_eq!(input.action_bits(), 0b10);
    }
}
