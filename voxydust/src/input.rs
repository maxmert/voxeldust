//! Player input processing: key/mouse state to network input packets.

use tokio::sync::mpsc;
use winit::keyboard::KeyCode;

use voxeldust_core::client_message::PlayerInputData;

/// Build and send the player input packet based on current key state.
pub fn send_input_with_dt(
    input_tx: &Option<mpsc::UnboundedSender<PlayerInputData>>,
    keys_held: &std::collections::HashSet<KeyCode>,
    engines_off: bool,
    is_piloting: bool,
    camera_yaw: f64,
    camera_pitch: f64,
    pilot_yaw_rate: &mut f64,
    pilot_pitch_rate: &mut f64,
    selected_thrust_tier: &mut u8,
    frame_count: u64,
    dt: f64,
) {
    if let Some(tx) = input_tx {
        let mut movement = [0.0f32; 3];
        // Engine cutoff: zero all thrust inputs. Ship goes ballistic.
        if !engines_off {
            if keys_held.contains(&KeyCode::KeyW) { movement[2] += 1.0; }
            if keys_held.contains(&KeyCode::KeyS) { movement[2] -= 1.0; }
            if keys_held.contains(&KeyCode::KeyD) { movement[0] += 1.0; }
            if keys_held.contains(&KeyCode::KeyA) { movement[0] -= 1.0; }
        }
        let jump = keys_held.contains(&KeyCode::Space);
        let action = if engines_off { 5 } // engine cutoff signal
            else if keys_held.contains(&KeyCode::Enter) { 7 } // warp confirm
            else if keys_held.contains(&KeyCode::KeyG) { 6 } // warp target
            else if keys_held.contains(&KeyCode::KeyT) { 4 }
            else if keys_held.contains(&KeyCode::KeyE) { 3 }
            else { 0 };

        // Thrust tier selection (1-5 keys).
        if keys_held.contains(&KeyCode::Digit1) { *selected_thrust_tier = 0; }
        if keys_held.contains(&KeyCode::Digit2) { *selected_thrust_tier = 1; }
        if keys_held.contains(&KeyCode::Digit3) { *selected_thrust_tier = 2; }
        if keys_held.contains(&KeyCode::Digit4) { *selected_thrust_tier = 3; }
        if keys_held.contains(&KeyCode::Digit5) { *selected_thrust_tier = 4; }
        let free_look = keys_held.contains(&KeyCode::AltLeft);

        // When piloting: send yaw/pitch rate (-1 to 1) for ship torque.
        // When walking: send absolute camera yaw/pitch.
        let (look_yaw, look_pitch) = if is_piloting && !free_look {
            (*pilot_yaw_rate as f32, *pilot_pitch_rate as f32)
        } else {
            (camera_yaw as f32, camera_pitch as f32)
        };

        // Decay pilot rates toward zero (virtual spring centering).
        // Frame-rate independent: decay_rate = -ln(0.95) * 60 ~ 3.08
        // At 60fps: e^(-3.08/60) = 0.95 (matches original).
        // At 30fps: e^(-3.08/30) = 0.9025 (correct 2-frame equivalent).
        let decay = (-3.08 * dt).exp();
        *pilot_yaw_rate *= decay;
        *pilot_pitch_rate *= decay;

        let _ = tx.send(PlayerInputData {
            movement,
            look_yaw,
            look_pitch,
            jump,
            fly_toggle: false,
            orbit_stabilizer_toggle: false,
            speed_tier: *selected_thrust_tier,
            action,
            block_type: 0,
            tick: frame_count,
        });
    }
}
