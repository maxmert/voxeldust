//! Camera computation: position, forward, and up vectors from game state.

use glam::{DVec3, Mat4, Vec3};

use crate::gpu::EYE_HEIGHT;

/// Computed camera parameters for the current frame.
pub struct CameraParams {
    pub cam_pos: DVec3,
    pub cam_fwd: Vec3,
    pub cam_up: Vec3,
    pub view: Mat4,
    pub vp: Mat4,
    pub aspect: f32,
    pub fov_y: f32,
    pub frustum: crate::gpu::FrustumPlanes,
    /// Camera position in system-space (accounts for ship origin offset).
    pub cam_system_pos: DVec3,
}

/// Compute camera parameters from the current game state.
pub fn compute_camera(
    player_position: DVec3,
    camera_yaw: f64,
    camera_pitch: f64,
    is_piloting: bool,
    current_shard_type: u8,
    ship_rotation: glam::DQuat,
    keys_held: &std::collections::HashSet<winit::keyboard::KeyCode>,
    gpu_width: u32,
    gpu_height: u32,
    latest_world_state: Option<&voxeldust_core::client_message::WorldStateData>,
    // Interpolated ship origin for smooth cam_system_pos (None = use ws.origin).
    interpolated_ship_origin: Option<DVec3>,
) -> CameraParams {
    use winit::keyboard::KeyCode;

    let free_look = keys_held.contains(&KeyCode::AltLeft);

    // First-person camera. Player position is in ship-local coords when in a ship.
    // Rotate into world space for consistent rendering with the view matrix.
    let cam_pos = if is_piloting || current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
        // Ship: player_position is ship-local. Rotate by ship_rotation to get world-relative.
        let player_local = player_position + DVec3::new(0.0, EYE_HEIGHT, 0.0);
        ship_rotation * player_local
    } else if current_shard_type == 0 && player_position.length_squared() > 1.0 {
        // Planet: eye height along radial (outward from planet center).
        let radial = player_position.normalize();
        player_position + radial * EYE_HEIGHT
    } else {
        // System/fallback: Y-up.
        player_position + DVec3::new(0.0, EYE_HEIGHT, 0.0)
    };

    let cam_fwd = if is_piloting && !free_look {
        // Camera locked to ship heading.
        let fwd = ship_rotation * DVec3::NEG_Z;
        fwd.as_vec3().normalize()
    } else if current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
        // Walking inside ship: camera_yaw/pitch are ship-local.
        // Rotate local look direction by ship_rotation for world-space rendering.
        let (sy, cy) = (camera_yaw as f32).sin_cos();
        let (sp, cp) = (camera_pitch as f32).sin_cos();
        let local_fwd = DVec3::new((cy * cp) as f64, sp as f64, (sy * cp) as f64);
        (ship_rotation * local_fwd).as_vec3().normalize()
    } else if current_shard_type == 0 && player_position.length_squared() > 1.0 {
        // Planet: camera yaw/pitch in the local tangent plane.
        // Build tangent frame from player position (same algorithm as server).
        let up = player_position.normalize();
        let pole = DVec3::Y;
        let east_raw = pole.cross(up);
        let east = if east_raw.length_squared() > 1e-10 {
            east_raw.normalize()
        } else {
            DVec3::Z.cross(up).normalize()
        };
        let north = up.cross(east).normalize();
        let (sy, cy) = (camera_yaw as f32).sin_cos();
        let (sp, cp) = (camera_pitch as f32).sin_cos();
        // Forward in tangent plane: yaw=0 faces north, increasing yaw rotates clockwise (east).
        // Standard geographic convention: fwd = north*cos(yaw) + east*sin(yaw).
        let local_fwd = north * (cy * cp) as f64 + up * sp as f64 + east * (sy * cp) as f64;
        local_fwd.as_vec3().normalize()
    } else {
        // System/fallback: camera_yaw/pitch are world-space Y-up.
        let (sy, cy) = (camera_yaw as f32).sin_cos();
        let (sp, cp) = (camera_pitch as f32).sin_cos();
        Vec3::new(cy * cp, sp, sy * cp).normalize()
    };

    let aspect = gpu_width as f32 / gpu_height as f32;
    let proj = Mat4::perspective_infinite_reverse_rh(70.0_f32.to_radians(), aspect, 0.1);

    let cam_up = if current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
        // Inside ship: up follows ship rotation (artificial gravity floor).
        (ship_rotation * DVec3::Y).as_vec3().normalize()
    } else if current_shard_type == 0 && player_position.length_squared() > 1.0 {
        // Planet: up = radial direction from planet center.
        player_position.normalize().as_vec3()
    } else {
        Vec3::Y
    };

    let view_mat = Mat4::look_to_rh(Vec3::ZERO, cam_fwd, cam_up);
    let vp = proj * view_mat;
    let frustum = crate::gpu::FrustumPlanes::from_vp(&vp);

    // Camera position in system-space for celestial body rendering.
    // In ship shard, cam_pos is only the player's offset from ship center;
    // we must add the ship's system-space position (ws.origin) to get the
    // true camera position for correct body offsets.
    let cam_system_pos = if current_shard_type == voxeldust_core::client_message::shard_type::SHIP {
        // Use interpolated origin if available (smooth), otherwise fall back to WorldState.
        if let Some(origin) = interpolated_ship_origin {
            origin + cam_pos
        } else {
            latest_world_state.map_or(cam_pos, |ws| ws.origin + cam_pos)
        }
    } else {
        cam_pos
    };

    let fov_y = 70.0_f32.to_radians();
    CameraParams {
        cam_pos,
        cam_fwd,
        cam_up,
        view: view_mat,
        vp,
        aspect,
        fov_y,
        frustum,
        cam_system_pos,
    }
}
