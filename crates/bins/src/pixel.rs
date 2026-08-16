//! THE PIXEL-GATE INSTRUMENT (Tier-B process glue; window lane Slice D — `docs/design/window_lane.md`
//! §2.8/§2.11). Everything the Slice-D acceptance gates need in order to say, of a REAL captured
//! frame, *which* realm's pixels are *where* and *who authored them* — with no full-frame search
//! and no fitted number anywhere.
//!
//! Three pieces:
//!
//! 1. [`straddle`] — the capture discipline. A screenshot is bracketed by two `DevState` polls and
//!    retried until they AGREE, so the reconstructed camera and the reported subject positions
//!    describe the same world the readback holds. The tolerance is the subject's own apparent
//!    radius (never a fitted pixel count): drift below the thing's own footprint cannot carry it
//!    out of its bracketed rectangle. This is the C1/C2 lesson made reusable — a capture-geometry
//!    defect there masqueraded as a flake for two slices.
//! 2. [`subject`] — the drawn-footprint reading. For each named realm it reports WHICH lawful
//!    author drew it (`look` vs `marker`), where its centre projects, and how many pixels of radius
//!    it covers — a LOOK by its streamed extent, a MARKER through the very same Tier-A pair the
//!    renderer scales its point sprite by (`vd_client::realm_scene::marker_look` →
//!    `vd_client_harness::camera::marker_world_radius`), so the asserted rectangle and the drawn
//!    footprint cannot disagree.
//! 3. [`Presence`] — THE PRESENCE LAW as one verdict: exactly one of {marker, body} per realm,
//!    never zero, never both. It is unrepresentable for a row to be both (the wire types make a
//!    third pixel source impossible), so what this actually measures is the ZERO side and the
//!    identity of the author.
//!
//! PNG decoding and the pixel verdicts themselves stay in the gate files (`image` is a
//! dev-dependency); this module is the state/camera/geometry half that all three gates share.

use std::time::Duration;

use vd_client::realm_scene::marker_look;
use vd_client_harness::camera::{
    CaptureCamera, DOT_MIN_APPARENT_RADIUS_PX, ScreenAabb, marker_world_radius,
    pilot_capture_camera,
};
use vd_client_harness::verdict::projected_point_aabb;
use vd_core::glam::{DQuat, DVec3};
use vd_core::pose::RealmId;
use vd_devproto::{DevRequest, DevResponse, DevState};

/// How many straddle attempts before a capture is declared un-gettable. Generous: each retry costs
/// one screenshot round trip, and a WEDGED feed (not a slow one) is the only thing that exhausts it.
const STRADDLE_TRIES: u32 = 40;
/// The pause between straddle attempts — one client step at the shipped rate is far shorter, so
/// this simply lets a moving world settle rather than spinning the listener.
const STRADDLE_PAUSE: Duration = Duration::from_millis(200);

/// WHICH LAWFUL AUTHOR drew a realm's pixels this frame — the client's own reported body kind,
/// parsed once. There is deliberately no third variant: the wire types cannot represent one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Author {
    /// The realm's OWN self-authored outline (`TAG_LOOK`) — a running realm draws itself.
    SelfLook,
    /// Its parent's photometric point-of-light datum (`TAG_LUMA`) — a sleeping realm is a marker.
    ParentMarker,
}

/// THE PRESENCE LAW's verdict for one realm in one frame: exactly one of {marker, body} is drawn —
/// never zero, never both.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Presence {
    /// Drawn, by the named author.
    Drawn(Author),
    /// NOT DRAWN — the realm has no row in the composed picture at all. Lawful only for a realm the
    /// composer never stated; a violation for one the gate is watching.
    Absent,
}

/// One watched realm's drawn footprint in one captured frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Subject {
    /// Which author drew it (or that nothing did).
    pub presence: Presence,
    /// Its drawn centre, in the composed picture's origin frame (the client's own report).
    pub centre_m: DVec3,
    /// Its projected screen rectangle, `None` when it falls behind the eye.
    pub rect: Option<ScreenAabb>,
    /// Its drawn footprint RADIUS in pixels (half the projected rectangle's width); `0.0` when the
    /// realm is not drawn or projects behind the eye. THE quantity the warp gate's growth curve and
    /// the handover gate's monotonicity assert on.
    pub radius_px: f64,
    /// The universe tick this realm's pose feed last delivered — the G-SHEAR pixel half's input
    /// (every FRESH drawn row of one composed frame carries one tick).
    pub newest_tick: Option<u64>,
}

impl Subject {
    /// The projected centre, `None` behind the eye.
    #[must_use]
    pub fn centre_px(&self) -> Option<(f64, f64)> {
        self.rect.map(|r| {
            (
                f64::midpoint(r.min.x, r.max.x),
                f64::midpoint(r.min.y, r.max.y),
            )
        })
    }
}

/// One straddled capture: the two agreeing state polls, the camera the renderer actually used, and
/// the written PNG's run-relative path.
#[derive(Clone, Debug)]
pub struct Straddled {
    /// The state poll taken BEFORE the screenshot.
    pub pre: DevState,
    /// The state poll taken AFTER it — agreeing with `pre` on the watched subjects' projections.
    pub post: DevState,
    /// The camera reconstructed from `post` — exactly the one the renderer used (the SAME Tier-A
    /// expression, fed the same delivered state).
    pub camera: CaptureCamera,
    /// The screenshot path the client reported (cwd-relative).
    pub shot: String,
    /// The measured straddle drift, in pixels, of the watched subject that moved most — printed by
    /// the gates so a regression is diagnosable rather than a mystery.
    pub drift_px: f64,
}

/// The own entity's delivered pose `(position, orientation)` — the pilot camera's whole basis.
/// `None` before the avatar has a delivered row.
#[must_use]
pub fn own_pose(state: &DevState) -> Option<(DVec3, DQuat)> {
    let own = state.own_entity.as_deref()?;
    let row = state.entities.iter().find(|r| r.entity == own)?;
    let [x, y, z, w] = row.orient;
    Some((
        DVec3::from_array(row.pos),
        DQuat::from_xyzw(x, y, z, w).normalize(),
    ))
}

/// The PILOT capture camera for a sampled state — the ONE Tier-A expression the renderer places its
/// capture camera by, fed the same delivered pose the client drew from.
///
/// # Panics
/// If the avatar has no delivered pose yet (a broken gate precondition — every pixel gate waits for
/// its login to land first).
#[must_use]
pub fn pilot_camera(state: &DevState, width: usize, height: usize) -> CaptureCamera {
    let (pos, orient) = own_pose(state).expect("the avatar has a delivered pose to look from");
    pilot_capture_camera(pos, orient, width, height)
}

/// Read one watched realm's drawn footprint out of a sampled state, through `camera`.
///
/// A LOOK is sized by its STREAMED extent (the composed row's own outline). A MARKER is sized by
/// the parent's photometric datum through `marker_look` and then the SHARED apparent-size floor —
/// the identical two calls the renderer makes to scale the point sprite, so this rectangle is the
/// drawn footprint rather than a guess at it.
#[must_use]
pub fn subject(state: &DevState, camera: &CaptureCamera, realm: RealmId) -> Subject {
    let label = format!("{realm:?}");
    let Some(row) = state.realm_boxes.iter().find(|b| b.realm == label) else {
        return Subject {
            presence: Presence::Absent,
            centre_m: DVec3::ZERO,
            rect: None,
            radius_px: 0.0,
            newest_tick: None,
        };
    };
    let centre_m = DVec3::from_array(row.center);
    let (presence, world_radius) = match row.body_kind.as_str() {
        "look" => (Presence::Drawn(Author::SelfLook), row.extent_m),
        "marker" => {
            let (class_code, luma_lsun) = row.luma.unwrap_or_default();
            let base = marker_look(class_code, luma_lsun).base_radius_m;
            let dist = (centre_m - camera.eye).length();
            (
                Presence::Drawn(Author::ParentMarker),
                marker_world_radius(base, dist, camera.fov_y, camera.height as f64),
            )
        }
        // The client reports exactly the two lawful arms; anything else is a corrupt surface, not
        // a third pixel source — treated as not-drawn so the presence assert fires loudly.
        _ => (Presence::Absent, 0.0),
    };
    let rect = projected_point_aabb(camera, centre_m, world_radius);
    Subject {
        presence,
        centre_m,
        rect,
        radius_px: rect.map_or(0.0, |r| (r.max.x - r.min.x) * 0.5),
        newest_tick: row.newest_tick,
    }
}

/// STRADDLE one screenshot with two state polls and retry until they agree on the watched subjects'
/// projections AND on the drawn roster, then return both polls plus the reconstructed camera.
///
/// `camera_of` is the camera the RENDERER used, rebuilt from the same delivered state: either
/// [`pilot_camera`] (the `--capture-pilot` runs) or `crate::scene_camera::live_scene_camera` (the
/// default scene-fitting framing). The gate must pass the one its client was launched with — two
/// derivations of the camera is exactly how a rectangle and its pixels would drift apart.
///
/// THE AGREEMENT TOLERANCE IS THE SUBJECT'S OWN FOOTPRINT (never a fitted pixel count): a drift
/// smaller than the thing's own apparent radius cannot carry it out of the rectangle the gate
/// brackets it with, and the floor is the shared minimum apparent radius every point of light is
/// drawn at. A capture taken while the picture moved under it is the defect class that cost Slices
/// C1 and C2 real debugging time; this is that cure, generalized.
///
/// # Panics
/// If the client's dev-control listener stops answering, if the screenshot fails, or if the polls
/// never agree within [`STRADDLE_TRIES`] — each a real fault (a wedged feed), never a soft skip.
pub fn straddle(
    devctl_port: u16,
    label: &str,
    width: usize,
    height: usize,
    watch: &[RealmId],
    camera_of: fn(&DevState, usize, usize) -> CaptureCamera,
) -> Straddled {
    let mut tries = 0u32;
    loop {
        let pre = poll(devctl_port);
        let shot = screenshot(devctl_port, label);
        let post = poll(devctl_port);
        let cam_pre = camera_of(&pre, width, height);
        let cam_post = camera_of(&post, width, height);
        let roster_pre: Vec<String> = pre.realm_boxes.iter().map(|b| b.realm.clone()).collect();
        let roster_post: Vec<String> = post.realm_boxes.iter().map(|b| b.realm.clone()).collect();
        let mut drift_px = 0.0_f64;
        let mut settled = roster_pre == roster_post;
        for r in watch {
            let a = subject(&pre, &cam_pre, *r);
            let b = subject(&post, &cam_post, *r);
            settled &= a.presence == b.presence;
            if let (Some((ax, ay)), Some((bx, by))) = (a.centre_px(), b.centre_px()) {
                let d = (bx - ax).hypot(by - ay);
                drift_px = drift_px.max(d);
                settled &= d <= a.radius_px.max(DOT_MIN_APPARENT_RADIUS_PX);
            }
        }
        if settled {
            return Straddled {
                pre,
                post,
                camera: cam_post,
                shot,
                drift_px,
            };
        }
        tries += 1;
        assert!(
            tries < STRADDLE_TRIES,
            "capture '{label}': the picture never held still across a screenshot \
             ({tries} tries, worst drift {drift_px:.2} px, rosters {roster_pre:?} vs \
             {roster_post:?}) — the feed is wedged or the subject is moving faster than its own \
             footprint per poll",
        );
        std::thread::sleep(STRADDLE_PAUSE);
    }
}

/// One `DevState` poll, panicking on a dead listener (a gate precondition, never a soft skip).
///
/// # Panics
/// If the listener does not answer with a state.
#[must_use]
pub fn poll(devctl_port: u16) -> DevState {
    match crate::dev_roundtrip(devctl_port, &DevRequest::State) {
        Ok(DevResponse::State { state }) => state,
        other => panic!("dev-control State failed: {other:?}"),
    }
}

/// Ask the client for one screenshot and return its reported path.
///
/// # Panics
/// If the capture fails (no GPU adapter, or a wedged render thread).
#[must_use]
pub fn screenshot(devctl_port: u16, label: &str) -> String {
    match crate::dev_roundtrip(
        devctl_port,
        &DevRequest::Screenshot {
            at_tick: None,
            label: Some(label.to_owned()),
        },
    ) {
        Ok(DevResponse::Captured { path, .. }) => path,
        other => panic!("screenshot '{label}' failed: {other:?}"),
    }
}
