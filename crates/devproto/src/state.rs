//! The `DevState` diagnosis substrate (HR6): the client's DECODED DELIVERED world
//! view + lifecycle + honesty counters, as a pure serde type SHARED by the client
//! (which builds it) and `vdctl` (which decodes it). The composited render rows are
//! exactly what pixels would show; the counters distinguish never-welcomed /
//! starved / live without exposing interpolation internals (P2 reshapes those).

use serde::{Deserialize, Serialize};

/// The client lifecycle phase (mirrors the client's `ClientPhase`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DevPhase {
    Connecting,
    AwaitingWelcome,
    AwaitingSubscription,
    Active,
    Closed,
    /// The gateway refused this build's world identity (slice 7): terminal.
    WorldRefused,
}

/// One rendered entity: its id (canonical `Display`) and composited world pose.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevEntityRow {
    pub entity: String,
    pub pos: [f64; 3],
    /// Composited world orientation (`x,y,z,w` — glam `DQuat` component order). Always
    /// finite (sanitized like `pos`); EVERY row carries it (no per-row `is_own` flag), so
    /// the look-at closed loop reads the OWN row's orient as the current facing. Through
    /// P3 this is a full world-space rotation (`orient * -Z` = world-forward), so the
    /// closed-loop nav needs no separate "up".
    pub orient: [f64; 4],
    /// INERT since the pure-renderer collapse (S6): a node-AGNOSTIC client no longer has a
    /// per-entity authoritative sub (it renders every entity latest-wins by `EntityId`, never
    /// learning which node owns it). Retained as a stable diagnostic field (always the inert
    /// `vd_client::view::RENDERED_SUB` = 0) so `vdctl`/process-parity decode the row unchanged.
    pub authoritative_sub: u32,
}

/// One DRAWN realm box (VU diagnosis): its realm id (canonical `Debug`) and the center it would
/// render at (its `RealmBox` centre flattened to metres through the one `draw_center` chokepoint —
/// there is nothing to reduce it against, because the shard chain already measured it from the realm
/// this session stands in, the same space [`DevEntityRow::pos`] is in, so a test may compare the two
/// directly). This is
/// the render-plane twin of [`DevEntityRow`]: it exposes WHICH realms the client is
/// drawing and WHERE, so a headless test can prove the streamed scene is present AND
/// moving (a planet whose center changes across ticks is orbiting; a static container
/// shell never moves). Always finite (sanitized).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevRealmBox {
    pub realm: String,
    /// The row's HIERARCHY PARENT as delivered on the composed scene row (`SceneRow.parent`,
    /// identity only — same `{realm:?}` rendering as `realm`; `None` at the root). The
    /// G-PARENT-TRUE process gate (look_horizon.md slice 0) reads THIS: every planet row's
    /// parent must be its star system, never the galaxy — the composed row now carries the
    /// parent explicitly from the fold that knows it.
    pub parent: Option<String>,
    pub center: [f64; 3],
    /// The row's facing (x, y, z, w) as the composer delivered it — the rotation between the
    /// realm's own frame and the picture's. A gate that reads a point in the realm's frame (the
    /// picture gate's eye over the recipe, slice 8p) rotates by THIS; it used to assume the identity,
    /// which a spinning planet breaks silently (the refuter's finding 5).
    pub facing: [f64; 4],
    /// The drawn EXTENT in metres (a sphere's radius; a box's half-diagonal length; 0 for a
    /// MARKER point) — read STREAMED, straight off the composed row's look bag (window_lane.md
    /// §2.11: camera reconstruction and harness verdicts read streamed extents; the regions.json
    /// join is dead with the boot file).
    pub extent_m: f64,
    /// WHICH LAWFUL AUTHOR drew this box (THE DRAW LAW, owner decision 10): `"look"` — the
    /// realm's own self-authored outline; `"marker"` — its parent's photometric point-of-light
    /// datum. The pixel gates' body-kind assert (§2.11) reads this.
    pub body_kind: String,
    /// THE PHOTOMETRIC DATUM `(class_code, luma_lsun)`. On a MARKER body: the parent's stated
    /// point-of-light datum. On a LOOK body: usually `None` — but a RUNNING star's OWN look bag
    /// lawfully carries `TAG_LUMA` beside `TAG_LOOK` (THE STAR-LOOK EXTENSION SEAM, owner ruling
    /// 2026-08-19: future star params are future tags, skip-unknown free), so a look row may
    /// state its own light.
    /// The pixel gates size a point sprite's projected rectangle from exactly this, through the
    /// SAME Tier-A `marker_look` + `marker_world_radius` pair the renderer scales the sprite by
    /// (window lane Slice D), so the drawn footprint and the asserted rectangle cannot disagree.
    pub luma: Option<(u8, f64)>,
    /// SHAKE DIAGNOSIS — the newest universe tick this realm's pose feed has delivered; `None` for a
    /// box the feed never streamed (it is sitting at its boot placement). Read against
    /// [`DevState::entity_feed_newest_tick`], this is what distinguishes the two candidate causes of a
    /// wobbling horizon: a box whose tick TRACKS the entity feed is being drawn from the same moment
    /// as the player, so any residual wobble is a render-path fault; a box whose tick drifts against
    /// it is being authored by a shard whose sense of universe time is running independently. It also
    /// separates a FROZEN box (tick stops advancing) from a live one, so a smoothness gate cannot be
    /// satisfied by a box that simply stopped updating.
    pub newest_tick: Option<u64>,
}

/// SLICE 6 S5 — how one feed's tracks classified at the render cursor. The shake was invisible for a
/// long time because nothing reported that the interpolation machinery, though correct, never ran: a
/// feed permanently on `clamped_old` means the cursor is falling behind the retained history, which IS
/// that condition. A healthy live feed reads mostly `blended`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DevWindowCensus {
    /// Tracks genuinely interpolating (the cursor sits between two delivered poses).
    pub blended: u32,
    /// Tracks whose cursor precedes their whole history — clamped to the oldest pose.
    pub clamped_old: u32,
    /// Tracks at or past their newest pose — FROZEN (required; the client never coasts).
    pub clamped_new: u32,
}

/// The (P2) transfer view — empty-but-present so P2 transfer diagnosis is purely
/// additive (a new variant), never a reshape of this type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DevTransferView {
    /// No transfer in flight (the only P1.5 state).
    None,
}

/// The decoded, delivered client state — wire truth, the agent's diagnosis surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevState {
    pub phase: DevPhase,
    /// Canonical `Display` of the session id (a string — JSON cannot hold a u128).
    pub session: Option<String>,
    /// Canonical `Display` of this client's own entity (from `ServerControlMsg::OwnEntity` — the
    /// node-agnostic own-avatar signal; the client never learns which node owns it).
    pub own_entity: Option<String>,
    /// The player's LOCATION — a human-readable realm label (e.g. "System 7", later
    /// named planets/ships) derived from the authoritative FrameRef of the own entity,
    /// NOT a raw shard id (the client never sees shard processes). `None` until the own
    /// entity has a delivered pose. Fence-validated; changes only on a real cross-realm
    /// move. The player-stats-HUD source + a P2 transfer-test hook.
    pub location: Option<String>,
    /// The continuous render cursor (universe-tick units); `None` before the first
    /// snapshot. Always finite (the builder sanitizes), so this state JSON-encodes.
    pub render_cursor: Option<f64>,
    /// The freshest APPLIED universe tick (integer, run-stable + join-independent);
    /// `None` before the first snapshot. This — NOT the session-relative
    /// `snapshots_applied` count — is what `screenshot --at-tick` aligns on, so a
    /// capture lands on the same world state across runs.
    pub universe_tick: Option<u64>,
    /// SHAKE DIAGNOSIS — the newest universe tick the ENTITY pose feed has delivered (across all
    /// tracks). `None` before the first snapshot.
    ///
    /// WHY BOTH FEEDS' TICKS ARE REPORTED SEPARATELY. A player standing on a planet sees two things
    /// that must agree: their own body, composed by the shard that owns the realm they are in, and the
    /// ground under them, whose placement is authored by that realm's PARENT and relayed down. Each
    /// shard advances its own sense of universe time only when a clock sync ARRIVES from the
    /// orchestrator — there is no local per-tick advance — so the two numbers can drift apart, and the
    /// relative motion that drift produces is visible as a wobbling horizon. Reporting both, next to
    /// [`DevState::render_cursor`], is what tells a wobble caused by the render path apart from one
    /// caused by two clocks. Measure before building: the fix spans four crates on a ~50-minute build.
    pub entity_feed_newest_tick: Option<u64>,
    /// SHAKE DIAGNOSIS — the newest universe tick the REALM pose feed has delivered (across all
    /// streamed realms). `None` before the first realm frame. See
    /// [`DevState::entity_feed_newest_tick`] for why this is reported alongside it.
    pub realm_feed_newest_tick: Option<u64>,
    /// The composited render rows — each entity once, exactly what pixels show.
    pub entities: Vec<DevEntityRow>,
    /// SLICE 6 S5 — how the ENTITY feed's tracks classified at the reported cursor.
    pub entity_windows: DevWindowCensus,
    /// SLICE 6 S5 — how the REALM feed's placements classified at the same cursor.
    pub realm_windows: DevWindowCensus,
    /// SLICE 6 S5 — the two feeds' newest ticks, differenced (entity minus realm). `None` until both
    /// have delivered. This is the arrival skew whose interaction with un-cursored drawing WAS the
    /// shake; after the fix it should be small and, more importantly, harmless — both feeds are read
    /// at one cursor regardless.
    pub feed_skew_ticks: Option<i64>,
    /// The DRAWN realm boxes — each realm the client is currently rendering, with its
    /// composited center, streamed extent and body kind. Proves the composed scene is present
    /// (a box appears once its level/delta lands) and LIVE (its center moves as the composed
    /// pose feed overlays the orbit).
    pub realm_boxes: Vec<DevRealmBox>,
    /// THE ORIGIN MARKER (window_lane.md §2.7/§2.11): the realm the current scene is composed
    /// in (canonical `Debug` label) and its epoch — `(origin, origin_epoch)` as the last level
    /// stated them. `None` before the first level. The pixel gates read this: origin == home
    /// realm at login; the epoch bumps EXACTLY once per crossing.
    pub origin: Option<(String, u64)>,
    /// THE SKY THE CLIENT HOLDS (S11): `(generation, stars)` of the whole catalogue it has assembled
    /// and proved, or `None` while it holds no whole sky.
    ///
    /// ★ THE COUNT IS THE CATALOGUE'S, NOT THE DRAWN CLOUD'S. Reporting the held number keeps the
    /// client honest about what it RECEIVED. What it DREW is [`DevState::stars_drawn`], and the two
    /// disagree exactly when the sky is broken — which is the reason the second field exists.
    pub sky: Option<(u64, u64)>,
    /// ★ THE STARS ON SCREEN (owner ruling 2026-09-02 R9 step 1): how many points of light the
    /// renderer's star cloud holds RIGHT NOW. Zero while nothing is drawn.
    ///
    /// MEASURED 2026-09-01: a player stood inside a hull under a black sky while `sky` reported the
    /// whole census, because `sky` counts what the client HOLDS and the cloud had been despawned by a
    /// crossing and never rebuilt. An instrument that reports the held count certifies its own
    /// silence. This field is what a gate must read to say "the sky is drawn".
    ///
    /// Written by the render thread, read by the core thread — one shared counter, like
    /// `dev_commands_dropped`. A client with no renderer reports 0, truthfully.
    pub stars_drawn: u64,
    /// ★ THE TERRAIN ON SCREEN (slice 7): how many terrain chunks the renderer draws right now. Written
    /// by the render thread like `stars_drawn`; 0 with no renderer or no rung flag. The one instrument
    /// a picture gate waits on before it captures a hill.
    pub terrain_chunks_drawn: u64,
    /// The terrain chunks still BUILDING (slice 7): the instrument a picture gate waits on for "the
    /// whole wanted set landed" — `terrain_chunks_drawn ge 1` says the first one did, this says
    /// the last one did. 0 with no renderer or no rung flag.
    pub terrain_chunks_pending: u64,
    /// ★ THE CAMERA-MODE INSTRUMENT (2026-09-06): which view the renderer holds right now —
    /// `"first-person"`, `"third-person"`, or `"none"` in a client with no renderer. Recorded so a
    /// flight can poll it through a hand-over and confirm or refuse a seen flip; the owner reported
    /// one on the twenty-ninth flight and nothing recorded it. What the player SEES, named.
    pub camera_mode: String,
    /// ★ THE STAR PROBE (2026-09-06): the renderer's OWN projection of its brightest drawn stars —
    /// realm, screen x/y through the live camera and projection, and the star law's amplitude. A
    /// sky gate compares its expectation against THIS (the math) and THIS against the pixels (the
    /// raster), so a disagreement is split in one run instead of argued. Empty with no renderer.
    pub star_probe: Vec<DevStarProbe>,
    /// ★ THE TERRAIN STAMP (slice 8p, ruling V14 D8-7): the measured facts the renderer writes on
    /// every picture that draws ground — altitude over the recipe's surface, horizon, radius drawn,
    /// rungs and chunk counts, the star's angles, the biome, the world identity, the tick, and the
    /// ruler it planted. Written by the render thread like `star_probe`; `None` with no renderer,
    /// no rung flag, or no body under the eye. The picture gate recomputes altitude and horizon
    /// from THIS state's own pose and asserts they agree (M8-4).
    pub terrain_stamp: Option<DevTerrainStamp>,
    /// THE CAPTURE FRAME (slice 8 step 6): the renderer's own frame index of the captured
    /// picture this dump sits beside — `None` for a plain poll. Two frames of a recorded pair
    /// state indices one apart.
    #[serde(default)]
    pub capture_frame: Option<u64>,
    /// ★ WHERE THE GALAXY IS (owner ruling 2026-09-02 R1): the origin realm's centre in the galaxy's
    /// frame, in metres, as the gateway last stated it on the realm lane — the anchor the star cloud
    /// is placed by. `None` until the observer chain reaches the galaxy, which is exactly the case
    /// in which no sky is drawn; a gate reads this to tell "the chain never reached the galaxy"
    /// from "the renderer never drew".
    pub sky_anchor: Option<[f64; 3]>,
    /// IS ANYONE STILL SPEAKING FOR THE SKY (S11)? `NeverHeard`, `Confirmed` or `Quiet`, as
    /// [`vd_client::star_sky::sky_watch`] reads it.
    ///
    /// ★ THE STARS STAY ON SCREEN IN ALL THREE. A stale sky is not a wrong sky — stars do not move, so
    /// a quiet lane leaves every one of them exactly where it was. SL1 clause 6 requires a consumer
    /// past its bound to "degrade and SAY SO", and for a picture that cannot go stale, saying so IS the
    /// degradation. Hiding the stars would invent a seam (SL8 names `sprite-cull` and `flicker`) to
    /// honour a law about readings that go wrong, which this one does not.
    ///
    /// ⚠ IT DETECTS ABSENCE, NOT WRONGNESS. A server beating a frozen catalogue reads `Confirmed`, and
    /// must — the gateway folds the sky once at boot, so an unchanging generation is the normal case.
    pub sky_watch: String,
    /// FAULT/diagnosis: composed datagram rows dropped for carrying a PREVIOUS scene epoch
    /// (§2.6.6 `stale_epoch_rows`). Brief at a crossing; steady growth means the feed and the
    /// reliable lane disagree about the current scene.
    pub stale_epoch_rows: u64,
    // The honesty counters, classified so an agent reads them right: a nonzero FAULT
    // counter is a real problem; THROUGHPUT/BENIGN ones are not.
    /// THROUGHPUT: snapshots accepted by the §6.3 gate (proves frames are landing).
    pub snapshots_applied: u64,
    /// THROUGHPUT: REALM frames accepted by the gate (FA-2c) — proves the moving-realm feed is landing
    /// (a `WaitUntil{RealmFramesApplied >= 1}` closed-loop e2e polls this). 0 at walk scale (no movers).
    pub realm_frames_applied: u64,
    /// BENIGN: snapshots dropped as foreign-sub or strictly-stale (the gate working).
    pub stale_frames_dropped: u64,
    /// THROUGHPUT: input datagrams that rode the wire.
    pub sent_input_count: u64,
    /// FAULT: undecodable control/snapshot payloads from the gateway.
    pub decode_errors: u64,
    /// BENIGN (forward-compat): messages of a class/variant not relevant in P1.5,
    /// tolerated without a crash (e.g. a P2 control variant).
    pub ignored: u64,
    /// FAULT: messages from a non-gateway peer — the one-connection invariant tripped.
    pub foreign_peer_drops: u64,
    /// FAULT: delivered poses carrying a non-finite (NaN/Inf) component, sanitized at
    /// ingress. Nonzero means a sender (gateway/shard) is shipping corrupt floats — a
    /// real fault, surfaced (not silently fixed).
    pub nonfinite_poses: u64,
    /// FAULT at steady state / BENIGN right after a removal: entity rows refused by the client's
    /// RESURRECT GUARD (a straggler for an entity the reliable remove message already evicted).
    /// A brief burst after a removal is the unreliable lane draining; GROWTH at steady state means
    /// a shard keeps emitting an entity it told this client was removed — OR the guard is wrongly
    /// armed and the entity is permanently undrawable (the frozen/invisible-figure class). Surfaced
    /// here (audit :304) so `vdctl state` can tell "the shard stopped emitting" from "the client
    /// refused every row it sent".
    pub resurrect_rows_dropped: u64,
    /// BENIGN during a crossing's grace window / FAULT at steady state: rows skipped by the
    /// ONE-SPACE rule, summed across the entity AND realm feeds — a pose stated in a space other
    /// than the one this client stands in. Steady-state growth means a shard ships entities/realms
    /// in a space its observer does not stand in.
    pub foreign_space_rows: u64,
    /// BENIGN during a crossing's grace window: own-entity rows dropped as the ECHO (the old home's
    /// still-draining copy of the leaver in the space just left). Growth OUTSIDE a crossing means
    /// the echo filter is eating live rows.
    pub echo_rows_dropped: u64,
    /// THROUGHPUT: dev-control actions dequeued and applied (incl. Close/Reset).
    pub dev_commands_applied: u64,
    /// FAULT (overload): dev commands shed because the bounded mailbox was full.
    pub dev_commands_dropped: u64,
    pub transfer: DevTransferView,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) fn sample() -> DevState {
        DevState {
            phase: DevPhase::Active,
            session: Some("sess-1".to_owned()),
            own_entity: Some("ent-7".to_owned()),
            location: Some("System 7".to_owned()),
            render_cursor: Some(101.5),
            universe_tick: Some(101),
            // Distinct from each other AND from `universe_tick`, so a round-trip that dropped or
            // transposed one of the two feed ticks cannot still pass.
            entity_feed_newest_tick: Some(102),
            realm_feed_newest_tick: Some(100),
            entity_windows: Default::default(),
            realm_windows: Default::default(),
            feed_skew_ticks: None,
            entities: vec![DevEntityRow {
                entity: "ent-7".to_owned(),
                pos: [1.0, 2.0, 3.0],
                orient: [0.0, 0.0, 0.0, 1.0],
                authoritative_sub: 0,
            }],
            realm_boxes: vec![DevRealmBox {
                realm: "Planet(7)".to_owned(),
                parent: Some("System(7)".to_owned()),
                center: [10.0, 0.0, 0.0],
                facing: [0.0, 0.0, 0.6, 0.8],
                extent_m: 4.0,
                body_kind: "look".to_owned(),
                luma: None,
                newest_tick: Some(100),
            }],
            origin: Some(("System(7)".to_owned(), 1)),
            sky: None,
            stars_drawn: 0,
            terrain_chunks_drawn: 0,
            terrain_chunks_pending: 0,
            camera_mode: camera_mode_name(CAMERA_MODE_NONE).to_owned(),
            star_probe: Vec::new(),
            capture_frame: Some(4242),
            terrain_stamp: Some(DevTerrainStamp {
                realm: "Planet(7)".to_owned(),
                rung_min: 0,
                rung_max: 3,
                chunks_per_rung: vec![(0, 100), (3, 69)],
                surface_m: 6_370_001.5,
                altitude_m: 3.4,
                horizon_m: 6_582.0,
                horizon_dip_deg: 0.06,
                drawn_radius_m: 372.0,
                chunk_nearest_m: 12.5,
                chunk_farthest_m: 401.0,
                chunks_drawn: 169,
                chunks_pending: 2,
                chunks_urgent: 0,
                chunks_revealed: 0,
                urgent_per_rung: vec![(1, 2)],
                urgent_frames: 4,
                frames: 600,
                built_chunks: 700,
                build_nanos: 2_800_000_000,
                harvested: 690,
                harvest_full: 12,
                harvest_nanos: 700_000_000,
                parent_hits: 5_000,
                parent_builds: 600,
                parent_waits: 40,
                lead_m: 0.0,
                build_rate_per_s: 0.0,
                eye_speed_mps: 0.0,
                ask_horizon_m: Vec::new(),
                frame_work_ns: Vec::new(),
                morph_fallbacks: 3,
                morph_seam: 0,
                vertices: 2_000_000,
                bytes_drawn: 90_000_000,
                shadow_casters: 52,
                shadow_bytes: 15_000_000,
                eye_body_m: [6_371_000.0, 2.0, 3.0],
                camera_body_xyzw: [0.0, 0.0, 0.6, 0.8],
                hud_rect_px: [10.0, 10.0, 900.0, 170.0],
                frame_ms: 34.0,
                passes_ms: vec![("main_opaque_pass_3d".to_owned(), 1.5, 20.0)],
                star: Some(DevStarAngles {
                    elevation_deg: 15.0,
                    off_nose_deg: 120.0,
                }),
                biome: "Grassland".to_owned(),
                world: "0x1234abcd".to_owned(),
                tick: Some(101),
                ruler: Some(DevRuler {
                    centre_m: [1.0, -2.0, -15.0],
                    radius_m: 0.55,
                    distance_m: 15.2,
                    rung: 0,
                    cell_m: 1.0,
                }),
            }),
            sky_anchor: None,
            // A fixture has heard no beat (S11).
            sky_watch: "NeverHeard".to_owned(),
            stale_epoch_rows: 8,
            snapshots_applied: 4,
            realm_frames_applied: 3,
            stale_frames_dropped: 1,
            sent_input_count: 9,
            decode_errors: 0,
            ignored: 0,
            foreign_peer_drops: 0,
            nonfinite_poses: 0,
            // Distinct non-zero values so a round-trip that dropped or transposed one of the three
            // row-drop counters cannot still pass (the same discipline as the feed ticks above).
            resurrect_rows_dropped: 5,
            foreign_space_rows: 6,
            echo_rows_dropped: 7,
            dev_commands_applied: 2,
            dev_commands_dropped: 0,
            transfer: DevTransferView::None,
        }
    }

    #[test]
    fn devstate_roundtrips_through_json_with_string_ids() {
        let state = sample();
        let json = serde_json::to_string(&state).expect("encode");
        // ids are strings (u128 cannot ride a JSON number).
        assert!(json.contains("\"session\":\"sess-1\""));
        assert!(json.contains("\"location\":\"System 7\""));
        // The orientation quat rides each row (x,y,z,w) — the identity here.
        assert!(json.contains("\"orient\":[0.0,0.0,0.0,1.0]"));
        // The drawn realm box rides its realm id + composited center + streamed extent + body kind.
        assert!(json.contains("\"realm\":\"Planet(7)\""));
        assert!(json.contains("\"extent_m\":4.0"));
        assert!(json.contains("\"facing\":[0.0,0.0,0.6,0.8]"));
        assert!(json.contains("\"body_kind\":\"look\""));
        // The origin marker rides as (label, epoch) — the pixel gates' J1 surface.
        assert!(json.contains("\"origin\":[\"System(7)\",1]"));
        assert!(json.contains("\"stale_epoch_rows\":8"));
        // The terrain stamp rides whole: its nested star angles and ruler included.
        assert!(json.contains("\"altitude_m\":3.4"));
        assert!(json.contains("\"off_nose_deg\":120.0"));
        assert!(json.contains("\"radius_m\":0.55"));
        assert!(json.contains("\"chunks_per_rung\":[[0,100],[3,69]]"));
        assert!(json.contains("\"urgent_per_rung\":[[1,2]]"));
        assert!(json.contains("\"urgent_frames\":4"));
        assert!(json.contains("\"harvest_full\":12"));
        assert!(json.contains("\"bytes_drawn\":90000000"));
        assert!(json.contains("\"shadow_casters\":52"));
        assert!(json.contains("\"shadow_bytes\":15000000"));
        assert!(json.contains("\"eye_body_m\":[6371000.0,2.0,3.0]"));
        assert!(json.contains("\"camera_body_xyzw\":[0.0,0.0,0.6,0.8]"));
        assert!(json.contains("\"hud_rect_px\":[10.0,10.0,900.0,170.0]"));
        assert!(json.contains("\"frame_ms\":34.0"));
        assert!(json.contains("\"passes_ms\":[[\"main_opaque_pass_3d\",1.5,20.0]]"));
        assert!(json.contains("\"parent_waits\":40"));
        assert!(json.contains("\"transfer\":{\"kind\":\"none\"}"));
        // The three row-drop honesty counters ride the surface (audit :304 — a wrongly-armed
        // resurrect guard must be VISIBLE to `vdctl state`), each with its distinct sample value.
        assert!(json.contains("\"resurrect_rows_dropped\":5"));
        assert!(json.contains("\"foreign_space_rows\":6"));
        assert!(json.contains("\"echo_rows_dropped\":7"));
        let back: DevState = serde_json::from_str(&json).expect("decode");
        assert_eq!(back, state);
    }

    #[test]
    fn phase_serializes_snake_case() {
        assert_eq!(
            serde_json::to_string(&DevPhase::AwaitingSubscription).expect("encode"),
            "\"awaiting_subscription\""
        );
    }
}

/// THE TERRAIN STAMP (slice 8p; see `DevState::terrain_stamp`): what the renderer measured about
/// the ground it drew, at the frame the state was sampled.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevTerrainStamp {
    /// The body under the eye, as `{:?}` of its `RealmId`.
    pub realm: String,
    /// The finest and the coarsest rung on screen (slice 8 step 2: every ring from the eye to the
    /// horizon), and how many chunks each rung draws, finest first.
    pub rung_min: u8,
    pub rung_max: u8,
    pub chunks_per_rung: Vec<(u8, u64)>,
    /// The recipe's surface radius under the eye, in metres, and the eye's height over it.
    pub surface_m: f64,
    pub altitude_m: f64,
    /// The smooth-sphere horizon of the sphere through the surface under the eye, from the eye's
    /// height over it: the distance along the line of sight, in metres, and its dip below level,
    /// in degrees.
    pub horizon_m: f64,
    pub horizon_dip_deg: f64,
    /// How far the ladder reaches, in metres: the horizon from the eye's height plus the horizon of
    /// the recipe's tallest ground (a peak behind the geometric horizon is still drawn).
    pub drawn_radius_m: f64,
    /// The nearest and farthest drawn chunk's origin from the eye, in metres (0 while none).
    pub chunk_nearest_m: f64,
    pub chunk_farthest_m: f64,
    /// THE DRAWN CAMERA (slice 8 step 6, the pop detector): the eye the frame was drawn from, in
    /// the body's own frame, metres, and the camera's rotation in that frame (x, y, z, w) — the
    /// camera the pixel model reconstructs for a frame, so two consecutive frames reproject into
    /// one another and a pixel's rung can be compared before and after a boundary crossed it.
    pub eye_body_m: [f64; 3],
    pub camera_body_xyzw: [f64; 4],
    /// Chunks on screen and still building — the same two counts the wait fields read.
    pub chunks_drawn: u64,
    pub chunks_pending: u64,
    /// THE BAND'S GAP (slice 8 step 4): wanted chunks inside their rung's own territory (nearer
    /// than its switch distance) that are NOT resident this frame. Zero means the ground the
    /// picture draws now is complete; M8-1 reads it every frame on a moving eye.
    pub chunks_urgent: u64,
    /// THE REVEALS' GAP: wanted chunks of peaks PAST the horizon (the skyline admits them) that
    /// are not resident — a peak the eye sees before it is built, which no lead predicts; the
    /// pop detector's, not the band's.
    pub chunks_revealed: u64,
    /// THE BAND'S GAP PER RUNG: `chunks_urgent` split by rung (rungs with no gap absent).
    pub urgent_per_rung: Vec<(u8, u64)>,
    /// THE FRAMES WITH A GAP since the client started: a gate reads the difference across a leg,
    /// so no frame escapes a poll (ruling V14 M8-1: never incomplete for one FRAME).
    pub urgent_frames: u64,
    /// THE THREE RATES' COUNTERS since the client started (M8-2a, ruling V15): the frames the
    /// terrain system ran (every frame, ground or none), the jobs the workers ran (with or
    /// without a geometry) and the wall nanoseconds they spent (a wait on a sibling's parent
    /// build included), the chunks the engine harvested, and the harvests that filled their
    /// per-frame cap. A gate reads the differences across a leg.
    pub frames: u64,
    pub built_chunks: u64,
    pub build_nanos: u64,
    pub harvested: u64,
    pub harvest_full: u64,
    /// The main thread's nanoseconds in the harvest loop since the client started (M8-2a): the
    /// cost of an upload, against the frame.
    pub harvest_nanos: u64,
    /// THE PARENT CACHE's counts since the client started (M8-2a): claims served from the cache,
    /// claims that built a parent, claims that waited on a build in flight.
    pub parent_hits: u64,
    pub parent_builds: u64,
    pub parent_waits: u64,
    /// THE LEAD the band ran on this frame, in metres: how far the freshest delivered eye stands
    /// from the drawn eye in the body's frame (one interpolation buffer of the eye's motion
    /// through the body — a pilot inside a flying hull stands still in the hull and moves through
    /// the planet). Zero on a still stand.
    pub lead_m: f64,
    /// ★ THE BOUNDED ASK (ruling F9 item 1): what the client measured about its builders and its
    /// own motion this frame, and what the ask was bounded to.
    ///
    /// `build_rate_per_s` is the builders' CAPACITY — the worker count over the mean wall time of
    /// a build, smoothed — never the chunks they happened to finish. `eye_speed_mps` is the eye's
    /// speed through the body under it, the lead's metres over the interpolation buffer's seconds
    /// (two delivered poses, SL10 clause 7). `ask_horizon_m` is each rung's DELIVERABLE HORIZON in
    /// metres, finest first: how far that rung is asked for. EMPTY while the bound does not bind,
    /// which is the tier rule's own switch distances and the ladder every earlier flight flew.
    pub build_rate_per_s: f64,
    pub eye_speed_mps: f64,
    pub ask_horizon_m: Vec<f64>,
    /// ★ THE FRAME'S OWN WORK, piece by piece, since the client started: the piece's name, the
    /// wall NANOSECONDS it cost in all, its worst SINGLE FRAME, and how many times it RAN. The
    /// pieces are the builders' throughput read, the bounded ask's arithmetic, the wanted set's
    /// DESCENT and the crossfade materials' rewrite. A gate reads the differences across a leg,
    /// so a frame rate that falls with the bound on names its own cause.
    pub frame_work_ns: Vec<(String, u64, u64, u64)>,
    /// The geomorph's counts over the drawn chunks (slice 8 step 3): vertices whose morph
    /// target fell back to the coarser field (no parent triangle on their radial within the
    /// sink bound), vertices on a face seam (the field by rule), and vertices in all.
    pub morph_fallbacks: u64,
    pub morph_seam: u64,
    pub vertices: u64,
    /// The drawn chunks' mesh bytes as the engine uploads them (M8-2's census) — the built mesh,
    /// so under the flat-shading switch (vertices duplicated per face) they count the duplicates
    /// while `vertices` counts the library's.
    pub bytes_drawn: u64,
    /// THE SHADOW LADDER (D8-8's shadow cost): the coarse casters on the shadow layer, and
    /// their bytes on the GPU.
    pub shadow_casters: u64,
    pub shadow_bytes: u64,
    /// THE OVERLAY'S RECTANGLE in the picture, pixels (left, top, right, bottom): where the HUD
    /// drew its lines this frame — the one part of a picture that legitimately differs between
    /// two runs of one code (its tick readout), so a picture compare leaves it out. Zero when no
    /// HUD drew.
    pub hud_rect_px: [f32; 4],
    /// THE FRAME'S ANATOMY (D8-8's product form, MEASURED before it is built): the frame's time
    /// in milliseconds, smoothed by the engine's own frame-time diagnostic.
    pub frame_ms: f32,
    /// Each render pass's time this frame, smoothed: (the pass, its CPU milliseconds encoding
    /// it, its GPU milliseconds running it — zero where the GPU offers no timestamps). What a
    /// frame is made of: the prepass, the shadows, the main pass, the copy out.
    pub passes_ms: Vec<(String, f32, f32)>,
    /// The star the ground is lit by, or `None` when a work light stands in (no luminous row).
    pub star: Option<DevStarAngles>,
    /// The biome under the eye, as the recipe names it.
    pub biome: String,
    /// The world identity the client declared (the generator tag), as hex.
    pub world: String,
    /// The freshest delivered universe tick at the frame.
    pub tick: Option<u64>,
    /// The ruler ball, when the centre ray met the ground within the drawn radius.
    pub ruler: Option<DevRuler>,
}

/// The star's angles at the stand (slice 8p): how high over the local level and how far off the
/// camera's nose, both in degrees.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevStarAngles {
    pub elevation_deg: f64,
    pub off_nose_deg: f64,
}

/// The ruler ball (slice 8p): a sphere of stated radius at a stated eye-relative centre, in the
/// composed picture's frame (metres from the eye, the frame's own axes) — what a gate projects
/// through the pilot camera to predict the ball's pixel radius.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevRuler {
    pub centre_m: [f64; 3],
    pub radius_m: f64,
    pub distance_m: f64,
    /// The rung the ball stands on (the tier rule at its distance) and that rung's cell, in metres:
    /// the probe states the ball's distance in these cells.
    pub rung: u8,
    pub cell_m: f64,
}

/// One star the renderer projected itself (see `DevState::star_probe`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevStarProbe {
    /// The star's realm, as `{:?}` of its `RealmId` (the same spelling the sky gate prints).
    pub realm: String,
    /// Screen position through the renderer's camera and projection, in pixels from the top-left.
    pub x: f64,
    pub y: f64,
    /// The star law's amplitude at the eye (the same Tier-A `star_draw` the shader mirrors).
    pub amplitude: f64,
}

/// The camera-mode codes the renderer publishes on its shared handle (an atomic byte the bin threads
/// from the render thread to the dev state, exactly like the star count) — one number per view.
pub const CAMERA_MODE_NONE: u8 = 0;
pub const CAMERA_MODE_FIRST_PERSON: u8 = 1;
pub const CAMERA_MODE_THIRD_PERSON: u8 = 2;

/// The view's name for the dev state. An unknown code names itself as such — never a guess.
#[must_use]
pub fn camera_mode_name(code: u8) -> &'static str {
    match code {
        CAMERA_MODE_NONE => "none",
        CAMERA_MODE_FIRST_PERSON => "first-person",
        CAMERA_MODE_THIRD_PERSON => "third-person",
        _ => "unknown",
    }
}

#[cfg(test)]
mod camera_mode_tests {
    use super::*;

    #[test]
    fn every_camera_code_has_its_name_and_a_stranger_says_so() {
        assert_eq!(camera_mode_name(CAMERA_MODE_NONE), "none");
        assert_eq!(camera_mode_name(CAMERA_MODE_FIRST_PERSON), "first-person");
        assert_eq!(camera_mode_name(CAMERA_MODE_THIRD_PERSON), "third-person");
        assert_eq!(camera_mode_name(7), "unknown");
    }
}
