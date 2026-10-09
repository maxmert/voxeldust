# The weather step (ruling B35 item 5; ruling B67): the design — DRAFT, revision 3, for the owner's questions

> **What this is.** ONE design for the whole weather step of ruling B35 item 5: the almanac, the weather list and its
> lane, the clouds and their shadows, rain and snow on screen, the wind on hulls and trees, lying snow after a storm and
> wet ground after rain. Ruling B67 (2026-10-08) orders the design NOW, AS DOCUMENTS ONLY. The BUILD keeps its B35 place:
> after G-LOOK, the waves and the water fill, and before the edit system.
>
> **How it was made.** Four readers (the rulings and the designs; the code today; the rendering research; the published
> laws) → the synthesis (revision 1) → three adversarial reviews (A, the laws and the rulings: 4 blocking, 14 important, 20
> minor; B, the code facts: 2 blocking, 12 important, 21 minor; C, the look, the seams and the cost: 7 blocking, 17
> important, 14 minor) → this revision. §16 gives every finding with its answer. The notes and the reviews stay in the
> session's scratchpad, outside the repository; this file carries every fact that it takes from them, with its mark.
>
> **What changed from revision 1, in short.** No ruling is claimed as kept where the design changes it: §1 now says KEPT,
> AMENDS or ASKS for each row. Lying snow has no client-derived collision (look only in the first step, or the server's
> diff: WQ3). HR4 runs on two shard profiles, as the code runs it (WQ6). The list is a closed form of the weather epoch
> and the tick, so a sleep and a restart lose no storm (WQ7). The random source, the told placement's spelling, the
> turning planet's real code, the Tier-A crates and the weather grid follow the code. The clouds composite after the
> reconstruction, the near pass converges in one frame, the shadow maps ride the clouds' drift, and the frame line for
> flight is restated honestly (2.3–3.5 ms, ESTIMATE).
>
> **What revision 3 changes (the verifier's check of the 13 blocking fixes).** The cloud march's samples now sit on
> footprint-sized steps along each ray, with the step's phase fixed to a lattice in the planet's frame, for the near pass
> and for the far pass inside the cloud layer; the split between the passes is blended by weight (C-B1). WQ4 (b′) now
> prices where the client gets the orbit's start angle and periapsis angle (three roads). The edge rule runs at the
> display size; the shafts continue in front of an opaque cloud; the steering level is a share of the body's own surface
> pressure; the rain generator's chain reads the storm list, with its two probabilities registered; and the minor points
> are fixed. §16 lists them as V-1 to V-20.
>
> **Date.** 2026-10-08. **The tree.** Branch `game` at `7b756dd` with its uncommitted terrain changes. Read-only. This
> file is the only file that the work wrote. No build, test or instrument ran.
>
> **The wire numbers, read for this design** (§4.7): `PROTO_MINOR` 36 and `PROTO_MINOR_FLOOR` 36
> (`crates/wire/src/version.rs:364`, `:395`); the append rule at `version.rs:8-14` and `:369-371`; `BulkMsg` arms 0–6
> (`crates/wire/src/channels.rs:398-485`); `ServerControlMsg` 0–19, `ArtifactPart` the last (`channels.rs:177-361`);
> `ClientControlMsg` 0–8, `ArtifactHeld` the last (`channels.rs:46-116`); `GatewayToShard::ArtifactWant` 8 and
> `ShardToGateway::WindowFramePart` 17 (`crates/wire/src/session_flow.rs:1558-1577`); `InterShardFlow::ChildFelt` 44, the
> last (`crates/wire/src/intershard.rs:632-633`; the closed test pins 45 arms, `crates/wire/tests/intershard_closed.rs`).
>
> **The marks** (§15): `path:line`; MEASURED (a tool printed it; the record and its date are named); DOC (a ruling, a
> design, a code comment, or a published source that a reader or reviewer of this team read at first hand on 2026-10-08);
> ARITH (arithmetic of this team on marked inputs); ESTIMATE (a judgement, or a published value that nobody of this team
> read at first hand); UNMEASURED (nobody printed it; §9 names the bench).
>
> **The labels.** WM1–WM8: the first step's parts (§3.3; reader 4's W1–W8, renamed so that nobody reads them as the water
> rulings W1–W18). WL1–WL42: the published laws (reader 4's L1–L42). CR1–CR9: the clouds' seam rules (§6.2). WA1–WA3: the
> SL6 asks (§4.8). WP0–WP6: the phases (§10). WC1–WC30: the conflicts (§12). WQ1–WQ24: the owner's questions (§13).
> A-…, B-…, C-…: the review findings (§16). "Ground Qn / An / §n": `docs/investigation/2026-10-02/ground_step_design.md`.
> "05 §n": `docs/investigation/2026-09-08/landforms/05_climate_biomes_weather.md`. "Temporal §n":
> `docs/investigation/2026-09-28/temporal_path_design.md`.
>
> **The words.** THE LIST: the weather systems that live now (lows and highs). THE STATEMENT: one message of the planet's
> shard at each weather step (the list and the anomaly's changed cells). THE ANOMALY: the live, coarse field of the storms'
> new snow and wetness. THE BASE: the static weather fields of the body by season, on the weather grid. THE WEATHER EPOCH:
> a server-private number that, with the tick, fixes the planet's weather (§3.6). THE ALMANAC: the closed forms of the
> tick (the sun, the day, the season). THE EYE'S FADE: the shortest fade whose largest step between two frames stays
> under the eye's rule (B27).

---

## 0. THE RECOMMENDATION IN ONE PAGE

**What the step builds.** The planet's own server makes the weather. Every player who sees the planet gets it. The client
draws the clouds and their shadows, the rain, snow and dust in the air, the wind in the trees, the lying snow and the wet
ground. The wind pushes hulls. The weather never changes the land's shape. Each item below is a PROPOSAL for your
decision, with the question that asks it.

**The twelve recommendations.**
1. **The weather comes from published laws that read the planet's own facts** (WQ1, WQ2, WQ9). The laws decide where
   storms live, how many there are, how large they are, where they go and how long they live. Arithmetic on the home
   planet's facts (not a measurement) gives storms about four times as wide as Earth's and about 4 to 12 systems at one
   time. Where no law exists yet, Earth fixes one constant, with its name and value written before any test runs.
2. **The weather is one formula of a secret number and the time** (WQ7, WQ8). So a planet that sleeps, or a server that
   restarts, comes back with the same storms. Nobody can predict the weather from the world's seed.
3. **The list reaches every player who sees the planet** (WQ10). It goes on its own lane, through the gateway, every ten
   seconds. Each storm has an identity number. A storm starts and ends at zero strength, so none appears or goes away in
   one frame.
4. **A late list holds still for a short time, then fades** (WQ11). It never jumps.
5. **Snow and wet ground are drawn, but in this step they have no collision** (WQ3). Your feet stand on the ground under
   the drawn snow. The collision comes later, from the server only.
6. **Both machines compute the season, as you ruled on 16 September** (WQ4). The client computes no storm. The price:
   the client must also know where the planet stands on its orbit, which today only the star system's generator knows,
   so the client gains one orbit function (WQ4 gives three roads).
7. **The clouds are one field around the planet** (WQ12, WQ17, WQ18). The game draws it the same way on foot, in flight,
   high up and from orbit. Far away, each small detail becomes its own average. The clouds grow, drift and decay.
8. **One drop law makes the rain near the eye, in the middle distance and in the far curtain** (WQ20). Drops land on glass
   only where a hull or a suit has glass.
9. **The wind pushes a hull through the planet's own physics**, with the drag numbers that the hull already states
   (WQ13). Trees sway in the same wind.
10. **The planet must turn in the picture first** (WQ5). Today the drawn planet does not spin, so the drawn sun stands
    still. Rulings disagree on who knows the spin, so you choose the road. After it, you see the sun cross the sky in about
    72 hours, and the seasons change. This changes the whole world, so you look at it before any cloud comes.
11. **The cost.** Today's frame has no measured room: 23.1 ms against the 16.7 ms of 60 frames a second, and that reading
    is not clean. On your Mac the clouds may need 2.3–3.5 ms in flight through clouds (an estimate), more than the 2.0 ms
    line. The benches measure it first, and the build waits for the speed work or for your word (WQ17).
12. **One hard rule needs your reading first** (WQ6). Hard rule HR4 asks for one test on two kinds of shard. No second
    kind of realm has air today. We propose the test with the air given by the test, as the existing crossing test does.

**The order.** The benches → the turning planet → the server's weather and its lane → the clouds → rain, snow on screen
and the wind → lying snow and wet ground → your look.

**Example.** A pilot crosses the home planet in its long winter. The planet's server holds one deep low over the northern
plains, 7 000 km wide, and states it every ten seconds. The pilot's client draws its cloud shield, the snow that falls
in front of the canopy, and the moving shadow of the deck on the plains. The pilot lands in snow to the ankles: in this
step the snow is drawn, and the boots stand on the ground under it. The pilot leaves for the next star system. Three hours
later the pilot comes back;
the planet slept, and the same low is there, older and further east. Nothing that the storm does changes the land.

**The questions to answer first** (§13 has all of them): WQ1 the circulation law; WQ2 the land's season; WQ5 the turning
planet; then WQ3 the snow, WQ4 the season on both machines, WQ6 the test rule, WQ7 the formula of the weather.

---

## 1. The binding rulings, and how this design treats each one

The newest ruling wins over every design (CLAUDE.md); the newest ruling file is
`docs/design/owner_decisions_2026-09-22_believability.md` (B1–B67). STATUS: KEPT (as written), AMENDS (the design changes
it, so the owner must rule: the question is named), or ASKS (the design offers roads, and the recommended road keeps or
changes the ruling as stated).

| Ruling or law | What it requires | How the design treats it | Status |
|---|---|---|---|
| HR1 sealed shards | State private; bytes only as reviewed arms | The planet's shard computes; the statement leaves as `BulkFor { audience: Realm }`; the orbit as `InterShardFlow::PlacementTold` (§4.7) | KEPT |
| HR2 generic transfer | One transfer machinery | The weather is not an entity; hulls cross by the same machinery in any weather | KEPT |
| HR3 one tooling | A feature by capability, never a kind `match` | One weather system on every shard, gated at boot by the body's facts (the surface's pattern) | KEPT |
| HR4 features once | One fixture through ≥ 2 shard kinds, or it does not land | The fixture through the planet and the station profiles with the air and sun facts injected by the test (`assert_feature_anywhere`'s shape). No second realm kind holds air | ASKS — WQ6 |
| HR5 100 % | Tier-A at 100 % | The process and the almanac in two new Tier-A crates; `crates/bins` holds only the worker's wiring (§3.7) | KEPT |
| HR6 agent-operable | vdctl stands, probes | Cloud and precipitation probe kinds; a stamp row; the harness fixes the epoch through the realm store and the tick through the clock seam (§10.3) | KEPT |
| SL1 clauses 1–2 | The parent authors and may state the child's placement | The star system authors the attitude and states the orbit and spin law once as `PlacementTold` | KEPT |
| SL1 clause 3 + B36 item 5 | A child never derives its placement; one fenced exception | Only the almanac module reads `PlacementTold`; a fence stops it from reading the orbit that the boot generates (§4.8). The turning planet's own frame angle (WQ5 (a)) is a reading for the owner | KEPT for the almanac; the frame angle ASKS — WQ5 |
| SL1 clause 4 | One hop; never told the parent's placement | Kept for planets. An airy moon has no lawful sun | ASKS — WQ21 |
| SL1 clause 5 | Placement, containment and crossing never name it | The existing module fence (`tests/tests/crate_isolation.rs:165-230`) extended; the vocabulary in `crates/core/src/pose.rs:20-30` | KEPT |
| SL1 clause 6 | A stale reading is refused; degrade and say so | The bound derived from the weather's own rate of change; then the eye's fade; the stamp row says so | ASKS — WQ11 |
| SL2 + its clarification | No occupant pose crosses; the gateway is not a realm | The want names a realm and digests; one crossing, shard → gateway | KEPT |
| SL3 + B35 item 6 | A realm calculates and draws itself | The planet computes; the client draws it as the planet's look. A child realm's open surfaces are a separate case | KEPT; children ASKS — WQ14 |
| SL4 | Physics and re-home apart | The wind is a term of the planet's drag; the crossing never reads it | KEPT |
| SL5 one world | No variant | Benches read the laws on Earth's REAL facts in the bench copy; no Earth body runs in a shard | KEPT |
| SL6 | Ask before new data or arms | WA1–WA3 (§4.8) | ASKS — WQ10 |
| SL7 | An occupied or watched realm stays awake | A storm that a player stands in never ends by a sleep | KEPT |
| SL9 | No per-child row per tick | One statement per realm; the orbit once on change | KEPT |
| SL10 (V1 items 6–7) | The client derives only the static shape; live state crosses as a diff; the client never derives a realm's state | No collision from client-derived snow: look only in the first step, or the server's diff (§7.4) | KEPT under WQ3 (a)/(b); WQ3 (c) AMENDS |
| T1 items 10–11, T4 | Clouds after the freeze | As amended by B35 item 5: before the freeze | KEPT (as amended) |
| T2, W3 | Waves as a closed form; "the swell the almanac's wind raised" | The waves read the almanac's wind (§7.6) | ASKS — WQ15 (recommendation keeps it) |
| T3 class 1 | Both hosts compute the almanac; no lane | Both hosts compute it (the client by one fenced integer function) | ASKS — WQ4 (recommendation keeps it) |
| T3 class 2 | The list on its own lane; a dormant planet by the dormant-world rule | The list is a closed form of (epoch, tick): the dormant-world rule (LAW-WL-1) by construction (§3.6) | KEPT; WQ7 confirms the form |
| T3 class 3 | No erosion; the land never changes | Look-only snow changes no cell. Live snow cells (WQ3 (b)/(c)) would change the collision surface after the freeze; road (b), recommended at the collider slice, is asked against this class (or read, in W2's pattern, as live substance rows and not the land's shape) | KEPT under WQ3 (a); road (b) ASKS — WQ3 |
| T5 + B36 item 1 | The seasonal circulation allowed | The weather follows it | KEPT |
| T9 | No drawn physical number; a named scatter | Every random term is a named scatter around a law's value; every Earth constant has one value and its body | KEPT |
| W2 | A dug hollow fills from connected water | Untouched; rain-fed water later (§3.8) | KEPT |
| B5 | The 55° tilt stays | Read, never tuned | KEPT |
| B6 | Numbers and a picture each phase; the status file | Each phase ends so (§10) | KEPT |
| B16, W16 | No flicker or shape change between rungs | CR1–CR9; rule M with its covariance for the far snow (§7.4) | KEPT |
| B18 | A sub-pixel feature is a tint; a late chunk rises in over 0.5 s | A sub-pixel cloud is a tint. A late first statement fades by the eye's fade, longer than 0.5 s | ASKS — WQ11 |
| B20, F6 | The terrain takes half the cores | The weather's client work lives in the game's half | KEPT |
| B21, V4 row 13 | One volume; `Air` the atmosphere; `Empty` the vacuum | Snow a substance; a cloud a look, never a cell | KEPT |
| B22 | 60 frames a second at full screen | No measured room today; the flight cases exceed line G (§8.1) | ASKS — WQ17 |
| B23 | No caps; laws on every body | The caps list (§11); the felt air under the sea's level is asked | KEPT; felt air ASKS — WQ13 |
| B24 + temporal §4.6 | Motion stated, reactive, or after the reconstruction | Clouds and near streaks draw after the reconstruction; shafts resolve in the main history | KEPT |
| B25, B26 | The air rests on the sea; no term grows under the sea's level | The drawn weather terms follow B26 (look) | KEPT for the look; physics ASKS — WQ13 |
| B27, B31 | The eye's rule; half a level at a swap | Every leg, update, fade, wake and boarding | KEPT |
| B29 | FSR 2.2's rectification; no amplified difference | The cloud history uses the same clamp; G-DETERMINISM expects the drift | KEPT |
| B34 | 23.1 ms, not clean | §8 starts from it | KEPT |
| B35 | The scope, the order, a realm calculates and draws itself | All of this design; the biome's seasonal paint is outside the scope (§3.8) | KEPT |
| B36 items 3, 4, 8, 9 | The Mac; the art later; the stands; the third body | The owner's M4 Pro; the trees wait; the same stands rule; three bodies | KEPT |
| B38, B41 item 2 | The right law; no knob; Earth only calibrates, registered | Benches only READ; each Earth constant written before any bench, registered as an interim | KEPT; ASKS — WQ9 |
| B39 | The look must not depend on the pass count | By its principle: G-WEATHER-CONVERGE at the step size and at C against 2C | KEPT |
| B67 | Documents only; the build in its place | This file; no bench now. U-W2 joins the ground step's climate bench only if WQ1 (a) is ruled | KEPT |
| V6 A1 | The seam passes G-MAPPING-TABLE and -ROUNDTRIP | The weather grid's new answers (area, Coriolis) under those gates | KEPT |
| V12 (2026-09-08) | "The parent authors the spin on the row, the recipe lives in the planet's own frame" | Kept by WQ5 (a); amended by WQ5 (b) | ASKS — WQ5 |
| The owner's ground rule (2026-08-09, memory) | "Planet doesn't know its spin or position"; "a planet realm has no concept of rotating at all" | WQ5 (a) gives the planet's physics its spin RATE from its own charter | ASKS — WQ5 |
| V13 A1, A2, L12, L16 | The charter in the surface statement; the orbit half down; the list | Kept; L16's cap of 64 goes (B23) | KEPT as amended by B23 |
| S7-1 (V12's table) | No motion crate on the client | WQ4 (b′) puts one fenced integer almanac function on the client, and the orbit's start and periapsis angles reach it by one of three roads (§5.4) | ASKS — WQ4 |
| S7-7 | The light from the composed picture | The client's light | KEPT |
| Seed ruling S1, S3, S5 | Live state is never seed-derived; the wiki test | Depends on the random source | ASKS — WQ8 |
| Visibility (2026-09-01), reach (2026-09-02) | A dormant realm is never drawn; reach from the look | No seam at a wake; the reach from the brightest look that the charter allows | KEPT |
| M2a, M-A (movement) | Drag is the parent's; the medium never crosses; never cap a speed | The wind in the planet's drag by an injected per-child ambient (§7.3) | KEPT |
| Suit ruling (2026-09-05) | A suit is like a hull | A visor shows drops as its material | KEPT |
| LAW-WL-1 (the dormant-world design) | The closed form is the master; live == dormant | The list is a closed form | KEPT |
| Conventions | No I/O outside the seam; no magic numbers; no prediction; Category C | The random source needs a seam that does not exist (§3.7); operational numbers in one config struct; one statement back | KEPT; random seam ASKS — WQ8 |
| The owner's standing rules | No test seam; a window only on the owner's word; no new library; vendored changes die at Bevy 0.19 | Harness controls only; WP6; WQ22; every vendored change in `vendor/bevy_pbr/PATCH.md` | KEPT |

---

## 2. What exists

### 2.1 The short answer

1. No code steps a weather or draws a cloud, rain or snow. The words appear in the solve's annual climate
   (`crates/terrain/src/climate.rs`), in a storm-wave law for the coast's band (`crates/terrain/src/solve.rs:178-181`,
   `:2846-2857`), in `Stratum::Snow` (`crates/terrain/src/strata.rs:19`) and in comments.
2. The drawn planet does not spin: `FramePlacement::moving` writes the identity orientation and zero angular velocity
   (`crates/core/src/frame.rs:92-101`; `crates/physics/src/motion.rs:49-52`). Its drawn pole is the system's +Z
   (`crates/recipe/src/height.rs:398-400`), so its drawn obliquity is the orbit's small Rayleigh-drawn inclination
   (`crates/physics/src/worldgen/generate.rs:160`), not the charter's 55°.
3. The crossing arithmetic already reads a parent's spin (`frame.rs:250-251`, `:273`), but the gateway's relayed hop
   writes it as zero: "angular velocity is not on the level wire … the rotated-hop composition is the ledgered P10 owe"
   (`crates/connection-plane/src/window.rs:1083-1097`).
4. The vendored air knows a medium that changes only with the radius (`vendor/bevy_pbr/src/atmosphere/functions.wgsl:190-195`).
   Its per-pixel march takes 16 samples spread linearly over the whole ray at a fixed 0.3 offset
   (`vendor/bevy_pbr/src/atmosphere/mod.rs:430`; `functions.wgsl:50`).
5. The reconstruction (behind `VD_TEMPORAL=1`) sends a pixel with zero depth OR the motion sentinel through a point at
   infinity, whatever motion was written (`crates/client-render/src/reconstruction.wgsl:162-172`).
6. MSAA 4× is the engine's camera default; the client inserts its own `Msaa` only under the bench's setting
   (`crates/client-render/src/lib.rs:1489-1491`), and the temporal path forces `Msaa::Off` (`measure.rs:150-155`).
7. `sim::io` has `Clock`, `Transport`, `Store` and `RealmSpawner` (`crates/sim/src/io/mod.rs:384`, `:433`, `:499`, `:550`)
   and NO random source: "The remaining ShardIo traits (Clock, Store, DetRng, Provisioner) land with their first
   consumers" (`crates/sim/src/lib.rs:36-38`). Its one design is seed-derived: "DetRng … seeded from `(universe_seed,
   node_id, purpose_tag)`" (`docs/design/test_harness.md:133`).
8. A dormant realm's shard is killed (`crates/sim/src/rlm.rs:417`), and a wake is a new process that boots from the realm
   store (`crates/bins/src/bin/shard.rs:330-337`): the same boot as a kill-9 restart.
9. The drag has no wind term, and the ambient is ONE pull and ONE density for the whole realm, today zero
   (`crates/sim/src/stub/drive.rs:108-119`, `:202-208`, `:435-464`; `crates/sim/src/stub/placement.rs:45-61`, `:80-85`).
10. The frame has no measured room for clouds (§2.3).

### 2.2 The facts the design rests on

| Fact | Where | What it means for the step |
|---|---|---|
| The charter's time words; `day_s` is "The rotation period" (sidereal) | `crates/core/src/look.rs:136-141`; home values `crates/terrain/src/home.rs:117-120` | The almanac's inputs; the solar day is longer (§5.1) |
| The home radius 6 341 670 m; N = 9 961 472 = 19 · 2¹⁹ | `home.rs:222-223` | Every arithmetic uses it; the weather grid comes from N (§3.7) |
| The universe tick: 50 Hz, real time | `crates/bins/src/lib.rs:463`, `:546-550` | The almanac runs in real time |
| The charter's draws in a frozen order: `u_day`, `u_family`, `u_cos`; no pole azimuth, no spin phase | `crates/physics/src/worldgen/charter.rs:530-538` | Two identity draws are new (§5.3) |
| The planet's shard generates its own charter, orbit included, at boot | `vd_bins::boot_charter` → `body_charter_in_subtree` (`crates/bins/src/lib.rs:3263-3278`) | The almanac must be fenced from that orbit (§4.8) |
| The SL1 vocabulary `ToldPlacement`, `PlacementTold`, `told_placement`; the wire pin's empty `DOWNWARD_STATEMENTS`; the module fence with a planted control | `crates/core/src/pose.rs:20-30`; `crates/wire/tests/intershard_closed.rs:982-995`, `:1093-1101`; `tests/tests/crate_isolation.rs:165-230` | WA3 takes that spelling (§4.8) |
| Seven producer-less-reliable arms are pinned | `intershard_closed.rs:1260-1267` | WA3 makes eight |
| The client draws chunks by the row's facing | `crates/client-render/src/terrain.rs:2481-2486`, `:3930` | A turning planet moves its ground by its row |
| The air's only shadow is the ground sphere; no aerial table under the raymarched mode | `functions.wgsl:206-233`, `:283-306`; `atmosphere/node.rs:145-164` | Each air reader needs a cloud term; the shafts need their own method (§6.6) |
| Eight vendored changes; the patch dies at Bevy 0.19 | `vendor/bevy_pbr/PATCH.md:1-118`, `:115` | Each cloud change is one more listed change |
| MSAA edge pixels march per sample in the sky pass | `PATCH.md:49-52`; `render_sky.wgsl:51` | The cloud term must sit in that path too |
| The sky light: a 512 px cube from the sky-view table at the eye's place; no graph edge | `crates/client-render/src/sky.rs:66-69`; `atmosphere/environment.wgsl:26-34`; `atmosphere/mod.rs:201` | One probe for the whole view (§6.6) |
| One air a frame | `client-render/src/sky.rs:150-166` | WQ19 |
| The sun's direction: the realm box with the largest `luma` | `terrain.rs:2957-2961` | S7-7 |
| A fixed exposure; four cascades to 3.5 km, off above; one ground colour | `terrain.rs:128-136`; `:110-116`, `:4009-4033`; `:2325` | The light phase's adaptation, the clouds' own maps, the material inputs |
| The reactive mask: history × 0.9, box / 10 | `crates/client/src/reconstruction.rs:94-96` | A noisy cloud in the transparent pass flickers |
| The sky share; the points pass after the reconstruction | `reconstruction.wgsl:91`, `:400-405`; `crates/client-render/src/points.rs:1-13` | The clouds' transmittance enters it |
| The motion census sees only pixels with depth | `crates/client-harness/src/motion_census.rs:1-21` | No census for a cloud in the main history |
| The eye's rule: Crumey 2014 on Blackwell's data; the Weber fraction 0.27–0.39 % on 2–200 cd/m² | `crates/client-harness/src/eye.rs:12-14`, `:175-200` | Every look gate reads it |
| The climate's season is amplitudes only; Held & Hou ≤ 6 cells with ΔH fitted to Earth | `climate.rs:415-476`; `:104`, `:106`, `:321-349` | Phase 2 replaces both |
| The static climate's Earth constants | `climate.rs:93`, `:98-99`, `:117-118`, `:122-123`, `:127`, `:129-130`, `:140-142`, `:151`, `:161` | §3.4 |
| The pyramid ships heights alone since B22 | `crates/bins/src/artifact_source.rs:8-12`, `:135-137` | The far view holds no climate |
| The drag words cross on change | `intershard.rs:1455-1477` | The wind needs no new data |
| The sim names only core and wire | `crates/sim/Cargo.toml:8-16` | The laws are injected |
| The artifact worker runs off the tick; the schedule is single-threaded; the dev tick is 20 ms | `crates/bins/src/artifact_worker.rs:70-165`; `shard.rs:728-780`; `crates/node/src/app.rs:108-109` | The weather's worker copies that shape |
| The realm store's families: berths and the body, one prefix byte each, append-only tags | `crates/sim/src/stub/built_store.rs:1-30` | The weather is a new family |
| The capability decides what a system does; the surface is a boot-time gate on the body's charter, "never a kind test" | `crates/sim/src/stub/register.rs:353-358`; `shard.rs:281-322`, `:285` | The weather's gate copies the surface's |
| G-IDENTICAL runs one feature through `NodeKind::Shard(planet)` and `NodeKind::Shard(station)` | `crates/sim/src/stub/tests/containment.rs:1076-1130` | HR4 as the code runs it (§3.7) |
| The Tier-A list holds no `vd-bins` | `justfile:12` | The process cannot live in `crates/bins` |
| `GridMapping` answers `addr_of`, `cell_center`, `cell_corners`, `neighbor`, `outward`, `domain` | `crates/core/src/grid/mod.rs:122-214` | Area, Coriolis and gravity are new answers |
| A rung's face edge is N >> rung | `crates/seed/src/ladder.rs:173-175` | Home: 152 or 76 cells a face, never 128 |
| The gap byte is 1/128 cell; the sub-metre seat is 1/8 cell | `crates/terrain/src/chunk.rs:21-24`, `:76`; `crates/terrain/src/seat.rs:1-6`, `:26` | The snow surface's quantum (§7.4) |
| The composition's five ranks | `crates/terrain/src/compose.rs:1-17` | Where snow cells would sit (§7.4) |
| The held arms: the receiver states what it holds | `channels.rs:63-83` (`SkyHeld`), `:107-115` (`ArtifactHeld`); `gateway/artifact.rs:654`, `:710` | `WeatherHeld` (§4.5) |
| The artifact want fires only for realms whose look bag states `TAG_ARTIFACT` | `crates/connection-plane/src/gateway/artifact.rs:405-428` | The weather want's trigger (§4.5) |
| `relay_bulk` gates `ArtifactPart` on the negotiated minor | `crates/connection-plane/src/gateway/shard.rs:457-540`, `:525` | `WeatherPart` is gated the same way |
| `Air` 0, `Snow` 2, `Ice` 3, `Empty` 18 | `strata.rs:16-41` | Snow exists as a substance |
| The gateway's pace: 1 MiB a 0.5 s beat a client | `artifact.rs:50`; `crates/connection-plane/src/gateway/config.rs:384-390` | A list behind the 5.9 MB pyramid waits about 3 s unless it goes first (ARITH) |
| A body with no solid surface states no surface; "the cloud shell of its own slice" is promised | `shard.rs:309-312` | Gas giants (§3.8) |

### 2.3 The frame today

| Reading | Value | Mark |
|---|---|---|
| The temporal path at s = 0.5 in the owner's window, full screen | 23.1 ms median, about 43 frames a second | MEASURED 2026-10-02 (B34 item 3; `docs/design/DEFERRED.md:11709-11712`); NOT clean |
| G-NO-STALL | over 33 ms in 18.5–35.8 % of the seconds on foot and in every second at dusk; 146 ms at the dusk boarding | MEASURED 2026-10-02 (`DEFERRED.md:11710-11715`) |
| MSAA off, s = 1 / 0.75 / 0.67 / 0.5 / 0.33 (span) | 22.2 / 17.0 / 15.5 / 12.7 / 10.5 ms | MEASURED 2026-09-29 (`DEFERRED.md:10941-10942`) |
| Per pass at s = 0.5 | ground 8.44 (about 6.9 vertex work, ARITH), air 1.95, shadows 1.83, bloom 1.10 ms | MEASURED 2026-09-29 (`DEFERRED.md:10929-10932`); the window was hidden and another GPU client put gaps; the document keeps these as the steady numbers |
| The reconstruction | 0.30 ns a display pixel, about 2.2 ms at full screen | MEASURED 2026-09-29 (`docs/investigation/2026-09-24/terrain_believability_status.md:1334`) |
| The frame interval at s = 0.5 / 0.33 | 15.9 ms (client CPU 107 %) / 12.4 ms (121 %) | MEASURED 2026-09-29 (`DEFERRED.md:10951-10953`) |
| The half-size span plus the reconstruction | 14.9 ms | ARITH. A span over-reads a cost, so 1.8 ms is at most a LOWER bound of the GPU room; the sum omits the prepass (about 0.7 ms, not proven, `DEFERRED.md:10943-10944`) and the points pass, sharpen and motion (1.1–2.4 ms, temporal §10.2). The CPU limit (15.9 ms at 107 %) leaves about 0.8 ms |

**No measured room exists for the clouds or the rain today. The 23.1 ms reading is the only measurement of the frame.**
The room depends on the speed work's line A (ground §7.3), which nobody has priced.

---

## 3. The weather model

### 3.1 The three layers

| Layer | What it holds | Computed by | How the client gets it |
|---|---|---|---|
| The static climate and THE BASE | Annual means per 8 192 m node; after phase 2 the seasonal amplitude (ground A7) and the circulation (ground A3). The base: those fields on the weather grid as functions of the season | The planet's shard, once per solve | The base by digest, once (WA2) |
| The almanac | The sun, the day, the season, the daily insolation, the daily swing | Both hosts (T3 class 1): the planet from `PlacementTold` (B36 item 5), the client from the seed by one fenced integer function (WQ4 (b′)) | Computed, never shipped |
| The live weather | The list (a closed form of the epoch and the tick) and the anomaly (snow, wetness) | The planet's shard only | The statement (WA1) |

Under WQ4 (a) instead, the client computes no almanac, and the planet ships the base evaluated for the season as keyframes
ahead of their ticks (§5.4).

### 3.2 What the laws read, and what they give on the home planet

The laws read the gravity, the spin rate, the radius, the surface pressure, the mean molecular mass, the scale height, the
grey infrared depth, the equilibrium and surface temperatures, the tilt, the year, the eccentricity, the water inventory
(zero water gives zero humidity: no water cloud, rain or snow, with no branch), the static climate, and, after phase 6, the
ground's roughness.

The home planet beside Earth (ARITH: reader 4's `calc.py` for the home planet; reviewer B's `earth.py` and `arith.py` for
Earth with Earth's own sunlight; nothing in the game measured either column; "Earth" means the laws evaluated on Earth's
real facts, never an Earth body):

| Reading | Home planet | Earth | Note |
|---|---|---|---|
| Spin, Ω / Ω⊕ | 0.332 | 1 | the rotation period 259 597 s |
| Today's Held & Hou (ΔH 0.388), sin²φ_H | 1.93: ONE cell | 0.235: 29° | `climate.rs:321-349` |
| Held 2000's eddy-onset edge, same Δ_v | about 52° | about 31° | an ESTIMATE form |
| Thermal Rossby number g·H·Δ_H/(Ωa)², H the scale height (with the tropopause height) | 0.50 (0.98) | 0.061 | the definition of the published values: Earth 0.06, Mars 0.2, Titan 18 (DOC, Guendelman & Kaspi 2018). The home planet sits a little above Mars, far below Titan |
| Tropopause: Robinson & Catling's pressure law / the skin law | 12.2 / 14.1 km | 12.4 / 11.3 km | U-W3 reads both |
| Deformation radius L_R = NH/f at 45° / 60° (N = 0.0113 s⁻¹) | 4 655 / 3 801 km | 1 238 / 1 011 km | |
| Eady wavelength 3.9 L_R at 45°; waves round the 45° circle | 18 150 km; 1.55 | 4 830 km; 5.86 | Earth's readings about 4 000 km (DOC, Showman et al. 2013) and 5–8 waves (ESTIMATE) |
| Annual-mean pole-to-equator contrast T₂ | +0.21 K | −25.0 K | the home planet's storm energy is seasonal; the code's own Earth check gives −25 K (`climate.rs:290-293`) |
| Seasonal amplitude at the pole: one column / land only / ocean only | 10.5 / 96 / 8.1 K | 6.6 / 62 / 4.7 K | the climate's own law |
| Eady e-fold at 60°: one-column / land-only season | 13.5 d / 1.5 d | about 1.5 d (the annual gradient) | σ = 0.31 g(∂T/∂y)/(NT) |
| Ekman spin-down at 60° (K = 5 m²/s) | 16 d | 8.1 d (45°, Earth's 11.3 km) | K is an Earth calibration (§3.4) |

### 3.3 The first step's eight parts

Each law reads the body's own facts, so another body gets other weather from the same code. A random term is a NAMED scatter
around a law's value, drawn from the weather epoch (T9). Every source below is DOC unless marked. Each Earth calibration has
ONE value, written before any bench runs (§3.4).

**WM1 — The season's temperature, land and sea apart** (T3 class 1).
- Ground A1 (a two-mode seasonal balance with each node's own heat capacity, North, Short & Mengel 1983) plus the daily
  swing (ground A7). The heat capacities of rock and water are laws of the substances.
- Why: the storms take their energy from the season (T₂ = +0.21 K). On today's one-column season a storm grows by e in 9–14
  days and the friction stops it in 11–16 days; on a land column it grows by e in 1–1.5 days (ARITH). Phase 2 builds A1
  (WQ2).

**WM2 — Where the storms live, and the rain belt** (closed form by season).
- WL12, the thermal wind `∂u/∂z = −(g/(fT)) ∂T/∂y` (Held & Hou 1980): the shear and the steering wind.
- WL13, the size and the growth: `L_R = NH/f`, the fastest wave `λ ≈ 3.9 L_R`, its growth `σ ≈ 0.31 g (∂T/∂y)/(N T)`
  (Eady 1949; Lindzen & Farrell 1980; Hoskins & Valdes 1990). On the home planet L_R/a ≈ 0.6–0.7, outside the f-plane law's
  strict range; U-W1 reads it.
- WL14, the storm zone: a wave lives where `σ > 1/τ_E`, with the Ekman spin-down `τ_E = H √(2/(K f))`. Reader 4 composed
  this rule from WL13 and the spin-down; U-W1 tests it against Earth's tracks (Hoskins & Hodges 2002). Near the equator,
  where f → 0, the equatorial β-plane form of each law replaces the f-plane form (no floor on f; §11 row 15).
- WL19, the rain belt follows the season (B36 item 1); the thermal Rossby number decides how far (Guendelman & Kaspi 2018).
  At 0.5–1.0 the home planet's solstice belt moves well away from the equator (ARITH on the published scale; how far is
  UNMEASURED; U-W2).

**WM3 — The systems: count, size, birth, drift, life, fronts** (the list; a closed form of the epoch and the tick, §3.6).
- WL16, the count: the continuous ratio `m = 2π a cos φ_s / λ` is the Poisson mean of the lows, and of the highs, of each
  storm belt and season (the M/G/∞ result: births at `λ_b = m / E[life]` keep the mean count alive at m; ESTIMATE source
  Takács 1962). No `round()`: a ratio of 1.5 gives a mean of 1.5, so two planets 2 % apart in spin differ by about 2 % in
  their storms. U-W1 also reads the integer wavenumber that the growth rate σ(k) chooses on the sphere, as a cross-check.
  Where along the belt a storm forms is a scatter around a computed value (uniform in longitude by the belt's symmetry),
  named, drawn from the epoch. N is a READING: about 25–30 on Earth's facts, about 4–12 on the home planet (ARITH); ground
  §8.2's area law (Gill 1982, ESTIMATE) gives 3–7, a cross-check.
- WL17, the drift: the steering flow at ONE stated level, 0.7 of the body's own surface pressure (Earth's 700 hPa practice
  as a share of its own surface pressure, §3.4 row 5; an absolute 700 hPa would lie at 2.4 % of the ground pressure on the
  28.7-bar world and would not exist on the 555 Pa icy moon, ARITH), plus the β drift (ESTIMATE form). Emanuel et al. 2006
  move synthetic storms the same way.
- WL18, the life: growth at σ (WL13, a law), decay at 1/τ_E (WL14). The wind is the shear velocity `Λ·H_tp` times a share;
  the share and the curve's shape are Earth calibrations (Hoskins & Hodges 2002). The chosen curve starts at zero and ends
  at zero, at a finite age, so a birth and a death are continuous on screen. The central pressure follows by gradient
  balance (WL27); for a high the life stays inside the gradient wind's own limit (§11 row 13).
- WL15, the fronts: the Norwegian life cycle as the row's stage (an Earth conceptual model, ESTIMATE source Bjerknes &
  Solberg 1922) and Margules's slope `tan α = f T̄ Δv / (g ΔT)`, a law; the warm front's shield reaches `H_tp / tan α` ahead
  of the surface front, about three times Earth's reach for the same contrasts (ARITH: f a third of Earth's).
- The class rule (WL41): independent systems, no merging or interaction in the first step. This keeps the list a closed
  form (§3.6).

**WM4 — The clouds: cover, base, top, kind, optical depth** (the client draws them from the base and the list).
- WL1, the cover `C = 1 − √((1 − RH)/(1 − RH_c))` for RH > RH_c (Sundqvist, Berge & Kristjánsson 1989). RH_c has no law:
  ONE value from its source before any bench (WQ9). Where a system's lift makes RH reach 1, the excess leaves as rain
  (WL22), never a clamp (§11 row 12).
- WL6, the base: the exact lifting condensation level (Romps 2017), a law of the condensable and the gas. Espy's 125 m/K
  holds only for Earth's air and gravity and is never used on another body.
- WL7, the top of deep cloud: Robinson & Catling 2014 (six calibration bodies) or the skin temperature `T_e · 2^-¼`
  (Pierrehumbert 2010); about 15 % apart (ARITH); U-W3 reads both, and the owner chooses on its numbers.
- WL9, the phase: liquid above 0 °C, mixed to −38 °C, ice below (Koop et al. 2000), a law of water.
- WL10, the kind as a CONTINUOUS MIX, never a switch: the convective share a smooth function of the parcel's buoyancy
  (CAPE), the ice share over the phase law's own mixed range (0 to −38 °C). A cell never changes from flat to heaped in one
  frame (G-KIND-SWEEP).
- WL11, the optical depth: `LWP = ½ Γ_ad H²` and `τ_c = 3 LWP / (2 ρ_w r_e)` (Brenguier et al. 2000; Stephens 1978); the
  cloud's extinction comes from this one law (no band of extinction values beside it: one model for one quantity, CR1).
  The droplet count N_d, and so r_e, is an Earth calibration (§3.4).
- The daily cycle of convective cloud: the convective cover reads the almanac's daily swing (WM1), so fair-weather cumulus
  grows by day and fades at night. On the home planet's 72 h day this is a strong signal.
- Organisation: rolls (cloud streets) against open and closed cells by the boundary layer's stability, the spacing by its
  depth (a published regime rule; ESTIMATE source, owed before WP3); the orographic cap and wave clouds from the wind and
  the ground's own slope (both hosts hold the slope, SL10).
- Stratocumulus decks under an inversion (WL5: Klein & Hartmann 1993; Wood & Bretherton 2006): the slope of the cover per
  kelvin is an Earth fit. WQ18 asks whether they come in the first step.

**WM5 — Rain and snow: the budget, the phase, the drops.**
- WL22, the rain as a STOCHASTIC WEATHER GENERATOR conditioned on the climate (Richardson 1981; Wilks & Wilby 1999),
  applied as the sources state it: the occurrence a two-state Markov chain, the amounts a gamma distribution, the
  climate's annual rain the condition. Where the wet-time share goes to zero, the generator's own form gives zero rain (no
  division, no floor; §11 row 16).
- THE CHAIN READS THE LIST, so the rain has one source. At each step and weather cell the chain's two transition
  probabilities (dry→wet and wet→wet) are those of its state: under a system's rain field (the shield, the fronts and the
  stage, by WL15 from the rows) or outside it; a generator conditioned on the circulation this way is a published class
  (Wilks & Wilby 1999's review; ESTIMATE until read). The four values (two states, two probabilities) are named Earth
  calibrations, one value each, written before any bench (§3.4 row 14); U-W5 only reads the dry spells beside Earth's
  stations. The chain's state of each cell (wet or dry, and its rate class) is part of the anomaly, so it ships in the
  statement. THE RAIN ON SCREEN takes the cell's shipped state and shapes it inside the cell by the systems' own rain field
  (the shield's edge, the fronts; `vd-weather`'s law on the same rows), so the drawn rain follows the drawn storm, and the
  server's wet ground and new snow follow the same cells.
- WL23, the intensity scatter: a gamma distribution on wet days, its shape an Earth fit; the extremes follow the ascent's
  saturation humidity (O'Gorman & Schneider 2009).
- WL24, rain or snow: the logistic in the air's temperature and humidity (Jennings et al. 2018). The melting physics is a
  law; the offset over 0 °C is an Earth calibration.
- WL26, the drops: Marshall & Palmer 1948, `N(D) = N₀ e^(−ΛD)`, `Λ = 4.1 R^−0.21` (N₀ an Earth fit), up to the largest
  drop that the published break-up law allows (ESTIMATE source owed; §11 row 17); the fall speed
  `v ≈ √((8/3) g D ρ_w / (C_D ρ_a))` reads g and the air. A snowflake's fall speed is the same terminal-speed law with the
  crystal's mass and area; only the crystal habit (the mass and area per size, Locatelli & Hobbs 1974) is Earth's.

**WM6 — The wind: gradient, surface, gusts.**
- WL27, the gradient wind `V²/r + fV = (1/ρ) ∂p/∂r`, ρ = pμ/(RT). With f a third of Earth's, a system needs a third of the
  pressure drop for the same geostrophic wind (ARITH).
- WL28, the free wind to the surface: the geostrophic drag law `u*/G = κ / √((ln(G/(f z₀)) − A)² + B²)` and the log profile
  (Garratt 1992, ESTIMATE source); z₀ over water by Charnock 1955 (it reads g); over land from the cover (after phase 6).
- WL29, gusts: `σ_u = 2.5 u*` (Panofsky & Dutton 1984, ESTIMATE source) with the full distribution, never cut at a peak
  factor (§11 row 18); a named scatter in the server's physics only.

**WM7 — Lying snow and wet ground** (look in the first step, WQ3).
- WL32, the snow water grows by the snowfall's mass. WL33, the new snow's density for the drawn depth (Hedstrom & Pomeroy
  1998, an Earth fit).
- WL34, the melt by the energy balance `Q_M = SW↓(1 − α) + LW_net + Q_H + Q_E + Q_G` (Hock 2005). The degree-day factor
  is an Earth fit to each site and is refused.
- WL35, the season's snow line meets the equilibrium line at the summer's end: a GATE against `ela_z` on glaciers (U-W7).
- WL36, the climatological pack `SWE_clim(t)`, a closed form per node and date: both hosts may compute it (T3 class 1
  names "the snow line this month" in the almanac). The storms' anomaly is live and comes in the statement.
- Wetness: the rain less the drying by FAO-56 Penman–Monteith (DOC ground §8.4), whose reference crop is an Earth
  calibration (§3.4).

**WM8 — The wake and the restart** (§3.6): the list as a closed form of (the epoch, the tick); the anomaly stored and
advanced over the gap by the same laws.

### 3.4 The named Earth calibrations (B41 item 2; B38 item 4)

Each one gets ONE value from its source, written into the code and the bench reports BEFORE any bench runs, and registered
in `docs/design/DEFERRED.md` as an interim in the pattern of D-TERRAIN-10 (the constant, its calibration body, the law
owed). A bench READS the result beside Earth's readings and never moves the value (B38 item 4). WQ9 asks the owner to
accept this list.

| # | Constant | Where | Its one value and source | Body | The law owed |
|---|---|---|---|---|---|
| 1 | RH_c, the humidity spread in a box | WL1 | Quaas 2012's satellite law of RH_c against the box size, fixed before any bench (ESTIMATE until read); its value at the grid level that U-W12 picks follows from that law, so no bench chooses it | Earth | Tompkins 2002's prognostic spread (WQ9 (b)) |
| 2 | The droplet count N_d | WL11 | about 100 cm⁻³ over the sea, a few hundred over land (ESTIMATE until read; ONE value each, written before WP3) | Earth's aerosols | a body's aerosol sources |
| 3 | The rain–snow offset over 0 °C | WL24 | Jennings et al. 2018's coefficients | Earth's 6 883 stations | the melting distance by the fall speed and the melting time |
| 4 | The new snow's density | WL33 | Hedstrom & Pomeroy 1998's fit | Canada | a crystal-habit law |
| 5 | The steering level; the life curve's shape and wind share | WL17, WL18, WL15 | the steering level at 0.7 of the body's own surface pressure (Earth's 700 hPa practice as a share of its surface pressure); Hoskins & Hodges 2002's curve (ESTIMATE until read) | Earth's reanalyses | an eddy life-cycle and steering law |
| 6 | K, the boundary layer's eddy viscosity | WL14 | 5 m²/s (Holton & Hakim 2013, ESTIMATE) | Earth | a boundary-layer law of the body |
| 7 | The drag-law constants A and B; Charnock's α; σ_u/u* | WL28, WL29 | A = 1.8, B = 4.5 (Garratt 1992); α from Fairall et al. 2003; σ_u = 2.5 u* (all ESTIMATE until read) | Earth; universal in principle | — |
| 8 | Marshall & Palmer's N₀; the gamma shape | WL26, WL23 | N₀ = 8 000 m⁻³ mm⁻¹; the shape from Wilks & Wilby 1999 | Earth | — |
| 9 | The snowflake's crystal habit | WL26 | Locatelli & Hobbs 1974's mass and area per size | Earth | a growth law of crystals in the body's air |
| 10 | The stratocumulus slope (if WQ18 (a)) | WL5 | Wood & Bretherton 2006's slope per kelvin of EIS | Earth's subtropical oceans | — |
| 11 | FAO-56's reference crop | WM7 | grass of fixed height, albedo and resistance (FAO-56) | Earth | the cover's own law (phase 6) |
| 12 | The dust constants A_N and γ (if WQ16 (a)) | WL31 | Shao & Lu 2000 | Earth's wind tunnels | the soil step's grain law |
| 13 | Under the weather, in the static climate: the moist lapse share 0.666; the OLR sensitivity B 1.9 and the diffusion D 0.44; the rain supply 990 mm at 288 K; the belt shares; the orographic gain from the Cascades; Ohmura's curve; the 50 m mixed layer; Earth's PET 1 000 mm; ΔH 0.388; the lift clamps | `climate.rs:93`, `:98-99`, `:117-118`, `:122-123`, `:127`, `:140-142`, `:151`, `:161`, `:104`, `:129-130` | as in the code | Earth | phase 2 (A1–A3) replaces several; the rest stay registered |
| 14 | The rain chain's transition probabilities: dry→wet and wet→wet, under a system's rain field and outside it (four values) | WL22 | Richardson 1981's chain, its values from Earth's station fits conditioned on the circulation (Wilks & Wilby 1999's class; ESTIMATE until read), one value each, written before any bench | Earth's stations | a precipitation law of the systems' ascent |
| 15 | The eye's integration time for a rain streak's length | §7.1 | about 0.1 s, the critical duration of Bloch's law in daylight (ESTIMATE until read), written before M-RAIN | Earth's observers | the integration time as a function of the adapting luminance, with the eye's rule's data |

### 3.5 Three findings to settle before the build

- **F2 — two circulation laws disagree.** Today's Held & Hou (ΔH fitted to Earth) gives the home planet ONE cell from pole
  to equator: an east wind everywhere and no storm belt. The eddy-onset law gives an edge near 52°, and general-circulation
  runs at a quarter to a half of Earth's spin keep a storm zone (Showman et al. 2013; Kaspi & Showman 2015). Ground A3 plans
  Held & Hou with Lindzen & Hou. The static climate and the weather MUST read one law. If WQ1 (a) is ruled, U-W2 reads each
  candidate against the published Hadley extents of Earth, Mars, Titan and Venus (readings) with the ground step's climate
  bench (M0-9, which B52 item 1 moved into the phase that uses it), and the owner chooses.
- **F3 — the storms need the land's season.** Ground A1 is the two-column season in substance; U-W1 runs on A1's output
  (WQ2).
- **F5 — a dry body uses the moist lapse.** With no water nothing releases latent heat, so the desert world's lapse is the
  dry adiabat; its tropopause, N and storms move. U-W11 measures it; the change belongs to the climate (§11 row 21).

### 3.6 The list as a closed form, and ONE boot rule for a wake and a restart

- **The facts.** A dormant realm's shard is killed (`rlm.rs:417`), and a wake boots a new process from the realm store
  (`bin/shard.rs:330-337`), the same path as a kill-9 restart. No general realm checkpoint exists; the store holds row
  families (`built_store.rs:1-30`).
- **The list is a closed form of (the weather epoch, the tick).** Births in each storm belt come per time bucket of one
  weather step: the count in a bucket is a Poisson draw at the law's rate for that season, from a counter hash of (the
  epoch, the belt, the bucket); each birth's place and scatter come from the same hash. Each system then evolves by its own
  laws only (the class rule, WM3): its drift integrates the steering flow of the static climate and the season along its
  own path from its birth, and its strength is its life curve at its age. So the systems alive at tick t are the births in
  the buckets of [t − L_max, t] whose lives cover t, where L_max is the life law's own longest life for the season (the law's
  value, never a cap). The shard evaluates it at each weather step (incrementally: one bucket more, the alive systems one
  step older), and at a boot from scratch.
- **This is the dormant-world rule** (T3 class 2; LAW-WL-1, `scripts/dormant_world_simulation_design.md:180-183`: "THE
  CLOSED FORM IS THE MASTER … Live == dormant becomes true by construction"). A dormant planet advances by no stepping; at
  the wake the shard evaluates the formula at the present tick. Nothing is drawn afresh and nothing is thrown away: the
  pilot who left a low over the northern plains three hours ago finds the same low, older and further east.
- **The anomaly is history, so it is stored.** The shard writes the anomaly (the storms' new snow and wetness) to its weather
  family in the realm store at each step. At a boot it advances the stored anomaly from its stored tick to the present by
  the same laws, with the systems of the gap given by the closed form:
  1. over a gap up to τ_exact, at the weather step itself (the catch-up costs about 0.4–0.7 s for an hour, ARITH on the
     ESTIMATED 1–2 ms a step), so a kill-9 restart states the same statements byte for byte as a run that never stopped
     (G-RESUME);
  2. over a longer gap, at the coarsest step that G-WEATHER-CONVERGE allows, and only over the last τ_snow, the melt law's
     own time for an anomaly to fall under its quantum at that place and season (computed by the law, not Earth's two
     weeks). A long sleep's anomaly then differs from a never-stopped run only within that gate's tolerance, and nobody saw
     the planet while it slept.
- **The epoch** is drawn once, at the planet's first boot ever, and kept in the weather family (WQ8 decides the source). It
  is never shipped. The code is public and enough observed births can in principle invert a plain hash (seed ruling S2:
  "enough observed world inverts to the seed"), so the counter hash is KEYED by the epoch, and the design does not rest
  on secrecy: the weather is not valuable today (S1), and a forecast that a player computes breaks nothing that pays.
- **A never-solved planet.** The weather's static inputs come from the first solve (minutes on the home planet, B56 and B65).
  Its first statement waits for the solve; its clouds then rise in by the eye's fade with its first look.
- **The store family**: one new prefix byte in the realm store beside the artifact's, a schema id, append-only tags
  (`built_store.rs:1-30`'s rule): the epoch, the generation, the anomaly's base and the changes since it, the last stored
  step and tick.

### 3.7 The stepper's place

- **The planet's own shard** computes the weather (B35 item 6); a moon's shard computes the moon's.
- **The gate.** The weather system is installed on every shard by the install rule (`register.rs:353-358`: the capability
  decides what a system does) and acts at boot by the body's facts, as the surface does (`shard.rs:281-322`, "a capability
  of the body, never a kind test"): a charter with air (`CHARTER_FLAG_HAS_AIR`, `look.rs:168`) and a lawful sun (WA3). An
  airless body has no zone and no condensable, and G-EMPTY asserts its empty list. The icy moon has air but no lawful sun
  (§5.5): its weather stays OFF, and its stamp row says why (G-EMPTY does not read it as empty by its laws).
- **Off the tick.** The anomaly's step and the boot's advance run on a worker in the artifact worker's shape; the tick loop
  polls it; the list's incremental step is microseconds (ESTIMATE). No unbounded work runs inside a tick.
- **The crates** (HR5: the Tier-A list at `justfile:12` grows by two):
  - `vd-weather` (new, Tier-A, both hosts): the shared laws (a row to its fields, the gradient and surface wind, the drop
    law, the advection's closed form, §6.8) and, under WQ4 (b′), the one fenced integer almanac function. No collision and
    no static shape reads it.
  - `vd-weather-shard` (new name proposed, Tier-A, server only): the closed-form list, the anomaly, the boot rule, the
    almanac's evaluation from `PlacementTold`, the per-child ambient provider (§7.3). It reads `vd-terrain`'s climate,
    `vd-physics`'s Kepler and `vd-weather`.
  - `crates/bins`: only the worker's wiring and the injected providers, as the solve does (`vd-terrain` holds the laws,
    `artifact_worker.rs` the shell).
  - Std f64 math serves the Category C host; no direct `libm` dependency (none exists today).
- **The random source.** No `sim::io` random seam exists, and its one design is seed-derived (§2.1 item 7). The epoch's
  source is WQ8. Every later random number is a counter hash of (the epoch, a purpose, an index), with a fixed order and no
  dependence on a sum's order.
- **Precision.** The shard holds positions in 64-bit and rounds only when it ships.
- **The weather grid comes from the body.** A rung's face edge is N >> rung (`ladder.rs:173-175`), and the home planet's N is
  19 · 2¹⁹, so 128 cells a face is no level of its grid. The weather grid is a level of the body's own macro pyramid: on the
  home planet 152 cells a face (138 624 cells of about 66 km) or 76 (34 656 cells of about 131 km) (ARITH). The coarsest
  level whose statistics agree with the next finer one within the stepper's own noise is chosen (G-WEATHER-CONVERGE at C
  against 2C, B39's principle). The new seam answers (a cell's area; the local Coriolis parameter from the realm's own spin;
  the local up) are new code under G-MAPPING-TABLE and G-MAPPING-ROUNDTRIP (V6 A1). `GridMapping` answers none of them today
  (`grid/mod.rs:122-214`).
- **HR4 as the code runs it** (`containment.rs:1076-1130`): the identical weather fixture drives the weather feature through
  the shard rig on `NodeKind::Shard(planet)` and `NodeKind::Shard(station)`, with the charter's air and sun facts injected
  by the test, in the shape of `assert_feature_anywhere` (which injects a synthetic child region). Both runs must state the
  same birth and death ticks, and the same drifts in each grid's own coordinates (V6 A1: no fixture gives equal numbers on
  a bent grid and a flat one); the weather feature reads its geometry (a cell's area, the Coriolis parameter, the up) ONLY
  through the seam. Example: one synthetic storm belt runs in a planet shard and in a station shard; both state a birth at
  the same tick, and each states the drift in its own grid. The fixture lives with the rig, with `vd-weather-shard` as a
  dev dependency of `vd-sim` (as `vd-physics` is today). **The fact the owner must read:** no second realm kind holds air
  in the world
  (`charter.rs:497`), so the station run's air comes from the test. If HR4 needs a real second kind with air, the build
  waits for one (WQ6).

### 3.8 What the first step does not deliver

Tropical cyclones (WL20); fog; slope winds, blocking and foehn; the eddy-set tropopause; squall lines; LIGHTNING (the storm
at dusk lacks its light; WQ18); lake-effect snow; sea ice growth; blowing snow and the snow's ageing; the humidity spread
and the condensate cover; rain-fed puddles, rising lakes and floods (W2 covers only the dug hollow); moonlight and starlight
on clouds; contrails; the sail's thrust; the wind on a walker on an open deck (§7.2); icing on hulls and wet suits; dust on
the desert world unless WQ16 says yes; THE BIOME'S SEASONAL PAINT (the leaf colour, the dry grass): outside B35 item 5's
scope and B36 item 5's fence, for the ground step's phases 5–6 and the art discussion (B36 item 4); A GAS GIANT'S CLOUD
BANDS (`shard.rs:309-312` promises "the cloud shell of its own slice"; its own laws, the banded jets by the Rhines scale,
priced later). Each one is a later law with its own price.

---

## 4. The weather list and its lane

### 4.1 The statement

| Part | Content | Size (ARITH, postcard) |
|---|---|---|
| Header | the realm (a `RealmId`, about 10 B); the GENERATION (u32: it changes when the epoch changes); the step index; the step's universe tick; the anomaly base's digest (a [u64; 2], about 20 B, as the artifact's); the row count (u16) | about 35–40 B |
| A row | `id` u32 (unique within a generation); `centre` 3 × i16, a unit direction in the planet's frame (about 194 m on the home planet); `radius` u16 as a share of half the body's circumference (about 304 m units on the home planet); `dp` i16 as a share of the body's surface pressure (a low can never pass p_s); `drift` 2 × i16 as a share of the body's gravity-wave speed √(g·H_tp); `stage` u8, the life's share; `kind` u8 | about 19–23 B (an i16 of a unit vector over 0.25 zigzags to three bytes) |
| The anomaly's changes | a run-length list of (cell, the changed words): the storms' new snow and wetness, the melt and the drying, and the rain chain's state | NOT zero in dry weather: after a storm the melt (WL34) and the drying change the cells at each step until they reach their quantum; up to about 24 KB a step for each snowing system (ARITH upper bound); MW-1 counts every kind of change |

- Each field is in the body's own units, so a field cannot reach its bound except by a defect. A value outside its field is
  REFUSED loudly and counted on the stamp row and in the log, never clipped. U-W1 reads the extremes on every body of the
  home system, the 25 407 km world with 28.7 bar of air included (its ground air is about 18 times as dense as the home
  planet's, ARITH: a home low of 2 000 Pa becomes about 36 000 Pa there, which a raw Pa field would not hold).
- The client draws the fronts and the shield from the centre, the radius, the drift and the stage by WL15.

### 4.2 Identity, generation, birth and death

The client matches two statements' rows by `id` inside one generation and interpolates each system's centre (along the
great circle), radius, pressure, drift and stage. A new generation (a new epoch after a lost store) drops the client's book,
so an old id never names a new storm. A new `id` starts from zero strength and a departing one ends at zero, because the
chosen life curve does (WL18): continuous on screen by that curve.

### 4.3 Timing: what interpolates and what steps

- One statement at each weather step (10 s, an operational number in the one config struct; G-WEATHER-CONVERGE proves that
  the look does not depend on it).
- THE SYSTEMS are drawn ONE STATEMENT BACK: the client interpolates between the last two statements and never
  extrapolates. They are 10–20 s old; a system that drifts 10 m/s is 100–200 m behind the server's, nothing that the eye
  sees on a 7 000 km storm (ARITH).
- THE ANOMALY'S LOOK is also drawn one statement back, interpolated between its last two states. In 10 mm/h rain a step at
  each statement would darken a whole plain by 1.1–2.2 % at once against an eye threshold near 0.3 % (ARITH, reviewer C);
  the interpolation turns that into a ramp. G-MAP-STEP reads the eye's rule on a plain at the onset of rain.
- THE SEASON is continuous: under WQ4 (b′) the client evaluates the base at its own almanac's date every frame; under WQ4
  (a) the planet ships the base for the next keyframe tick (one planet day ahead) before that tick, and the client
  interpolates the two keyframes on its render clock.
- A COLLISION STATE (only under WQ3 (b)) would step at a stated tick: the step's tick plus a lead of one gateway beat and
  the transport's bound (about 0.6 s, ARITH), applied by both hosts at that tick, the client on the render clock that also
  places the walkers' poses (`channels.rs:547`).

### 4.4 A late or stale list (WQ11)

- THE BOUND (SL1 clause 6, derived): the time over which the last two statements, run forward by their own rate of change,
  stay within the eye's threshold of the picture at the eye's place, and never less than two weather steps plus one beat
  (20.5 s, ARITH). That lower end is the lane's own law, not a chosen floor: drawing one statement back needs two
  statements, and the next one is due a step and a beat later. The weather changes slowly, so this bound is long for a
  broad deck and short near a moving cloud edge.
- UNTIL THE BOUND the client HOLDS the last state: the systems still, the noise's advection still running, so the sky stays
  alive (no extrapolation of the systems).
- PAST THE BOUND the client refuses the list (no new motion, no new system), and the picture fades to the season's base over
  THE EYE'S FADE, defined per frame, so its seconds follow the frame rate. By the eye's rule a 3× change takes about 6 s and
  a 10× change about 13 s at 60 frames a second, 8.6–18 s at 43 (ARITH, reviewer C on `eye.rs`). A whole storm dissolving in
  about ten seconds passes the eye's rule but is not weather (B35's bar): the hold is the design's answer, and the owner
  judges it.
- The stamp row says "held" or "stale" and gives the age.
- A LATE FIRST STATEMENT rises in by the same fade. B18 item 6 rules 0.5 s for a late chunk; a 0.5 s rise of a whole storm
  breaks the eye's rule, so the weather takes the eye's fade. On the normal path the statement goes first in the gateway's
  beat and arrives no later than the pyramid's first part, so the first picture of the planet already holds its weather
  (M-SERVE measures it).

### 4.5 The lane

```
 the planet's shard (computes on a worker)              the gateway                        the client
   each 10 s: WeatherStatement   ──BulkFor{Realm}──▶   keeps per realm: the base, the   ──WeatherPart──▶  the weather book:
   WeatherMap: the base, the anomaly base (by digest)     anomaly base, the statements                      two statements, the
   ◀── WeatherWant { realm, held digests, held step } ── each beat for a drawn realm with air   ◀── WeatherHeld ──   base, the fade, the stamp
```

1. **Shard → gateway.** `ShardToGateway::BulkFor { audience: BulkAudience::Realm(realm), bytes }` exists
   (`session_flow.rs:552-556`, `:600-606`). Two new `BulkMsg` arms: `WeatherStatement`, and `WeatherMap { kind, digest,
   part, parts, bytes }` for the static base (WA2) and for the anomaly's base, in parts of at most 32 KiB, COARSEST LEVEL
   FIRST (each level the mean of the finer, rule M), so a far planet's clouds come with its first look; a finer level that
   lands while the planet is drawn rises in over B18 item 6's ramp.
2. **The want.** `GatewayToShard::WeatherWant { realm, held_base, held_anomaly, held_step }`, discriminant 9, each beat for
   every drawn realm whose look bag states air (`CHARTER_FLAG_HAS_AIR`, read with `charter_of_bag`, `look.rs:405`), never for
   stars, hulls or airless bodies. The shard answers only with what is newer, and keeps no list of subscribers.
3. **The gateway.** A new branch beside `cache_artifact_part` (`artifact.rs:316` today calls such bulk a stray) keeps, per
   realm, the newest base, the newest anomaly base and the statements since it. THE RE-BASE RULE: the shard states a new
   anomaly base when the changes since the last one reach that base's own size, so the statements since a base never exceed
   one base. The gateway's memory is bounded once per gateway (ground §7.6); when the bound binds, the realm that no session
   drew for the longest drops its weather first and asks again when drawn. THE STATEMENT GOES FIRST in each beat's budget.
4. **Gateway → client.** A new `ServerControlMsg::WeatherPart { bytes }`, discriminant 20, carries an encoded weather
   `BulkMsg` on the Control stream, gated on the negotiated minor as `relay_bulk` gates `ArtifactPart` (`gateway/shard.rs:525`):
   a session below the weather's minor never receives it.
5. **The receiver states what it holds.** A new `ClientControlMsg::WeatherHeld { realm, base_digest, anomaly_digest }`,
   discriminant 9, as `SkyHeld` and `ArtifactHeld` do, because "the sender is not the party that knows" (`channels.rs:63-83`).
   A pilot whose session moves to a second gateway states the held bases and gets the statements only.
6. **The receivers.** Every session whose picture holds the planet, not only its occupants (the owner, 2026-09-20; ground
   C19).

### 4.6 Sizes and rates (ARITH; postcard)

| Item | Home planet | On Earth's facts |
|---|---|---|
| Systems N | about 4–12 | about 25–30 |
| A statement with no anomaly changes | about 110–290 B | about 0.6–0.7 KB |
| The steady lane, per client, per planet | about 11–29 B/s | about 60–70 B/s |
| The anomaly's changes in a storm | up to about 2.4 KB/s for each snowing system | the same per system |
| The static base (WA2), 8–16 B a cell | 1.1–2.2 MB raw at 152 cells a face; 0.28–0.55 MB at 76 | — |
| The anomaly base, 2–5 B a cell | 0.28–0.69 MB at 152; 0.07–0.17 MB at 76 | — |
| A base at the gateway's pace | one to three beats (0.5–1.5 s) | — |
| Compressed sizes | UNMEASURED (MW-1) | |

### 4.7 The arms and the minor

- Today minor 36, floor 36. The ground step's A0 arm takes `BulkMsg` 7 at the next free minor if it lands first (ground C5).
- This step APPENDS six arms: `BulkMsg::WeatherStatement` and `BulkMsg::WeatherMap` (the next two free, 8 and 9 after A0);
  `GatewayToShard::WeatherWant` (9); `ServerControlMsg::WeatherPart` (20); `ClientControlMsg::WeatherHeld` (9);
  `InterShardFlow::PlacementTold` (45, WA3). Appended enum variants gated on the negotiated minor keep the floor
  (`version.rs:8-14`, `:369-371`); the minor moves to the next free number.
- `PlacementTold` is classified like `ReachStated` and `ChildFelt`: `EffectClass::FireAndForget` (a stated fact that
  commands nothing; an effect-free arm carries no idempotency key, `intershard.rs:1018-1020`) and
  `FlowDurabilityClass::ProducerLessReliable`, pushed retained, so a once-only statement is replayed and never lost. The
  golden pin moves from seven producer-less-reliable arms to eight (`intershard_closed.rs:1260-1267`), and the closed test
  from 45 arms to 46.

### 4.8 The SL6 asks (WQ10)

**WA1 — the weather statement, its want and its held arm.**
- Data: the header, the systems' rows, the anomaly's changed cells (§4.1); the anomaly's base by digest.
- Directions: the planet's shard → the gateway → every client whose picture holds the planet; the want gateway → shard; the
  held arm client → gateway. No realm-to-realm crossing (the connection plane is not a realm, SL2's clarification). A moon's
  statement goes from the moon's shard to the gateway.
- Why the receiver cannot compute it: live state (T3 class 2); the epoch is server-private; the client may not simulate it
  (SL10, V1 item 7).
- Without it: no weather at all.
- Size and rate: about 110–290 B each 10 s on the home planet, plus up to a few KB a step in a storm (ARITH).
- Arms: `WeatherStatement`, `WeatherMap` (the anomaly kind), `WeatherWant`, `WeatherPart`, `WeatherHeld`.

**WA2 — the static weather base.**
- Data: per weather cell, the climatological fields as functions of the season (mean cover, base height, top height, kind
  mix, precipitation, the cloud-level wind, the temperature and the precipitation for WL36's pack), as an annual mean and
  harmonics (the count of harmonics by the bench). No biome paint (§3.8).
- Direction: as WA1, by digest, once per solve.
- Why the receiver cannot compute it: the client holds climate bytes only near the eye (the rows reach only the finest
  ring), never for the globe.
- Without it: no clouds from orbit or at the horizon that follow the climate; a planet with uniform noise clouds.
- Size: 0.28–2.2 MB raw once (ARITH, by the grid level); compressed UNMEASURED.
- Arm: `WeatherMap` (the base kind). Under WQ4 (a) it becomes the season-evaluated base, shipped as keyframes ahead.

**WA3 — the orbit elements and the spin law** (ground A12; ruled in substance by B36 item 5).
- Data: the planet's Kepler elements, the spin rate, the pole's direction and the spin's phase at the epoch: a few dozen
  bytes (DOC ground A12). The pole's azimuth and the spin's phase are two NEW identity draws (T9), appended after `u_cos` in
  the charter author's frozen order (`charter.rs:530-538`), so no earlier word moves; a world change with its pins (WP1).
- Direction: the star system → the planet, once on change, stamped (SL1 clause 2).
- THE SPELLING: `PlacementTold`, from the SL1 vocabulary (`pose.rs:20-30`). It joins `DOWNWARD_STATEMENTS`
  (`intershard_closed.rs:990-995`), the one arm allowed to carry a placement down, and the orbit elements' type joins the
  wire pin's placement types (`intershard_closed.rs:982-986`), so no other arm carries Kepler elements unseen.
- THE FENCES. The module fence (`crate_isolation.rs:165-230`) keeps scanning its nine placement, containment and crossing
  modules (`:176-186`, `crates/sim/src/stub/placement.rs` among them), and none of them may name `PlacementTold`. So the
  star system's SENDER of `PlacementTold` lives in its own module OUTSIDE the nine. The names that may spell the arm are an
  allow-list with a planted control: the wire enum and its two classifications, the closed tests, the star system's
  sender, the planet's receive path, and the almanac module, which alone READS the elements. A second fence keeps the
  almanac from calling the boot's generated orbit (`boot_charter` → `body_charter_in_subtree`,
  `crates/bins/src/lib.rs:3263-3278`).
- Why the receiver cannot compute it: SL1 clause 3.
- Without it: no day and no season in the planet's weather and melt.

**Never crosses:** the medium (M2a); the planet's sun or its season's phase as a number (B36 item 5); the weather epoch; a
per-observer weather window (the discussion's A5); an occupant's pose (SL2); a weather datum from realm to realm.

---

## 5. The almanac and the planet's own sun

### 5.1 The closed forms

- The planet's place on its orbit (Kepler, `crates/physics/src/celestial.rs:53`, `:204`, `:229`); the sun's direction in the
  planet's frame (the orbit's place turned by the spin law); the day's phase; the season's phase (the solar longitude); the
  sub-solar point; the daily mean insolation (Berger 1978, ESTIMATE source); the daily swing (ground A7); the climatological
  pack (WL36); the season's base (from WA2's harmonics).
- The home planet (ARITH on `home.rs:117-120` and the 50 Hz tick): the rotation period (sidereal) 259 597 s = 72.11 h; the
  SOLAR day 261 552 s = 72.65 h; a tilt of 55.1°; an eccentricity of 0.0155; a year of 401.9 days of real time, 133.8
  rotations or 132.8 solar days; a season of about 100 days; the sun moves about 4.96° an hour; an equatorial night lasts
  about 36 h.

### 5.2 Who computes what (under the recommended roads)

| Who | What | Rule |
|---|---|---|
| The planet's shard | the almanac from `PlacementTold` (B36 item 5) | used only for the weather, the melt and the light; stated on no lane; fenced from the boot's own orbit |
| The client | the light's sun from the composed picture (S7-7); the season from the seed by one fenced integer function (T3 class 1; WQ4 (b′)), with the orbit's two angles by WQ4's road | the function feeds only the season's look, WL36's pack and the waves' seasonal wind (§7.6); never a pose, a crossing or a placement |
| The star system | the planet's attitude in the placement that it authors (V12); `PlacementTold` once | the parent is the only writer (SL1 clause 1) |
| Placement, containment, crossing | nothing of the almanac | the module fence with its planted control |

G-ALMANAC-PIN: the client's integer almanac and the planet's give the same integers at the same tick on every stand; it
can fail, because the planet reads `PlacementTold` and the client reads the seed.

### 5.3 The turning planet (WQ5)

**The facts in the code.**
- The placement that the star system authors has the identity orientation and no spin (`frame.rs:92-101`); the code says
  "in the planet's own frame while the planet does not spin" (`crates/bins/src/lib.rs:1129-1132`). The drawn obliquity is
  the orbit's small inclination (§2.1 item 2).
- The crossing conversion already reads a parent's angular velocity on both sides (`frame.rs:250-251`, `:273`).
- The gateway's relayed hop writes the angular velocity as zero, a ledgered P10 owe (`window.rs:1083-1097`): a composed
  velocity would miss up to 153.5 m/s at the home equator (ARITH).
- The drive integrator has no turning-frame terms (`drive.rs:139-182`).
- The client draws chunks by the row's facing (`terrain.rs:2481-2486`, `:3930`).

**The rulings that meet here, quoted.**
- V12 (2026-09-08): "the parent authors the spin on the row, the recipe lives in the planet's own frame, only the light
  changes on the surface".
- The owner's ground rule (2026-08-09, the memory `feedback_only_parent_knows_positions`): "Planet doesn't know its spin or
  position"; "Spin is done also by the Star System realm, not the Planet"; "a planet realm has no concept of rotating at
  all".
- SL1 (2026-08-24): clause 2, a parent MAY state the placement as a stamped read-only reading; clause 3, a child never
  derives its placement; clause 5, the placement, containment and crossing machinery may not name a realm's own position.
- V13 L12 (2026-09-09): the charter's "about twenty quantised integers (gravity, spin, tilt, …) authored ONCE, stored, and
  stated by the realm in its own surface statement".
- B36 item 5 (2026-10-02): the planet evaluates its own sun from the orbit and spin law stated once, fenced to "the weather,
  the melt, the light".
- M2 (the movement contract): the placement that goes down holds "position, velocity, facing, spin".

**The physics of a turning frame.** If the realm's frame turns with its ground (V12), a hull that hovers co-rotates, and the
planet's physics needs the turning frame's two terms: the Coriolis term (0.026 m/s² at 528 m/s) and the centrifugal term
(3.7 mm/s² at the equator) (ARITH). Without them, orbits inside the planet's realm turn with the ground: the moon's authored
orbit would gain the planet's spin, about ω × r in speed (about 24 km/s at 10⁹ m, ARITH by reviewer A; with ω²r ≈ 0.59 m/s²
there, more than the planet's pull).

**The roads.**
- (a) THE FRAME TURNS (V12 kept). The planet's physics takes the two terms from ITS OWN CHARTER WORDS, the spin rate
  (`day_s`) and its own pole (+Z), words that V13 L12 already lets it state about itself, never from `PlacementTold`. It
  authors its children (the moon, a low orbit) in its turning frame with its own phase origin, so it never needs the
  parent's spin phase: the star system's phase only turns the whole subtree, which keeps every orbit a true orbit. SL1's
  fences see no read of the told placement. It AMENDS the 2026-08-09 ground rule ("no concept of rotating"), which V13 L12
  already narrowed for the rate. It also computes a turning angle of its own frame from its own origin (never the parent's
  phase): whether that is "computing its own placement" under SL1 clause 3 is the owner's reading, and WQ5 asks it.
  G-SPIN-AGREE: the star system's authored angular velocity equals the charter's rate on every body.
- (b) THE FRAME STAYS INERTIAL, AND THE GROUND TURNS INSIDE IT, by the planet's own charter (SL3: how it looks). The physics
  stays inertial, and the children's orbits stay plain Kepler. It AMENDS V12 (the recipe in a turning sub-frame) and the
  2026-08-09 rule (the planet turns its own ground). Every surface occupant stands on a floor that moves at up to 153.5 m/s
  in the realm's frame, every collision and chunk lookup turns by the spin angle at each tick, and the wind co-rotates.
- (c) NO TURNING: the drawn sun stays still while the weather follows the almanac's sun. A seam; refused, unless the
  weather step waits.

**What it touches** (the turning planet's own design, with reviews, prices each item):
1. The gateway's relayed hop must carry the angular velocity (the P10 owe, `window.rs:1083-1097`).
2. The drive integrator gains the two terms (road (a)), or the ground's turn enters every surface lookup (road (b)).
3. The crossing at the equator (153.5 m/s), AND at the edge of the planet's realm, where the conversion is largest (about
   24 km/s at 10⁹ m, ARITH): M-TURN and G-CROSSING measure both, end to end.
4. The moon's placement in the turning frame; a low orbit; a boarding.
5. The parent's two new identity draws (the pole's azimuth, the spin's phase), appended after `u_cos`, with new pins.
6. The stands pin a place AND a tick (two ticks for the two suns, B36 item 8; §10.3).
7. The frozen references change once, on the owner's look (B29, B32).

It is a WORLD-VISIBLE CHANGE: the sun crosses the sky in about 72.65 h, the seasons move the sub-solar point over ±55°,
and every outdoor picture changes. The owner looks at it before any cloud.

### 5.4 The almanac on both hosts (WQ4)

- **The true dates.** V1 item 7 (2026-09-07: the client never derives a realm's state) and S7-1 (2026-09-08: no motion
  crate on the client) are OLDER than T3 (2026-09-16), whose file says "Newest ruling file; it wins over every earlier
  design document". T3 class 1: "both hosts compute it from the seed and the tick; no state, no lane. The sun, the
  seasons, the day, the moon, the tides, and THE ALMANAC: the snow line this month, a river's stage in the melt." So T3
  class 1 already widened V1 item 7 for the closed forms of time. B36 item 5 (2026-10-02) rules the PLANET's road and says
  nothing of the client; after it, the planet holds its orbit by a statement that SL1 clause 4 forbids it to pass on, so the
  client needs its own source.
- **The roads.**
  - (b′) T3 CLASS 1 AS RULED: the client evaluates the planet's orbit from the seed by ONE fenced integer almanac function
    in `vd-weather` (both hosts; a no-drift gate, G-ALMANAC-PIN). S7-1's isolation row narrows to admit that function,
    which feeds only the season's look, WL36's pack and the waves' seasonal wind, never a pose. The season is then
    continuous on the client, and WA2 is static.
  - (a) THE PLANET SHIPS THE SEASON'S LOOK: the client computes no date; the base comes as keyframes ahead of their ticks.
    It AMENDS T3 class 1's client half. Its further price: a base evaluated at the planet's date puts the date on a lane in
    another form, which B36 item 5 ("nothing states it on any lane") also asks about; and the daily cycle of cumulus needs
    keyframes shorter than a day (the solar day is 72.65 h), so more bytes.
  - (c) THE COMPOSED PICTURE: the season read from the drawn sun. Floats, never equal on both hosts; refused for anything
    that collision would read.
- **The price that (b′) must also pay: the orbit's two angles.** The client's season needs where the planet stands on its
  orbit at a tick: the orbit's start angle (the mean anomaly at the epoch) and its periapsis angle. Of the orbit, the
  charter holds the year and the eccentricity only (`look.rs:155-158`; its time words `:136-158`), never these angles.
  They are float draws of the star system's generator (`planet_element_draws`,
  `crates/physics/src/worldgen/generate.rs:161-163`), and a planet may not state them about itself (SL1 clause 3; B36
  item 5). Three roads give them to the client:
  1. THE CLIENT RUNS THE GENERATOR'S ORBIT DRAWS: the same code (no port), linked into the client for the element draws
     and the part of the system's generation that seeds each planet's stream, and quantised by the same integer rule that
     the star system uses for `PlacementTold`. S7-1's isolation row narrows further (the draws, their seeding and the
     almanac). G-ALMANAC-PIN proves that the client's integers equal the planet's. The size of that generation part on the
     client is UNMEASURED.
  2. A SECOND IMPLEMENTATION of the draws on the client: REFUSED by V1 item 2 (one generator; a port is forbidden).
  3. EVERY ORBIT TO ONE INTEGER FUNCTION on both hosts: the star system's orbit authoring itself moves to the fenced
     integer function, so the parent, the planet and the client read one source. A world change in every orbit, with new
     pins, and the crossing and placement code then read the integer orbit.
  A fourth road, the angles in the planet's charter, is REFUSED: the planet would state its own orbital phase on a lane
  (SL1 clause 3; B36 item 5).
- **The recommendation: (b′) with road 1**, because it keeps T3 class 1 as written and moves no world. Its price: S7-1's
  row narrows to admit the element draws and the integer almanac, and G-ALMANAC-PIN. Road 3 is the cleaner single source,
  for the owner to choose if the world may move once more.

### 5.5 A moon's sun (WQ21)

A moon's sun needs its own orbit round the planet AND the planet's orbit round the star. SL1 clause 4: "you are never told
your parent's placement … Never fold a realm's absolute from the root". B36 item 5 fences an exception to clause 3 for a
planet only. The roads: (a) the moon receives what its sun needs: an exception to SL1 CLAUSE 4 (told its parent's placement;
a two-level fold); (b) no weather on airy moons: against B35 item 2 ("other bodies … from the same laws") and item 6 ("a moon
calculates and draws itself"). NEITHER ROAD KEEPS EVERY RULING. SL6 asks for a local formulation first; none is found yet,
and the search is owed before the icy moon (`Planet(61143963952882988)`, 555 Pa, MEASURED 2026-10-02, bench M0-5) becomes a
stand. The home moon has no air and needs none.

---

## 6. The clouds

### 6.1 The recommendation: ONE SHELL, ONE MARCH (kept), with revision 2's changes

The planet's clouds are ONE density field in a shell around the planet, marched at every height with ONE light model; the
far look is the filtered near look. The client draws it as the planet's own look (SL3) from the base, the list and the
composed sun. Its parts come from shipped sources (DOC, read by reader 3 and reviewer C):
1. **The shell** (Hillaire 2020): from the lowest base to the highest top, in the planet's own frame; WHERE IT RAINS, the
   march and the skip structure reach the ground (rain shafts and virga). A thin cirrus slab above it is its own layer, with
   its own shadow map (§6.7).
2. **The density** (Horizon Zero Dawn 2015, Nubis 2017, Frostbite 2016), driven by the laws: the cover, kind mix,
   precipitation, base and top from WM4; the base shape from 3D noise times the kind's height profile; the edges eroded by
   detail noise; the extinction from WL11; the phase function from the droplet size for liquid (Jendersie & d'Eon 2023) and
   a published ice-crystal phase function for cirrus (source owed before WP3).
3. **The march**: a conservative skip over clear air; FOOTPRINT-SIZED STEPS ALONG EACH RAY whose phase is fixed to a
   lattice in the planet's frame (CR5), so a still eye sees still samples and a level ray inside a cloud has its samples
   next to the eye; Frostbite's energy-conserving step; in EYE-RELATIVE coordinates (§6.10). The light toward the sun: a cone
   of 5–6 samples over the first kilometre or two, then the Beer shadow map.
4. **Multiple scattering**: Wrenninge's octaves, calibrated against a path-traced reference of the same density, the air
   under a deck and the ambient path included (M-CLOUD-CAL; a harness tool), never tuned by eye (B38).
5. **Resolution and time**: a FAR pass and a NEAR pass, sized against the DISPLAY (the far pass at a quarter of the
   display's width and height, the near pass at an eighth: half and a quarter of the render size at s = 0.5), split by the
   parallax law and BLENDED BY WEIGHT across a band that both passes march (§6.4); the near pass converges in ONE frame.
6. **The composite after the main reconstruction**, at the display size, from the cloud pass's own resolved history (§6.5).
7. **Shadows** in the clouds' drift frame, by layer, handed over by the pixel's size (§6.7).
8. **Rain and snow**: particles near, streaks in the middle, the medium in the march far, matched on one look (§7.1).

Why: the only option with ONE representation from the ground to orbit, so the seam law is a property of the filter rules;
its author designed the vendored air; no new dependency. REFUSED: a 2D far layer cross-faded to a 3D volume (two light
models give one cloud two brightnesses). Kept as the bench's alternative A1: Red Dead Redemption 2's froxel volume near the
eye with the march beyond it ("NO POPPING", 1.31 ms march + 0.23 ms reconstruction on a PS4, DOC).

### 6.2 The seam rules (B16: every level refines and never contradicts)

- **CR1 — one field.** Every view reads one density function. The far shadow is the same field's transmittance (§6.7), not a
  second model.
- **CR2 — the mean-preserving filter, with the clumpy correction NOW.** An octave under the footprint becomes its expected
  effect, and the transmittance through the replaced octaves is the gamma-weighted mean `(1 + τ̄/ν)^−ν` (Barker 1996,
  ESTIMATE), with ν from the variance of the replaced octaves, which the noise's own spectrum gives. Jensen's inequality
  (E[e^−τ] < e^−E[τ]) makes the plain mean too opaque on every climb. No shipped system does this, so it is a NEW METHOD,
  benched in G-CLOUD-NEST at two render scales before WP3. The footprint is the DISPLAY pixel's, so the render scale (the
  valve, the MSAA path) never changes the clouds' detail.
- **CR3 — no distance cut.** If the planet is drawn, its clouds are drawn; Unreal's "tracing start" limit is refused.
- **CR4 — the split is a resolution change only, with no history**: the split distance comes from the parallax law, and the
  two passes are BLENDED BY WEIGHT across an overlap band that BOTH passes march (§6.4). No dither: a dither needs a history
  to resolve it, and none survives at speed.
- **CR5 — sample positions fixed in the PLANET's frame, at footprint-sized steps along each ray.** Each ray steps at the
  size that its footprint asks for, and the step's PHASE is fixed to a lattice that stays in the planet's frame: for
  example, the samples sit where the ray crosses fixed planes perpendicular to its main axis, at a spacing taken from a
  ladder of sizes (each half the last). Where the footprint lies between two sizes of the ladder, the two marches are
  blended by weight, so a size change never pops; where the ray's main axis changes across the screen, the two lattices
  blend across a narrow band, read by the eye's rule. So a still eye sees still samples, a moving eye sees samples that stay
  on the world, and a level ray inside a cloud has its samples next to the eye. No animated jitter where no history
  survives. Revision 2's step shells at fixed radii are REFUSED: a level ray inside a cloud meets the next shell only
  3.6–11 km out for shells 1–10 m apart (ARITH, the verifier: √(2·r·Δ) at r = 6 344 km), so the middle band of the view
  would show clear air inside a tower.
- **CR6 — shadows nest**: fixed world texels snapped to their grid; a hand-over where the smallest cloud shadow falls under a
  pixel, never at a distance; cascades added by that rule and ramped in (§6.7).
- **CR7 — the field moves smoothly**: the systems by identity, one statement back; the anomaly's look interpolated; the
  season continuous or by keyframes ahead; the stale and late paths by the hold and the eye's fade.
- **CR8 — no lattice and no repeat**: the cover field C¹ across the weather grid and across face edges, WITH A STATED RULE
  AT THE CUBE'S EIGHT CORNERS (a plain bicubic filter is not C¹ there); the noise placed by stochastic tiling (Heitz & Neyret
  2018, ESTIMATE), so no tile's cells repeat from orbit; between the tile and the weather grid, the organisation terms of
  WM4 give structure (B35 item 1: no scale left as plain noise). G-GRID and G-TILING on the cover from orbit.
- **CR9 — one rain LOOK**: the near particles, the middle streaks and the far medium match on the per-pixel mean radiance,
  the transmittance and the variance, never on σ alone (§7.1).

### 6.3 The walk from the ground to orbit (the home planet, radius 6 342 km)

| Where the eye is | What happens | The seam risk and its cure |
|---|---|---|
| On foot under a deck (base about 1.5 km) | Grazing rays reach the cloud horizon about 138 km out (ARITH); the front air marched to the cloud's weighted depth with the full count | Far shimmer → CR5; the ridge against the deck → the edge rule (§6.5) |
| On foot, cumulus over the plains, a 15 m/s wind | Shadows ride the drift; clouds grow and decay over their own time | Shadow stutter → §6.7; rigid sliding clouds → §6.8 |
| A hull at 240 m/s under the deck | The split by the parallax law: one render pixel of error needs about 1.7 km for a 500 m cumulus, 3.3 km for a 2 km tower (ARITH, reviewer C) | Trails → §6.4 |
| Inside and between clouds at 528 m/s | The near regime reaches about 2.5–4.9 km; a cumulus 200 m aside moves about 58 render pixels a frame (26 at 240 m/s) on the 45° seat (ARITH) | One-frame convergence (§6.4); the cost (§8.1); the cockpit veil → the start depth (§6.5) |
| Above the deck at 6–12 km, cirrus above | The deck's sunlit top under thin cirrus | The one-front shadow map would read the deck black (T ≈ 10⁻²⁰ against a true 0.61, ARITH) → a map per layer (§6.7) |
| 30–100 km up | Octaves become their means with the clumpy correction (CR2); cascades added by the pixel rule | A slow level drift on the climb → CR2 |
| Orbit (400 km) | Each disc pixel marches 10–20 km of shell; a 1 km cumulus shadow is 3.5 render pixels at nadir (ARITH) | Repeats → CR8; shadows near the terminator → §6.7, priced in M-CLOUD-COST |
| A far body with air | The same march on its own field | Its first statement late → the base's coarse level, then the eye's fade |

### 6.4 The near pass at speed (the leg between towers at 528 m/s)

- **The split distance from a law.** A cloud pixel reprojects by one depth; the front and the back of a cloud Δd deep at
  distance d move apart by (v_rel/60)·Δd/d² radians a frame, with v_rel the eye's speed relative to the cloud (the wind
  included). The split distance keeps that under one render pixel for the deepest cloud in view. At 528 m/s that is about
  2.5 km for a 500 m cumulus and 4.9 km for a 2 km tower (ARITH); Nubis³'s 200 m was for a flying mount. The two passes are
  blended by weight across an overlap band that both march, so the split needs no history.
- **The near pass converges in one frame.** Its samples sit on footprint-sized steps along each ray with the step's phase
  fixed to the planet-frame lattice (CR5), so the pattern stays on the world and does not change between frames, and a
  level ray through a tower has its samples from the canopy's glass outward; it takes a depth-aware spatial filter at its
  own size. Animated jitter needs a history, and none survives at this speed: a near ray covers 8 × 8 display pixels, where
  the eye sees a one-frame change of 0.83–1.11 % (ARITH, `eye.rs`).
- **The far pass inside the cloud layer.** When the eye is inside the layer (the leg between towers), the far pass takes the
  same rule: footprint-sized steps along the ray from the band onward, phase fixed to the lattice, so a level ray finds the
  towers between the split (2.5–4.9 km) and the horizon.
- **No valve at a cloud's edge.** A lower resolution inside a cloud would switch the sharpness when the hull enters (and its
  16 × 16 blocks lower the threshold to 0.57–0.69 %, ARITH). Any lower quality at speed is a continuous rule of the speed,
  judged by the eye's rule (WQ17).
- **The bench decides the representation.** M-CLOUD-COST and the eye's rule on the 528 m/s leg BESIDE and INSIDE clouds,
  this near pass against alternative A1 (the froxel volume), before WP3 builds either.

### 6.5 The composite after the reconstruction, the air join, the edges and the cockpit

- **Why after.** A thin cloud over moving ground holds two layers that move differently, and the main reconstruction takes
  one motion a pixel: a threshold on the opacity switches between them and leaves ghost trails at a deck's edges. The code
  also ignores a written motion where the depth is zero: `motion_at` takes the sky's motion when `z == 0.0` OR at the
  sentinel (`reconstruction.wgsl:162-172`). So the clouds composite AFTER the main reconstruction, at the display size,
  from the cloud pass's own resolved history (its own upsample and its own motion: the weighted depth plus the wind's drift),
  as the points pass does for the stars. The main history then holds only the ground and the sky, each with a true motion,
  and `motion_at` needs no change. Price: one display-size pass, about 0.44 ms at 273 GB/s (ARITH, reviewer C). The bench
  (G-CLOUD-GHOST over a broken deck at 240 and 528 m/s) compares it with the other road: the clouds before the
  reconstruction with a continuous reactive weight and a cloud depth, which needs `motion_at` changed (a written cloud
  motion wins at zero depth) and a census of depth-zero pixels.
- **The air join, with no double count.** The reconstructed picture holds the background with its whole air. The cloud
  pass marches the air IN FRONT of the cloud itself, from the eye to the cloud's weighted depth, with the full sample count
  (the vendored march's 16 linear samples would put one or two in front of a cloud 100 km away; ARITH: 71 km a step at the
  horizon). The composite is `final = T_c · L_bg + (1 − T_c) · L_front + T_front · L_cloud`, with L_bg the reconstructed
  background, L_front and T_front the front air's in-scatter and transmittance, L_cloud the cloud's light and T_c its
  transmittance. The cloud shadow enters the air only through the single-scattering term (the shafts, §6.6); a cloud factor
  (the mean cloud transmittance over the sample's upper hemisphere) multiplies only the multiple-scattering term ψ_ms
  (`functions.wgsl:283-306`). Frostbite's Eq. 22 is not applied on top: it was made for a table with no cloud shadow.
  G-CLOUD-AIR reads the result against the per-step air transmittance (Hillaire 2020, slide 41, DOC). One known
  difference stays: the formula takes the background's front air (the vendored 16-sample march) and the cloud pass's front
  air (the full count) as the same air, so a thin cloud mixes two estimates of it; the result is continuous in T_c, so it
  makes no seam, and G-CLOUD-AIR reads its size.
- **The shafts continue in front of a cloud.** Where a cloud is opaque, the air in front of it comes only from the cloud
  pass's L_front. So the cloud pass's front-air march carries the SAME shaft integral as the air pass (§6.6) and reads the
  same shadow maps. Example: at dusk a shaft runs through a gap toward a far cloud bank; it continues to the bank's face,
  and it does not stop at the bank's edge and come back through its thin rim.
- **The edges, at the display size.** The far pass marches at a quarter and the near pass at an eighth of the DISPLAY's
  width and height (§6.1 item 5) against sharp edges (a ridge, a tree line, a canopy strut). The composite runs at the
  display size after the reconstruction, where only the render-size depth exists, so a display pixel at a ridge takes its
  cloud in three parts:
  1. at the render size, the cloud pass marks the EDGE pixels (those whose low-resolution footprint holds depths more than
     one march step apart) and marches them at the render size against their own depth, with the reconstruction's jitter,
     as the vendored MSAA patch marches its own edge pixels (`render_sky.wgsl:51`);
  2. the cloud history adds up those jittered render-size results over frames into its display-size buffer, as the
     reconstruction adds up the ground's edges;
  3. where the sky lies behind all ground, the sky share serves the clouds, as it serves the stars.
  The rest of the pixels upsample by the nearest render depth. The edge share is UNMEASURED. G-CLOUD-EDGE: the eye's rule on
  a still eye at the ridge stand and on the 240 m/s leg along a skyline.
- **The cockpit.** The canopy draws after the air pass, so with no rule the march would start at the eye and lay a veil of
  7.7–21 % in a 2 m cockpit inside a cloud (ARITH). The hull writes its outer skin's depth into a START-DEPTH target before
  the cloud and air passes (SL3: the hull draws its own inside); the cloud march and the rain begin there. Gate: a cockpit
  probe inside a cloud reads no cloud light in front of the canopy.

### 6.6 The light

- **The sun on a cloud.** The transmittance table gives the sun's colour at the sample's height; the planet's shadow in the
  air is the sea's horizon (B25 law 3), so cloud tops stay lit after the ground's sunset.
- **The clouds' ambient** comes from a CLOUD-FREE probe (today's sky-view table), so the clouds never light themselves
  through a probe that holds their own picture; Wrenninge's octaves carry the light inside the cloud.
- **The ground's sky light** comes from the clouded probe AND a top-down cloud occlusion map read per pixel (Unreal's cloud
  ambient occlusion), so a far valley under a deck takes its own shaded sky, not the eye's clear one. One probe at the eye
  (`environment.wgsl:26-34`) would brighten and darken the whole landscape as a hull passes under each cumulus (a 1 km
  cumulus 1.5 km overhead holds about 10 % of the sky's light, ARITH). The probe's cloud part refreshes each frame at a small
  size, or its faces interpolate in time; its node gets a stated order.
- **The ground's sun.** V_cloud multiplies V_sun in the light sum (ground K14), as a vendored change at
  `vendor/bevy_pbr/src/render/pbr_lighting.wgsl:906-926` (the engine's light-texture feature is outside Bevy's default set;
  a new engine feature is the owner's word).
- **Light shafts.** Today's air march cannot give them (16 linear samples at a fixed 0.3 offset: 8.8–71 km a step).
  Hillaire 2020 (DOC, slide 38) names two ways: "Quality: Per-pixel tracing + jitter + TAA", or "Fast: Stored in LUTs
  (SkyView and AP)". On the TEMPORAL path: a dedicated shaft integral over the segment where cloud shadows can exist (from
  the eye to where the ray leaves the shell or meets the ground), its sample count from the shadow map's texel size, jittered
  and resolved by the main history (the shafts are part of the air, in the main picture). In front of a cloud the cloud
  pass's front-air march carries the same integral (§6.5). About 32–64 fetches a pixel inside the layer (ESTIMATE). On the
  MSAA path no history exists and the raymarched mode has no aerial table: NO SHAFTS there, registered, until D14 decides
  the path. Gate: the eye's rule at dusk under drifting cumulus, the eye still.
- **The eye's adaptation.** Under an overcast deck the ground's light falls to 6–41 % (ARITH, WL11 with τ ≈ 13–150): the
  light phase's adaptation law (ground caps row 44) must exist before WP3.
- **The stars and the sun's disc** dim by the sky share, which carries the clouds' transmittance.
- **A named limit**: the air's multiple-scattering table takes the ground's albedo; over a deck seen from orbit it misses
  the deck's bounce (§11 row 26).

### 6.7 The shadows

- **In the clouds' drift frame.** A cloud and its shadow move with the wind; a map refreshed once in N frames would hold a
  texel, then jump by w·N/60 (1 m at N = 4, 4 m at N = 16 at 15 m/s; a shadow edge 50 m from a walker would jump 28–110
  render pixels across the view, ARITH). The map is built in the drift frame of the local cloud-level wind over its area:
  the receiver looks a texel up at (x − drift(t) + drift(t_refresh)), so the refresh follows only the clouds' slow change
  of shape (the advection rule, §6.8). The shafts read the same map.
- **Beer shadow maps** (Hillaire 2020, slide 37: the front depth, the mean extinction, the maximum optical depth), ONE MAP
  PER DISJOINT LAYER at least: the cirrus slab and the main shell apart, so a deck under thin cirrus stays sunlit
  (§6.3); a column where the kind mix gives two layers takes the layered form. An exponential shadow map is refused.
- **The hand-over by the pixel's size.** The cascades end where the smallest cloud shadow they hold falls under a pixel;
  past them, the far transmittance comes from the same field with the clumpy mean (CR2), so no resolved shadow vanishes
  into a tint. The cascade count follows the eye's height by that rule; a new cascade ramps in over B18 item 6's 0.5 s.
- **Orbit at the terminator.** Shadows there need cascades reaching 1 000–2 000 km at texels of about 300 m along a grazing
  light ray through up to 780 km of shell (ARITH): priced as its own case in M-CLOUD-COST.
- G-CLOUD-SHADOW reads the eye's rule on a still walker under a moving cumulus field, the cascade swaps, the hand-over and
  the shafts at dusk.

### 6.8 Advection, growth and decay; two players, one cloud (WQ12)

- **Two-phase advected noise** (Neyret 2003; Perlin & Neyret 2001, ESTIMATE): two copies of the detail noise advect by the
  LOCAL cloud-level wind, each restarting every period P and cross-faded with offset phases so the mean and the variance
  stay. P is a law's value: the convective time scale (the boundary layer's depth over the convective velocity) for cumulus,
  longer for stratiform cloud. So each cumulus grows and decays, the drift may vary by place with no shear (a single global
  drift would shear 86 km a day for a 1 m/s difference across a cell, ARITH), and the clouds move with the wind that the
  trees and the drag read.
- **A closed form of shipped words and the tick.** Each phase's offset is the local wind (from the base and the systems'
  rows, interpolated one statement back) times (t − t_phase), where t_phase = ⌊t/P⌋·P is a function of the universe tick
  alone. Every client computes the same offset from the same statements and the same tick, whenever it joined. The large
  shapes (a system's shield) ride the system's shipped centre.
- **Two players at one place see the same clouds** within the GPU's rounding: one statement, one set of noise bytes (fixed
  integer hashes), one tick. No gameplay reads a cloud's fine shape (WQ12).

### 6.9 The cloud kinds that the target pictures need

| Target | What the model gives | Status |
|---|---|---|
| A storm over plains | the system's shield and cumulonimbus by the kind mix | lightning missing: later (§3.8; WQ18) |
| A fair-weather cumulus field | the daily convective cycle; streets and cells by the regime rule; growth and decay | FIXED (WM4, §6.8) |
| A stratus deck from above | a front's shield; decks under an inversion only with WL5 | WQ18 |
| Cirrus at sunset | an ice phase function; noise stretched along the upper shear's direction | FIXED (§6.1) |
| The planet from orbit | the cover's structure at every scale; stochastic tiling; the corner rule | FIXED (CR8) |

G-KIND-SWEEP sweeps the buoyancy through zero under a deck and reads the eye's rule (WL10's continuous mix).

### 6.10 Precision, the MSAA path, reflections, boarding and determinism

- **Precision.** The march runs in eye-relative coordinates; each noise lookup splits into the eye's place in 64-bit,
  wrapped by that noise's own period on the CPU, plus the ray's offset in 32-bit. At 6 342 km a 32-bit position holds 0.5 m
  (ARITH), which would make the near wisps swim.
- **The MSAA path.** The cloud pass's size is stated against the DISPLAY, the same on both paths (§6.1 item 5: the far pass
  at a quarter and the near pass at an eighth of the display's width and height; on the MSAA path the render size is the
  display size, so a size stated as "half the render size" would be four times the temporal path's rays, ARITH). The cloud's
  own history resolves its noise on both paths; MSAA edge pixels march per sample in the sky pass (`PATCH.md:49-52`), and the
  cloud term sits there too. M-CLOUD-COST prices both; D14 (the temporal design's T8) is decided before WP3, or the clouds
  ship on one path only.
- **Reflections (a contract for the water fill's design).** The clouds reach a reflection through the cloud pass's own
  output: on-screen clouds by a screen-space reflection that runs after the cloud composite; off-screen clouds by the
  clouded probe; the wet ground and the canopy glass by the same rule. Stand: a calm lake at dusk under cumulus.
- **Boarding.** The cloud history turns by the air's rotation at a boarding, as the main history does (temporal §6.3).
- **G-DETERMINISM** on the cloud history expects the clouds' drift between two ticks a few frames apart.

### 6.11 What the client derives, and what it never derives; the far look; the second air

- **It derives, as the planet's look, from shipped state and its own almanac:** the cloud density field; the rain, snow and
  dust rate and phase at a point; the wind for the trees and the particles' slant; the snow cover and the wetness paint;
  the far snow share (§7.4).
- **It never derives:** a system's birth, drift or death past the interpolation; any prediction; the weather epoch; a pose;
  anything that a player's collision, flight, sensor or AI reads.
- **The far look and the reach.** The same march draws the clouds from orbit and from far bodies; a sub-pixel planet shows the
  tint of its clouds' true share (rule M). The reach comes from the brightest look that the charter allows (ground §8.8
  item 9), so the weather never moves a reach statement.
- **The second air.** One air a frame (`client-render/src/sky.rs:150-166`). From the airless moon the home planet is the only
  airy body. An airy moon over an airy planet lacks one air, and a far airy body's clouds take the bare sun at their
  terminator under that rule (WQ19).

---

## 7. Rain and snow on screen, the wind, lying snow and wet ground

### 7.1 One rain look across the whole screen

- **One law.** WL26's drop sizes and fall speeds give the extinction `σ(R)` (ARITH at 10 mm/h: Λ = 2.53 mm⁻¹,
  σ ≈ 1.56 × 10⁻³ m⁻¹, a visibility near 1.9 km).
- **Three bands, one look** (CR9): NEAR, a GPU particle field round the eye, anchored in the world and wrapped (at 10 mm/h
  about 894 drops over 0.5 mm stand in each cubic metre, ARITH, so it draws only the nearest); MIDDLE, layered streak bands
  (Remember Me), each a depth band, following the relative wind at 240–528 m/s; FAR, the precipitation medium in the cloud
  march, reaching the ground. The bands match on the per-pixel mean radiance, the transmittance AND the variance, because a
  drop's streak refracts a wide part of the sky (Garg & Nayar 2007, ESTIMATE) while the medium only scatters: the same σ is
  not the same brightness. G-RAIN-R9 reads the eye's rule at each hand-over.
- **The streak's length** is the drop's speed relative to the eye times an exposure from a stated law: the eye's own
  integration time, about 0.1 s in daylight (the critical duration of Bloch's law, ESTIMATE until read; §3.4 row 15, a
  named calibration written before M-RAIN), independent of the frame rate (at 23.1 ms instead of 16.7 ms a frame-tied
  streak would be 38 % longer, ARITH).
- **Where they draw.** The near streaks draw AFTER the reconstruction at the display size, with analytic coverage, as the
  stars do: a 0.5 mm drop at 2 m is 0.69 display pixels wide (ARITH), which the reconstruction loses, and the reactive
  mask's binary test would leave faint streaks unmarked.
- **The air volume.** The temporal design (§4.6, §7.5) and the ground step (§8.8 item 5) put the weather's near fog, dust and
  rain into T5's air volume. This design keeps that for the NEAR medium (if T5 builds the volume; else the per-pixel air
  takes the term), and puts the FAR rain in the cloud march, because the rain shaft hangs under its own cloud in one field.
- Snow and dust follow the same rule with their own size and fall laws (WM5; dust if WQ16).
- Line H's 0.25–0.7 ms comes from walking speed (Remember Me); at 528 m/s the overdraw grows, so M-RAIN prices line H at
  528 m/s before line H is ruled.

### 7.2 Where rain falls and where drops land

- A top-down depth map over the eye (256²–512²) hides rain under roofs, overhangs and inside hulls; a hull draws its own
  inside (SL3).
- **Child realms open to the sky** (an Area's built square, a hull's open deck): WQ14. In the first step the recommendation
  draws no weather paint on a CHILD's own surfaces (the child authors its look, SL3); the planet's rain in the AIR over them
  is the planet's look and shows. Snow paint with no collision on a child's floor is not recommended. "On whole hulls" means
  the wind's drag on a hull as a whole (§7.3), never snow lying on a hull.
- Drops on glass are a MATERIAL of a hull's canopy or a suit's visor (SL3; the suit ruling), never a camera effect. On foot
  without a helmet no glass exists, so no drop lands on the view (WQ20).

### 7.3 The wind on hulls and in trees

- **The drag.** The planet's own physics: `½ ρ v² Cd A / m` (`drive.rs:202-208`) reads the hull's velocity RELATIVE TO THE
  AIR, `v − w(x, t)`.
- **The ambient becomes a per-child function of position.** Today `Ambient { pull, density }` is one value for the realm,
  built once a tick and passed to every child (`drive.rs:108-119`, `:435-464`; `placement.rs:45-61`). It becomes a function
  of the child's position (the density by height, the wind by place), injected by the composition root like `MotionFn` or
  `TileSource`, and read inside the tick from the weather worker's latest state. The sim names no law. The hull's wind is a
  lawful indirect use of the fenced almanac (B36 item 5's "the weather").
- **w(x, t)**: the static prevailing wind at the season, each system's gradient wind (WL27), down to the surface by WL28,
  plus gusts (WL29, a named scatter, uncut), from the planet's own state at full precision.
- **ρ(z)**: the barometric law from the charter's pressure and scale height (`look.rs:126-131`). Under the sea's level the
  felt air follows the law (it grows: about 1.6 times the sea's density 4 km down with an 8.5 km scale height, ARITH by
  reviewer A), while the DRAWN air holds its sea-level value by B26 (a look ruling). WQ13 asks whether the physics takes the
  law (recommended) or B26's hold (a named cap, §11 row 23).
- No new data crosses (the hull states its mass, cross-section and drag coefficient on change). No speed cap. High up the
  density falls by its law, so the wind needs no boundary near orbit (05 Q4). A sail later: its force will be the parent's
  arithmetic on the hull's stated surfaces.
- **In trees**: a closed-form sway from the same local wind (look), with exact motion (ground P10). The art waits for the
  forest discussion (B36 item 4); DEFERRED registers the sway.

### 7.4 Lying snow (WQ3)

- **The fact that decides the road.** SL10 (V1 items 6–7) lets the client derive only the static shape; "everything the seed
  does NOT decide crosses the wire as a DIFF from the owning realm"; "the client never derives a pose, a velocity, or any
  state of any entity or realm". Lying snow is live state. Seed ruling S6 names the argument "it is deterministic, both
  sides get the same answer, so it is safe" as the one that failed before. So the client may DRAW snow from the shipped
  anomaly (a look, as T3 lets it draw clouds from the list), but it may not DECIDE the snow that feet stand on.
- **The look (all roads).** One snow field: the season's pack (WL36, a closed form both hosts may compute, T3 class 1) plus
  the storms' anomaly (live, from the statement, interpolated one statement back). A column's drawn snow water is the field
  at its height through a C¹ filter (CR8), times its static redistribution (the slope, Bernhardt & Schulz 2010; the lee,
  Winstral, Elder & Davis 2002; ground K12), with the depth by WL33. In the first step the drawn snow is a displacement of
  the ground's own mesh plus the snow material share: no new cell and no chunk rebuild.
- **The far snow (rule M at every rung).** The far rung shows the picture's mean, not only the share's mean: snow lies more
  on lee and shaded slopes, so the far fold carries the covariance of the snow share and the slope's shading (the closure,
  P13, with the live snow field), and G-SNOW-FOLD judges the picture by the eye's rule at a low sun.
- **The collision.** WQ3: (a) none in the first step (recommended): the boots stand on the ground under the drawn snow and
  sink by its depth (in a deep drift a walker stands buried); the collider (slice 11) comes later in any case; (b) the
  planet's shard AUTHORS the snow cells and ships them as a one-hop diff (SL10's own road): a new compose rank after the
  terrain edits (rank 2) and before the catalogue blocks (rank 3) (`compose.rs:1-17`), its surface at the gap byte's 1/128
  cell quantum (`chunk.rs:21-24`), never the sub-metre seat's 1/8 (12.5 cm steps beside 7.8 mm ground, ARITH); each
  snow-driven rebuild ramps in over B18 item 6's 0.5 s, and the client's paint must equal the server's cells where both
  exist (G-SNOW-PIN, a gate that can fail); (c) both hosts derive the cells from the shipped field by one fenced law: it
  AMENDS SL10 (V1 items 6–7). Under (b) and (c), live snow cells change the collision surface in time after the freeze,
  which T3 class 3 names ("the only shape change is the authored override"). So road (b), recommended at the collider
  slice, is ASKED against T3 class 3 (WQ3): either an amendment, or the reading in the pattern of W2's live water (a live
  substance's rows, not the land's shape).
- How a dig INTO live snow is recorded waits for the edit system (B22 step 5), after this step.

### 7.5 Wet ground

The anomaly's wetness (the rain under the systems less the drying, WM7), interpolated one statement back. The look is
Lagarde's porous model (the diffuse darker by porosity; the gloss toward 1 at half the wet level; metals and closed skins
only a film; DOC). Its far fold filters the gloss as a reflection lobe (a Toksvig-type filter, ESTIMATE), because the mean of
a gloss word is not the mean of the reflections. It needs the light phase's material input `w`. Look only; puddles later.

### 7.6 The waves' wind (WQ15)

The ocean's waves (8o; T2) are a closed form both hosts compute. W3's own example is "the swell the almanac's wind raised".
**The recommendation: the waves read the almanac's seasonal wind** (a closed form both hosts compute under WQ4 (b′); T2
and W3 kept as written). A storm's own swell then waits: with the live list's wind, the drawn waves (the client, one
statement back) and the server's buoyancy (the newest statement) would disagree. The client computes no buoyancy.

---

## 8. The costs

### 8.1 The frame (GPU, s = 0.5, full screen, the p99 on moving legs)

- **The base.** 23.1 ms median (MEASURED 2026-10-02; NOT clean). The ground step plans 4.5 ms more for its own lines B–F
  (DOC proposal), and its line A must fall to about 7.4 ms (DOC target, ground §7.3; unpriced). **On today's base the weather
  step cannot pass G-FRAME-TOTAL.** The build waits for a priced and landed line A, or for the owner's word on the frame
  (WQ17).
- **Line G stated honestly.** Line G (2.0 ms) is Nubis³'s 2.1 ms on a PS5 (ground §7.3) with NO scaling to the instrument.
  With one ratio (the M4 Pro at 0.6–0.9 of a PS5 for this work, ESTIMATE), Nubis³'s flight case is 2.3–3.5 ms on the M4 Pro
  for the march alone, and Nubis Evolved's superstorm (2–3 ms on a PS5) is 2.2–5.0 ms (ARITH, reviewer C). Ground views:
  about 0.3–1.0 ms (Red Dead Redemption 2's 1.54 ms on a PS4 scaled between the bandwidth ratio 1.55 and the FP32 ratio
  about 4.3–4.9, ARITH on a community TFLOPS figure). M-CLOUD-COST decides.
- **The items** (ESTIMATE until M-CLOUD-COST):

| Item | M4 Pro (ESTIMATE) |
|---|---|
| Ground views and above the deck (far pass, shadows, sky light) | 0.3–1.0 ms |
| Inside and between clouds at 528 m/s (the march alone) | 2.3–3.5 ms |
| Orbit, including the terminator's shadows | 0.7–2.0 ms (no anchor for the terminator; priced in the bench) |
| The display-size composite (§6.5) | about 0.44 ms (ARITH) |
| Shafts on the temporal path | 32–64 fetches a pixel inside the layer: priced in the bench |
| Rain, snow and dust at walking speed (line H) | 0.25–0.7 ms; at 528 m/s UNMEASURED (M-RAIN) |

- **The real choice for the flight cases (WQ17):** (a) a flight budget (line G rises in and near clouds, paid by line A's
  cut); (b) the cheaper near representation (A1, the froxel volume); (c) a lower near quality at speed by a continuous rule
  judged by the eye's rule. No valve switches at a cloud's edge.
- **The hand-off that this engine can have.** "Inside a cloud the ground costs less" is Nubis³'s statement about its own
  engine; here about 6.9 ms of the ground's 8.44 ms is vertex work (ARITH on MEASURED pass times), which a cloud does not
  shrink. The engine's form: the cloud pass's opacity, one frame old, feeds the geometry step's occlusion culling, so a chunk
  behind opaque cloud is not drawn (inside a cloud with 25–75 m of visibility nearly every chunk is hidden). The geometry
  step's design prices it.

### 8.2 The client's CPU and memory

- CPU: under 0.1 ms a frame on average (ESTIMATE): the statement's decode, the field's rebuild each 10 s, the almanac (under
  WQ4 (b′)); the particles live on the GPU. The line is 0.5 ms (DOC ground §7.4); the CPU already limits the frame below half
  size (MEASURED 2026-09-29, `DEFERRED.md:10951-10953`), and the CPU limit leaves about 0.8 ms (§2.3).
- Memory: the noise volumes about 8 MiB; two weather-field copies about 25 MB; the shadow maps (now one set per layer) about
  6–12 MB; the cloud targets and histories about 15–20 MB; the display-size cloud buffer about 30 MB at RGBA16F (ARITH):
  about 85–95 MB in all, beside today's 2.5–2.9 GB on foot (MEASURED, status log 2026-09-28).

### 8.3 The shard and the lane

| Item | Cost | Mark |
|---|---|---|
| The list's incremental step | about 100 operations a system | ESTIMATE |
| The list from scratch at a boot (the buckets of L_max, each alive system's path) | about 10⁶ operations: milliseconds | ESTIMATE |
| The anomaly's step | about 2·10⁶ operations: about 1–2 ms a step, about 0.02 % of a core at one step each 10 s | ESTIMATE |
| The boot's catch-up | about 0.4–0.7 s for an hour's gap; seconds for the last τ_snow at the coarse step | ARITH on the ESTIMATED step costs |
| The store | the anomaly's changes each step; a base at each re-base | writes UNMEASURED |
| A bench's long run (ten home years, 34.7 M steps) | the list alone about 6 min; with the anomaly about 14 h on one core | ARITH on the ESTIMATED costs |
| The lane | §4.6; the gateway keeps per drawn realm the base, the anomaly base and the statements since it, bounded once per gateway | ARITH |

T3's "about 4 ms a step" (05's ESTIMATE for a field of 393 216 cells) is replaced (WC18). U-W10 measures on the owner's Mac.

---

## 9. The benches before any build

Every bench runs on the owner's Mac (B36 item 3) with Docker stopped, one job at a time, headless unless the owner gives the
window, with the GPU clock's freshness check (the memory notes: the clock lied until 2026-09-29; a span over-reads when
frames overlap) and per-pass readings. A bench's control lives in the harness or a standalone bench binary, never as a
switch in the game. Earth's numbers are readings; a bench READS every named calibration and never moves it (B38). B67 forbids
code now, so the benches wait for the step's place, except U-W2 if WQ1 (a) is ruled. **The frame's verdict waits for line
A**: no bench here can show that the frame fits.

### 9.1 The laws' benches (on the bench copy; "Earth" is the laws on Earth's real facts)

| ID | What | Home planet | Desert world | Moon | Decides |
|---|---|---|---|---|---|
| U-W1 | The storm zone and N on A1's season: σ − 1/τ_E by month; the continuous ratio and the growth rate's integer wavenumber; radii; the row's extremes on every home-system body (the 28.7-bar world included); Earth's facts as the law check (λ about 4 000 km, 5–8 waves, an e-fold of 1–3 days) | the count and size by season | its day and tilt | empty at every tick (G-EMPTY) | reads the laws of WM2–WM3 |
| U-W2 | The circulation law against the published Hadley extents of Earth, Mars, Titan and Venus, then the home planet | one cell or two; the belt's latitude | the same | — | WQ1 (only if ruled) |
| U-W3 | The tropopause, both laws | both | both, dry lapse | — | the owner chooses on its numbers |
| U-W4 | The clouds over ten home years: the cover, the kinds, the base histogram, READ beside Earth's about two thirds | report | zero water cloud | zero | nothing is tuned |
| U-W5 | What the generator does not force: the wet-time share, the intensity distribution and the dry-spell lengths against the generator's published forms and Earth's station records (readings); the annual sum as a consistency check only. The chain's four probabilities (§3.4 row 14) are written before this bench and only read | report + check | zero rain | — | reads WL22's application; moves no value |
| U-W6 | The surface wind by cover; the gust factors | report | report | — | WM6; the drag's inputs |
| U-W7 | The snow-covered area by month; the summer's-end line against `ela_z` | GATE (G-ELA) | no snow | — | WM7 |
| U-W8 | The boot rule: a kill-9 restart within τ_exact against a never-stopped run (byte for byte); a long sleep's wake within G-WEATHER-CONVERGE | test | test | empty | WQ7 |
| U-W9 | Determinism: 10⁶ replayed steps byte for byte on one host | test | test | test | the epoch's stream |
| U-W10 | The cost: the steps, the boot's catch-up, the bytes over a winter | measure | measure | zero | §8.3 |
| U-W11 | The dry lapse | — | measure | — | §11 row 21 |
| U-W12 | G-WEATHER-CONVERGE at the step and its half, and the grid at C and 2C | test | test | — | the grid level; the step |

05's U10 (one tick dormant against woken, per pixel) is moot: a dormant realm is never drawn.

### 9.2 The rendering benches (a standalone bench binary with synthetic fields, then the game path)

| ID | What | PASS LINE / decides |
|---|---|---|
| M-CLOUD-COST | On every moving leg: the ground (clear, broken, overcast); under the deck at 240 m/s; BETWEEN towers and inside clouds at 528 m/s; above the deck at 10 km; a deck under cirrus at sunset; 100 km; orbit at 400 km over the terminator; rain at 528 m/s; both paths at a display-stated size; the near pass against A1; the frame's p99 with and without clouds on the same leg | the cloud passes' p99 ≤ line G on each leg, read by the GPU clock with its freshness check; WQ17 |
| G-CLOUD-NEST | The picture at a distance against a full-detail reference filtered to the pixel, at TWO render scales; the climb from the deck to orbit | the clumpy correction (CR2) |
| G-CLOUD-GHOST | A hull's outline against clouds; a broken deck over ground at 240 and 528 m/s; the composite after vs before the reconstruction | the composite's place (§6.5) |
| G-CLOUD-EDGE | A still eye at the ridge stand; the 240 m/s leg along a skyline | the edge rule (§6.5) |
| G-CLOUD-SHADOW | A still walker under a moving cumulus field; cascade swaps; the pixel-size hand-over; shafts at dusk | §6.7 |
| G-CLOUD-AIR | The composite against the per-step air transmittance on the dusk and aloft stands; the air under a deck | §6.5 |
| G-KIND-SWEEP | The buoyancy swept through zero under a deck | WL10's mix |
| M-CLOUD-CAL | The octaves and the powder against a path-traced reference of our own field, the air under a deck and the ambient path included | a calibration, never a knob (B38) |
| M-RAIN | Particle count, overdraw and cost at display size; the streaks at 0, 240 and 528 m/s; the hand-overs by the eye's rule | line H |
| M-STATIC | The adapter's 16-bit and 3D compressed formats; every new pipeline warmed (ground FR7) | G-NO-STALL |

### 9.3 The interface benches

| ID | What | Decides |
|---|---|---|
| G-ALMANAC-AGREE, G-ALMANAC-PIN, G-SPIN-AGREE | The drawn sun against the planet's almanac; the client's integer almanac against the planet's; the authored spin against the charter's | WP1 |
| M-TURN | The crossing at the equator and at the realm's edge in a turning frame; a boarding; a low orbit; the moon | WQ5's road |
| MW-1 | The base, the anomaly base and the changes, raw and compressed | WA1, WA2 |
| MW-2 | (WQ3 (b) only) the chunk rebuilds under deep snow on the low fast legs, with the eye's rule on a stand where snow builds in a time-lapse | the snow's collision road |
| M-SERVE | The statement's and the base's arrival under a pyramid download | the serve order |
| G-MAP-STEP | The eye's rule on a plain at the onset of rain, and across a season keyframe (WQ4 (a)) | §4.3 |

---

## 10. The phases, the stands and the gates

### 10.1 What must exist before the build

1. The water fill (B22 step 4): clouds over a dry sea floor are judged wrongly (B26).
2. The ground step's light phase: V_cloud in the light sum, the eye's adaptation, `s` and `w` as material inputs, the sky
   share, the probe kinds, the pipelines warmed (ground §8.8).
3. The corrected climate from phase 2: ground A1, A3 (by WQ1's law) and A7.
4. A PRICED AND LANDED LINE A (the speed work), or the owner's word on the frame (WQ17).
5. D14 decided, or the clouds on one path only (§6.10).
6. The soil step's grain size if WQ16 (a); the forest art for the trees' sway, or the sway registered.
7. The owner's readings of WQ5 (the turning planet) and WQ6 (HR4): each blocks a phase.

### 10.2 The phases

Every phase ends with the numbers on the pinned stands of the three bodies, ONE picture set on those stands, the living
status file and DEFERRED updated in the same change, and the frame, CPU and memory lines (ground §7.7). One job at a time.
**From WP3 to WP6 every outdoor picture changes, so the frozen references go red** and change only on the owner's
acceptance at each phase's look (B29, B32).

- **WP0 — the benches and the answers.** §9 at the step's place; the owner's answers to §13. No world change.
- **WP1 — the almanac and the turning planet** (its own design with reviews first). The road of WQ5; the two new identity
  draws; `PlacementTold` with its fences; the relayed hop's angular velocity; the integrator's terms (road (a)); the
  integer almanac with the orbit's two angles by WQ4's road (the generator's element draws on the client, or every orbit
  to one integer function); stands by place and tick. Gates: G-ALMANAC-AGREE, -PIN, G-SPIN-AGREE, G-CROSSING at the equator
  and the realm's edge, the eye's rule at dawn and dusk, the fences' planted controls. End: a stand at dawn, noon and dusk;
  the seasons from orbit; the references re-frozen on the owner's look.
- **WP2 — the weather's formula and its lane.** The weather system and its gate; the worker; the store family and the boot
  rule; the epoch's source (WQ8); WM1–WM3, WM8; the base (WA2); the statement and its six arms; the gateway's cache and its
  re-base rule; the client's book. Gates: U-W1, U-W5, U-W8, U-W9, U-W10, U-W12, G-EMPTY, G-IDENTICAL (WQ6), HR5. End: N by
  season, the sizes, the costs; one census picture (a globe map of the list and the base from the stamp rows).
- **WP3 — the clouds.** §6 whole; each vendored change in `PATCH.md`; the pipelines warmed; the second air priced. Gates:
  §9.2's cloud gates; the eye's rule on every leg; line G; G-NO-STALL. End: the weather stands, four views, two suns,
  three bodies; the legs from orbit to the ground and through a deck.
- **WP4 — rain, snow and dust on screen; the wind.** §7.1–7.3. Gates: G-RAIN-R9, M-RAIN, the temporal contract, line H,
  U-W6. End: pictures in rain, snow and dust; a hull in a gust with the drag's numbers.
- **WP5 — lying snow and wet ground** (look, WQ3 (a)). The snow field, the displacement, the far covariance, the wet
  material. Gates: G-SNOW-FOLD, G-ELA, G-GRID on the snow line, G-MAP-STEP. End: a valley white at dusk after a storm; a wet
  plain after rain; the snow by season from orbit.
- **WP6 — the owner's look** (ground Q18's second look; WQ23): every weather stand, four views, two suns, three bodies, the
  flight legs in the window on the owner's word at that moment. The living status file goes only after this look.

### 10.3 The stands

- The ground step's rule holds (B36 item 8): stands pinned once, four views, two suns, three bodies (the home planet, the
  home moon, the desert world). The icy moon joins as a stand with its weather OFF and the reason on its stamp row.
- With the turning planet a stand is a PLACE and a TICK (two ticks for the two suns).
- **The tick, lawfully.** In-process tiers inject the universe clock through `sim::io::Clock` (`crates/sim/src/io/mod.rs:384`).
  For process-tier stands (production binaries), the cluster must start its universe clock at a stated tick through its own
  world-start configuration, never a shard switch; whether that exists is UNMEASURED and asked at WP1.
- **The weather, lawfully.** The harness writes the weather family's epoch row into the planet's realm store before its
  shard boots (the store is the harness's directory, as the artifact store is), so a stand's weather is fixed with no
  switch in the game.
- A stated rule adds weather stands at the pinned tick and epoch: under a warm front, in a cold sector, under a high, in the
  rain belt, in falling snow, on lying snow after a storm, on a wet plain after rain, above a deck at 6–12 km, a deck under
  cirrus at sunset from above, between towers on a 528 m/s leg, a calm lake at dusk under cumulus, from orbit over the
  terminator, the desert world in its strongest wind, the moon (empty), the icy moon (off).
- HR6: probe kinds 3 (cloud: the weighted depth and the optical depth class) and 4 (precipitation) (`probe.rs:49-54`); a
  weather stamp row beside `DevSkyStamp` (`crates/devproto/src/state.rs:1040-1054`: the generation, the step, its age, N,
  the held or stale state, the fade weight, the digests, the refusals); a cockpit probe inside a cloud.

### 10.4 The gates

| Gate | What goes red |
|---|---|
| The eye's rule (B27) | a pop that the eye sees on any leg, at an update, a hold, a fade, a wake, a boarding |
| G-CROSSING (B31) | over half a level at the turning frame's crossings or at a boarding under clouds |
| G-ALMANAC-AGREE, G-ALMANAC-PIN, G-SPIN-AGREE | the drawn and the almanac's suns apart by more than the disc's radius; the two hosts' almanac integers differ; the authored spin is not the charter's |
| G-IDENTICAL (HR4) | the planet and station runs state different systems |
| HR5 | under 100 % in `vd-weather`, `vd-weather-shard` or the client's book |
| G-EMPTY | a row on an airless body; a water cloud, rain or snow on a dry one |
| G-WEATHER-CONVERGE (U-W12) | the statistics at one step and its half, or at C and 2C, disagree beyond the stepper's noise |
| G-RESUME (U-W8, U-W9) | a restart within τ_exact that differs; a replay that differs |
| G-ELA (U-W7) | the summer's-end snow line off the glaciers' line |
| G-STALE-FADE, G-MAP-STEP, G-RAIN-R9 | a frame step of a fade, a map change or a rain hand-over over the eye's threshold |
| G-CLOUD-NEST, -GHOST, -EDGE, -SHADOW, -AIR; G-KIND-SWEEP; G-DETERMINISM | §9.2 |
| G-SNOW-FOLD, G-GRID (snow line), G-TILING | the far picture's mean off the near; a grid mark in the snow line or the cover; a repeat from orbit |
| G-NO-STALL, G-FRAME-TOTAL | a pipeline compiled after loading; the running total over 16.7 ms |
| The SL1 fences | the machinery names `PlacementTold`; the almanac reads the boot's orbit (each control fails when it should) |
| The shape pins | any static-shape pin moves (the weather never touches the land) |

---

## 11. The caps list (B23): every clamp, ceiling and floor on this step's path

| # | Cap | Where | Fate |
|---|---|---|---|
| 1 | 64 weather systems a body | 05 §8.5; V13 L16 | GOES: the count law's continuous Poisson mean |
| 2 | Snow at most one cell | 05 §8.3 | GOES: no depth limit (look in the first step; cells under WQ3 (b)) |
| 3 | A fixed relaxation toward the climate | 05 §8.5 | REPLACED: the life law |
| 4 | A forcing seeded by (seed, step) | 05 §8.5 | REPLACED: births at the law's rate from the epoch (WQ8) |
| 5 | `round()` on the count | WL16 (revision 1) | GOES: the continuous ratio |
| 6 | The row's fields | §4.1 | Body units; an overflow is refused and counted, never clipped |
| 7 | The stale bound | §4.4 | Derived (SL1 clause 6); its lower end (two steps plus a beat) is the lane's own law, because drawing one statement back needs two statements, not a chosen floor |
| 8 | The fade's time | §4.4 | Derived from the eye's rule, per frame |
| 9 | A tracing start distance; a 16-frame near amortization | Unreal; Horizon Zero Dawn 2015 | REFUSED (CR3; temporal §4.6) |
| 10 | The near/far split at 200 m | revision 1 | REPLACED: the parallax law (§6.4) |
| 11 | The ground's shadow reach of 3.5 km | `terrain.rs:110-116` | The clouds' own cascades by the pixel rule (CR6) |
| 12 | RH clamped at 1 | WL1 | REPLACED: the excess leaves as rain (WL22) |
| 13 | The gradient wind of a high | WL27 | A LAW-LIMIT: no real solution past a pressure gradient of about ρ f² r / 4 (the gradient-wind balance, Holton & Hakim 2013, ESTIMATE); the life law stays inside it |
| 14 | L_max, the longest life | §3.6 | The life law's own value, never a cap |
| 15 | A floor on f near the equator | WL13, WL14, WL15, WL28 | REFUSED: the equatorial β-plane form of each law |
| 16 | A floor on the wet-time share | WL22 (revision 1) | REFUSED: the weather generator gives zero rain by its own form |
| 17 | The largest drop | WL26 | A LAW-LIMIT: the published break-up size (source owed before WP4) |
| 18 | The gust's peak factor 3 | WL29 | The full distribution, uncut |
| 19 | A fixed weather grid C = 128 | revision 1 | REPLACED: a level of the body's own pyramid by the convergence gate |
| 20 | Hadley cells ≤ 6; lift [0.1, 4.0]; rain [1, 10 000] mm/yr; ΔH 0.388; one seasonal column | `climate.rs:106`, `:129-130`, `:104`; `solve.rs:56-60`; `climate.rs:445-470` | Phase 2 (ground rows 20–22, 25, 49) |
| 21 | The moist lapse share 0.666 on a dry body | `climate.rs:93` | The dry adiabat where no condensable exists (U-W11) |
| 22 | 990 mm of rain at 288 K | `climate.rs:117-118` | The energy constraint (O'Gorman et al. 2012, DOC) in the climate's step; a named calibration until then |
| 23 | The felt air held at the sea level's density under the sea | §7.3 (revision 1) | WQ13: (a) the barometric law in the physics, B26's hold in the look only; (b) a named cap |
| 24 | The degree-day factor | ground §8.4 | REPLACED: the energy balance |
| 25 | A fixed exposure; the Bond albedo "that includes clouds" | `terrain.rs:128-136`; D-TERRAIN-6 | The eye's adaptation law; the drawn albedo plus the cloud field |
| 26 | The air's multiple scattering with the ground's albedo under a deck | §6.6 | A NAMED LIMIT, registered |
| 27 | One air a frame | `client-render/src/sky.rs:150-166` | Registered; WQ19 |
| 28 | The gateway's 1 MiB a beat | `artifact.rs:50` | A shared pace in the one config struct; the statement first |
| 29 | Snow's seat at 1/8 cell | revision 1 | REPLACED (WQ3 (b)): the gap byte's 1/128 |

**Kept on purpose:** the representation bounds of the map's words (asserted); the render-scale floor; the operational rates
(the step, the beat, the lead) in the one config struct.

---

## 12. The conflicts, and how each one is resolved

| # | The older statement | Decided by | The result |
|---|---|---|---|
| WC1 | 05 §8.8: a tag on the self-look bag, to the own occupants | T3; the discussion's A4; the owner 2026-09-20; ground C19 | Own arms through the gateway to every session (WA1) |
| WC2 | 05 §8.3: snow derived from the list, capped at one cell | B35 item 5; B23 | The pack plus the live anomaly; no cap |
| WC3 | 05 §8.5, V13 L16: at most 64 systems | B23, B38 | The count law |
| WC4 | 05 §8.5: a fixed relaxation; a forcing seeded by (seed, step) | T9; B38; seed ruling S3 | The life law; the epoch (WQ8) |
| WC5 | 05 §8.5: an anomaly field of 10 B a cell for the dynamics | Reader 4's finding | The list as a closed form plus a coarse anomaly |
| WC6 | 05 §8.1, T1 item 10, 8b §4.2: the almanac derived, no lane; the shard's own orbit at boot | B36 item 5; SL1 clause 3 | `PlacementTold`; a fence against the boot's orbit |
| WC7 | `docs/design/sealed_shards.md:134`, PLAN P8: a WeatherDragPort | M2a | The wind in the parent's drag |
| WC8 | Weather as f(seed, tick, position) (`sealed_shards.md`; `scripts/dormant_world_simulation_design.md:2152`) | T3; SL10 V1 item 7; S3 | Live state of an epoch, shipped |
| WC9 | T1 items 10–11, T4: clouds after the freeze | B35 item 5 | Before the freeze |
| WC10 | 05 §8.4: a dormant planet drawn as a point of light | Visibility; reach | Never drawn |
| WC11 | 05 §8.4, ground §8.3, ground Q17 (b): zero at the wake | T3 class 2; LAW-WL-1 | The closed form (WQ7) |
| WC12 | Revision 1 §3.6: a stationary draw at every wake | T3 class 2; reviews A and B | REFUSED: it threw checkpointed storms away |
| WC13 | Ground §8.4: the degree-day melt | B41 item 2 | The energy balance |
| WC14 | Ground §8.2: the count by area calibrated on Earth's count | B41 | Eady's ratio; the area law a cross-check |
| WC15 | Ground §8.2: a stale list snaps to the almanac | B16; B27 | The hold, then the eye's fade (WQ11) |
| WC16 | Ground §8.1: the client takes the season from the composed picture | T3 class 1 | The client's integer almanac (WQ4 (b′)) |
| WC17 | Revision 1 WQ4: "three later facts" against T3 | The true dates | T3 is newer than V1 and S7-1 (§5.4) |
| WC18 | T3's "about 4 ms a step" | The model | Microseconds and 1–2 ms (ESTIMATE) |
| WC19 | DEFERRED D-TERRAIN-8: "THE CLOUDS … WHEN: 18's state model" (`docs/design/DEFERRED.md:10167`) | B35 | The weather step; fixed in the NEXT change that touches DEFERRED (this pass writes no other file) |
| WC20 | Ground §0 and C5: minor 35, floor 34 | The code: 36 and 36 (B55) | The arms at the next free minor |
| WC21 | T5: no monsoon | B36 item 1 | The seasonal circulation |
| WC22 | Ground FR8, Q10: the average machine owed | B36 item 3 | The owner's M4 Pro |
| WC23 | Revision 1: the snow's collision derived on both hosts | SL10 V1 items 6–7; S3; S6 | Look only, or the server's diff (WQ3) |
| WC24 | Revision 1: HR4 on two mappings with a DEFERRED row | HR4 | Two shard profiles (WQ6) |
| WC25 | Revision 1: the weather map's seasonal paint | B36 item 5; B35 item 5 | Removed (§3.8) |
| WC26 | Revision 1: `ChildOrbit` under a new name | SL1's vocabulary fences | `PlacementTold` |
| WC27 | Revision 1: the stepper in `crates/bins` | HR5 | Two Tier-A crates |
| WC28 | Revision 1: the snow's seat at 1/8 cell "as on bare ground" | `chunk.rs:21-24`; `seat.rs:26` | 1/128 under WQ3 (b) |
| WC29 | Revision 1: the near pass to 200 m with animated jitter; a valve at a cloud's edge | B16; B27 | The parallax law; one-frame convergence; no edge switch |
| WC30 | Revision 1: `ArtifactPart` reused or not | The owner's standing rule against repurposed messages | A new `WeatherPart`; `ArtifactPart` is already a generic relay, so this is a naming choice that the rule supports |

---

## 13. The owner's questions

Each question has its options, a recommendation and its price. NOW: the answer changes another step's work or blocks a
phase. AT THE STEP: it can wait for the weather step's place. AT G-LOOK: asked at that look.

1. **WQ1 — THE CIRCULATION LAW** (NOW; F2; ground A3; B38).
   - (a) A bench (U-W2) reads Held & Hou with Lindzen & Hou, Held 2000's eddy-onset edge and the thermal Rossby number's
     solstice theory against the published Hadley extents of Earth, Mars, Titan and Venus; the static climate and the
     weather then use the one law that you choose from its numbers. It runs with the ground step's climate bench, before A3
     is built. (b) Build A3 as planned, with no bench.
   - **Recommendation: (a).** Price: a headless bench of a few days (ESTIMATE); no world change; the chosen law can move the
     deserts, the ice and the stands. Example: today's law gives the home planet one wind cell and no storm belt; the
     eddy-onset law gives a storm belt near 52°. The home planet's thermal Rossby number is 0.5–1.0, a little above Mars.
2. **WQ2 — THE LAND'S SEASON** (NOW; F3; ground A1).
   - (a) Ground A1 is the two-column season; the weather builds on it, and U-W1 runs only on A1's output. (b) A separate
     land-and-sea law for the weather only: two laws for one season.
   - **Recommendation: (a).** Price: the weather step waits for A1 (no ruling answers ground Q4 yet; B52 item 3 lists phase
     2's ruled laws without A1–A3).
3. **WQ3 — LYING SNOW AND ITS COLLISION** (AT THE STEP; SL10 V1 items 6–7; T3 class 3).
   - (a) Look only in the first step: the snow is drawn from the season's pack and the shipped anomaly, and the boots stand
     on the ground under it (in a deep drift a walker stands buried). (b) The planet's shard authors the snow cells and
     ships them as a one-hop diff (SL10's road): a new compose rank, the gap byte's quantum, a rebuild ramp, a gate between
     the paint and the cells. (c) Both hosts derive the cells from the shipped field by one fenced law: AMENDS SL10.
   - **Recommendation: (a) now, (b) at the collider slice** (slice 11 comes later in any case). Price of (a): drawn snow
     with no collision, registered; of (b): the diff's bytes and the rebuilds, UNMEASURED (MW-1, MW-2). Under (b) and (c),
     live snow changes the collision surface after the freeze, which T3 class 3 names: road (b) is ASKED against T3 class 3
     (an amendment, or your reading, in W2's pattern, that live snow is a live substance's rows and not the land's shape).
4. **WQ4 — THE SEASON ON BOTH MACHINES** (AT THE STEP; T3 class 1; S7-1; B36 item 5).
   - (b′) T3 class 1 as ruled: the client evaluates the planet's orbit from the seed by one fenced integer function, with a
     no-drift gate. (a) The planet ships the season's look as keyframes ahead: AMENDS T3 class 1; it also puts the date on
     a lane in another form (B36 item 5 asks about that), and the cumulus's daily cycle needs keyframes shorter than a day.
     (c) The client reads the season from the drawn sun: refused for collision.
   - The client's season under (b′) needs the orbit's start angle and periapsis angle, which the charter does not hold and
     which are float draws of the star system's generator (§5.4). The roads for them: (1) the client runs the generator's
     orbit draws (the same code, no port); (2) a second implementation: REFUSED by V1 item 2; (3) every orbit moves to one
     integer function on both hosts: a world change in every orbit, with new pins. The angles in the planet's charter are
     REFUSED (the planet would state its own orbital phase: SL1 clause 3; B36 item 5).
   - **Recommendation: (b′) with road (1).** Price: S7-1's isolation row narrows to admit the generator's element draws and
     one integer almanac function on the client; the gate G-ALMANAC-PIN; the static base (WA2) once by digest. Road (3)'s
     price: one more world move, for one integer source of every orbit.
5. **WQ5 — THE TURNING PLANET** (NOW, for its substance and its place).
   - The rulings meet (§5.3): V12 ("the parent authors the spin on the row, the recipe lives in the planet's own frame") and
     the 2026-08-09 ground rule ("a planet realm has no concept of rotating at all").
   - (a) The frame turns (V12 kept); the planet's physics takes its spin RATE and pole from its own charter (V13 L12),
     never from the told placement, and authors its children in its turning frame by a turning angle from its own origin:
     AMENDS the 2026-08-09 rule, and you read whether that angle falls under SL1 clause 3. (b) The
     frame stays inertial and the ground turns inside it: AMENDS V12 and the 2026-08-09 rule; every surface occupant stands
     on a moving floor. (c) No turning: refused (a seam).
   - **Recommendation: (a), as its own design with reviews in WP1.** Price: the relayed hop's angular velocity (the P10 owe,
     `window.rs:1083-1097`); the integrator's two terms; crossings up to 153.5 m/s at the equator and about 24 km/s at the
     realm's edge, measured end to end; the moon in the turning frame; two new identity draws with pins; stands by tick; one
     re-freeze.
6. **WQ6 — HR4 FOR THE WEATHER** (AT THE STEP; it blocks WP2).
   - (a) The identical weather fixture through the planet and the station profiles, with the air and sun facts injected
     by the test, as `assert_feature_anywhere` injects its child region: HR4 kept as the code runs it. (b) The build waits
     for a real second realm kind with air (a station hall with an air plant, an O'Neill cylinder). (c) Another form that
     you name.
   - **Recommendation: (a).** Price: the fixture beside the shard rig; no world change. (b)'s price: the weather waits for a
     new realm kind.
7. **WQ7 — THE WAKE AND THE RESTART** (AT THE STEP; T3 class 2; LAW-WL-1).
   - (a) The list as a closed form of (the epoch, the tick), and the anomaly stored and advanced over the gap by the same
     laws: a sleep and a restart lose no storm; G-RESUME passes. (b) A fresh draw from the long-run mix at every wake: it
     throws checkpointed storms away. (c) A start from zero (05): an empty sky for days.
   - **Recommendation: (a).** Price: no merging or interacting systems in the first step (the class rule); a boot's catch-up
     of about a second.
8. **WQ8 — THE WEATHER'S RANDOM SOURCE** (AT THE STEP; seed ruling S1, S3, S5; the conventions).
   - The facts: no `sim::io` random seam exists, and its one design (DetRng) is seed-derived. S1 forbids only VALUABLE facts
     from being seed-only, and the weather is not valuable today; S3's table says live state is "not seed-derived at all",
     and T3 class 2 calls the weather live state, so S3 leans to (a).
   - (a) A new entropy seam: a trait in `sim::io`, an OS-entropy impl in the production io (reading the OS's random source
     by file I/O, or a crate on your word), a fixed impl in `io::mem`; the epoch drawn once per planet and stored. (b) DetRng
     as designed, seeded from (the universe seed, the realm, "weather"): no new seam; the weather becomes a function of the
     seed, a wiki fact (S5 item 4). (c) A hash of (the seed, the planet's first-boot tick) through the existing clock seam:
     no new seam; not seed-only, but predictable from public facts.
   - **Recommendation: (a).** Price: one trait, two impls and the harness's epoch row; no crate unless you approve one.
9. **WQ9 — THE NAMED EARTH CALIBRATIONS AND RH_c** (AT THE STEP; B38 item 4; B41 item 2).
   - Every Earth constant of §3.4 gets one value from its source, written before any bench, registered as an interim; the
     benches only read; the rain chain's four probabilities and the eye's integration time are on the list. For RH_c: (a)
     Quaas 2012's law of the box size, fixed before any bench, its value at the grid level that U-W12 picks following from
     that law, a named Earth calibration; (b) Tompkins 2002's prognostic spread, the law (a humidity field with two more
     moments a cell).
   - **Recommendation: accept the list; (a) for RH_c now, (b) registered as the owed law.** Price of (b): the closed-form
     list grows a field.
10. **WQ10 — THE LANE** (AT THE STEP; SL6 asks WA1–WA3).
    - Six appended arms (`WeatherStatement`, `WeatherMap`, `WeatherWant`, `WeatherPart`, `WeatherHeld`, `PlacementTold`);
      the want only for drawn realms with air; `WeatherPart` gated on the minor; the statement first in each beat; the static
      base once per solve.
    - **Recommendation: yes.** Price: about 11–29 B/s a client a planet; the base 0.28–2.2 MB raw once; one minor.
11. **WQ11 — A LATE OR STALE LIST** (AT THE STEP; SL1 clause 6; B16; B18; B27).
    - (a) Hold the last state up to a bound derived from the weather's own rate of change, then fade over the eye's fade,
      defined per frame; a late first statement rises in by the same fade, longer than B18's 0.5 s for a chunk. (b) Fade at
      once over the eye's fade: a storm dissolves in about 6–13 s, which passes the eye's rule and is not weather. (c) Snap:
      forbidden by B16 and B27.
    - **Recommendation: (a).**
12. **WQ12 — TWO PLAYERS, ONE CLOUD** (AT THE STEP).
    - (a) The same clouds as look, by the closed-form advection of shipped words and the tick; no gameplay reads a cloud's
      fine shape. (b) Also gameplay (a hull hides in a cloud from a sensor): the cloud field becomes a fenced function on
      both hosts with a no-drift gate.
    - **Recommendation: (a) now, (b) when a sensor needs it.**
13. **WQ13 — THE FELT AIR UNDER THE SEA'S LEVEL** (AT THE STEP; B26; T9; B23).
    - (a) The drag reads the barometric law; B26's hold stays a rule of the look. (b) B26's hold in the physics too: a named
      cap.
    - **Recommendation: (a).** Example: a hull 4 km down in a dry basin feels about 1.6 times the sea's air (ARITH).
14. **WQ14 — CHILD REALMS OPEN TO THE SKY** (AT THE STEP; SL3; R-16; M2a).
    - (a) The child borrows its parent's weather as an instrument, as an Area borrows its parent's grid (V5): an SL6 ask
      against M2a. (b) No weather paint on a child's own surfaces in the first step; the seam at an Area's edge registered.
      (c) The planet's live snow crosses as the Area's own diff: against R-16.
    - **Recommendation: (b) in the first step, (a) asked with the Areas' own design.**
15. **WQ15 — THE WAVES' WIND** (AT THE STEP; T2; W3).
    - (a) The almanac's seasonal wind (T2 and W3 as written). (b) The live list's wind: the drawn waves (one statement
      back) and the server's buoyancy (the newest statement) disagree; the client computes no buoyancy.
    - **Recommendation: (a).** A storm's own swell waits.
16. **WQ16 — DUST ON THE DESERT WORLD** (AT THE STEP).
    - (a) Yes, if the soil step gives the grain size by law (the dust constants named). (b) No: the desert world shows only
      wind.
    - **Recommendation: (a)**, else (b) with the gap stated.
17. **WQ17 — THE FRAME AND THE FLIGHT CASES** (AT THE STEP; B22; ground Q10).
    - Line G (2.0 ms) is Nubis³'s PS5 number unscaled; on the M4 Pro the flight cases need 2.3–3.5 ms for the march alone
      (ESTIMATE). (a) A flight budget (line G rises in and near clouds, paid by line A's cut). (b) The cheaper near
      representation (A1, the froxel volume). (c) A lower near quality at speed by a continuous rule.
    - **Recommendation: bench first (M-CLOUD-COST with its pass line); then (b) if the near pass is over its line.** The
      build waits for a priced and landed line A, or for your word on the frame.
18. **WQ18 — THE CLOUD KINDS FOR THE TARGET PICTURES** (AT THE STEP).
    - (a) Stratocumulus decks under an inversion (WL5) in the first step, their slope a named Earth calibration. (b) The
      "sea of clouds from above" waits. Lightning waits in both, so the storm at dusk lacks its light.
    - **Recommendation: (a).**
19. **WQ19 — THE SECOND AIR** (AT THE STEP).
    - Price it in WP3 (UNMEASURED); build it only when a stand reaches an airy moon over an airy planet.
    - **Recommendation: yes.**
20. **WQ20 — DROPS ON FOOT WITHOUT A HELMET** (AT THE STEP).
    - (a) No drops on the view: an eye has no glass. (b) A lens effect: not physical.
    - **Recommendation: (a).**
21. **WQ21 — A MOON'S SUN** (when an airy moon becomes a stand; §5.5).
    - (a) An exception to SL1 CLAUSE 4 (the moon told its parent's placement; a two-level fold). (b) No weather on airy
      moons: against B35 items 2 and 6. Neither keeps every ruling; the search for a local formulation (SL6) comes first.
    - **Recommendation: the search first; then (a), asked with the icy moon's stand.**
22. **WQ22 — OUR OWN CLOUDS, NO LIBRARY** (AT THE STEP).
    - No cloud crate fits (a sky box with no flight; Bevy 0.19, a gated upgrade; none uses the Bevy atmosphere).
      `bevy_hanabi` (Bevy 0.18) could carry particles.
    - **Recommendation: our own code on the published methods**; a crate only after your joint investigation.
23. **WQ23 — G-LOOK's SCOPE** (AT G-LOOK; ground Q18; B36's last paragraph).
    - G-LOOK judges the ground and the light without clouds and with the dry sea floor; a second look after the weather step
      judges the clouds and the water; the status file stays until then.
    - **Recommendation: yes.**
24. **WQ24 — A DAY AND A YEAR FOR A PLAYER** (AT THE STEP; 05 Q1).
    - The solar day is 72.65 h (the rotation 72.11 h); the year is 401.9 days of real time; a season lasts about 100 days
      (ARITH). No knob exists (SL5; the time multiplier never touches an orbit).
    - **Recommendation: accept** (a law's outcome).

---

## 14. What is unmeasured

1. The circulation law on the home planet; the rain belt's solstice latitude (U-W2).
2. The storm zone, N and the sizes on A1's season; the row's extremes on every home-system body (U-W1).
3. The tropopause law (U-W3); the cloud cover, kinds and bases (U-W4); the generator's unforced statistics (U-W5); the
   surface wind (U-W6); the snow by month (U-W7).
4. The boot rule's exactness and the catch-up's cost; τ_exact and τ_snow on each body (U-W8, U-W10).
5. The step's and the grid's convergence (U-W12).
6. Every cloud and rain cost on the M4 Pro at every stand and on both paths, against the pass line; the near pass against A1
   (M-CLOUD-COST, M-RAIN); today's clean full-screen frame (M0-11).
7. The clumpy correction's level error at two render scales (G-CLOUD-NEST); the sample lattice's blends (between two step
   sizes, and across the band where a ray's main axis changes) by the eye's rule on a still and a moving eye.
8. Ghosting, edges, shadow stability, the air join (G-CLOUD-GHOST, -EDGE, -SHADOW, -AIR); the depth-edge share.
9. The multiple-scattering calibration (M-CLOUD-CAL).
10. The base, the anomaly base and the changes, compressed (MW-1); the rebuilds under deep snow if WQ3 (b) (MW-2).
11. The crossing in a turning frame end to end, at the equator and the realm's edge (M-TURN).
12. Whether the cluster can start its universe clock at a stated tick (WP1).
13. The statement's arrival under a pyramid download (M-SERVE).
14. The second air's price.
15. The cloud look's drift across GPUs.
16. Whether wgpu overlaps the cloud compute with the ground's draw on the M4 Pro.
17. The engine's light-texture feature, the 16-bit and 3D compressed formats on the M4 Pro (M-STATIC).
18. The client's CPU line for the weather.
19. The hold's bound and the eye's fade on the stands (G-STALE-FADE).
20. The sources still owed before their phases: RH_c's law of the box size (Quaas 2012), N_d, the life curve (Hoskins &
    Hodges 2002), the rain chain's four probabilities by circulation state, the eye's integration time, the ice phase
    function, the cumulus regime rule, the drop break-up size, the snow's size law.
21. The believability of one or two storm waves a hemisphere: the owner's look.

---

## 15. The marks

| Mark | Meaning |
|---|---|
| `path:line` | Read in today's code at `7b756dd` plus its uncommitted changes. This revision re-read: the wire numbers; `frame.rs:82-101`, `:244-276`; `home.rs:104-122`, `:218-224`; `climate.rs:90-165`; `drive.rs:106-120`, `:196-209`; `placement.rs:70-86`; `sim/src/lib.rs:30-40`; `sim/src/io/mod.rs` (its traits); `test_harness.md:128-138`; `justfile:10-14`; `rlm.rs:410-420`; `bin/shard.rs:305-340`; `pose.rs:15-35`; `intershard_closed.rs:980-1000`, `:1093-1102`, `:1260-1268`; `crate_isolation.rs:165-180`; `window.rs:1080-1098`; `bins/src/lib.rs:3260-3280`; `charter.rs:525-540`; `containment.rs:1076-1105`; `ladder.rs:170-176`; `grid/mod.rs` (its public answers); `chunk.rs:18-26`, `:74-78`; `seat.rs:1-8`, `:24-28`; `compose.rs:1-20`; `reconstruction.wgsl:158-175`; `atmosphere/mod.rs:425-434`; `functions.wgsl:48-52`, `:280-308`; `environment.wgsl:26-34`; `eye.rs` (its laws); `channels.rs:63-83`; `gateway/artifact.rs:400-430`; `gateway/shard.rs:525`; `built_store.rs:1-30`; `look.rs:134-142`; `artifact_source.rs:8-12`, `:133-138`; `client-render/src/lib.rs:1486-1493`; `measure.rs:148-156`; `scripts/dormant_world_simulation_design.md:176-186`; the owner's ground rule in the memory. Other lines come from the readers' and reviewers' notes of the same day |
| MEASURED | A tool printed it in an earlier run; the record and its date are named. Nothing ran for this design |
| DOC | A ruling, a design document, a code comment, or a published source that a reader or reviewer of this team read at first hand on 2026-10-08 |
| ARITH | Arithmetic of this team on marked inputs: reader 4's `calc.py`, reviewer B's `arith.py` and `earth.py`, reviewer C's `review_c_calc.py` (all in the session's scratchpad), and this synthesis |
| ESTIMATE | A judgement, or a published value that nobody of this team read at first hand; the source is named and is read before a gate uses it |
| UNMEASURED | Nobody printed it; §9 names the bench |

---

## 16. The reviews' answers

Every finding of the three reviews, one line each. FIXED: the design changed (where). ASKED: an owner question carries the
decision, and the design no longer claims the ruling (which question). ANSWERED: no change, with the reason. A finding's
own words quote revision 1's question numbers (its WQ9 was the random source, its WQ14 G-LOOK, its WQ20 the moon's sun);
each answer gives revision 2's numbers.

**Review A — the laws and the rulings**

| Id | Finding | Answer |
|---|---|---|
| A-B1 | The client derives lying snow's collision cells from a live field (SL10, S3) | ASKED — WQ3; the design no longer derives collision on the client (§7.4; §1 SL10 row) |
| A-B2 | HR4 replaced by a two-mapping fixture with a DEFERRED row | ASKED — WQ6; the fixture now runs through two shard profiles (§3.7) |
| A-B3 | U-W4 "decides" RH_c beside Earth's cover | ASKED — WQ9; U-W4 only reads; one value written before any bench (§3.4, §9.1) |
| A-B4 | Seasonal paint widens B36 item 5's fence and B35's scope | FIXED — removed from WA2, registered outside the step (§4.8, §3.8); the "only" wording corrected |
| A-I1 | §0 "decisions"; §1 claims "kept" for amended rulings | FIXED — §0 recommendations tagged with questions; §1 status column |
| A-I2 | WQ4's dates wrong; T3 class 1's own road not offered | ASKED — WQ4 with the true dates and road (b′) recommended (§5.4) |
| A-I3 | The wake's draw discards checkpointed weather; Earth's two weeks | ASKED — WQ7; the list is a closed form, nothing is discarded (§3.6) |
| A-I4 | The turning planet's physics reads its own spin; WQ5 asked only the order | ASKED — WQ5 with the rulings quoted, the spin's one source and the realm's edge (§5.3) |
| A-I5 | WQ20 (a) breaks SL1 clause 4 without naming it | ASKED — WQ21 names the clauses; neither road keeps every ruling; SL6 search first (§5.5) |
| A-I6 | B26 carried into the drag's physics: a cap | ASKED — WQ13; §11 row 23 (§7.3) |
| A-I7 | The caps list incomplete | FIXED — §11 rows 5, 12–19, 23; G-WEATHER-CONVERGE at C and 2C (§9.1) |
| A-I8 | `round()` decides the storm count | FIXED — the continuous ratio as the Poisson mean; the integer by σ(k) as a cross-check (WM3) |
| A-I9 | WL22 is no law; U-W5 cannot fail | FIXED — the stochastic weather generator as the law; U-W5 reads the unforced statistics (WM5, §9.1) |
| A-I10 | Earth constants missing, given as bands, not registered | FIXED — §3.4 table with one value each, registered; snow's fall by law; extinction from WL11 only |
| A-I11 | The row's units may bind on a dense world | FIXED — body units; overflow refused and counted; the 28.7-bar world in U-W1 (§4.1) |
| A-I12 | Child realms painted by the parent's weather; snow paint with no collision | ASKED — WQ14; "on whole hulls" defined (§7.2) |
| A-I13 | "The wind times the time" is no closed form; two players differ | FIXED — the two-phase advection as a closed form of shipped words and the tick (§6.8) |
| A-I14 | The stepper and almanac in `crates/bins`, outside Tier-A | FIXED — `vd-weather` and `vd-weather-shard` Tier-A; bins holds wiring only (§3.7) |
| A-M1 | "except U-W2" reads as decided | FIXED — "only if WQ1 (a) is ruled" (§1, §9) |
| A-M2 | WQ14 (G-LOOK) asked "NOW" against B36's schedule | FIXED — WQ23 asked at G-LOOK |
| A-M3 | "identity choice" for an epoch draw | FIXED — "a scatter around a computed value, named" (WM3) |
| A-M4 | B36 item 2 claimed for the wake | FIXED — removed with the new §3.6 |
| A-M5 | WQ20 (b) cites only B23 item 3 | FIXED — WQ21 (b) cites B35 items 2 and 6 |
| A-M6 | The late first fade against B18's 0.5 s ramp | ASKED — WQ11 states why and asks (§4.4) |
| A-M7 | WA1 omits the want's direction; WA2's "only" | FIXED — §4.8 directions; wording corrected |
| A-M8 | No epoch in the header; ids across a wake | FIXED — the GENERATION in the header; the book drops on a change; the count's width (§4.1, §4.2) |
| A-M9 | The gateway's statements grow without a re-base rule | FIXED — the re-base rule and the bound's victim (§4.5) |
| A-M10 | The stands' tick needs a lawful seam | FIXED — the clock seam in-process; the cluster's world start asked at WP1 (§10.3) |
| A-M11 | The wind's way into the sim's drag | FIXED — an injected per-child ambient; the lawful indirect use (§7.3) |
| A-M12 | The icy moon's weather is OFF, not empty by its laws | FIXED — §3.7, §10.3 and the stamp row |
| A-M13 | Gas giants missing | FIXED — §3.8 |
| A-M14 | "BY THE LAW, with no knob" | FIXED — "by the chosen curve" (§4.2, WM3) |
| A-M15 | §7.4's composition says two things | FIXED — one rule: a new rank after rank 2 under WQ3 (b); no cells under (a) (§7.4) |
| A-M16 | DEFERRED D-TERRAIN-8 stale | ANSWERED — this pass may write only this file; fixed in the next change that touches DEFERRED (WC19) |
| A-M17 | The frozen references red from WP3 to WP6 | FIXED — §10.2 |
| A-M18 | WQ9 (b): S3 and T3 lean to (a) | FIXED — WQ8 says so |
| A-M19 | "Earth's charter" reads as an Earth body | FIXED — "the laws on Earth's real facts in the bench copy" (§3.2, §9.1) |
| A-M20 | WQ3 does not cite T3 class 3 | FIXED — WQ3 cites it |

**Review B — the code facts**

| Id | Finding | Answer |
|---|---|---|
| B-B1 | No `sim::io` random seam; DetRng is seed-derived | ASKED — WQ8 with the facts and three priced roads (§3.7) |
| B-B2 | `ChildOrbit` invisible to the SL1 fences | FIXED — `PlacementTold`, `DOWNWARD_STATEMENTS`, the module fence extended, a fence against the boot's orbit (§4.8) |
| B-I1 | `ChildFelt`'s classes misread | FIXED — FireAndForget + ProducerLessReliable; pins 7→8 and 45→46 (§4.7) |
| B-I2 | A written cloud motion is ignored at zero depth | FIXED — the composite after the reconstruction keeps clouds out of the main history; else `motion_at` changes (§6.5) |
| B-I3 | The crossing reads the spin; the relayed hop writes zero | FIXED — the facts replace UNMEASURED; in WQ5's price (§5.3) |
| B-I4 | No code draws the pole's azimuth or the spin's phase | FIXED — two identity draws appended after `u_cos`, with pins (§4.8, §5.3) |
| B-I5 | The ambient is one value for the realm | FIXED — a per-child function injected by the root (§7.3) |
| B-I6 | The stepper outside Tier-A; the fixture is not G-IDENTICAL | FIXED — Tier-A crates (§3.7); HR4 ASKED — WQ6 |
| B-I7 | C = 128 is no level of any body's grid | FIXED — a level of the body's pyramid; the new seam answers listed (§3.7) |
| B-I8 | No receiver statement of what it holds | FIXED — `WeatherHeld` (§4.5) |
| B-I9 | A wake and a kill-9 are the same boot | FIXED — one boot rule; the store family (§3.6) |
| B-I10 | The Earth column used the home planet's sunlight | FIXED — Earth's own numbers (§3.2) |
| B-I11 | R_t on a different definition | FIXED — the published definition: home 0.50 (0.98), Earth 0.061 (§3.2, WM2, WQ1) |
| B-I12 | The snow's seat names the wrong quantum | FIXED — the gap byte's 1/128 under WQ3 (b) (§7.4) |
| B-M1 | The floor's citation | FIXED — `version.rs:8-14`, `:369-371` |
| B-M2 | MSAA 4× default cited wrongly | FIXED — the engine's default; the client's setting; the temporal path's `Msaa::Off` (§2.1) |
| B-M3 | `PATCH.md` has 118 lines | FIXED — `:1-118` |
| B-M4 | Drifted lines | FIXED — every cited line moved to today's |
| B-M5 | MEASURED rows without dates; the hidden-window run | FIXED — dates and the note (§2.3) |
| B-M6 | `day_s` is sidereal | FIXED — the solar day 72.65 h and the rest (§5.1) |
| B-M7 | The pyramid ships heights alone | FIXED — §2.2 |
| B-M8 | Postcard does not shrink this row | FIXED — header 35–40 B, row 19–23 B, statement 110–290 B (§4.1, §4.6) |
| B-M9 | The surface is a boot gate, not a profile field | FIXED — the install rule and the boot gate (§3.7) |
| B-M10 | MSAA edge pixels march per sample | FIXED — the cloud term sits there too (§6.10) |
| B-M11 | No direct `libm` | FIXED — std f64 (§3.7) |
| B-M12 | More Earth constants in the static climate | FIXED — §3.4 row 13 |
| B-M13 | A base arrives in beats | FIXED — one to three beats (§4.6) |
| B-M14 | The first statement waits for the first solve | FIXED — §3.6 |
| B-M15 | "An upper bound of the room" is inverted | FIXED — a lower bound; the omissions; the CPU's 0.8 ms (§2.3) |
| B-M16 | The want's trigger | FIXED — drawn realms with air only (§4.5) |
| B-M17 | Where snow sits in the five ranks | FIXED — after rank 2 under WQ3 (b) (§7.4) |
| B-M18 | Weather words also in the solve and strata | FIXED — §2.1 |
| B-M19 | `ArtifactPart` is already a generic relay | ANSWERED — the new arm is kept as a naming choice that the owner's standing rule supports (WC30) |
| B-M20 | "No tilt": the drawn obliquity is the orbit's inclination | FIXED — §2.1, §5.3 |
| B-M21 | A new client arm must be gated on the minor | FIXED — §4.5 |

**Review C — the look, the seams and the cost**

| Id | Finding | Answer |
|---|---|---|
| C-B1 | The near pass flickers at 528 m/s; the 200 m split trails | FIXED in revision 3 (V-1) — the parallax law; the passes blended by weight across the band (CR4); footprint-sized steps along each ray with the phase fixed to a planet-frame lattice, for the near pass and for the far pass inside the layer (CR5, §6.4); A1 benched first |
| C-B2 | The map's look steps at its ticks | FIXED — the look interpolates; collision alone would step; the season continuous or keyframes ahead; G-MAP-STEP (§4.3) |
| C-B3 | Shadow maps stutter under a moving field | FIXED — the maps in the clouds' drift frame (§6.7) |
| C-B4 | Shafts cannot come from today's air march | FIXED — Hillaire's per-pixel way on the temporal path; none on MSAA until D14 (§6.6) |
| C-B5 | Low-resolution clouds halo every skyline | FIXED — edge pixels at the render size, added up by the cloud history at the display size, the sky share behind all ground (V-3); nearest-depth upsample; G-CLOUD-EDGE (§6.5) |
| C-B6 | Line G cannot hold the flight cases | ASKED — WQ17; the cost restated as 2.3–3.5 ms ESTIMATE with its bench (§8.1) |
| C-B7 | Part-clear cloud over moving ground ghosts | FIXED — the composite after the reconstruction; the other road benched (§6.5) |
| C-I1 | The air join counts the cloud shadow twice; clouds light themselves | FIXED — single scattering only, ψ_ms by a cloud factor, a cloud-free probe for clouds (§6.5, §6.6) |
| C-I2 | One sky-light probe for the whole landscape | FIXED — a top-down cloud occlusion map per pixel (§6.6) |
| C-I3 | One or two air samples in front of a far cloud | FIXED — the front segment marched with the full count, priced (§6.5) |
| C-I4 | The far shadow erases resolved shadows | FIXED — the hand-over by the pixel's size; the clumpy far value; the ramp; the terminator priced (§6.7) |
| C-I5 | A deck under cirrus reads black | FIXED — one map per disjoint layer; a stand (§6.7, §10.3) |
| C-I6 | CR2 is a new method; the clumpy correction now; the footprint | FIXED — CR2 with the gamma mean, the display footprint, two render scales |
| C-I7 | One drift; no growth or decay | FIXED — two-phase advected noise with a regeneration law (§6.8) |
| C-I8 | Missing cloud kinds and scales for the targets | ASKED — WQ18 for the decks; the rest FIXED (WM4, §6.9, CR8) |
| C-I9 | A cloud veil inside the cockpit | FIXED — the hull's start depth (§6.5) |
| C-I10 | One σ is not one look | FIXED — matched on radiance, transmittance and variance; an exposure law; streaks after the reconstruction (§7.1) |
| C-I11 | The far snow and wet gloss; rebuilds | FIXED — the covariance, the gloss lobe, the ramp (§7.4, §7.5) |
| C-I12 | No contract for clouds in reflections | FIXED — the contract for the water fill (§6.10) |
| C-I13 | The MSAA path costs four times | FIXED — sizes against the display; both priced; D14 first (§6.10) |
| C-I14 | 32-bit positions hold half a metre | FIXED — eye-relative marching; the 64-bit split (§6.10) |
| C-I15 | The bench plan cannot decide the frame | FIXED — pass lines, the missing cases, the verdict waits for line A (§9) |
| C-I16 | A storm dissolving in ten seconds is not weather | ASKED — WQ11 with the hold (§4.4) |
| C-I17 | The cloud kind is a switch | FIXED — continuous mixes; G-KIND-SWEEP (WM4, §6.9) |
| C-M1 | 40 px uses a 90° view | FIXED — 58 and 26 px on the 45° seat (§6.3) |
| C-M2 | `motion_at` ignores written motion at zero depth | FIXED — not needed after the composite moves; else changed before PREPARE (§6.5) |
| C-M3 | "Never four times" is false at MSAA edges | FIXED — §6.10 |
| C-M4 | CR5's space | FIXED — the planet's frame (CR5) |
| C-M5 | The cloud history across a boarding | FIXED — turns by the air's rotation (§6.10) |
| C-M6 | The far rain under the shell's base | FIXED — the march reaches the ground where it rains (§6.1) |
| C-M7 | A finer map level that lands while drawn | FIXED — B18 item 6's ramp (§4.5) |
| C-M8 | The span's bound inverted | FIXED — §2.3 |
| C-M9 | "Under 10 µs" is not ARITH | FIXED — ESTIMATE (§8.3) |
| C-M10 | The air volume dropped without saying | FIXED — near medium in the volume, far rain in the march, and why (§7.1) |
| C-M11 | The first solve's order | FIXED — §3.6 |
| C-M12 | G-DETERMINISM must expect the drift | FIXED — §6.10 |
| C-M13 | The air's ground albedo under a deck | FIXED — a named limit (§6.6, §11 row 26) |
| C-M14 | A far airy body's clouds under one air | FIXED — stated with WQ19 (§6.11) |

**The counts.** 111 findings (A: 4 blocking, 14 important, 20 minor; B: 2, 12, 21; C: 7, 17, 14): FIXED 95; ASKED 14
(A-B1, A-B2, A-B3, A-I2, A-I3, A-I4, A-I5, A-I6, A-I12, A-M6, B-B1, C-B6, C-I8, C-I16; B-I6's HR4 half rides with A-B2 on
WQ6); ANSWERED 2 (A-M16, B-M19). Every BLOCKING finding is FIXED or ASKED: A-B4, B-B2, C-B1–C-B5 and C-B7 are fixed in the
design; A-B1, A-B2, A-B3, B-B1 and C-B6 are owner questions, and the design no longer claims the ruling that each one
touches.

**The verifier's check of revision 2 (12 of the 13 blocking fixes HOLD, C-B1 PARTLY; five important gaps; minor points) —
answered in revision 3**

| Id | The verifier's item | Answer |
|---|---|---|
| V-1 | C-B1 / N1: CR5's shells at fixed radii leave a level ray inside a cloud with no sample for 3.6–11 km; two step rules; a dithered split needs a history | FIXED — footprint-sized steps along each ray, the phase fixed to a planet-frame lattice; the far pass inside the layer the same; the passes blended by weight (§6.1 items 3 and 5, CR4, CR5, §6.4) |
| V-2 | N2: WQ4 (b′) does not price the orbit's start and periapsis angles | FIXED — three roads, a refused fourth, road 1 recommended, each price (§5.4, WQ4, §0 item 6, §1 S7-1 row) |
| V-3 | N3 / the C-B5 note: the edge rule at the display stage; render against display sizes | FIXED — the three-part rule at the display size; all sizes stated against the display (§6.1 item 5, §6.5, §6.10) |
| V-4 | N6 / the C-B4 note: the shafts stop at an opaque cloud's edge | FIXED — the cloud pass's front-air march carries the same shaft integral (§6.5, §6.6) |
| V-5 | N5: the steering level as an absolute 700 hPa | FIXED — 0.7 of the body's own surface pressure, a named calibration (WL17, §3.4 row 5) |
| V-6 | N4 / check 3: two sources of rain; the chain's probabilities unnamed | FIXED — the chain reads the list, the rain on screen follows the shipped cells shaped by the drawn storm; four values in §3.4 row 14; U-W5 only reads (WM5, §9.1) |
| V-7 | The A-B1 note: road (b) after the freeze | FIXED — road (b) ASKED against T3 class 3, or read in W2's pattern (§1, §7.4, WQ3) |
| V-8 | The A-B2 note: equal drifts on a bent and a flat grid | FIXED — the same birth and death ticks, the drifts in each grid's own coordinates; geometry only through the seam (§3.7) |
| V-9 | The A-B3 note: RH_c read at a grid that a bench picks | FIXED — Quaas's law of the box size fixed before any bench; the value follows at U-W12's level (§3.4 row 1, WQ9) |
| V-10 | The B-B2 note: "only the almanac module names `PlacementTold`"; the sender's module; the detector's types | FIXED — the allow-list, the sender outside the nine fenced modules, the orbit elements' type in the wire pin (§4.8) |
| V-11 | The C-B2 note: "zero in dry weather" | FIXED — the melt and the drying change cells after a storm; MW-1 counts every kind (§4.1) |
| V-12 | The C-B7 note: two estimates of the front air | FIXED — stated as continuous in T_c, read by G-CLOUD-AIR (§6.5) |
| V-13 | Check 1: §0 item 6 hides WQ4's price | FIXED — §0 item 6 |
| V-14 | Check 2: "disagree on buoyancy"; the client's almanac fence omits the waves' wind | FIXED — the client computes no buoyancy; the fence lists the waves' seasonal wind (§7.6, WQ15, §5.2, §5.4) |
| V-15 | Check 3: the eye's integration time has no value or row; U-W1 "decides" | FIXED — about 0.1 s (ESTIMATE until read), §3.4 row 15; U-W1 "reads the laws of WM2–WM3" (§7.1, §9.1) |
| V-16 | Check 5: two MEASURED marks without dates | FIXED — both dated 2026-09-29 (§2.3, §8.2) |
| V-17 | V13 L16 "KEPT" while its cap goes | FIXED — "KEPT as amended by B23" (§1) |
| V-18 | WQ4 (a) states the date on a lane; the daily cycle's keyframes | FIXED — both in (a)'s price (§5.4, WQ4) |
| V-19 | "No client can predict the weather" | FIXED — a keyed hash, and the design rests on S1 (not valuable), not on secrecy (§3.6) |
| V-20 | The stale bound's lower end is a floor | FIXED — named as the lane's own law (§4.4, §11 row 7) |

All 20 are FIXED. The coordinator's rule (no gendered pronoun for any person anywhere) is checked: none remains.
