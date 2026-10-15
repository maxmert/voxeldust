# Voxeldust — Greenfield Rebuild (transfer-first)

Multiplayer voxel planet MMO — "Star Citizen meets Minecraft". Spherical voxel planets,
player-built ships you walk inside while flying, Newtonian physics, distributed shard
architecture. **This worktree is the approved total-greenfield rebuild** (June 2026);
since 2026-09-07 `main` IS this tree (fast-forwarded from the `warp` worktree at Step 25); since 2026-10-14 `main` is fast-forwarded from the `game` worktree and THE WORK CONTINUES IN THE MAIN CHECKOUT (dev-cluster slot 52); the old code lives on `ecs-system` as reference/spec ONLY.

## THE binding specs — read before designing anything

- `docs/design/PLAN.md` — the approved plan: context, 6 hard rules, unified architecture, P0–P11 roadmap.
- `docs/design/*.md` — the hardened subsystem designs (connection_plane, transfer_protocol,
  test_harness, identity_persistence, generic_transfer, sealed_shards, coverage_e2e).
- `docs/design/integration.json` — 19 binding cross-design conflict resolutions + glossary.
- `docs/design/owner_decisions_*.md` — ★ THE BINDING RULINGS, and **the NEWEST FILE WINS**. Read the
  newest one BEFORE relaying any plan or design content: these are later than every design doc, and a plan
  agreeing with itself proves nothing. Newest first:
  - `owner_decisions_2026-09-22_believability.md` — ★ B1 STOP AND REACH BELIEVABILITY INSIDE 8d BEFORE 8e
    (owner: "Agreed"): the order — lakes from a water budget → the summer ice line + the crater age → the flat
    receivers by distance + the scratch measurement → the river lines, the carve and the floodplain down to
    the fine rungs (the cure for the dunes) → the lake side word; each step ends with the census's numbers and
    ONE picture on the three stands (the belt, the coast, a plain); 8e starts on the owner's word. ★ B3 THE
    GATES as numbers with a body each: G-LAKE-SHARE ≤ 4 % of land, G-LAKE-COUNT, G-BASIN (the inventory
    balances), G-ICE ~10 % of land, G-BUZZSAW 1 500 m, G-CRATER ~190, G-GRID (no lattice direction),
    G-HYPSO, G-RIVER, G-LOOK. ★ B5 the home planet's TILT (55°, a seed identity choice) STAYS — a hand-set tilt
    is a seam, refused; G-ICE tests the LAW on an Earth-like charter, the home planet's 4 % is a reading; the
    caps question waits for 8e's two-column seasonal balance; the lawful road to caps is the seed search.
    ★★ B6 (2026-09-24, owner: "In general I agree … make sure that visually we are moving towards this target
    picture") — the dunes, the teeth and the water positions have ONE root: the solve knows only 8 192 m nodes,
    under them the ground is noise to 32 km and the water is a node square. THE PLAN: measure first (gates that
    can see the dunes and the teeth) → the lake budget from the whole watershed → R2 (one eroded 1 024 m level
    in the artifact) + the noise only below it → one shoreline from one field on the R2 ground → G-GRID again
    → outlets, deltas, R3 → G-LOOK. EVERY step also shows a FOURTH stand framed like the reference picture
    (the Crimson Desert vista, kept outside the public repo) with its skyline numbers. THE DRESSING (snow,
    ground colour, forest, cliffs, far shadows, water look, clouds) is DEFERRED D-TERRAIN-8, not forgotten;
    B6.5 (amended): the far shadow is WANTED but not needed to judge the pictures (owner: "We can go without long
    shadows for now"); each picture is captioned "no far shadow".
    ★ B7: an inland pocket under the sea's level is a closed basin by its budget (the sea over the connected ocean
    only); the discharge row is the runoff over the final receivers; step 2 is built while R2 is designed.
    ★ B8 (2026-09-25, after step 2's pictures showed every small lake as an 8 km square): THE EDGES BEFORE R2 (one
    shoreline from one field, sea and lake, on today's ground); THE RIVERS FOLLOW THE WATER (receivers and the
    discharge row follow the budget's water — amends B7 item 2); R2's SLICE 0 FIRST (the bench + a hillshade for the
    owner's look, before any bake service; the coarser ground stands until a bake is done); step 2 committed.
    ★ B9 (2026-09-25): step 2b committed (generator 21: the rivers follow the budget's water; G-RIVER-DESCENT 0);
    the lake share 4.828 % kept RED and judged after the shoreline step; the flat directions' lattice lean cured BEFORE R2.
    ★ B10 (2026-09-26): the shoreline BENCH committed (N1 steady at rungs 0–9, O1 a W16 breach, every rule red at the far
    swaps); ONE MORE BENCH ROUND (N2) before any kernel; the WASHBOARD (D8's eight-direction bias) measured with two cures —
    seeded Rho8 and D∞ — BEFORE R2, and FIRST (then N2, then build both, then fly all stands against both references).
    ★ B11 (2026-09-26): the WASHBOARD is the DEPOSIT's (one-node lines along the flat's paths from pass 1; the washboard
    bench, its solve copy proven exact); a SLOPED basin fill with a published slope law, measured first; D∞ adopted for the
    erosion passes in the same change (the world moves once); the washboard bench committed.
    ★ B12 (2026-09-26): the sloped fill (Paola's fans) measured — clean floors, fans kept by a skip bit, the facing halved,
    but giant flat lakes and barely changed pictures (rain-cut vs runoff-fans; 11.9 km-deep pockets); THE BASINS PARKED until
    after R2 (R2 decides the cut's water and measures D∞); NEXT the shore round N2, then R2.
    ★ B13 (2026-09-26): N2 IS THE SHORE (N1's side + a fade PER OCTAVE over its own reach, faded by every nearby water + a
    30 m BAND on the node words + the SIDE HOLD); the deciding number passes (1.40 px); the fine octaves' lift measured
    (571.8 m at share one = N1's flat discs); the 30 m band KEPT (its price: a steep coast's line midway, deep small lakes by
    their node's footprint); the filled troughs ACCEPTED until R2 and R4; BUILD N2 (the integer shore law, both hosts), THEN
    FLY all stands against both references; narrow water at the far swaps stays RED for slice 0b (SL6).
    ★ B14 (2026-09-26): the build's first four questions on MEASURED numbers — SEND THE SHORE BAND (SL6 yes: 0.77 MB of
    band heights and lake levels with the head); the REACH IS A LAW (C_smooth 0.7723, C_ridged 0.9107) and the spectrum's
    conflict is registered (the noise realizes a fifth of the stated slope); SEND THE FAR FOLD'S COUNTS TOO (0.64 MB; no
    empty far globe; W15.1 item 5 amended); the POCKET EXEMPTION kept as a stated rule. CORRECTED after the ruling by S0
    item 5: the band 0.81 MB with the lake levels, the counts 1.10 MB per NODE SQUARE (re-confirmed by B15).
    ★ B15 (2026-09-26): S0 items 4–6 measured — CURE THE FAR LAKE FLICKER FIRST (the one naming rule holds rungs 8–13 but
    a big lake's ragged edge flickers in one-column fragments at 13–18; the bench measures a non-flickering fold BEFORE
    S1); the WET TIE and the LONE HOLD join N2; B14's corrected sizes stand; Q4–Q6, Q8–Q10, Q12 accepted as recommended.
    ★★ B16 (2026-09-26, owner: *"No flicker or changing of the shapes between rungs at all"*): none of the three measured
    far folds (the box, the tent, no lone column) is taken — each thresholds counts on nested footprints, so a small
    feature changes at a swap; FIND THE BEST SOLUTION (the fold alone may not be the place), bench it against W16 in full,
    and bring it before S1. THE ANSWER, design D (`docs/investigation/2026-09-26/far_water_seamless_design.md`): the
    rung-free SKIN (the far water's colour is the share of N2's samples under the pixel) + the LID from rung 14.
    ★ B17 (2026-09-27, on slice K0's numbers, the design §9): Q9 THE SHADER CHOOSES THE SKIN'S LEVEL ITSELF (the Vulkan
    rule, the level of P_max/η, and its own probes) with the design's half-cell windows — MEASURED 0 steps at every front
    and climb, 17–79 MB of windows, the same picture on every GPU; the GPU's own anisotropic filter is not used for the
    skin. Q8 THE RIM IS SIZED AT 86.4° NOW and judged in the flight (wider margins only if the flight shows it). K0
    committed. ★ B18 (2026-09-27, after plain answers): Q2 THE LID YES (a height at every rung — MEASURED: dry columns
    keep their ground, the relief within 1–3 % of today's); Q3 THE DRY WORD SHIPS (SL6 yes, 48.9 KB); Q4 R2 NOW, the
    near slides judged in the flight; Q5 the hand-over by the pixel's size (no rung decides it); Q6 SLICE 0b RETIRES;
    Q7 the arrival ramp 0.5 s; Q1 THE FAR LOOK YES (a soft far coast, sub-pixel water a tint). Design D ruled whole;
    NEXT slice R1 (the renderer's own pops).
    ★ B19 (2026-09-27, the walk on the solved ground): the gateway's artifact delivery fixed now; the gate solves the
    home planet itself and stands on solved land (the stand on the circle where the star stands 15° up); a berth
    states its FACING (optional stored field); a tick's picture only from COMPLETE frames — SL6 YES: the new
    `ShardToGateway::WindowFramePart` (index 17, minor 35, the floor stays 34) carries the part number and count,
    the gateway reads the older `WindowFrame` as a whole one-part frame.
    ★ B20 (2026-09-27, the owner's first look in the window: "quality and speed quite terrible"): MEASURED the observer
    104 m over the sea near a coast (it keeps its starting radius) — the coast drew a hole under the land, skirts as
    walls, the land's undersides. THE COAST FIRST: a gate 100 m over the sea looking at the coast that FAILS today,
    cured until it passes, then the owner looks again; the walk on the ground after. The terrain's core share is HALF
    (7 of 14; amends F6's quarter).
    ★★ B21 (2026-09-27, the owner: "you don't build a generic solution, but some parts that you try to fit together"):
    STOP ADDING TERRAIN RULES; N2 parked. DESIGN THE ONE-VOLUME MODEL FIRST — one volume of material cells per body,
    generated as a SOLID pass then a FILL pass (sea, lakes, air/vacuum, other substances), edits change the same cells,
    every distance a copy made by one rule, a shore is where water cells touch ground cells; do NOT connect it to
    today's special shore/water rules. Multi-agent design with reviews → `docs/investigation/2026-09-27/
    one_volume_design.md`; the far-coast question (exact vs a small smooth change) goes to the owner with prices.
    ★★ B22 (2026-09-28, THE NEW ORDER OF WORK): (1) switch OFF the old water and shore rules (no water drawn at all; the
    dry sea floor shows; the solve's water CALCULATION stays); (2) SPEED, no world change — 60 fps at FULL SCREEN, no
    stall; (3) THE GROUND with biomes, believable, no flicker at any distance; (4) THE WATER FILL (one-volume, option (b):
    exact from ~450 km, smooth nearer; measure the smaller hand-over for "no flickers at ANY distance"); (5) THE EDIT
    SYSTEM as its own step. Q3/Q4/Q6 accepted; Q5 (the hills' average) decided in the ground step.
    ★★ B23 (2026-09-28, owner: "Ideally we don't do any caps and see what's happening … fully generated, not custom
    built, as all those laws should be applied to other planets and moons"): NO CAPS — the gates' bands are READINGS,
    never stops; every clamp/ceiling inside an operator goes or is replaced by its law (a limit that IS a law stays,
    named with its calibration body); the caps of ice come only from the climate law; the owner's look judges. THE
    THIRD REFERENCE PICTURE (outside the repo; status §9). FULLY GENERATED: no hand placement, no per-planet tuning;
    each ground step judged on the home planet AND other real bodies (a planet, a moon), never a test-only variant.
    ★★ B24 (2026-09-29, owner: "Ok, let's build temporal pass now."): D1 YES — BUILD THE TEMPORAL PATH by
    `docs/investigation/2026-09-28/temporal_path_design.md`; D2–D4 as the design recommends (T0 measure → T1 motion → T2
    the first picture at the sizes that ship → T3 the one geometry pass only after the edge look D5; our own
    reconstruction on FSR 2.2's structure, MetalFX only on D12); MSAA stays the default until D5/D14; every look the owner's.
    ★★ B25 (2026-09-30, the owner low over the dry sea floor: the haze vanished; all four laws approved, built before the
    rest of T2): (1) THE AIR RESTS ON THE SEA — the charter's words hold at the sea level the realm states (the datum on
    a dry body); (2) THE AIR REACHES EVERY GROUND — no eye and no drawn ground under the air; the molecules grow by the
    pressure law with no cap, the haze holds its sea-level value under the sea; (3) the planet's shadow in the air is the
    sea's horizon; (4) THE HAZE SCATTERS 0.9 (Bevy 0.18.1's Earth medium swaps scattering and absorption). Built as the
    air's sphere at the sea + the per-pixel march from the TRUE eye to its ground (vendored sixth change).
    ★ B26 (2026-09-30, the owner from orbit: "Can we not increse dencity below the sea level?"): AMENDS B25 law 2 — under
    the sea level EVERY term holds its sea-level value (no growth). MEASURED: the dry basins stay pale (0–5 levels less),
    since a 7 km basin holds about the whole column's air even at sea-level density; the cure is the water fill (B22 (4)).
    ★ B27 (2026-10-01, owner: "Yes, as it stands"; "1,2 first please, then rest of T2 gates"): THE EYE'S RULE IS THE SEAM
    LAW of the flight gate (the display picture under the engine's dither; a pop = the jump between two still sides, four
    frames a pair; patches beyond the 8-bit rounding + windows ≥ 1°; Blackwell/Crumey thresholds, the 45° seat, 200 cd/m²,
    seen at level 1); every flight leg must hold no pop the eye sees, the landing is reported; the level rule retires as a
    law. ORDER: the walk's stepping eye, then the flickering crack (cause, then cure), then the rest of T2's gates.
    ★ B28 (2026-10-01, three answers "Cure it now", "Wait for the solve", "Find the amplifier"): THE TILE REACH CURED
    NOW (one rule, W17: the shard reads the eye's height over the solved ground its artifact holds, as the client does;
    the gateway, which relays tiles undecoded, serves the whole tile ring; the tile square counts the nodes' own chords);
    THE PICTURE GATE WAITS FOR THE SOLVE (every stand on the solved ground, as B19; the four references change only on
    the owner's acceptance); G-DETERMINISM: find why the history turns a world a few ticks apart into 4-level steps.
    ★ B29 (2026-10-01): THE SOLVED PICTURES ACCEPTED AND FROZEN (owner: "Accept and freeze"; all stands, the default
    path); THE AMPLIFIER FOUND — the history's clip toward the box's centre carried near colours apart (the hill 7
    levels); the owner asked for "the most robust and production ready solution": FSR 2.2's rectification (a clamp
    into the box cut to the samples' range, 16-bit history kept) BUILT for the owner's look; the terrain crate's 52
    older coverage misses close inside each believability change (the owner's word owed); the window asked again.
    THE LOOK GIVEN (scale 1.0 "Good"; flying at 0.67 "All good!").
    ★ B30 (2026-10-01): ONE COMMIT NOW of B28 + B29 on `game`, no push ("Commit now"); the timing readings (the
    window's cost, G-NO-STALL) with Docker stopped, the window on the owner's word; G-GHOST and the V switch's
    crossing reading MOVE TO T4 ("Yes, move them"); the boarding's and the departure's crossings stay in T2.
    ★ B31 (2026-10-02): THE BOARDING'S THREE CURES (the window rows carry the attitude; every drawn row reads the
    camera's moment; the vendored sky table's eye — ON FOOT THE GROUND HAD NO SKY LIGHT, now it does: the vendored
    eighth change); G-CROSSING's tolerance HALF A LEVEL over the pairs beside the swap, one in ten thousand past 3
    levels (owner: "if half-level provides that, then it's ok"); the sky's light judged IN THE WINDOW, the references
    frozen again on the owner's word; the commit after the look.
    ★ B32 (2026-10-02, the owner in the window: "It looks great!!! freeze and commit please!"): THE SKY'S LIGHT ON FOOT
    ACCEPTED; the solved references FROZEN AGAIN on it (every stand, the default path); one commit, no push.
    ★ B33 (2026-10-02, "Readings to T4"): T2's readings-only items MOVE TO T4 (the departure's crossing, the sky's lag
    on the descent/climb, the T0 meters at two sizes, the still records' grid, a tool's jump); T2 CLOSES after ONE
    window session (Docker stopped): D5 at half size, D6 the stars' light, the frame cost, G-NO-STALL; THEN the terrain
    step's discussion (first: the waves in flight).
    ★ B34 (2026-10-02): T2 CLOSES — D5 GIVEN ("Looks amazing"), D6 GIVEN FOR TODAY'S SPRITE ("Looks amazing!": the
    star sprite after the reconstruction, through the air table's transmittance); the readings RED (full screen
    23.1 ms ≈ 43 fps; a frame over 33 ms in 18–36 % of the seconds at the ground stand and in every second at dusk,
    146 ms at the dusk boarding with a pipeline compiled then, 94 ms at the full-screen switch; AirPlay, the poller
    and the local cluster UNMEASURED) and MOVED TO T4 with a CLEAN READING FIRST; the
    window crash on a quick shrink CURED (the frame decision now runs after the render scale; a schedule test);
    one commit after a check; THE WAVES WAIT for the proper terrain.
    ★★ B35 (2026-10-02, owner: "Ok, agree with the plan then. Please get into the context the ground rules and laws
    before designing the plan. Remember that realm calculate and draw itself"): THE PICTURE SETS THE QUALITY, NOT A
    LIST ("it's not only about the bioms or things we have in the picture") — seven qualities everywhere, at every
    height, on every body; THE CATALOGUE of every kind of country the body's facts call for (each family priced
    first, T5); stands chosen by a rule on the home planet and two more bodies; THE ORDER: phase 0 ONE DESIGN WITH
    REVIEWS → the shape's foundations → R2 → R3 → the processes → biomes and materials → the cover → THE LIGHT (the
    far shadows join; amends B6.5) → G-LOOK → the waves → the water fill → ONE WEATHER STEP (almanac, weather list,
    clouds and their shadows, rain/snow on screen, wind, lying snow; amends T1 items 10–11 and T4: clouds, rain on
    screen and the forest BEFORE the freeze) → the edit system; A REALM CALCULATES AND DRAWS ITSELF (SL3).
    ★ B36 (2026-10-02, the phase-0 design `docs/investigation/2026-10-02/ground_step_design.md` written; five answers):
    Q3 the seasonal circulation that follows the sun ALLOWED (amends T5's "no monsoon"); Q7 the first visit "only by
    law, no manual law changes" (no exception to B16; a slow bake is solved by another law with its numbers); Q10 the
    target machine IS the owner's current Mac (M4 Pro, 14 cores, 48 GB); Q14 the art discussed when its time comes;
    Q17 (a) the orbit stated ONCE, the planet evaluates its own sun (a fenced, named exception to SL1 clause 3);
    the other questions one by one when their answer changes the work: Q1 YES (rule M the average, rule S the
    outline, the zero-mean gate, no ridged centring), Q2 YES (Earth/Moon/Mars data outside the repo as the measure),
    Q16 YES (stands pinned once, four views, two suns, three bodies; G-ICE on Earth's real data, amends B5 item 1);
    item 9 THE THIRD BODY = the desert world `Planet(17782101822166660494)` (the rule plus the owner's condition: its
    node is the one the 8 192 m target sets); Q4, Q5, Q6, Q8 when phase 1 ends; Q9, Q11-Q13, Q15, Q18 at their phases.
    ★ B37 (2026-10-02, after the Earth reading and the cause bench: "Design now, build in phase 2"): at 8.2 km Earth holds
    no hills either (the scale); away from closed basins the home planet holds ONE FIFTH of Earth's mountains and half its
    relief; NO cap binds — THE BELTS BARELY RISE (769 m at most: a belt's room is the relief less the planet's TALLEST
    platform, a cap, caps-list row 53) and THE EROSION TURNS THEM INTO LOWLANDS (8.4 times their uplift over the system's
    whole age); the rebound leaves 131 km grid steps. THE BELTS' LAW (convergence over the belt's own active time, limited
    per node only by the crust's strength) and THE LANDSCAPE'S TIME (erosion since the ground was last raised) are
    DESIGNED NOW and BUILT in phase 2's macro change before the first 1 km bake; THE SMOOTH REBOUND goes in phase 1.
    ★ B38 (2026-10-02, after two adversarial reviews of the belts' design found the cure larger than two laws): ONE
    BENCH (a copy of the macro solve, every item switchable in the copy only, its control the pinned digest); the
    binding laws and rules unbroken; a NEW PHYSICAL LAW is welcome where it adds realism and must be the RIGHT one
    (published, applied as stated, its own calibration body); NO CAPS AND NO ARTIFICIAL CHANGES (no clamp, no tuning
    knob; Earth's numbers are readings, never a target a constant is turned toward).
    ★ B39 (2026-10-03, the bench found today's solve NOT converged in time: 40 / 80 / 160 even passes give mountains
    0.47 / 0.11 / 0.02 %, so the world's relief is partly a numerical error): THE WORLD CONVERGES IN TIME — a BINDING
    GATE (the readings at N and 2N passes agree within a stated tolerance; as many passes or as fine a step rule as it
    needs), built WITH the new laws in phase 2 (converged today's laws grade the planet flat); the slower first solve
    accepted.
    ★ B40 (2026-10-03): THE BELT WIDENS UNDER A FORCE BALANCE (a belt's mean height stops where the plates' force holds
    it — about 5 km for Earth's numbers, Tibet 5 023 m — and the extra rock widens it into a plateau, the width growing
    by the influx less the erosion's efflux; peaks keep the strength law; AMENDS B37 item 1); A PLATE'S MEAN SPEED IS
    EARTH'S 50 mm/yr (the constant corrected to its own meaning, checked against Earth's 53.4 mm/yr ridge opening; the
    craters and the belts read the one speed; a world change in phase 2).
    ★ B41 (2026-10-03, after runs 9–10: the two belt laws cut the collisions' rock to a fifth; the Andean margins now
    81 % of the belts' rock; THE GATE FAILS ON THE LAND'S HEIGHT, 214 → 414 passes move the land over 1 km by 8–10
    points): THE GATE'S CAUSE FIRST (where a finer step erodes more), the Andean law's research BESIDE it; the erosion
    law (F49) waits; PROPER LAWS, NEVER EARTH (owner: "Are you trying to build Earth?") — a law reads the body's own
    facts, an Earth place only fixes a constant it cannot compute yet (India's decoupling depth, Tibet's slowing time,
    until a heat law per body); the owner's word on that answer owed; one commit, no push.
    ★ B42 (2026-10-03, the gate's cause MEASURED: the NUMBER OF ROUTINGS sets the rivers' cut — 1 / 2 / 4 / 8 a pass
    leave 35.7 / 21.0 / 8.4 / 0.9 % of the land over 1 km, routed often today's laws grade the planet flat — and the
    history does not settle even at one routing every 3 Myr): FIND THE MECHANISM (split each pass's cut: drains changed,
    lakes drained, flats with a new outlet), then the cure with its numbers; THE THREE DEFECTS CURED IN THE COPY (the
    ice's per-call thickness cap, the ice's cut missing from the rebound, the rebound by pass count); THE ANDEAN MARGINS
    PARKED (today's share a registered interim); one commit, no push.
    ★ B43 (2026-10-03, THE MECHANISM FOUND: the game's sweep reads every node it has just cut as a lake — its ground
    under its stale flood level, `routing_base` — so the implicit sweep climbs one node per routing; cured in the copy,
    the routing converges and today's laws grade the planet flat): THE BASE CURE GOES INTO THE GAME'S SOLVE IN PHASE 2
    with the new laws, never alone; F49 RULED — THE JOINT LAW (Adams et al. 2020, `E = K_lp·Q·S^2.2`) keeps the plains
    (land 17.5–19 % of the globe, land over 1 km 20–21 %); NEXT the three defect cures on the cured base, then how Earth's
    old continents keep their height (B37's landscape time); one commit, no push.
    ★ B44 (2026-10-03, the defect cures measured: the stepped ice first chopped its glaciers by the same stale-flood rule,
    fixed — water only where the routing found a lake; uncapped it cuts 4.8 against 3.5 × 10¹⁵ m³ and keeps the gate;
    THE REBOUND'S SPACING HAS NOT CONVERGED, the land over 1 km ~20 / 25.6 / 28.8 % at 125 / 25 / 12.5 Myr): THE ICE CURE
    GOES INTO PHASE 2'S SOLVE; NEXT the old-plains research with the rebound's convergence measured beside it; one commit.
    ★ B45 (2026-10-03, the rebound's spacing MEASURED — without its four floors it settles from 12.5 Myr and the gate
    passes; the old plains RESEARCHED — Earth's 485 m against the home planet's ~110 m, low because nothing raises them
    again for 4.8 Gyr): THE EXACT REBOUND (no floors, fractions carried, ≤ 12.5 Myr) GOES INTO PHASE 2; BENCH THE
    LANDSCAPE'S TIME (B37: a plate's plains erode only since its ground was last raised; the raise time from a reworking law
    calibrated on Earth's continental ages, Artemieva 2006); one commit, no push.
    ★ B46 (2026-10-04, the landscape's time BENCHED: the old plains 586–680 m against Earth's 485 m (310 m without it),
    the mountains 19.9 → 11.6 %, the land turned to tablelands; the gate fails by the lakes — a lake's level is fixed
    between routings): THE LANDSCAPE'S TIME GOES INTO PHASE 2 WITH THE LAKES' CURE; NEXT CURE THE LAKES (a lake's level
    follows its outlet inside the sweep), then the gate again; one commit, no push.
    ★ B47 (2026-10-04, the lakes' cure MEASURED — a lake's level follows its outlet inside the sweep; the routing
    dependence gone; the whole law set passes 214 → 414 passes): THE LAKES' CURE GOES INTO PHASE 2; THE GATE'S TOLERANCE
    (B39) — every share within 2 points, every height within 10 %, at N and 2N passes and twice the routings; NEXT THE
    MOUNTAINS (11–13 % against Earth's 2.39 %: which cause carries them, then each one's law); one commit, no push.
    ★ B48 (2026-10-04, the mountains MEASURED: the belts and their plateaus carry 88 % — Earth's share of the land but
    49–68 % mountains inside against Earth's orogens' 8–12 %, 1–1.8 km higher; the old plains match Earth's 516 / 318 m
    against 485 / 349 m; the craters minor): RESEARCH THE BELTS' SHAPE (the critical taper, a plateau's low inner relief,
    the forelands, the Andean height law), then bench the laws found; one commit, no push.
    ★ B49 (2026-10-04, the belts' slopes MEASURED: 7.7–10.8 % at the median against Earth's young belts' 3.2 %, the joint
    law's steady state under a bump that lifts the whole belt at once): BENCH THE BELT AS A CRITICAL WEDGE WITH A PLATEAU
    (Roe et al. 2008; the taper of Davis, Suppe & Dahlen 1983 calibrated on Taiwan; the crest to the force ceiling, then a
    plateau) and read where the plateaus' rivers end; one commit, no push.
    ★ B50 (2026-10-04, the critical wedge BENCHED: mountains 11.7 → 5.4–6.2 % (Earth 2.39 %), the ramps near Earth's, the
    gate passes; the plateaus carry ~64 % of the rest — each copies its old platform's tilt and drains to the sea): THE
    WEDGE GOES INTO PHASE 2'S BELTS' LAW; NEXT A LEVEL PLATEAU (one level along its margin, the lower crust's flow, Clark &
    Royden 2000) and whether dry plateaus' lakes close under the solve's own water budget; one commit, no push.
    ★ B51 (2026-10-04, runs 26–30: the level plateau 6.20 → 5.45 % mountains, the gate passes; the joint law reads the RAIN,
    `Q = A·R`; the water's own paths change the mountains little, raise the old plains (median 356 m, Earth 349 m) and widen
    the closed basins under the sea (7–8 % of the land); 87 % of the plateau land is ANDEAN): THE LEVEL PLATEAU AND THE
    WATER'S OWN PATHS (the law's own rain on the area that truly drains to a node; the budget's lakes in the passes; a
    closed basin keeps its rock, filling from its floor) GO INTO PHASE 2; NEXT THE ANDEAN MARGINS' LAW (research, then
    bench); one commit, no push.
    ★★ B52 (2026-10-04, owner: "when the implementation part will start?" → "Build now, Andean beside"): THE BUILDING
    STARTS NOW — the "before" record on the pinned stands (headless) → PHASE 1 (the flat directions, the smooth rebound,
    G-MEAN) → PHASE 2'S MACRO CHANGE with every ruled law in the generator's integer form under the float fence, one change
    at a time; the Andean margins' law researched BESIDE by a read-only agent, joining phase 2 as one change (the interim
    until then); the other phase-0 items move into the phases that use them.
    ★ B53 (2026-10-04, owner on phase 1's pictures: "very hard to judge … I didn't see the mountain region … let's do
    [phase 2] and then look at the end result"): BUILD PHASE 2 NOW; ONE LOOK AT THE END RESULT with stands that hold a
    mountain region and the other kinds of country (never only today's five plain stands); the frozen references stay
    until then; no commit until the owner asks.
    ★ B54 (2026-10-05, after the gate's cause was measured: 17 % of the rock vanished past closed hollows' spills — CURED,
    the rock kept exactly; the heights converge from 414 to 814 passes; the plains' share carries its own noise of about
    2.6 points at one resolution, the young plateaus' drainage layout being sensitive): THE CURE FIRST (find whether the
    sensitivity is numerical — the suspect, the routing rationing how fast a valley captures flat plateau ground), THEN
    A NOISE RULE (each reading's own noise measured at one resolution; N and 2N agree within it where it exceeds the
    tolerance); 214 PASSES STAY until the cure.
    ★ B55 (2026-10-05, owner: "Agree, go"): THE BAND'S TWO ENDS IN THE ARTIFACT'S HEAD (SL6 YES) — the shard states the
    solved ground's lowest and highest point once and every host reads them, so the band is one value on every machine
    and never changes in flight (the skyline's walls, cracks and lids were the summits over the band's top); THEN the
    level handover's pops on steep ground and the far summits the culling drops MEASURED planet-wide, headless; per-tile
    or per-point ranges only where that shows a defect.
    ★ B56 (2026-10-05, owner: "414 steps (Recommended)"): B54 ANSWERED BY MEASUREMENT — a second rock defect (an endless
    loop past full hollows) CURED, the rest passes each hollow once, no rock lost; the sensitivity is NOT numerical (the
    chaos twins' plains spread 1.4 / 1.6 / 5.0 points at 214 / 414 / 814 passes, plate 10's young plateau); under the
    noise rule the gate FAILS 214 → 414 on the heights and PASSES 414 → 814 on every reading: THE SOLVE'S AXIS IS 414
    PASSES (every first solve ~1.6× as long); then the world moves once (generator and artifact key 25, the pins), the
    census and pictures on three bodies, the owner's one look.
    ★ B57 (2026-10-06, owner: "Yes, proceed", after the generator-25 pictures): THE LOOK AT PHASES 1 AND 2 GIVEN (not
    G-LOOK); R2 NEXT in the ground step design's order (§5.2) — ITS SLICE 0 FIRST: the bench (M2-1: R1's loop as the
    control, the new operators without caps, the bake's rate, G-MEAN) and a shaded relief at 100–150 km on the three
    bodies for the owner's look, before any bake service, world change or data crossing; Q5's parts, Q6, Q8 and the SL6
    asks A1/A2 come with measured numbers at the slice that needs them; no commit until asked.
    ★ B58 (2026-10-06, owner: "I would take the recommendation which will remove the strait or 45 percent angled rivers
    (or staicased), as it hurts believability"): R2's slice 0 measured — the 1 km law passes Hack (n 0.29 → 0.52); on a
    smooth cone D8 runs down the grid's rays, D∞ makes an eight-armed star, RHO8 cuts valleys at every angle; the macro
    ground carries the grid's mark (straight trunks, 45° side valleys, stepped coasts) — THE MACRO ROUTING'S GRID LEAN IS
    CURED FIRST: Rho8 in the macro routing and erosion sweep (amends B11's D∞), measured in the bench copy, then built as
    generator 26 with new pins and the gate; the target is the owner's look (no straight, 45° or staircased rivers or
    coasts); then R2's bake.
    ★ B59 (2026-10-06, all four recommendations): the remaining marks have ONE ROOT, the first land (87 % exact flats,
    no relief between 8 and 375 km; the plates' steps one node wide) — BENCH THE FORMS OF THE FIRST RELIEF (a crust-
    thickness scatter through Airy, the same through flexure, the older landscape's relief by Earth's spectrum), THE
    OWNER CHOOSES FROM THE PICTURES; THE MARGINS BY KIND by their published laws (each node its cell's mean; rifted by
    the necking instability, Andean by the interface–Moho crossing at 23°, transform at Earth's measured 50 km); a
    SOURCED per-plate crust spread measured beside [0.7, 1.4]; THE WORLD MOVES ONCE (generator 26 = Rho8 + the relief +
    the margins, new pins, the gate), then R2's bake.
    ★ B60 (2026-10-06, the owner chose (c) over the recommended (a)): THE FIRST RELIEF IS THE OLDER LANDSCAPE'S — Earth's
    continental spectrum (β 2.09) sized to 131 m at 110 km, a named scatter by the crust share; with Rho8, the margins by
    kind (cell mean; the Andean wedge from the law's coast; the trench's inner wall) and the sourced spread it makes
    generator 26; the gate (414/828 + twins) first; THE SECOND CLIFF (a coast inside a plate filled ~4 km and ending in a
    3.5 km cliff, also in today's world) measured and brought back before the world moves.
    ★ B61 (2026-10-06): the (c) flooding was a BENCH DEFECT (the relief added after the freeboard law; corrected, 24.6 %
    land); THE BELTS' LAW READS THE CRUST SHARE CONTINUOUSLY in generator 26 (the second cliff: the wedge stopped at the
    continent's half-share line); THE GATE'S CAUSE FIRST (plates 5 and 13, old low plains, −10 to −12 % at 828 passes).
    ★ B62 (2026-10-07): the gate's two causes are NUMERICAL — EIGHT SWEEPS A PASS (the implicit step's error) and THE
    CLOSED BASIN'S CAPTURE CURED INSIDE THE SWEEP (a basin opened only at the next routing, so the coarser axis kept rock
    in captured hollows); designed and benched, then the gate again.
    ★ B63 (2026-10-07, B62's gate: thirteen of fourteen readings pass, plate 5 cured, PLATE 13's MEAN −11.9 % against 10 %;
    generator 26 built, its first land the benched world's byte for byte): PLATE 13's CAUSE FIRST (the ledger by plate,
    then the cure; the pins wait for the gate); A SINKING FOREARC IS DEEP SEDIMENT (continental crust that sinks where
    plates push together; rift basalt only where they pull apart — 12 609 home-planet nodes were misnamed); THE WINDOW
    NOW AS A PREVIEW (the generator-25 cluster stops, generator 26 built and solved, then the owner's word to open it).
    ★ B64 (2026-10-07, owner: "Can we give more cores for passes to build the realm for the first time?"): BUILD THE
    PARALLEL FIRST SOLVE NOW, measure after plate 13's runs; THE SAME BYTES ON ANY NUMBER OF CORES (the digest does not
    move — a pure speed change, the pin gate proves it); RUST'S OWN SCOPED THREADS, no new dependency.
    ★ B65 (2026-10-07, plate 13 FOLLOWS THE ROUTING COUNT ALONE — at 1 656 routings 414 and 828 passes give it 214 and
    215 m — and converges): FOUR ROUTINGS A PASS in generator 26 (two of the eight sweeps each; +~64 min a first solve on
    one core, the parallel solve cuts it), then the gate with twins; THE GATE READS EACH TRACKED PLATE'S OLD PLAINS' MEAN,
    never its whole-land mean (its dry basins under the sea level swing it: plate 5 reads 61–365 m under one law).
    ★ B66 (2026-10-08, B65's gate: plate 13 CURED by four routings, PLATE 5's OLD PLAINS FAIL −14.3 % — 345 / 305–329 /
    294–303 / 252–263 m at 414 / 828 / 1 656 / 3 312 routings, not converging): PLATE 5's CAUSE FIRST (414 passes routed
    eight times, the ledger by plate on both axes); the pins wait.
    ★ B67 (2026-10-08, the owner: "Is there any way how we can parallelize work? … Clouds?"): THE WEATHER STEP DESIGNED NOW,
    AS DOCUMENTS ONLY (almanac, weather list, clouds and their shadows, rain/snow on screen, wind, lying snow; one design
    with reviews), BY AGENTS FROM THIS SESSION; its BUILD keeps its B35 place (after G-LOOK, the waves and the water fill).
    ★ B68 (2026-10-08, plate 5's cause found — TWO NUMERICAL PARTS: the river's step over the long history (16 sweeps and 8
    routings a pass reach the 828-pass axis within 3 m) and the sea re-solved only every 20 passes near the present (a
    53–60 m jump that lands at another time on each axis)): BENCH BOTH CURES (the river's step ×2; the sea and the climate
    every pass), then the gate with twins; the benches first, the coverage gate after them.
    ★ B69 (2026-10-09, the cheap cure passes B39's gate on every reading, both cures fail by the noise's size): THE CHEAP
    CURE in generator 26 (the climate and the running sea every pass, `CLIMATE_EVERY` 20 → 1; the river step unchanged);
    THE NOISE RULE AS RULED; GENERATOR 26 CLOSES BY A FIXED LIST (the cure, a clean timing, the pins and the pin gate, the
    coverage gate, the census and pictures on three bodies for the owner's look); anything else is REGISTERED and does not
    stop the move to R2's bake unless it breaks a hard rule or a ruling.
    ★ B70 (2026-10-09, the owner: "migrate to bevy 0.20 and make sure that we are using all benefits that bevy 0.20 is
    provide, while not breaking anything we have today"): AFTER GENERATOR 26 CLOSES AND THE OWNER'S WORD, THE MOVE TO BEVY 0.20
    (before R2; lifts the 2026-08-27 upgrade gate, met); every benefit listed with its price for the owner to choose; nothing
    broken (suite, coverage, process and pin gates, the frozen pictures, a clean frame reading before and after); its design
    starts now as documents only.
    ★ B71 (2026-10-09, the owner: "If pictures quality improves in comparison to frozen ones, let's use new ones"): THE FROZEN
    PICTURES FOLLOW QUALITY — a change that improves them re-freezes them (side by side, no new defect by the gates, better;
    the owner sees the pairs); a worse picture is a defect to cure, never a re-freeze.
    ★ B72 (2026-10-09, the owner: "most of the time we are waiting … suggestions to improve that, while don't compromize on
    quality?"): THE FASTER SOLVE (the flood, the lakes' hierarchy and the deposit on all cores, the same bytes; built beside
    generator 26's close, the release pin gate proving pins and bytes at once) and A PARALLEL FEATURE TRACK (AMENDS T1: a
    feature that does not depend on the terrain's final look moves in its own worktree and session with its own gates;
    the owner names it); not adopted: standing rules, bench checkpoints, nightly-only coverage, a rented machine (not now).
    ★ B73 (2026-10-09; the pins recorded on one core, the release pin gate GREEN on 14 cores with the faster solve, the same
    bytes): THE ROUTING'S UNREAD WORK SKIPPED FOR NOW (in a wet body the routing's flood, flat walk, order and discharge are
    replaced by the budget's tree before anything reads them; only the ocean label is kept); THE FASTER SOLVE STOPS HERE (the
    steps still on one core registered); DOCKER STOPPED for the coverage gate and the clean timing.
    ★★ B74 (2026-10-10, the owner: "we are waiting for tens of hours instead of building the game"): THE FASTER COVERAGE
    BUILD (light optimisation, proved first on the terrain crate: the same misses) and THE SHARED TEST SOLVES (one moon solve
    per kind and test binary, on all cores) ADOPTED; ★ TESTS ONLY AFTER A SUBSTANTIAL CHANGE AND ASK FIRST — no suite, gate or
    coverage run starts without the owner's word, each ask states why it is needed now; THE VISUAL TESTS (glitch, crack,
    flicker) run more often, beside other work where the machine allows; the parallel feature track NOT NOW.
    ★★ B75 (2026-10-10, the owner after the window look: "VERY far from believable terrain … fast proceeding with the game
    development"; the shapes, the surface, no water and no cover the worst): LOOK FIRST — GENERATOR 26 CLOSES ON THE WINDOW
    LOOK (the census registered, not done); BEVY 0.20 WAITS (amends B70); R2's SMALL FIRST SLICE NOW (one region eroded at
    1 km on the bench + a shaded picture); A FIRST SURFACE LOOK BY LAW BESIDE IT (rock on the steep ground, soil in hollows,
    snow by the climate law; amends B35's order); checks by look, heavy gates only on the owner's word; feature track not now.
    ★★ B76 (2026-10-10, the owner on R2's 256 m pictures: "still a lot of strait lines … Are you sure we are using good
    algorithms?" → "we rely on the visuals first, even when we are building the game"): THE PICTURES DECIDE FIRST, on the
    bench AND in the build; R2's order: the literature on straight channels → bench the three causes one at a time (the
    start from the game's own fine relief, routing that splits the water, the finer cell) → pictures side by side with a
    straight-line number → R2's build only on the owner's word that they look believable.
    ★★ B77 (2026-10-11, the owner on the balance's readings — the land almost still and dry — with four new reference
    pictures: "it's not good enough. The terrain should be very believable"): LANDFORMS BY LAW BELOW 8 KM (amends B57/B58's
    R2 bake): the macro world says WHERE, the one generator makes the forms at each point with no bake from published laws
    (eroded mountains by the stream law's analytic solution on a drainage network built first; cliffs, mesas, towers by the
    beds' hardness and the cliff law; valley floors by the discharge); a design with reviews first, then one family on the
    bench beside the references; the macro world's rain and rise MEASURED BESIDE (a read-only study), the shapes not waiting.
    ★ B78 (2026-10-11): the macro study found the dryness a DEFECT (the rain belt's share never renormalised) and a MISSING
    LAW (the 55° tilt's seasonal circulation, B36 Q3, never built), plus the wrong height law; the owner chose "Later" —
    THE LANDFORMS FIRST, the rain's cure (generator 27) discussed again after their first pictures; the defects registered.
    ★ B79 (2026-10-11): the landforms design REVISION 2 (`docs/investigation/2026-10-11/landforms_by_law_design.md`,
    three reviews answered) — Q1 EARTH AS A MEASURING STICK YES (bench only, data outside the repo, no constant from it);
    Q3 THE LAW'S OWN WATER SHIPS (SL6 yes: one byte a node of rain × area over the tree, 8.9 MB, tile +10 %, a re-solve when
    built); Q10 THE RIVER TREE IS LAWFUL (no water word); Q4 NOTHING RETIRES BEFORE THE OWNER'S LOOK; BENCH STEP 1 ON EARTH
    STARTS (Taiwan, Oregon, a Colorado Plateau tile), then the owner's look; Q2, Q5–Q9 at their steps.
    ★★ B80 (2026-10-11, the owner on the pairs: "I like it WAY BETTER! What's next?"): THE ROOT WAS THE REGULAR GRID under
    every erosion method below 8 km; THE DIRECTION IS THE JOINT LAW'S STEADY STATE ON IRREGULAR POINTS (fastlem's point graph,
    MPL-2.0, bench only; each node's uplift solved so the fine mean equals the node's own); the ladder and grid bakes stop.
    THE ORDER (owner's word owed): bench defects + an eye-height view → finer ground to 2–4 m → a short design for the way
    into the game (SL6, SL10, region seams) → the build on the belt stand → the rain (B78) from spacesim's seasonal structure.
    ★★ B81 (2026-10-11, the owner: "Let's accept recommendations if they are not breaking ground rules and laws and if we
    don't build seams — all our solutions should be generic and reusable"): the way-in design's eight recommendations
    ACCEPTED — Q1 THE TILE SOLVE below 8 km (AMENDS B77's "no bake"; no noise under the finest level), Q2 the planet's
    shard solves/stores/ships, Q3 SL6 YES (one bulk part variant, the client acknowledges), Q4 idle L1 near arrival places,
    Q5 the coarser rung stands until live, Q6 a standing body moves with the ground, Q7 the bounds MEASURED first, Q8 F2
    AMENDED (a new fine law may move unbuilt ground; built ground never moves); NO SEAMS (one generic stitch, G-SEAM
    BLOCKING); ONE GENERIC TILE MACHINERY for every level, body and landform family; NEXT B0 (bench only).
    ★ B82 (2026-10-11, the owner: "Agree, let's try"): THE THRESHOLD BY THE LEVEL'S SCALE — tan 35° at 128 m (Burbank
    1996), S_c 1.25 at 16 m (Roering 1999; amends A6, refuted: 52 % of the 128 m ground over 45° at 1.25); THE STRAIGHT
    VALLEYS ARE THE 8 km WORLD'S DEFECT (an 8 km river along a node column), cured there with the rain (generator 27), the
    seeded displacement OFF; THE 128 m HOLD IS B80's ROUNDS (coarse start + three rounds + the additive hold; the design's
    linear solve refuted, amends §3.3); NEXT the slow tiles, P-COST, step 4.
    ★ B83 (2026-10-11, the owner: "Agree with the fix if it's performant, seamless and will be equal on BE and client on
    all devices"): THE HASHED POISSON-DISK POINTS; THE 128 m TILE SERVES RUNGS 7–12 (its mean pyramid); THE BAND ±4.1 km
    (0.70 × the node's half-diagonal); THE ADAPTIVE RICE CODER; THE 16 m LEVEL MUST REFINE THE 128 m GROUND (solved under
    its drainage, held per 128 m cell) — fast, seamless, the same bytes on server and every client — before B3; NEXT
    step 5 (the plain and the belt at 128 m for the owner's look).
    ★ B84 (2026-10-11, the owner: "Proceed"): THE 128 m LEVEL IS THE LAW ALONE — NO ADDITIVE HOLD (amends B82 item 3: the
    hold dug hollows up to 1.2 km deep; without it no pit, each node's mean ≤ 1 233 m off its row = 0.15 px at the rows'
    switch); G-MEAN BECOMES "THE SWAP TO THE COARSER LEVEL UNDER A PIXEL AT ITS SWITCH" at every level; NEXT the 16 m fix.
    ★ B85 (2026-10-11, the owner: "Agree with your proposal if we don't sacrifice quality"): THE SHARD SHIPS THE FINISHED
    TILE (joined to its neighbours, with its coarser copies; amends the design §4.4: no cut paths, the client only reads);
    THE CORNER RULE (each edge's cut in two halves from each corner point to the edge's midpoint; a corner point from its
    four tiles, a midpoint from its two) — every finished tile depends on its eight neighbours only; judged against the
    bench's stitch at a four-tile corner before the build takes it — MET (the same slope breaks to the third decimal, no
    line at the corner; the finished tile 303 KB against the raw 463 KB).
    ★★ B86 (2026-10-12, the owner on B2's flats: "Yes please, add those … no teleport … put player at the starting
    position immediately … show the loading screen over the gameplay"): THE FINAL GROUND FROM THE FIRST FRAME (no
    temporary ground in the normal case; the client's own solve refused for the speed) — THE ADVANCE CALCULATION (the
    planet's shard solves every fine tile once per world version: its occupants', its arrival places', then the rest;
    how an unvisited planet completes it OPEN at the build), FAR COPIES FIRST (a tile's levels ship apart), SENT AHEAD
    WITH THE SPEED, THE PLAYER'S DISK CACHE (keyed by the input word; B81 Q3's acknowledgement names tiles from disk
    too), THE LOADING SCREEN OVER THE RUNNING GAME AT LOGIN ONLY (no teleport, no special login path: the player at the
    start at once, every machine as in play, the client judges "complete" from what it holds), THE SAFETY NET = cure B
    (the 8 km ground fills only empty places; amends the 2026-09-30 own-height stand-in for the fine level); bandwidth
    ESTIMATES (login ~10–15 MB; ~10 Mbit/s least, 25 comfortable). ORDER: the face-edge cut → the safety net → the
    owner's look → far copies → send-ahead → loading screen → advance calculation → disk cache.
    ★★ B87 (2026-10-12, the owner: "based on the distance from the rung, not in which realm I'm. And it's true for ANY
    realm"): THE FINE GROUND REACHES EVERY VIEWER BY DISTANCE — one rule (each level to its rung's ask edge, bounded by
    the ground seen), any realm, any viewer (on foot, aboard a hull, in orbit); ONE CALL (`fine_needs_of_head`): every
    viewer outside the realm through the gateway's realm cache (its VIEW on the one tile path, `ArtifactWant { view }`;
    the shard names, solves, answers one level a record), the realm's own occupants by its shard at their poses (the
    gateway cannot see a pose inside a realm); coarse first; no new data class, no wire arm.
    ★★ B88 (2026-10-12, the owner: "one way only - through gateway!"; "exactly the same machinery, despite of which
    realm I currently belong"; AMENDS B87 item 3): ONE PATH — every tile (artifact and fine) comes from the gateway's
    realm cache by the one call; ONE EYE from the picture's own inputs (the eye in the player's realm carried by each
    drawn realm's row; in the player's own realm the avatar's position, read by the gateway off that realm's snapshot,
    once a beat; elsewhere the realm's centre, the SL7 proxy); no shard ships a tile to a session; the interest width
    and the gateway's relayed-tile record gone; no new data class, no wire arm.
    ★ B89 (2026-10-12, the owner: "do all 5 from the speed work … check performance, but not run full gates … judge
    visually in the window … I don't want to spend week on this"): THE SPEED STEP — the measurement, a frame cap + the
    render scale, smaller vertices, one draw for many pieces, ask only what can be seen (the planet's and the hills'
    horizon around the eye), fewer vertices on flat ground (our own, no new library); headless readings + the owner's
    look; no gates.
    ★ B90 (2026-10-12, the owner: "let's take the most optimized version please"): THE NEW DETAIL RULE STAYS (the fine
    ground's rungs by their curve's step; the level edges at 20 km/s are the LOADING LIMIT, cured by B86 item 4's
    send-ahead); THE RECONSTRUCTION AT 0.67 IS THE WINDOW'S DEFAULT (amends B24; 16.8 ms against 32 ms; `VD_TEMPORAL=0`
    for a run; the capture keeps its reference defaults); THE FREEZES OPEN (~2 a second 50–150 ms, outside the terrain).
    ★ B91 (2026-10-12, the freezes' cause MEASURED — two far planets' first solves restart at every login and never land,
    taking the cores; A/B 247 → 2 late frames): THE SOLVE THREADS BELOW THE TICK AND THE WINDOW (macOS "utility", Linux
    nice 10; `libc` approved for the shard program; the generator's helpers take the host's start step); A STARTED FIRST
    SOLVE LANDS — SL7 AMENDED (a realm stays alive while its first solve has not landed; it does not report itself empty;
    a failed solve ends the reason); WHEN a solve starts stays the advance calculation's (B86 item 6, must not be hurt).
    ★ B92 (2026-10-13, the owner on the pictures: "I think they all look amazing, so take the most optimal one"): THE
    128 m LEVEL'S DRAINAGE IS BUILT ONCE (B77's "network built first"; the rebuilt drainage kept capturing and every
    fine round ended at its pass limit) — one exact pass + the half step a round, the SETTLE LAW (5 m) ends the rounds
    within 180, the slopes in logarithms; a tile 6.5 → 3.2 s, the node means nearer their rows; FINE_VERSION 2; a far
    copy solves only when no whole tile waits.
    ★ B93 (2026-10-13, the owner saw walls and a spire on B92's ground: "Let's try (1) … if nothing works, can we
    adopt something from the list"): MEASURED built-once drainage = 1–1.7 % of links over 63°, peaks to 24 km; THE
    DRAINAGE BUILT AGAIN AFTER EVERY SECOND ROUND (0.02–0.05 %, peaks ≤ 5.6 km, settles 78–132 rounds, 4–6 s a tile;
    FINE_VERSION 3); every ground change judged FROM THE SIDE (the census + the window), never top-down or a mean alone;
    if it fails the look, ADOPT A PROVEN METHOD from the reviewed list (fastlem's own solver first), never more rounds.
    ★ B94 (2026-10-13, the owner: "Ok, please proceed with your plan"): THE FLAT LAND HAS TWO SCALES — a tile's rivers
    leave at the node MEAN, so no valley cuts below it (exits 100 / 300 m lower: Kansas- / Tennessee-like numbers);
    STEP 1 the fine level's BASE LEVEL by a PUBLISHED LAW (Earth measures and checks it, gives no constant; FINE_VERSION
    4); STEP 2 the 8 km world's valleys (B78's dry-world cure on the macro copy, generator 27, on the owner's word).
    ★★ B95 (2026-10-13, the owner on the day-side hills: "Actually it looked very good! Hills as hills." — then "the
    quality should be amazing on all levels" → "Yes, let's proceed please"): THE 128 m FORM ACCEPTED (its roughness a
    reading); THE 16 m LEVEL BUILT NOW (design step B3, the generic tile machinery, seamless, the same bytes) ON ALL LAND A
    PLAYER WALKS ON (amends §3.6's "only where the land rises"); then the 2–4 m level; the first look on foot at hull 17.
    ★ B96 (2026-10-14, the owner: "Can you adjust the formula and law, so we don't break what we already have today?" →
    "Go"): THE 16 m GROUND = TODAY'S DRAWN 128 m GROUND + THE LAW'S DETAIL FINER THAN 128 m (the law solved alone, no hold;
    amends B95 item 2); THE PIPELINE BUILT END TO END through the one tile machinery, then the first look on foot.
    ★ B97 (2026-10-14, the owner on the cure's pictures: "Yes, I accept the cure"): the 16 m level made needles and walls on
    steep ground (the dry world's slow rate + Oregon's creep sank the law's ground under today's; the pinned channels stood);
    THE 16 m LAW IS THE 128 m LAW (B80's, B93 item 3: the joint law's steady state, single-flow water, no pinned channels;
    amends B96 item 1) UNDER tan 35° (amends B82 item 1's 1.25 at 16 m: the creep's own limit); B96's form stays; the solve
    stops by convergence; only the 16 m tiles move. Benched against Earth's 1 m lidar at 16 m (San Gabriel, Oregon).
    THE LOOK GIVEN in the window at hull 19's valley: "It looks amazing!" (9 680 pieces, loading ~80 s).
    ★ B98 (2026-10-15, the owner on B5's bench: "Agree with your recommendation. But after colors are done, I'd like to
    test 4m with colors again"): THE 4 m LEVEL WAITS (it matched Earth on the mountain, overshot on the hills with a comb;
    its code stays bench-only: `fine::refine` — the 16 m level runs through it byte for byte —, `fine::l3`,
    `planet_stand l3`); THE GROUND'S COLOURS BY LAW NEXT (B35's biomes and materials on B75's first surface look; research,
    a design with reviews, the art's questions, B36 Q14); THEN THE 4 m LEVEL TESTED AGAIN WITH COLOURS.
    ★ THE LIVING STATUS FILE `docs/investigation/2026-09-24/terrain_believability_status.md`: READ IT FIRST for
    the arc's state; UPDATE IT IN THE SAME CHANGE as any work on its items; DELETE it when G-LOOK is given.
  - `owner_decisions_2026-09-21_water.md` — ★ W1 the roughness factor reads the SOLVE (built, GENERATOR_VERSION
    8), then 8d. ★ W2 (answers L20): a player's dam never moves the seed's river (no re-solve, the river an
    endless source); a dug hollow BELOW a connected water's level FILLS to that level as LIVE STATE — a bounded
    local flood in the owning shard, rows of the block store (slice 9, water = a substance with liquidity).
    ★ W3 amends T2: waves AND the CURRENT are required (8o; priced first); pressure and draining maybe, later.
    ★ W4: EVERY 8d ask APPROVED (the Earth-calibrated width law, small streams, one lake mechanism, the
    PROVINCE WORD on the row (SL6 yes), the override's row by the owner's tool only, the crease decided on
    8d's cliff pictures). 8d's order: beds' hardness → rock map + fixed-radius substance → rivers → the carve
    → lakes → the override's read path. ★ W5 step 1 LANDED (each bed's own hardness; GENERATOR_VERSION 9), the
    cap share HALF is ASSUMED — the owner's number owed. Step 2 LANDED (the province word, ARTIFACT_VERSION 5,
    GENERATOR_VERSION 10). ★ W6 THE SEA DECIDES THE SHORE (built, GENERATOR_VERSION 11): the fine octaves may
    not carry a column across its water — after the bench the surface is held on its GROUND's side by a QUARTER
    of the ground's height (a stated choice, the owner's word owed); a dry row reads the body's sea; MEASURED:
    the shore's step at a ring swap 234 m/848 m → 0 m. A lake's shore is owed the same law at step 5.
    ★ W7 (2026-09-22) THE LAKES TRIED: the flood every pass + the deposit (the cut laid in the first hollow
    downstream) KEPT, pits 137 k → 7 k; FOUND UNDER IT: the old solve eroded a tenth of the age by a stale-
    flood defect, the true erosion grades every continent to the sea and the re-solved sea DROWNS it (1.4 %
    land at any age; hard rock 1 %; a cratonic uplift 0.1 %). THE ASK (T9): the basins' depth from the oceanic
    crust's isostasy, the inventory against that volume, then uplift-over-erodibility for the plains. UNTIL
    RULED the tree's solve drowns the home planet; the pins are NOT re-recorded; the window runs generator 11.
    ★ W8 (2026-09-22) THE THREE STEPS BUILT (GENERATOR_VERSION 12, pins re-recorded): the FREEBOARD LAW
    (Wise 1974: the continental thickness solved so the sea stands at the crust's shelf quantile, Earth's 29 of
    40) + the running sea + the erodibility by the rocks' tensile strength (Sklar & Dietrich 2001) + the
    envelope a per-node clamp (the global scale was the drowning's other half). MEASURED: land 8.6 % → 22 %,
    sea 78 %, pits 170 k → 14 k; the remaining lakes are GLACIAL (the ice line over more than all the land:
    8e's climate). OWED: the ice line, a delta at the sea, the cratons' transient, the owner's look.
    ★ W9 (2026-09-22) THE ON-FOOT GROUND: two defects measured and cured — `dev-cluster down` reaped the realm
    stores (now `realms/<slot>` beside the slot, `vd_bins::realm_store_dir`, ONE place), and the gateway dropped
    a shard's bulk for a session still activating (now HELD on the session and delivered at activation). A
    headless ground probe MUST pass `--pilot`. THE FAR-RUNG SHORE measured: the coast steps ~a level node at
    each pyramid-level swap (11.5 km median at rung 9→10); the cure is a coast MASK (one bit per node, 1.1 MB,
    shipped with the head) giving the water's SIDE at every rung — an SL6 ask, owed the owner's word.
    ★★ W16 (2026-09-23) ONE SHAPE AT EVERY DISTANCE — THE BINDING LAW OF THIS FILE'S NEWEST RULING: at EVERY
    distance the water and the land are ONE shape from ONE field; a finer rung REFINES a coarser one and never
    CONTRADICTS it; a RUNG BOUNDARY IS INVISIBLE (no shape change, no water appearing or vanishing, no colour
    or brightness step). Built (GENERATOR_VERSION 19, ARTIFACT_VERSION 7, PROTO_MINOR unchanged at 34):
    (A) step 4's DRAWN RIVERS ARE RETIRED — they were the near flicker (190 wet columns flipped at one ring
    swap, the water's edge jumping 5.4 km); the width law and the meander belt stay for the SOLVE.
    (B) the coast mask carries a THREE-WAY SIDE (land / sea / LAKE, two bits a node, the count pyramid folding
    both) — before it NOT ONE column of 300 000 stood under a lake at any rung, because the shore law drained
    every lake; after it 279 989 do, at every rung. (C) the water SHEET is built on the GROUND'S OWN GRID: a
    flat block is a CHORD, and its dip (170 km at rung 18) buried the whole globe's sea from rung 17 up — the
    model read 78.5–79.7 % wet at every rung all along, so "no water" was a DRAWING fault.
    ★★ W18 (2026-09-23, the owner: *"Not fixed"*, *"It changes still when I'm leaving farer"*) THE WATER IS
    DRAWN ON THE GROUND'S OWN LATTICE: the sheet was a quad per cell on the corner COLUMNS while the ground
    is a surface-nets mesh whose vertices sit INSIDE the cells — two lattices, two chords, and MEASURED
    (`sheet_poke`) 8.7 % of the sea-floor vertices at rung 14 and 7.4 % at rung 18 stood ABOVE the drawn
    water (by up to 3.8 km): the speckle and the 400 km teeth on the far coast, changing with the rung. Now
    `position::water_sheet` takes the ground mesh and builds ONE water triangle per ground triangle on the
    ground vertices' own directions, so the two can cross only at the shore; W16's block is retired; the
    hide bound stays; the sheet is client-only, no pin moves. MEASURED after: the teeth and the wide speckle
    gone on the same stands; a shelf then DISSOLVED into dots — MEASURED (`mask_at_pixel`): every column
    under its water, the mask says sea, yet 84 % of the DRAWN vertices at rung 13 stand a metre over it,
    because the extractor's gap byte (cell/128, 64 m at rung 13) is coarser than the shore law's depth on a
    shelf; a depth tie-break and a buffer-step nudge were REFUTED; now a vertex's side is its COLUMNS' word
    and the water is held one extractor's step off the drawn ground on that side. AND THE WATER MORPHS WITH
    ITS GROUND: in a crossfade band the ground morphs toward the coarser mesh (at its 2–8 km step) and the
    water carried a morph of zero — a dot screen over every continent at 118 000 km, the planet's shimmer
    at 408 000 km; each water vertex now morphs to the same placing rule read from its ground's target.
    OWED: the owner's look, the frame cost, a water skirt. ★ PERFORMANCE (2026-09-23): the frame numbers were
    the 60 Hz PACE (headless loop and vsync alike), never the work — `VD_CLIENT_UNPACED=1` measures the work;
    the sheet's hide bound cut nothing at the far rungs (661 168 sheet triangles = the land's), so the sheet
    is now cut EXACTLY from each vertex's radius and morph target, and the sea floor no ray reaches under
    deep water is not drawn (the visible depth = the extractor's step today; 8o replaces it with the water's
    clarity so shallow water shows its floor); the gateway's held bulk is bounded in BYTES with the NEWEST
    refused (a bound made right, no loss measured); the twelve chunks that never settle on a still stand are
    two FAR PLANETS whose artifact has not arrived (a pop-in seam, owed: draw a far body from its coarsest
    level alone the moment it lands).
  - `owner_decisions_2026-09-16_terrain.md` — ★ T9 (2026-09-18) NO PHYSICAL NUMBER IS DRAWN — every physical
    fact (pressure, temperature, water, spin, crust, relief cap) is COMPUTED by a published law with a stated
    calibration body; a draw only for the seed's identity choices, an unrecoverable history (a tilt), or a
    NAMED scatter around a computed value; a band standing in for a missing law is a defect. T8 the drowning:
    the sea's mechanism built, 8a's pictures judged DRY until 8c. ★ THE TERRAIN ARC BEFORE SLICE 9 (T1): the WHOLE of 8 lands
    before the block store — M1b → 8a the spectrum → 8b the charter → 8s the sky (moved INTO 8, the light's
    exception) → 8c the solve → 8d water and rock (+ the authored override) → 8e biome and soil → 8f the third
    dimension → 8o THE OCEAN (T2: the sea surface as a shape, the bathymetry kept, tides from the almanac;
    buoyancy in 11, waves as look, real ships that sail) → 18's state model → then 9, 10, 11, then 14 the
    freeze. T3 TIME: closed form from the tick (the sun, the seasons, the tides, the almanac) / live state
    stepped by the shard and shipped as a WEATHER LIST / NEVER the land's shape (no erosion in the game; the
    authored override is the only shape change after the freeze). The reference (crimson desert) is ONE zone
    of ONE planet; zones come from charter → climate → biome. L25: 8a measures its densest chunk first.
  - `owner_decisions_2026-09-12_frozen_patches.md` — ★ THE FROZEN PATCH (F1): a built area keeps the
    octave COUNT it was built on (a small integer stored with the realm, shipped as a block-store row);
    the client and the server derive that patch with its count and the rest with the newest; a BLEND
    BAND at the edge. F2 the added-octave tolerance = HALF A METRE at the 1 m rung (ten band cells × a
    slope of 1/20), unless the owner states another. F3 the patch's extent, thaw and band shape are for
    the slice 9 discussion. F4 the GPU question ANSWERED (integers agree on the GPU; the 64-bit floats do
    not exist there; no speed gain measured; V18 stands). F5 still owed: the wall's defaults, R-18. ★ F6 THE AVERAGE MACHINE: the terrain workers get a SHARE of
    the cores, never all. ★ F7 THE RECIPE GOES INTEGER-ONLY AND THE GPU DOES THE CLIENT'S CHUNK WORK TO
    THE MAXIMUM (one source, two targets; the server computes the same integers on the CPU; the pictures
    re-judged); supersedes V18's "the recipe stays 64-bit". Order: the bench, the discussion, the build;
    slice 9 and the frozen patch follow the new recipe. ★ F9 (2026-09-13) THE BOUNDED ASK (the client asks
    for the finest ring only as far as its builders can deliver it before the ground reaches the screen;
    the next rung stands whole meanwhile; a hole and a pop stay defects) AND THE CARD AS A BUDGETED SECOND
    BUILDER (a slice of each frame, a knob; the self-check decides; the CPU share is the fallback); G2 (the
    mesh on the card) SKIPPED on measurement. ★ F8 THE FIVE DESIGN DECISIONS RULED (owner:
    "agreed, please implement"): 40 fraction bits in 64-bit words (bench part 3: within 0.02 mm of the
    float bend), the CPU path of the same source as the fallback, the runtime GPU self-check gate, the
    order (CPU recipe → pictures → G1–G4 → slice 9), rust-gpu one crate two compilations. The design:
    `docs/investigation/2026-09-08/landforms/slice_08_integer_recipe_design.md`.
  - `owner_decisions_2026-09-07_voxels.md` — ★ THE VOXEL FOUNDATION. SL10 THE SEED-SHAPED WORLD: the
    client MAY derive the world's STATIC SHAPE from the seed (a function of seed and address, never of
    time or state) with ONE generator crate compiled into both hosts (a port is forbidden), no-drift as a
    byte-for-byte MEASURED gate on every target, collision on the same shape, everything the seed does not
    decide as a one-hop DIFF from the owning realm. Supersedes the 2026-08-27 S6 "no client-side
    derivation" for the static shape ONLY; the star field stays shipped once. Plus the owner's nine
    requirements for the voxel foundation (smooth terrain on voxels, trees as one object, ~20 square
    shapes, sub-metre blocks inside the 1 m cell, no-volume attachments, params + client style, themes,
    engine-agnostic render seam, seamless). ★ V4 = THE OWNER'S REVIEW of the proposal
    (`docs/investigation/2026-09-07/00_proposed_voxel_foundation.md`, NOT binding): trees, grass and
    decoration are ART ASSETS blended dynamically, never drawn primitives (modular record + server
    skeleton kept); the removal kind is `Empty`, `Air` is the atmosphere; cross-realm edits POSTPONED;
    a hull or station MAY hold terrain; a future O'Neill-cylinder realm means gravity is a per-realm
    FUNCTION; common ore NOT public by default; the record, the pyramid bits and EVERY SL6 row are still
    OPEN, and no slice starts before its own detailed discussion. Bevy continues; the client stays
    replaceable (fully server-authoritative), and no other engine is named in any plan.
  - `owner_decisions_2026-09-05_suit.md` — ★ THE SUIT: a character WEARS its way to move in space; there
    are different suits; WITHOUT A SUIT a character cannot move in space; a suit is SIMILAR TO THE HULL —
    it states its rating (acceleration, cruise) as facts about what it IS, per entity, never a constant
    tied to the server. Deletes the wake horizon as the walk's time constant and the approach ceiling on
    the player's stick. The gates fly the shipped path (board a hull / wear a suit), never a walking dot.
  - `owner_decisions_2026-09-02_reach.md` — ★ REACH: one radius per realm, TESTED BY ITS PARENT. Makes the
    2026-09-01 ruling concrete. The galaxy is ALWAYS visible and NEVER re-anchored (the star field is the
    galaxy's own look, shipped once; the client places ONE cloud at the galaxy's composed placement; parallax
    is the camera). A parent POSITIONS its children and NEVER draws them (the per-child marker is deleted,
    LAST). A realm's reach is BIGGER than its bound and usually bigger than its parent's bound; it covers
    its whole subtree by definition (`reach = max(own look radius, max over children(distance + reach))`),
    and it comes from the realm's LOOK (size AND brightness). The decision is tree-independent; the
    consequence (every ancestor runs, to author placements) walks the tree. ★ NEW DATA APPROVED (SL6): a
    child states its REACH to its parent, on change. Windows ship children IN RANGE only. Binding order of
    work in R9; five measurements owed in R8.
  - `owner_decisions_2026-09-01_visibility_radius.md` — ★ VISIBILITY IS A RADIUS, NOT A TREE WALK. Every
    realm states one radius; inside it the realm boots and draws itself; a dormant realm is never drawn by
    anybody; a realm in somebody's area of interest is not killed; ONE machinery for every realm (a
    player-built ship is not a special case); the occupancy bit keeps liveness only and no longer decides
    what anybody can see. MEASURED cause: 96 000 drawings parked behind an empty vouch list.
  - `owner_decisions_2026-08-27_galaxy_shape.md` — ★ THE GALAXY'S SHAPE, ITS DENSITY, AND HOW GALAXIES
    DIFFER. Written before S12 starts. **EVERY number the placement reads is SEED-DERIVED** — a density
    knob is not a tuning parameter, it is a lever that moves every star a player has ever seen (owner:
    *"otherwise any tiny change might change positions"*). The shape is BELIEVABLE and consumes the
    taxonomy's own galaxy-kind draw; a shell, a ring, a plain filled ball and a lattice-AS-POSITION are
    each refused by name (a lattice stays lawful as a LOOKUP). Placement is stable forever: grow the
    world and nothing already placed moves. ★ GALAXIES DIFFER in what you SEE and how you TRAVEL —
    shape, density, size, stellar population — and NEVER in how much they PAY, because a seed-derived
    rich galaxy is a treasure map and the seed ruling forbids it. **An expensive gate buys ROOM, not
    treasure:** a new galaxy is worth reaching because nobody has taken it yet, which is live state.
    S12 builds ONE galaxy properly; the between-galaxy differences are designed now and built when a
    second galaxy is reachable.
  - `owner_decisions_2026-08-27_movement_answers.md` — ★ THE ANSWERS TO THE MOVEMENT QUESTIONS. Three
    items closed. (1) THE UNGOVERNED ROCK: the containing realm owns the speed of a thing that governs
    nothing itself, **but a transfer NEVER changes that speed** — a destination realm's ceiling limits
    what it may ADD, never what a thing arrives with (a re-clamp is a jump, and a jump is a seam).
    (2) WAKING UP IN TIME: **grow the interest radius with the closing speed; never cap the speed** —
    a cap makes the game unfair, because the player would pay for a server's start-up cost.
    ★ (4) THE CEILING LEAVES THE FLIGHT PATH COMPLETELY — no clamp, no refusal, no stop; a player never
    STATES a speed (a throttle makes a force), so nothing there can be refused. A fence may stand ONLY
    where CODE states a speed (a spawn, a fixture, a tool), where a bad number is a defect and a test
    goes red. CONTAINMENT BECOMES SWEPT: the test reads the LINE from last tick to this tick, so nothing
    must be slowed to be caught. The band never protected the player — it protected a SNAPSHOT, which is
    why deleting the governor broke the bands. Slowing down is GAMEPLAY: a ship's own safety block slows
    it (and a player may switch that off), and air slows a rock. MEASURED: the galaxy's ceiling is
    ~170 million times the speed of light, so it never binds in open space and binds only near SMALL
    bodies — exactly where it felt wrong. ⇒ THE BAND RE-SOLVE DOES NOT RUN YET: fix the test first, and
    the bands shrink on their own.
    (3) THE INTERIM ENGINE RATING: yes, a ship may state its rated cruise speed and acceleration as
    facts about what it IS. ★ Plus THE TEMPORARY CONTROL SEAM: attach the controls to the ship realm,
    make forces directly in the ship's shard (no signals, no functional blocks, no hull), and send them
    up the lane the movement contract already defines — so the whole path is testable NOW. Still open:
    the band re-solve, the HR4 second shard kind, the wake-up constant.
  - `owner_decisions_2026-08-27_seed_and_secrecy.md` — ★ WHAT A SEED MAY DECIDE. Anything a fixed seed
    alone determines must be SAFE TO PUBLISH; anything VALUABLE must depend on world state that CHANGES.
    Binds the block system BEFORE it starts: a block's substance may not be a pure function of
    (position, seed). Secrecy is not the cure — one world + many players makes any static map public by
    wiki, and a shipped generator inverts to its seed. ⚠ Does NOT authorise client-side derivation;
    that collides with the client-only-renders law and is OPEN.
  - `owner_decisions_2026-08-26_movement.md` — ★ THE MOVEMENT CONTRACT. Two lanes upward: **acceleration
    and torque ONLY, per tick, in the child's own frame** (what I am DOING), and **mass + cross-section +
    drag coefficient ON CHANGE ONLY** (what I AM). **Velocity NEVER crosses upward** — it is half a
    placement, and only the parent writes those. Downward: the authored placement, stamped, read-only,
    one hop. The width-over-three-minutes speed rule is RETIRED for band-width ÷ tick. Gravity never
    needed a child's mass (it cancels); drag and collisions do. Still open: the ungoverned rock, the
    wake-up radius. **WARP IS A DECLARED STATE** with a FIVE-TEST GATE and a CLOSED, reviewed set —
    a declared state says what a ship IS, never what it wants; it must survive an empty ship; and it must
    not move you by itself. An autopilot is software pressing the stick and gets NO special path.
  - `owner_decisions_2026-08-26_s9.md` — the S9 walk (hold/derive/send, identity naming, the star cap).
  - `owner_decisions_2026-08-24.md` — the SL1 rewrite, the SL2 clarification, SL9, the movement law
    (a parent never sets a speed), and the nine Universe/Galaxy/sky answers.
- `docs/design/DEFERRED.md` — the binding registry of every interim/stub: WHAT proper solution is missing,
  WHERE it lives, WHEN (which slice/phase) it lands. A phase isn't done until its entries flip to 🟩.
- `docs/audit/` — evidence for why the old architecture was unfixable (root causes R1–R10).

## Hard rules (user-mandated; violations are defects)

1. **HR1 sealed shards** — a shard's World/rapier/redb is private; inter-shard bytes exist
   only as `InterShardFlow` arms (one reviewed file in `vd-wire`); sim cross-feeds are
   `EffectFree` `CouplingPort`s; gameplay cross-shard data rides the Signal system.
2. **HR2 generic transfer** — ANY entity kind crosses shards via the `TransferableKind`
   registry; Durable vs Transient is policy fan-out on ONE machinery (batched `TransientGo`).
3. **HR3 one tooling** — ONE transfer FSM/envelope/Fence/registry; ONE `shard` binary;
   shard types are `ShardProfile` capability configs. Never `match` on a shard kind in features.
4. **HR4 features once, run anywhere** — capability DAG + the stateless `GridMapping` geometry
   seam (renamed from `FrameSpace` by the owner on 2026-09-07, ruling V6 A2; any physics anchor sits
   BELOW the seam); every feature ABOVE the seam passes the identical fixture on ≥2 shard kinds
   (G-IDENTICAL) or it doesn't land; the seam ITSELF passes G-MAPPING-TABLE (exhaustive at N = 62,
   the cube net) and G-MAPPING-ROUNDTRIP (at the largest legal N), because no fixture can give equal
   results on a bent grid and a flat one (ruling V6 A1).
5. **HR5 100% coverage** — Tier-A crates at 100% region+branch (`just coverage-fast`);
   exemptions only via `#[cfg_attr(coverage_nightly, coverage(off))]` or `coverage-exemptions.toml`.
   Generic-code gotcha (learned in SPIKE-0a/P0.3): llvm counts regions PER
   MONOMORPHIZATION — a branch inside a generic fn must be exercised for every
   instantiated type AND in every test binary that instantiates it. Discipline:
   (a) **generic fns are branchless shims** — serialize/lookup as straight-line
   expressions, ALL branching (`?`, `if`, `match`, error closures) in monomorphic
   helpers (see `core/src/tlv.rs` for the canonical shape; `field_decode_err` for
   hoisting closures out of generic bodies); (b) cover a crate's generics completely
   in its own unit tests; (c) integration tests exercise the full surface or none;
   (d) in tests prefer `assert_eq!`/`expect_err` equality over `assert!(matches!(…))`
   (the false arm is an uncoverable region) and split `assert!(a && b)` (short-circuit
   branches).
6. **HR6 agent-operable E2E** — client ships the `dev-control` harness (`vdctl`): input
   injection at the input-resource seam, wgpu readback screenshots/video, `runs/` manifests.

## Standing laws (owner-stated; SAME WEIGHT as HR1–HR6; breaking one is a defect)

Each of these was stated by the owner after a live defect. They are not preferences.

7. **SL1 — A REALM IS TOLD WHERE IT IS; IT NEVER DECIDES WHERE IT IS.** (Rewritten by the owner
   2026-08-24, deliberately reversing the 2026-08-05 ruling that a realm may never be told its own
   placement — see the REVERSAL note below. The earlier text read "ONLY THE PARENT KNOWS POSITIONS"
   and forbade the datum outright, on the reasoning that a field's PRESENCE is the leak.)
   1. A parent authors its direct children's placements in its own frame. **The parent is the ONLY
      writer, always.** Conversions happen in the parent: going down it subtracts before shipping;
      going up the child ships its own-frame pose and the parent adds.
   2. A parent MAY state to a child the placement it authored for that child. The child holds it as
      a **stamped, read-only reading** — an instrument, never a fact it owns.
   3. A child NEVER derives, adjusts, computes or states its own placement — not to its parent, not
      to its children, not on any lane. A placement a child asserts about itself is refused.
   4. **ONE HOP ONLY.** What you are told about yourself, you NEVER pass on; and you are never told
      your parent's placement. **Never fold a realm's absolute from the root** — the chain does not
      exist anywhere a realm can reach it, which is what makes the iron rule structural rather than
      remembered.
   5. The PLACEMENT, CONTAINMENT and CROSSING machinery may not NAME a realm's own position.
      Enforced by a crate/module dependency rule with an observed-failing control — the same shape
      that fences motion off the crossing path (SL4), never by care.
   6. A STALE reading is REFUSED, never used. Every statement carries its instant; past its derived
      bound a consumer degrades and says so. Absent data stops an autopilot; stale data flies it
      into something.

   **WHY THE REVERSAL (owner, 2026-08-24).** The old law forced TWO mechanisms for one job: a screen
   was sent "the contents plus your own eye", a realm was sent "a view composed for you". Same job,
   two paths, two cost models — a fork decided by what KIND of thing receives, which is what these
   rules exist to prevent. Worse, the composed-per-observer form costs contacts × observers, and the
   sky proved that product fatal (150,000 rows per player per tick). Telling a thing where it is
   turns the product into a SUM for everyone. The clause given up — "a field's presence is the leak"
   — was never about correctness; it was about people. Clauses 4 and 5 replace that taboo with a
   fence the compiler and a test enforce, which is strictly stronger.
8. **SL2 — NO OCCUPANT POSE CROSSES A REALM BOUNDARY.** Not up, not down, not for culling, warming,
   or rendering. The realm you are INSIDE warms what comes next, from occupants it already holds;
   travel is always out into the shared parent and in again, never sibling-to-sibling. Liveness
   needs one bit per level: a live child keeps its parent alive.

   **CLARIFIED 2026-08-24 (owner) — this law governs REALM-TO-REALM, and the connection plane is not
   a realm.** A shard handing its OWN occupants to the gateway was never a crossing: it is how any
   player has ever seen anything. So SEEING another realm's people is lawful and needs no ruling —
   each shard streams its own occupants to its own subscribed clients, and the GATEWAY composes one
   ready-to-draw picture per observer (it alone holds both sides, and it is not a realm). What stays
   forbidden is unchanged: no occupant pose may enter another REALM's simulation. Hence the line:
   **CONTACTS ARE REALMS, PEOPLE ARE SEEN** — a ship's systems track ships, stations and bodies
   (placements the parent authors and may state), never individual people. You SEE someone in a
   spacesuit because the composer draws them; your targeting computer does not hold their position.
   Acting on them is a projectile CROSSING a boundary, which is the transfer machinery, not a leak.
9. **SL3 — A REALM DRAWS ITSELF.** The parent authors WHERE a realm is; the realm itself authors
   HOW IT LOOKS (extent, surface, detail, eventually meshes at a detail level it chooses). A
   parent's per-child message carries a PLACEMENT and nothing else, and must shrink toward that,
   never grow. A realm that is not running cannot be drawn — which is WHY visibility is the
   spin-up trigger.
10. **SL4 — PHYSICS AND RE-HOME ARE SEPARATE MACHINERY, one-way.** Physics (orbits, gravity,
    thrust, drag — a per-realm capability) produces A PLACEMENT: where each child is, in my frame,
    at this tick. Re-home/containment/hand-off CONSUMES placements and may never ask HOW a thing
    moves. No orbit/gravity/thrust symbol on the crossing path — a ship, a station, a moon and a
    rock cross by identical code because that code cannot tell them apart. A "does this child have
    orbital elements?" test inside a placement lookup is STILL a specific and is forbidden. A
    static flag may decide WHETHER TO RECOMPUTE, never WHAT ANYONE READS, and is derived from
    whether a child actually has motion — never declared per shard kind. Enforce structurally (a
    module/crate dependency rule), not by care.
11. **SL5 — ONE WORLD.** One universe, generated from the seed, used by the game AND every test
    including e2e. No scale knob, no preset, no reduced or test-only variant, no second generator.
    Numbers change on THE world; a variant is never created. (True astronomical scale + generated
    stations/areas + the galaxy cell lattice are owed changes to this one world.)
12. **SL7 — AN OCCUPIED REALM IS ITS OWN OCCUPANTS' PROXY, AT ITS PARENT'S SCALE.** How area-of-
    interest works at any nesting depth without a pose ever crossing, and without anything central:
    - LIVENESS, decided by the realm itself, looking only at itself and ONE level down: *do I hold
      occupants? do I have a live direct child? do I hold a first solve that has not landed (B91,
      2026-10-12)?* Any ⇒ stay active. None ⇒ shut down after the cooldown. A realm never reaches
      upward and has zero control over its parent.
    - AREA OF INTEREST, decided by the PARENT, never by the realm about itself: for each of MY
      direct children, is it within the interest of an occupant I hold, or of an OCCUPIED CHILD
      treated at the placement I authored for it? Yes ⇒ demand it stays alive.
    - So an occupied child stands in for whoever is inside it. The error is bounded by the child's
      own size, which is exactly the resolution at which the parent's decision is meaningful —
      a child is small relative to its parent, which is what nesting means.
    - The parent also authors its children's VELOCITY, so warming-ahead needs nothing told to it.
    - What crosses: ONE BIT of occupancy, upward, which already crosses today (a sealed parent
      cannot see inside a child, so the child reports its own emptiness). Nothing else. No occupant
      pose, no entity set, no central assembly of a global picture, no depth or hop count anywhere.
13. **SL6 — ASK BEFORE NEW DATA CROSSES A REALM BOUNDARY, and before adding a wire arm.** Default
    NO. Find the local formulation first; there usually is one. State what data, from which realm
    to which, why the receiver cannot compute it from what it legitimately holds, and what doing
    without costs.
14. **SL9 — A PARENT'S CHILD COUNT IS UNBOUNDED.** Owner-stated 2026-08-24. A realm may hold ANY
    number of direct children: six moons, six hundred ships and stations, or a galaxy's hundred and
    fifty thousand star systems. There is no cap, no reserved width and no "bounded child set"
    anywhere. EVERY mechanism that touches children must therefore be written for an unbounded
    count: never a fixed-width bitset over children, never a per-tick walk of all of them, never a
    per-child row on a per-tick lane. Membership is a SHORT LIST of the realms an occupant is
    inside (its own chain, plus the edge of a crossing) — never one bit per watched realm. Finding
    which child holds a point is a LOOKUP, never a scan; sibling realms may not overlap, so at most
    one child can hold a point and the answer is O(1)-ish, not O(children). A cost that grows with
    the number of children is a defect, and it must be measured on a realm with many, not argued.
    (This law does NOT relax SL7: the parent still decides area of interest for its direct children
    — it must simply do so without touching all of them.)

15. **SL10 — THE SEED-SHAPED WORLD: ONE GENERATOR, TWO HOSTS, NO DRIFT.** Owner-stated 2026-09-07
    (`owner_decisions_2026-09-07_voxels.md` V1). The client MAY derive the world's STATIC SHAPE from
    the seed: what `(seed, address)` alone decides, never a function of time or live state. ONE Rust
    generator crate is compiled into the server and into every client; a port is FORBIDDEN. No drift is
    a MEASUREMENT: a gate compares server-built and client-built chunks byte for byte on every shipped
    target. The server computes collision on that same shape. Everything the seed does not decide
    (edits, placed blocks, attachments, growth, damage, live-state deposits) crosses as a one-hop DIFF
    from the owning realm. This does NOT relax "the client only renders": no pose, no velocity, no
    entity state is ever derived on the client. ★ WIDENED 2026-09-09 (ruling V13 L12): the static shape
    is a function of the seed, the address AND THE BODY'S CHARTER — about twenty quantised integers
    (gravity, spin, tilt, water, air, age, the orbit-derived insolation) authored ONCE, stored, and
    stated by the realm in its own surface statement. Both hosts read the same integers; no float fact
    ever enters the recipe.

    *Note: this file has no SL8. The seamless law (SL8) was written in the `new-system` worktree and
    is not committed there yet; that number is reserved for it so the two trees do not disagree.*

## Workspace

```
crates/core              vd-core   pure domain (ids, Fence, kind registry, TLV, celestial math, bands)
crates/wire              vd-wire   frozen wire contract + InterShardFlow + seam contracts
crates/sim               vd-sim    pure ECS/FSMs + sim::io ShardIo traits + io::mem test impls
crates/node              vd-node   build_app(NodeKind, cfg, io); step_tick(); universe clock
crates/connection-plane  vd-connection-plane  gateway internals (lib; gateway bin is a shell)
crates/harness           vd-harness  Topology, FaultFabric, ControlOracle, WireMonitor, ChaosRunner
tests                    vd-tests  accumulated scenario suites — never delete a scenario
```
Dependency rule: bins → node → sim → wire → core → seed; harness → node + sim::io::mem; nothing
depends on a bin. Every node is lib + 4-line bin. `io-prod` / `tests-process` appear later.
The voxel foundation (slice 5, ruling V9; ruling F7 2026-09-12): `crates/recipe` (vd-recipe) is THE
INTEGER RECIPE — the fenced integer word `Gi`, the two-word product, the root and the reciprocal, the
bend at 40 fraction bits, the noise at 28, the octave sum — ONE SOURCE for the server's CPU, the
client's CPU and (through rust-gpu, owed) the client's GPU, depending on nothing; `crates/seed`
(vd-seed) is the LEAF two hosts must compute identically — the hash (stepped through the recipe),
the digest, the face bend (the integer direction beside the float inverse), the ladder — and
`crates/terrain` (vd-terrain) is THE ONE GENERATOR (SL10), which depends on the recipe, the leaf and
nothing else; its recipe path names no float (the body's once-per-body draw and six seam doors are
`Gf`'s last callers). Today `bins`,
`connection-plane` and `vd-tests` depend on terrain; `sim` and `physics` carry no terrain edge yet (the
block store, slice 9, and the collider slice add theirs); the client's edge is slice 7's. Both new
crates are Tier-A and carry the float fence (`clippy.toml`, `Gf`, the link scan).

## Non-negotiable conventions

- **No I/O outside the seam**: sim/node code gets time/rng/persistence/transport ONLY through
  `sim::io` traits (clippy `disallowed-methods`/`disallowed-types` enforce; `HashMap` with the
  default hasher is banned in sim/node — use `BTreeMap` or `DetHashMap`).
- **No magic numbers**: world/sim params seed-derived; entity props per-entity; operational
  params in ONE config struct (`TransportTuning`/`TransferTuning`), never inline literals.
- **NO client-side prediction** (interpolation on a 100–150 ms buffer). Players physically collide.
- **Fence discipline**: every authoritative action carries its `Fence`; receivers reject stale;
  the directory CAS is the only commit point. Source authority retained until dest Committed.
- **postcard v1 everywhere** (codec flag bit reserved); entity blobs are TLV-framed;
  decode-to-Default is BANNED for Durable kinds.
- **Determinism**: closed-form `f(seed, universe_tick)` for celestial math (Category A);
  rapier state is checkpoint-carried, never re-simulated cross-host (Category C); every
  physics→control boundary quantized to integer grids.
- **No `git commit` / `git push` without an explicit user request for that exact action.**

## Build & gates

```bash
cargo test --workspace      # full deterministic suite (fast: virtual clock, no sockets)
just gate                   # fmt + clippy(-D warnings) + tests + coverage — the pre-merge gate
just coverage-fast          # Tier-A 100% region+branch (HR5 inner loop)
just coverage-html          # see the uncovered region
```
Coverage runs on a pinned nightly (`VD_COVERAGE_TOOLCHAIN`); product builds on stable 1.94.1.

A NEW WORKTREE STARTS WITH THIS CHECKOUT'S SOLVED WORLD (2026-10-14): `scripts/worktree-new.sh <name>` makes it under
`.claude/worktrees/` and clones (copy-on-write, no disk at first) this checkout's realm stores into its slot and the build
folder into it — no first solve of the home planet or the far planets; `scripts/worktree-remove.sh <name>` removes it and
its slot's stores. A store keeps every old fine version's tiles: `examples/store_prune.rs` drops them (cluster down).

## Roadmap position

P0 (foundation: harness + frozen wire + FSMs + clock) → P1 stub shards → P1.5 client+HR6
harness → P2 THE transfer (one class, ghosts, CUT_MARKER, fence CAS) → P3 all classes +
kill-9 crash/chaos matrix → only THEN voxels (P4 terrain, P5 physics, P6 blocks, P7
checkpoints, P8 ships, P9 signals, P10 warp, P11 combat). Transfers are proven robust
before features pile on — the structural inversion of the old project's failure.
