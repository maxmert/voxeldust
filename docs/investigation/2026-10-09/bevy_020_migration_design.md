# The move to Bevy 0.20 (ruling B70; ruling B71): the design — DRAFT, revision 3, for the owner's questions

> **What this is.** ONE design for the move of every crate that names Bevy from Bevy 0.18.1 to Bevy 0.20.0: the ECS of
> every shard (`bevy_ecs` in vd-sim, vd-node, vd-connection-plane, vd-harness), the window and the capture
> (`vd-client-render`: `bevy`, `bevy_egui`, a direct `wgpu`), the vendored `bevy_pbr` with its sky changes, and every crate
> that the move pulls along (wgpu, naga, glam, the toolchain). Ruling B70 (2026-10-09) orders it after generator 26 closes
> by B69's fixed list and on the owner's word; it lifts the upgrade gate of 2026-08-27; it asks for every benefit with its
> price, and for nothing broken. Ruling B71 (2026-10-09) sets how the frozen pictures follow the move.
>
> **How it was made.** Three readers (reader 1: the step 0.18 → 0.19, its release notes and its 103 migration entries;
> reader 2: the step 0.19 → 0.20, its 75 entries, and the benefits of both; reader 3: our own Bevy surface) → the
> synthesis (revision 1) → three adversarial reviews (A, the rules and the rulings: 3 blocking, 13 important, 11 minor;
> B, the code and Bevy facts, read in the published crates that it downloaded: 2 blocking, 5 important, 16 minor; C, the
> plan and its estimate, counted bottom up: 3 blocking, 11 important, 15 minor) → revision 2 → a verifier's check of the
> eight blocking fixes (six HOLD; A-3 and C-B2 PARTLY; four new items N1–N4) → this revision. §11 gives every finding with
> its answer (V-1 to V-15 for the verifier's). The notes and the reviews stay in the session's scratchpad.
>
> **What changed from revision 1, in short.** One fixed closing list ends the move (§4.15), and §0 says what it
> guarantees. A spike phase S0 (2–4 machine hours, ESTIMATE) settles the two possibly fatal unknowns before any other work. The path
> offers four roads, with a bottom-up estimate, machine hours and a calendar in weeks. A refused GPU module now gives the
> CPU builder, and a GPU error now ends in a failure code, as the code must make them. No order tool reaches a shipped or
> gated binary. Bevy's two sky changes are your choice before UP5. The compiler's move runs the cross-target legs (SL10).
> glam takes an exact patch. "Better" has a definition. The window gets a checklist flight. Many facts are corrected: the
> Metal limit, glam's version, the capture's cure, the exit code, the lock.
>
> **What revision 3 changes (the verifier's items).** Every gate of §4.17 now names the §4.15 item that governs it; the
> stall, pipeline and tick-time gates compare with the before record in the frame rule's form, never as absolute gates; a
> red on the list always ends in a cure or in your ruling (§4.15, "the end of a red"); UQ7 (b) has its price in days; the
> cut module of the REFUSED control keeps the SPIR-V's magic word; S0 also checks the real module's words against the CPU,
> with a prepared question for a FAILED verdict (UQ17); every time count and every MEASURED mark has its mark and its date.
>
> **Date.** 2026-10-09. **The tree.** Branch `game` at `7b756dd` with the uncommitted generator-26 changes. Read-only. This
> file is the only file that the work wrote. No build, test, check or instrument ran.
>
> **The marks** (§10): `path:line` (read in today's tree); MEASURED (a tool printed it; the record and its date are named);
> DOC (a ruling, a design, a code comment, or a published source that a reader, a reviewer or this synthesis read at first
> hand on 2026-10-09); ARITH (arithmetic on marked inputs); ESTIMATE (a judgement); UNMEASURED (nobody printed it; §9).
>
> **The labels.** S0.1–S0.5: the spikes. UP0–UP11: the phases (§4). UB1–UB30: the benefits (§5). R1–R56: the risks (§6;
> R1–R20 keep reader 3's numbers). UQ1–UQ17: the owner's questions (§8). G-…: the gates (§4.17). A-n, B-…, C-…: the
> reviews' findings; V-1 to V-15: the verifier's items (§11).
>
> **The words.** THE MOVE: Bevy 0.18.1 → 0.20.0 in the whole workspace. THE BEFORE: the record of UP0. THE STOP: the green
> state at Bevy 0.19.1 (UP6). A QUIET CHANGE: an engine change that moves a pixel, an order or a behaviour with no compile
> error. A PAIR: a stand's new picture beside its reference (B71). A CAMPAIGN: one run of the full gate set. THE WINDOW:
> the game client's window on the owner's Mac. THE CAPTURE: the headless client that takes the gates' pictures. THE SKY
> PATCH: `vendor/bevy_pbr` and its changes (`vendor/bevy_pbr/PATCH.md`). THE STORED WORLDS: the solved artifacts that the
> realm stores already hold.

---

## 0. THE RECOMMENDATION IN ONE PAGE

**What the move does.** Bevy is the engine under two parts of the game. Each shard keeps its state in Bevy's ECS, and the
window draws the world with Bevy. The move changes how the window draws and how the shards store their state. It does not
change what the game is. Nothing that a pilot sees or does may become worse.

**What is guaranteed, and what is not.**
- **Guaranteed.** The move ends when ONE fixed list is green (§4.15). The list does not grow. Only a defect that a gate on
  the list finds, or your look, reopens the move. Every gate names the list item that governs it (§4.17), and every red
  on the list ends: one working day to find its cause, then a cure, or your ruling on a road (§4.15). A difference with no
  known cause that is not worse gets the same day; then you see it, and we register it. The first phase (S0, 2–4 machine
  hours, ESTIMATE) tells you whether the game keeps its GPU chunk builder and its GPU clock on the new versions, before any
  other work starts.
- **Not guaranteed.** The number of days: it is an estimate, and the gates' machine time depends on the first-solve
  timing that B69's list measures. A green S0. That Bevy 0.20's first weeks bring no defect on our path. That every slower
  frame or worse picture has a cure inside the move: then you rule.

**The twelve recommendations.**
1. **S0 first** (§4.2). Short tests in a scratch area show whether the new GPU libraries accept our GPU recipe and whether
   our GPU clock still reads. If one is red, you choose the road before any phase (UQ3).
2. **Record "before"** (§2): the full pre-merge gate, the GPU gates, the pins, the cross-target legs, the pictures and a
   clean frame reading on Bevy 0.18, after generator 26 closes and is committed on your word.
3. **Two steps, with a green stop at 0.19.1** (UQ1). The other roads: directly to 0.20; the shards straight to 0.20; or
   0.19.1 now, the terrain's R2 bake next, and 0.20 later.
4. **The compiler, glam and the written-down defaults in ONE campaign on Bevy 0.18** (UP1–UP3), with a rule that finds
   which change moved a result.
5. **Every order that we depend on, written down** (UQ4). Example: today Bevy, not our code, decides that a shard reads
   the universe clock after it writes its realm rows. A new Bevy can change that with no error. A test now holds it.
6. **The gates reuse the stored worlds** (UQ13). Today each picture gate solves the planets again, about an hour a run
   (the home planet's first solve took 3 514 s in its shard, MEASURED 2026-10-07). Reuse saves about 75–140 machine hours
   (ESTIMATE).
7. **Bevy now offers two of our sky changes. You choose before UP5** (UQ7), with the pairs and a window look as the proof.
8. **The pictures follow B71.** A picture is BETTER when it is better in at least one reading and worse in none, with no
   new defect. A better picture becomes the reference, and you see the pairs. A worse picture is a defect. Anything mixed
   goes to your look.
9. **A refused GPU module or a GPU error never passes in silence** (UQ9). A refused module gives the CPU builder, and the
   client runs on. A GPU error ends the run with a failure code.
10. **The window gets a checklist flight** at the stop and at the close: keys, mouse look, cursor grab, full screen, app
    switch, the HUD. Each window opens only on your word at that moment.
11. **Every new feature is your choice** (§5, UQ14).
12. **The timing** (UQ2): start after generator 26 closes and on your word, with no wait for 0.20.1. The 0.20 step takes a
    stable HUD library (`bevy_egui`) if one exists then; else it waits at the stop, or takes the test release on your word.

**The order.** S0 → the before record → compiler + glam + defaults (one campaign) → shards on 0.19.1 → window on 0.19.1 →
THE STOP → shards on 0.20 → window on 0.20 → pictures and frame → the features that you chose → close.

**The price** (ESTIMATE; review C's bottom-up count, §4.1). Road (b) with UQ7 (a): 31–45 working days with the levers of
UQ13 and UQ16 (34–50 without them); machine time 90–160 hours (165–300 without). The project works every day, so this is
about 4.5–6.5 calendar weeks (5–7 without the levers), plus 2–4 days of your time: five looks and four window sessions.
UQ7 (b) adds about 2–4 working days (33–49 with the levers). R2's bake starts after the move (B70 item 1).

**Example.** A pilot boards hull 10 at the berth on the home planet's plains, flies to orbit, and looks back at the dusk
line. On Bevy 0.20 the planet's shard runs its systems in the same order as on 0.18, so every realm row carries the same
clock value. The window draws the same ground, the same haze over the dry sea floor and the same sun at the horizon. The
dusk sky's new table is smoother and worse nowhere, so that picture becomes the new reference, and you see it beside the
old one. Five runs of each build, in turn, in one session, read the frame's cost: the same, or less.

**The questions to answer first** (§8 has all of them): UQ1 the path; UQ2 the timing; UQ4 the shard's clock order; UQ13
the stored worlds. UQ3 and UQ17 are prepared now and asked only on a red S0 or a FAILED self-check. The others come at
their phases.

---

## 1. The binding rulings and rules, and how this design treats each one

The newest ruling wins over every design (CLAUDE.md); the newest ruling file is
`docs/design/owner_decisions_2026-09-22_believability.md` (B1–B71). STATUS: KEPT (as written), AMENDS (the design changes
it, so the owner must rule: the question is named), or ASKS (the design offers roads; the recommended road keeps the
ruling as stated).

| Ruling or rule | What it requires | How the design treats it | Status |
|---|---|---|---|
| HR1 sealed shards | State private; bytes only as reviewed arms | `bevy_ecs` stays inside each shard; no arm, no new data; glam's wire bytes held by the new G-WIRE-BYTES (UP0) | KEPT |
| HR2 generic transfer | One transfer machinery | Untouched; the transfer suites are in the before record | KEPT |
| HR3 one tooling | One shard binary; never a kind `match` | One schedule constructor for every profile and every test rig (UP3) | KEPT |
| HR4 features once | G-IDENTICAL; the seam's G-MAPPING gates | In the suite at every campaign; nothing above the seam changes | KEPT |
| HR5 100 % | Tier-A at 100 %; exemptions only by the two named mechanisms | `coverage-fast` at the end of every phase that touches a Tier-A crate and before every commit; the io-prod floors in the full gate (§4.17); the 0.19 resource derive measured first (R3) | KEPT; an exemption road ASKS — UQ15 |
| HR6 agent-operable | vdctl, readback, manifests | The readback becomes systems; every vdctl render command proven; G-CAPTURE-SAME-FRAME tests the pixel copy's own cure (§4.13) | KEPT |
| The workspace rule | bins → node → sim → wire → core → seed; nothing depends up | The one schedule constructor lives in vd-sim, the lowest crate that holds `bevy_ecs`; no new edge | KEPT |
| SL1–SL4, SL7, SL9 | Placements, containment, crossings, liveness, unbounded children | No placement, containment or crossing code changes; the SL1 fences (`tests/tests/crate_isolation.rs`) re-run | KEPT |
| SL3 a realm draws itself | The parent authors where; the realm authors its look | The sky's centre comes from the realm's statement as today (`sky.rs:211-253`) | KEPT |
| SL5 one world | No variant | No test-only world; the order record builds the production schedules | KEPT |
| SL6 | Ask before new data crosses | No new data crosses a realm boundary; no new arm | KEPT |
| SL10 no drift | A gate compares the chunks byte for byte on every shipped target | The integer world links no Bevy and no glam (`crate_isolation.rs:264-`); `just terrain-legs` (aarch64 Linux, x86-64 emulated) in the record and at UP1 after the compiler moves; F8's self-check and gpu-drift on wgpu 29 and 30. The emulated leg can find a drift but cannot prove its absence (DEFERRED D-TERRAIN-1) | KEPT |
| Determinism; the frozen wire; postcard v1 | Closed-form celestial math; no silent byte change | glam alone and at an exact patch (UP2); the ephemeris pin bit for bit; G-WIRE-BYTES byte for byte | KEPT |
| No I/O outside the seam | Time only through `sim::io` | Bevy's delayed commands read seconds: not used on the shards | KEPT |
| F7, F8 | The integer recipe on the GPU; the runtime self-check; the CPU path as the fallback | S0.1 first; the self-check behind an error scope gives a REFUSED verdict and the CPU path (UP3; review B's B1) | KEPT; a red S0.1 ASKS — UQ3; a FAILED verdict ASKS — UQ17 |
| V4 | Bevy continues; the client stays replaceable | Bevy continues at 0.20 | KEPT |
| SL8 (reserved), B16, W16 | A seam is a defect; no flicker between rungs | The ground shader's port gated by terrain-moving-eye, gpu-seam and the eye's rule on every flight leg | KEPT |
| B22 | 60 frames a second at full screen | The frame rule of §4.15: the same or better, or your ruling | KEPT |
| B24 the temporal path | Our reconstruction; MSAA the default until D5/D14; MetalFX only on D12 | Neither 0.19 nor 0.20 has an FSR or MetalFX path (DOC); MSAA 4 stated (UP3); Bevy's sky motion turned off until benched | KEPT; the sky motion ASKS — UQ8 |
| B24 item 3, B30 item 2, memory | A window opens only on the owner's word at that moment | Every window session and look in this design opens only on your word at that moment; an earlier answer approves the plan, not the opening | KEPT |
| B25, B26 the air | The air rests on the sea; no growth under it; the haze scatters 0.9 | Change 6 re-based (§4.9); the haze stays in the game's own medium (Bevy 0.20's Earth preset still swaps the terms, DOC) | KEPT |
| B31, B32 | The sky light on foot, judged in the window | Retiring change 8 is your choice (UQ7), with a window look at the ground stand | ASKS — UQ7 |
| B27 the eye's rule | The display picture under the engine's dither | The dither copy checked by a canary (§4.12); the formula is the same in 0.18.1, 0.19.1 and 0.20.0 (DOC, review B) | KEPT |
| B29 the rectification | FSR 2.2's clamp, 16-bit history | Ported unchanged; motion-truth and the temporal gates | KEPT |
| B28, B29, B32, B53 + B71 the frozen pictures | A better picture with no new defect re-freezes, the owner sees the pairs; a worse picture is a defect; a first reference needs the owner's acceptance (B28) | BETTER defined (§4.14); the generator-26 re-freeze in B69's close; UP8 against UP6's run folder; the temporal path's first references by your acceptance (UQ6) | KEPT (as amended by B71) |
| B34 the frame readings | Red and not clean; T4 starts with a clean reading (AirPlay off, a lighter meter, the cluster's load known) | B70 item 4 moves a clean reading into the move; the protocol names B34's three causes (§2) | AMENDED by B70 item 4 (as ruled) |
| B35 as amended by B70 | The move before R2's bake | After generator 26 closes, on the owner's word | KEPT |
| B36 Q10 | The target machine is the owner's M4 Pro | Every frame and GPU reading there | KEPT |
| The commit rule (CLAUDE.md; the user's global rule; B53) | No commit or push without an explicit request for that action | A commit only on your request; the commit units in §4.1 | KEPT |
| B69 | Generator 26 closes by a fixed list | The record stands on the closed world; its commit is your request, a precondition; its re-freeze by B71 happens in its close | KEPT |
| B70 items 1–5 | The order, the target 0.20, every benefit, nothing broken, the design now | §2, §4, §5; a rest at 0.19.1 or road (c') changes item 1 | KEPT; a rest or road (c') AMENDS item 1 — UQ1, UQ2 |
| B70 item 3, the library rule | Nothing adopted that the owner did not choose | Each opt-in row of §5 asked; the two sky changes asked before UP5 (UQ7); the crates Bevy brings listed (UB30) | KEPT; ASKS — UQ7, UQ14 |
| The 2026-08-27 upgrade gate | No upgrade before a ship is flown in a window | Lifted by B70 item 1; its star-sprite condition checked: the new sprite features are 2-D only | KEPT (as lifted) |
| One job at a time (memory) | No reading while anything builds | One job on the machine; the next phase's code prepared in a second worktree with no cargo during a campaign; Docker stopped for timing | KEPT |
| No test seams in the game (2026-10-01) | Controls live in the harness | No order tool reaches a shipped or gated binary; the conflict report runs in tests on built schedules; each test scene is an in-process test `App`; no new `VD_` switch | KEPT |
| Test exactly production (memory) | The shipped path | The binaries' schedule assembly lifted into library functions that the binaries and the tests both call (UP3) | KEPT |
| Never revert WIP (memory) | No checkout over uncommitted work | Unfinished phase work goes to its own branch on your word, never out by a checkout | KEPT |
| Read the code (memory) | The source is the truth | Every engine name is read from the 0.19.1 and 0.20.0 sources at its phase | KEPT |

---

## 2. THE BASELINE FIRST

**Why.** Reader 3's R20: the tree is not green today. The suite has seven known reds (MEASURED 2026-10-04 on `7b756dd`,
`docs/investigation/2026-09-24/terrain_believability_status.md:1606-1607`), and no document names them (review A, GREP).
The frozen pictures are generator 22's while the tree's world is generator 26 (B71 item 3). The last frame reading is red
and not clean (MEASURED 2026-10-02, B34). So "nothing broken" (B70 item 4) can only mean "the same as a record taken on
the same world with Bevy 0.18".

**When and where.** After S0, after generator 26 closes by B69's five items, and after you ask for its commit (B69 does not
say "committed"; the commit is your request). The move works in its own worktree with its own target directory (R37).
UP0 starts only on a green `coverage-fast` (B69 item 4). If generator 26 closes with the 52 older misses open (B29 item 3),
the record holds the exact miss list, and the move adds none.

**The generator-26 references come first, in B69's close (B71 item 3).** Each stand's generator-26 picture stands beside
its generator-22 reference. BETTER (§4.14) with no new defect: re-frozen, and you see the pair. WORSE: a generator-26
defect, and the move waits. Neither: your look. UP0 starts only when every stand holds a valid reference.

**What the record holds.**
1. **`just gate` in full** (`justfile:494`): `fmt-check`, `lint`, `lint-combos`, terrain-pin, terrain-link-scan,
   terrain-fence-control, the suite, the process and GPU recipes that it names, and `coverage` — which is `coverage-fast`,
   `coverage-io-prod` and `coverage-io-prod-hooks` (`justfile:69`).
2. **The seven known reds, named one by one**, each with its exact failure: the assertion, the values and the message
   (UQ5 rules what happens to them). Today their names are UNMEASURED.
3. **The gates outside `just gate`**: terrain-pictures, terrain-pictures-card, picture-census (+control), motion-truth
   (+control), air-under, gpu-seam, gpu-drift, terrain-moving-eye, boarding-storm, flights; `crates/bins/tests/star_sky_pixels.rs`
   (no recipe today); charter-pin, artifact-pin, the ephemeris pin; `just terrain-legs` on both platforms (Docker up, its
   own job, then Docker stopped); `just image-smoke` in the same Docker window. The k3d path is outside the record: its
   proof on today's binaries is already owed (memory), and DEFERRED registers it.
4. **The process gates as each recipe is written.** Nine run in release; orch-crash, chain-latency, rlm-proc-spawn and
   rlm-kill9 run as debug builds (`justfile:123-126`, `:190`, `:438`, `:447`; DOC, review B). Chain-latency's tick numbers
   are R1's reading.
5. **G-WIRE-BYTES, new.** Golden postcard bytes of a `StampedPose` (position, velocity, attitude) and of each glam type on
   the wire, built on glam 0.30, with a control: one changed field goes red. Today no test holds these bytes: the cited
   tests check widths and arm indices only (review A-13).
6. **The pictures.** The 9 stands × (picture + probe) and the 5 overlay rectangles. The picture gate runs TWICE on the same
   build with `VD_PICTURE_IDENTICAL`, to prove that one build gives the same pixels twice. If it does not, the strict phases
   use "within one level" instead, and the record says so.
7. **The temporal path's pictures** in a run folder (`VD_PICTURE_EXACT_DIR`; no `exact_temporal/` exists,
   `terrain_pictures.rs:1145`); frozen only after your acceptance (UQ6).
8. **The pipeline census and the flag table.** Every pipeline and its variant key that each stand creates (the pipeline
   meter, `pipeline_meter.rs:60`), and from it a table: each shader flag → the gate whose picture or control draws it
   (§4.11). A flag that no gate draws gets a control, or the table marks it "compile only" for your knowledge.
9. **The frame** by the clean protocol below; the release binaries of this build kept for UP9's runs in turn.
10. **The orders and the build facts**: the order record and the conflict lists (UP3 writes them, still on 0.18); the
    server build's dependency tree (no full `bevy`, no `wgpu`, no `winit`; the images rely on it); the 12k-tick gate's
    time in a debug build (`Cargo.toml:109-123`); the lock's resolved versions.

**A red with no explanation** at UP0: we register it with what we know (B69's rule). It does not stop the move unless it
breaks a hard rule or a ruling.

**The clean frame protocol** (B34 items 3–4; memory: the window frame steps; a busy machine lies; one job at a time).
- B34's three causes, removed or known: AirPlay off; a lighter meter (the client's own frame record read after the run,
  not the poller's five state reads a second); the cluster's load known (its CPU share measured during the reading, the
  same cluster for both builds).
- `uptime` load under 1.5 before each reading and before each process gate. Docker stopped; nothing builds; no coverage
  run; you away from the screen. The window opens only on your word at that moment.
- Full screen 3 456 × 2 168 on the M4 Pro (MEASURED 2026-10-02, B34). The window's frame time moves in whole 8.33 ms steps
  of the 120 Hz display (MEASURED 2026-09-28), so a cost comes from the GPU clock, never from the window's frame time.
- The clock's freshness first: its health line and the distinct count (memory, 2026-09-29).
- **One shared instrument** that does not change with the move: the unpaced headless frame interval
  (`VD_CLIENT_UNPACED=1`) on the same stands. The rebuilt clock (§4.8.2) is trusted only where it and this instrument agree
  within the noise rule of §4.15.
- The stands: on foot (ground), hill, aloft, orbit, dusk, the berth boarding; paced and unpaced; B70's example flight.
- **The stall readings, as numbers, never as an absolute gate.** For each stand: the share of seconds with a frame over
  33 ms, and the count of pipelines compiled after loading (the pipeline meter, `pipeline_meter.rs:5`). B34 measured both
  red on today's build (a frame over 33 ms in 18–36 % of the seconds at the ground stand and in every second at dusk; three
  pipelines compiled after the loading at each of the two later stands; MEASURED 2026-10-02) and gave their cure ("every
  pipeline warmed before play") to T4, not to the move. So the record holds today's values, and the move compares with
  them by the frame rule's form (§4.15 item 7).
- The last readings, for scale: full screen 23.1 ms (about 43 frames a second); a frame over 33 ms in 18–36 % of the
  seconds at the ground stand and in every second at dusk; 146 ms at the dusk boarding with a pipeline compiled; 94 ms at
  the full-screen switch (all MEASURED 2026-10-02, B34; not clean).

**Example.** On the day that generator 26 is committed, the capture flies the nine stands twice and gets the same pixels,
the chain-latency gate records "one tick a level", the legs run in the Linux image, and — on your word at that moment —
the window reads the berth on foot with AirPlay off and Docker stopped. Every later phase compares against these numbers.

---

## 3. THE PATH

**What review B found about "one Bevy version in the lock".** Revision 1 said that each step keeps one Bevy version in the
lock. That is false. Cargo resolves ONE lock for all members, and `--exclude` removes a package from the build, not from
the lock (DOC, review B's I3). While the shards move and the window waits, the lock holds two `bevy_ecs` versions, and
`vd-bins` with `render` (which `just lint-combos` builds, `justfile:87-88`) compiles both into one package. Cargo accepts
this, and no workspace lint refuses duplicates. **One build must prove** four things before a mixed state counts as green:
the workspace and `lint-combos` build; no ECS type crosses between the shard crates and the window crate (the client
binary names one server item, `vd_connection_plane::tickets::mint_login`, `crates/bins/src/bin/client.rs:37`, not an ECS
type); `cargo tree -d` lists only the expected duplicates; G-SERVER-TREE stays clean. If that build fails, the shard phase
and the window phase that follows it become one commit unit, and the shard phase runs the server half of the suite.

**The four roads** (working days and calendar weeks: ESTIMATE, review C's bottom-up count with the levers of UQ13 and
UQ16, under UQ7 (a); UQ7 (b) adds about 2–4 working days to each road; §4.1 gives the phase rows).

| | (a) DIRECT 0.18 → 0.20 | (b) TWO STEPS, a green stop at 0.19.1 | (b2) TWO STEPS FOR THE WINDOW, the shards straight to 0.20 | (c') 0.19.1 NOW, R2's bake, 0.20 LATER |
|---|---|---|---|---|
| The steps | S0–UP3, one server phase, one window phase, UP9 | S0–UP3, UP4–UP5, THE STOP, UP7–UP8, UP9 | S0–UP3, one server phase to 0.20, the window to 0.19.1, a stop, the window to 0.20, UP9 | S0–UP6 (with UP9's look at 0.19.1), then R2's bake; later UP7–UP9 |
| Suspects for a changed picture | Every 0.19 and 0.20 change at once | Half at each step; 0.20's step is a translation, checked against UP6's run folder | As (b) for the window | As (b), split in time |
| Double work | None; but a bisect for a confounded red, 2–5 days | The sky patch re-based twice (Rust and shader hunks across about eight upstream changes); wgpu 29 then 30; egui 0.42 then 0.43; two server phases | One server phase fewer; two ECS stacks in the render build until the window reaches 0.20 | As (b), and every shader that R2 adds is ported to WESL later |
| A safe place to stop | None until 0.20 is green | 0.19.1, all stable releases (bevy_egui 0.42.0 exists); 0.19.1 is the last 0.19 release (R55) | A mixed stop (shards 0.20, window 0.19.1) | 0.19.1 by design |
| Working days | 27–43 | 31–45 | 30–44 | 23–34 now, then 10–14 |
| Calendar (every-day pace) | about 4–6 weeks | about 4.5–6.5 weeks | about 4.3–6.3 weeks | about 3.5–5 weeks now, then about 1.5–2 |
| Rulings | KEPT | KEPT (a rest at the stop AMENDS B70 item 1) | KEPT | AMENDS B70 item 1 (target and order) |

**Recommendation: (b).** The owner's own reason of 2026-08-27 stands: "a red suite would not say which change broke it".
Most failures in this move are silent: a shader fails only when its pipeline is built at run time (R9); the weak render
order changes no compile result (R27); the quiet changes move pixels (R25). The stop gives 0.20's shader translation a
strict test against UP6's own pictures. In expectation (b) costs about 2–4 working days more than (a), because (a) pays a
bisect for a confounded red (ESTIMATE, review C). About half of the extra is porting, not machine time. (b) also meets
0.20 later: UP7 starts 21–31.5 working days (about 3–4.5 weeks) after the move starts (ARITH on §4.1's rows), and the move
starts only when generator 26 closes, a date that is UNMEASURED; so the 0.20 step meets 0.20 at least 3–4.5 weeks after
its release (ESTIMATE), when its first patches for a new shader language and for wgpu 30 can exist.
(c') reaches R2's bake about 8–11 working days sooner (ARITH on the rows), but it amends B70's target, and its later port grows with each new shader;
most benefits that today's game can use arrive with 0.19 already (UB1, UB2, UB5–UB8, UB12–UB14), and 0.20 adds UB3, UB4,
UB9 and UB28. (b2) saves about one day and leaves a mixed stop. We recommend (c') only if the terrain's start matters more
to you than the target of B70.

---

## 4. THE PHASES

### 4.1 The phases in order

Each phase is ONE job on the machine (memory). While a campaign runs, the session prepares the next phase's code in a
second worktree, with no cargo there. A commit comes only on your request for that action. The commit units: UP0's new
gates; UP1–UP3; UP4 (or UP4 with UP5, §3); UP5; UP6; UP7 (or UP7 with UP8); UP8; each UP9 cure; UP11.

| Phase | What it does | Its gate | Working days | Machine hours |
|---|---|---|---|---|
| S0 | The spikes (§4.2) | Each spike's verdict | 1–1.5 | 2–4 |
| UP0 | The before record (§2) | Complete; every red named | 2.5–3.5 | 13–24 |
| UP1–UP3 | The toolchain (§4.4), glam (§4.5), every default written down (§4.6): ONE campaign | Nothing changes; each new gate's control goes red | 5–7 | 13–22 |
| UP4 | The shards on `bevy_ecs` 0.19.1 (§4.7) | The record's server half; G-ORDER-RECORD; coverage | 1–1.5 | 5–8 |
| UP5 | The window on Bevy 0.19.1; the sky patch re-based (§4.8, §4.9) | The GPU gates; the pairs; G-SELF-CHECK; G-CLOCK-SUM | 9.5–15 | 16–30 |
| UP6 | THE STOP (§4.10) | The full list at 0.19.1; one look; the window checklist | 2–3 | 13–22 |
| UP7 | The shards on `bevy_ecs` 0.20 (§4.7) | As UP4 | 1–1.5 | 5–8 |
| UP8 | The window on Bevy 0.20 (§4.11) | Pictures within one level of UP6's run folder | 5–7.5 | 11–23 |
| UP9 | The pictures, the frame, the flights (§4.14) | The closing list (§4.15) | 2.5–3.5 | 10–17 |
| UP10 | The features that you chose (§5) | Each its own measured change | per feature | per feature |
| UP11 | The close (§4.16) | The documents current | 1 | — |

**Totals** (ESTIMATE, review C §5; ARITH on the rows): 31–45 working days and 90–160 machine hours with the levers; 34–50
days and 165–300 hours without them. A campaign costs 13–22 machine hours with the levers and 21–38 without; a machine day
is 18 hours under a monitor. The basis: nine recipes solve the planets on every run today (`crates/bins/src/pixel.rs:320-325`:
"a gate on a fresh fixture now waits over an hour for its first solves"; the home planet's first solve 3 514 s in the
shard, MEASURED 2026-10-07, the status file), and the test's own `solve_home` runs on one thread (`solved.rs:114-127`,
`artifact_worker.rs:36-41`, `par.rs:18-25`). The clean first-solve timing of B69's list (item 3 (2)) replaces these numbers.

**Under UQ7 (b).** These totals assume UQ7 (a): review C's two re-bases (W9 at UP5, W16 at UP8) count six kept sky
changes. Under UQ7 (b) the re-bases carry eight, so road (b) costs about 2–4 working days more (ESTIMATE: about 0.5–1 day
for each of the two extra changes at each of the two re-bases) and a few machine hours for the sky gates that stay
(G-SKY-CONTROL): 33–49 working days with the levers (36–54 without), about 4.7–7 calendar weeks (ARITH).

**Your time** (2–4 days, ESTIMATE). Five looks: the generator-26 pairs in B69's close (before the move); the temporal
path's first references at UP0 (UQ6); the ground stand's sky light in the window at UP5 (UQ7); the pairs at the stop; the
pairs at the close. Four window sessions, each opened only on your word at that moment: the protocol's first run at UP0;
the ground-stand look at UP5; the checklist flight at UP6; the runs in turn, the checklist and B70's flight at UP9. One
commit request for each commit unit.

### 4.2 S0 — the spikes, first

S0 runs after your word to start the move and before UP0. Each spike runs outside the tree, in the scratchpad, with its
own target directory, one job at a time, with a neutral identity on every fetch. Machine time: 2–4 hours for S0.1–S0.3
(ESTIMATE, review C).

| Spike | What it runs | Green: the plan | Red: what changes |
|---|---|---|---|
| S0.1 | naga 29.0.4 and 30.0.1 (the versions that the 0.19.1 and 0.20 locks take) on the shipped rust-gpu SPIR-V (the `.spv` that `crates/client-render/build.rs` writes): the SPIR-V front end, the validator, the Metal back end — with the settings that the game uses, stated in the spike's record: the validator's capabilities as wgpu sets them from the device's features (at least `SHADER_INT64`), and the Metal language version that wgpu-hal picks for this Mac (wgpu-hal 30.0.1 picks it by the OS, `src/metal/adapter.rs:707-724`: 4.0 on macOS 26, which this machine's Darwin 25.5.0 suggests; wgpu-hal 29.0.4 tops at 3.2, `:655`). Then S0.3's setup runs the REAL module once with one dispatch on the M4 Pro and compares its words with the CPU recipe's (the spike links the tree's `vd-recipe` and `vd-terrain` by path, as `crates/bins/tests/gpu_no_drift.rs` does): the self-check's own verdict | The plan as written | naga 29 refuses: the move cannot keep F7 as built; UQ3 before any phase. Only naga 30 refuses: the 0.19.1 step proceeds; the 0.20 step waits for a cure (UQ3). Accepted, but the words differ (FAILED): UQ17 before any phase |
| S0.2 | wgpu 29.0.4 and 30.0.1 on the M4 Pro: two empty passes stamped at their edges around a known workload, read back three frames later (the GPU clock's method; `crates/bins/examples/gpu_spike.rs` is the pattern) | The clock keeps its method | The clock needs a new method before UP5; until then the frame verdict uses the shared instrument alone; you hear it before UP0 |
| S0.3 | On wgpu 29.0.4 and 30.0.1: a CUT SPIR-V module and its pipeline, created inside an error scope (in wgpu 28 and later the scope is a guard, `wgpu-30.0.1/src/api/device.rs:542`). The cut falls on a word boundary and keeps the magic word (or one instruction word is corrupted): `make_spirv` panics BEFORE any error scope when the length is not a multiple of 4 or the magic word is missing (wgpu 27.0.1 `src/util/mod.rs:53-90`; wgpu 30.0.1 `src/util/spirv.rs:16-22`), and that panic would read as a red scope for a reason that is not the scope | The scope holds the error: UP3's self-check design as written | The error escapes the scope: the self-check first validates the module on the CPU with `wgpu::naga` (the same naga that wgpu links, re-exported from wgpu-core at `wgpu-30.0.1/src/lib.rs:319`; no new dependency) and creates nothing on a refusal |
| S0.4 (if time allows) | `ladder_fade.wgsl` and one sky hunk as WESL against 0.20 | UP8's size holds | UP8's estimate is restated |
| S0.5 (if time allows) | Bevy 0.19.1's and 0.20.0's own atmosphere example on the M4 Pro: the GPU preprocessing line | UB8's mode as expected | UB8 restated |

**Example.** S0.1 hands the recipe's SPIR-V to naga 30. If naga refuses one 64-bit conversion, you learn it on the first
day, not on day 20 after the window's port. If naga accepts it but the M4 Pro computes one height word of the home
planet's golden columns differently from the CPU, you learn that on the first day too.

### 4.3 UP0 — the before record

§2 gives it whole. UP0 also builds G-WIRE-BYTES, proves `VD_PICTURE_IDENTICAL` on two runs, and writes the flag table.

### 4.4 UP1 — the toolchain (UQ10)

- **The fact.** The `bevy` crate 0.20.0 states `rust-version = "1.97.1"`; `bevy_ecs` 0.20.0 states `"1.95.0"` (DOC:
  fetched 2026-10-09, §10.2). Cargo refuses a package whose `rust-version` is newer than the toolchain. Rust 1.97.1 exists
  (2026-07-16), and so does the image tag `rust:1.97.1-slim-bookworm` (Docker Hub, updated 2026-08-11) (DOC, reviews B
  and C).
- **What moves**, in one change: `rust-toolchain.toml:5`; `Cargo.toml:29`; `docker/server.Dockerfile:21` and its comment
  at `:29`; `docker/agent.Dockerfile:8`; `scripts/terrain_legs.sh:18` (the image) and `:26` (`RUSTUP_TOOLCHAIN`);
  CLAUDE.md's sentence "product builds on stable 1.94.1".
- **The new rustfmt.** If `fmt-check` goes red, `cargo fmt` runs as its own change, with no other edit in it.
- **The new lints.** Fixed in this phase, on Bevy 0.18 (R17).
- **The coverage nightly.** `nightly-2026-06-06` is rustc 1.98.0-nightly (DOC, `justfile:5`); it passes Cargo's version
  check for `bevy_ecs` 0.19.1 and 0.20 (1.95.0). That it builds them is UNMEASURED. The pin is deliberate (`justfile:5-7`),
  so the nightly moves only if it fails to build, with the failure written down.
- **rust-gpu.** `crates/recipe-gpu` builds on its own nightly through cargo-gpu and links no Bevy; it does not move.
- **The gate** (in the UP1–UP3 campaign): `just gate` in full; the pins; the ephemeris pin; G-WIRE-BYTES; `just
  terrain-legs` on both platforms with Docker up as its own job, then Docker stopped; `just image-smoke` in the same
  Docker window; the pictures identical.

### 4.5 UP2 — glam alone, at an exact patch

- **Why alone.** One glam serves Bevy and six Tier-A crates: vd-core (it re-exports glam, `crates/core/src/lib.rs:52`),
  vd-physics, vd-sim, vd-node, vd-client, vd-client-harness. glam values cross the wire (`crates/core/src/pose.rs:918-924`),
  and the ephemeris pin holds exact f64 bits through glam's rotation.
- **The exact patch.** Bevy 0.20's requirement `0.33.2` is a caret requirement, so a fresh lock takes the newest 0.33 patch:
  0.33.12 today (DOC, review B's I1). The workspace pins it exactly, `glam = { version = "=0.33.N", features = ["serde"] }`,
  where N is the newest patch whose change log we read at UP2. A `cargo update` then cannot move it in silence; a new patch
  is its own measured change.
- **The result changes up to 0.33.12** (DOC, the change logs): 0.32.1 `1.0 / m` is the element-wise reciprocal; 0.33.4
  scalar `recip()` replaced by division in vector code (for example `DVec3::project_onto`); 0.33.7 `Vec3A` and `Vec4`
  round half-way cases away from zero in the SIMD code, a NEON sine in `Quat` and `Vec3A` `slerp`, a special affine
  determinant in `Mat4::to_scale_rotation_translation`; 0.33.8 the `Mat3`/`DMat3` determinant and inverse aligned with
  `Mat4`, and fused multiply-add in the SIMD code where the target has it; 0.33.11 `FloatExt::lerp`'s formula; 0.33.12 a
  finite inverse only from `try_inverse`/`inverse_or_zero`.
- **Where they can reach us.** The shards' crates use the f64 types; review A found no `Vec3A`, `Vec4` or `slerp` in core,
  physics, sim, node or connection-plane. The ephemeris pin's own operations are identical in source (MEASURED by review B,
  a source diff of 0.30.10 against 0.33.12). Our Tier-A crates call `.inverse()` 21 times and `.lerp(` 4 times outside the
  window (review B, grep): their bits are UNMEASURED and the gates read them. The f32 changes reach Bevy's transforms only
  when Bevy itself takes glam 0.33 (UP8).
- **The deprecations.** glam 0.33.2 moves the camera constructors to its `camera`/`dcamera` modules. Our glam has one such
  call, `DMat4::look_at_rh` in a test of the camera copy (`crates/client-harness/src/camera.rs:971`); the window's
  `Mat4::perspective_infinite_reverse_rh` (`reconstruction.rs:845`) uses Bevy's glam and moves at UP8 (MEASURED
  2026-10-09, grep).
- **The glue.** Two glam versions live in the window's build from UP2 to UP8: ours at 0.33.N; Bevy's at 0.30 until UP5,
  then 0.32 (`bevy_math-0.19.1`), then one glam again at UP8. The window converts at 11 calls of `as_vec3()`/`as_quat()`
  (10 in `lib.rs`, 1 in `terrain.rs`; MEASURED 2026-10-09). Each call goes through the float array, which keeps every bit.
  One helper in `vd-client-render` (Tier-B) holds the glue; UP5 re-checks it against glam 0.32; UP8 deletes it; DEFERRED
  registers it with that date. UQ11 offers glam 0.32 first instead.
- **A moved pin.** A moved byte or bit is a finding, and it takes §4.15's time box: one working day to find its cause
  (the change-log entry and the function). A cause with a cure: we cure it. A known cause with no cure inside the move
  (for example glam's own numeric fix): you rule, by UQ11's rule (re-record the pin with its cause named, or hold the older
  patch). No cause after the day: we register it with what we know, and you rule the re-record. We never re-record by
  habit. The ephemeris pin is host-local by its own header ("re-capture on that host",
  `crates/physics/tests/celestial_ephemeris_pin.rs:1-20`).
- **The gate** (in the UP1–UP3 campaign): G-WIRE-BYTES byte for byte; the ephemeris pin bit for bit; the determinism gates;
  the digests; the integer world's pins as controls (they link no glam); `coverage-fast` of the six crates; the process
  gates; the pictures identical.

### 4.6 UP3 — every default written down, on Bevy 0.18

Each item changes NOTHING on Bevy 0.18. Its purpose: the move cannot change it in silence. The record proves "unchanged";
each new gate has a control that goes red when it should (memory: prove an instrument before trusting it).

**The shards.**
1. **The assembly in one place (review A-17).** Each binary's schedule assembly (`build_app` and its registration calls in
   their order: `crates/bins/src/bin/shard.rs:121`, `:454`; `gateway.rs:47`, `:182`; `orchestrator.rs:454`) moves into a
   library function that the binary and the tests both call.
2. **G-ORDER-RECORD (R1).** A test builds the three server schedules through those functions, initializes them, and
   compares each schedule's system order AND its list of conflicting pairs (Bevy's own report on the built schedule; an
   existing setting, no new feature) with committed lists. Control: a swapped entry in the expected list goes red.
3. **The total order (UQ4).** `observe_clock_syncs` has no edge to the chains; Bevy 0.18 runs it after `emit_realm_frames`
   (DOC: a measurement recorded with no date in the test's comment, `tests/tests/frame_conversion_e2e.rs:3319-3330`).
   `retry_pending_flushes` runs after
   `redrive_stranded_crossings` and before `emit_frames`, and its place among the four systems between them is Bevy's choice
   (`crates/sim/src/stub/register.rs:253-257`). We state both places exactly as the record shows them, and we correct the
   three comments that say the follower runs first (`register.rs:178`, `:214`, `:636`); the gateway's follower gets the
   same treatment. After this, the conflict list holds no pair without a stated order.
4. **One schedule constructor, in vd-sim (R2; review A-15).** Ten test rigs build `Schedule::default()` with no executor;
   under `cargo test --workspace` Bevy's `multi_threaded` feature is on, so they run the multi-threaded executor, while
   production pins the single-threaded one (`crates/node/src/app.rs:109`). The constructor lives in vd-sim, the lowest crate
   that holds `bevy_ecs`; `build_app` and every rig call it; no crate gains an upward edge.

**The window.**
5. **MSAA stated (R11)**: the window camera and the capture's picture camera state today's default `Msaa::Sample4`.
6. **The tone mapping and the dither stated** at today's defaults (the eye's rule copies the dither).
7. **The adapter's features stated (R6)**: the render plugin (`lib.rs:37-48`) states the priority that gives every
   adapter feature, so `SHADER_INT64` does not hang on a default; the startup line says whether the device holds it.
8. **The formats stated**: the capture's targets state the format that `TextureFormat::bevy_default()` returns today.
9. **The window's own order (R13, R43)**: a test reads Bevy's conflict report on the client's built app schedules
   (`input_system` and `place_camera` both write `CameraState`, `lib.rs:1434-1461`; `serve_captures`, `:3294-3327`), and
   the code states the order that 0.18 runs. The test builds the schedules in memory and loads no window system, as the
   capture does; it opens no window.
10. **The render world's hidden waits (R27)**: `choose_slot` before the render and `read_back` after it
    (`gpu_clock.rs:315-320`); `count_pipelines` after the render (`pipeline_meter.rs:60`); `prepare_points` (`points.rs:106`);
    `write_band_tables` (`lib.rs:1235`). The two view systems already state their edges (`measure.rs:268-270`,
    `reconstruction.rs:195-198`; review B). Schedule tests read the built render schedules, in the pattern of the engine
    canary at `reconstruction.rs:801`.
11. **The self-check behind an error scope (R5; review B's B1).** Today the module and the pipeline are created with no
    error scope (`gpu_check.rs:169-173`); FAILED means "the words differ", SKIPPED means "no 64-bit integers"; neither
    catches a refused module, which today panics the client (wgpu 27's default handler) and on 0.19/0.20 makes Bevy quit
    it. The cure: `gpu_check::run` takes the SPIR-V bytes as an argument (production passes the built-in bytes); it creates
    the module, the pipeline and the first dispatch inside validation and internal error scopes; a captured error gives a
    new verdict, REFUSED, and `attach_card(..., false)` runs the CPU path of the same source (F8); the client runs on.
    Control: an in-process GPU test passes a CUT SPIR-V — cut on a word boundary with the magic word kept, or one
    instruction word corrupted, because `make_spirv` panics before any error scope on a length that is not a multiple of 4
    or on a missing magic word (wgpu 27.0.1 `src/util/mod.rs:53-90`; wgpu 30.0.1 `src/util/spirv.rs:16-22`) — and expects
    REFUSED, a running process, and the CPU builder; a second control passes the real bytes and expects PASSED.
12. **The exit code carried (R26; review B's B2).** `vd_client_render::run` returns Bevy's `AppExit` (today it returns
    `()`, and `run_windowed` and `run_capture` drop the value of `.run()`: `lib.rs:1313-1322`, `:1467`, `:3328`).
    `crates/bins/src/bin/client.rs:415-436` maps `AppExit::Error(n)` to a failure exit code and `AppExit::Success` to
    success. On 0.18 nothing else changes. G-EXIT's form: a unit test of the mapping and, from 0.19.1, an in-process test
    `App` that makes one device error in test code. No client switch and no vdctl command makes a GPU error.
13. **The canaries (G-CANARY; R10)**: CPU tests for the Earth medium, the projection and `looking_at`, and the jitter; one
    in-process GPU test for the dither (§4.12).
14. **G-LAYOUT (R8)**: a CPU test compares the points pass's copy of the atmosphere's binding numbers (`points.rs:151-161`,
    `:434-443`) with the vendored layout.
15. **G-PIPELINES-OK (R9, R50)**: every pipeline on every stand and path reaches Ok and none is in Err, as in the record;
    the count of pipelines compiled after loading is compared with the record's by the frame rule's form (§4.15 item 7),
    never as "none", because B34 measured three on today's build at two stands (§2); the flag table of §2 item 8 says which
    gate draws each flag.
16. **G-CAPTURE-SAME-FRAME (R33; review B's I4)**: written now on the graph node, so it must pass on 0.18 (§4.13).

**The UP1–UP3 campaign and its bisect rule.** The three changes stay three separate patches. One campaign runs after all
three. If no result moved, each change holds alone (an exact cancellation of two changes is improbable). If a result moved,
a second worktree builds the first patch, then the first two, to find the change.

### 4.7 UP4 and UP7 — the shards on `bevy_ecs` 0.19.1, then 0.20

The shards use a World, a Schedule and resources only: 37 production systems, 68 `#[derive(Resource)]` in the Tier-A
crates, no entities, no queries, no commands, no events, no observers (MEASURED 2026-10-09, reader 3; confirmed by review
B). No change detection, no `clear_entities` and no entity walk on the server crates (MEASURED 2026-10-09, grep).

- **The lock.** The window keeps its Bevy, so the lock holds two `bevy_ecs` versions; the proof build of §3 runs first.
  If it is green, the full suite and lint run; if it is red, the server half runs and this phase joins the next window
  phase as one commit unit.
- **At 0.19.1 (UP4).** `ExecutorKind` is removed (#23414): the vd-sim constructor calls
  `set_executor(SingleThreadedExecutor::new())`. Resources become components on their own entities; the derive now emits
  a Component impl (`bevy_ecs_macros-0.19.1/src/resource.rs:27`, DOC). `coverage-fast` on the first build shows whether new
  regions appear at the 68 sites (R3, UQ15). The opt-level list of Bevy's crates in the dev profile (`Cargo.toml:112-123`)
  is read against 0.19.1's names (R19).
- **At 0.20 (UP7).** Exclusive function systems are unified (`&mut World` is a system parameter, #25507); the eight server
  functions that take `world: &mut World` keep working. 0.20's weak set orders (#25128) touch Bevy's built-in schedules;
  ours are our own, and G-ORDER-RECORD (order and conflicts) proves that nothing moved.
- **No shuffle.** The conflict list proves the same thing as Bevy 0.20's schedule shuffle, without its `debug` feature, so
  no build carries that feature. If you still want the shuffle (UQ4), it lives only in a test crate outside the workspace,
  as `crates/recipe-gpu` is, with its own target directory; `cargo tree -e features` then proves that no workspace binary
  carries `bevy_ecs/debug`.
- **The gate (both).** The record's server half (or all of it); G-ORDER-RECORD; `coverage-fast`; the process gates as
  written; G-TICK-TIME; G-SERVER-TREE; G-WIRE-BYTES; `just image-smoke` at UP7 when Docker may run.

### 4.8 UP5 — the window on Bevy 0.19.1

#### 4.8.1 The render graph becomes systems

Bevy 0.19 removes the render graph: a pass is a system in a camera's schedule (`Core3d`), and the root work is a system in
the `RenderGraph` schedule, whose `camera_driver` runs each camera's schedule (#22144). The command buffers are flushed in
the systems' topological order (`render_context.rs:70-72`, DOC). Our ten nodes:

| Node today | Where | Its place today | Its system on 0.19.1 |
|---|---|---|---|
| `OpaqueCopyNode` (view) | `reconstruction.rs:214` | after the transmissive pass and the sky pass, before the transparent pass (`:151-177`; the sky edge added in `finish`) | `Core3d`, after the vendored `render_sky` (re-exported, §4.9) and the opaque pass, before `main_transparent_pass_3d` |
| `ReconstructionNode` (view) | `reconstruction.rs:681` | after `StartMainPassPostProcessing`, before `Bloom` | post-processing, before `bloom`, placed as Bevy's own `temporal_anti_alias` |
| `PointsNode` (view) | `points.rs:366` | after the reconstruction, before `Bloom` | after our reconstruction, before `bloom` |
| `RenderPictureNode` (view) | `render_picture.rs:114` | before the reconstruction | before our reconstruction |
| `ImageCopyDriver` (root) | `lib.rs:3946` | after `CameraDriverLabel` | `RenderGraph`, after `camera_driver` |
| `MotionCopyNode` (root) | `motion_capture.rs:143` | after `CameraDriverLabel` | `RenderGraph`, after `camera_driver` |
| `PriceG2Node` (root) | `measure.rs:368` | after `CameraDriverLabel` | `RenderGraph`, after `camera_driver` |
| `BeginNode` (root) | `gpu_clock.rs:596` | before every root node | `RenderGraph`, before `camera_driver` |
| `PassStampNode` (camera graph) | `gpu_clock.rs:625` | after every working node, one chain | §4.8.2 |
| `EndNode` (root) | `gpu_clock.rs:678` | after every root node | `RenderGraph`, after `camera_driver` and the three copies |

The systems themselves state each order, so the first-frame build and the `finish`-time edge go (R14). Each order gets a
schedule test. We read the exact pass names in the 0.19.1 sources at the phase.

**The other 0.19 changes in the window** (DOC, reader 1 and review B): `ManageViews` splits into `CreateViews`, `Specialize`
and `PrepareViews`, and `prepare_view_targets` sits in `PrepareViews` (`bevy_render-0.19.1/src/view/mod.rs:195-199`), so
our two view systems keep their stated edges; we re-write the canary at `reconstruction.rs:801-836` for the new sets and
for `System::type_id` → `System::system_type` (#23326). `ExtractComponent` stays, and its derive now writes the
`SyncComponent` impl; our six derives need nothing, and the one manual impl (`points.rs:70`) needs a new `impl
SyncComponent for PointsCloud`. `view.hdr` moves to `ExtractedCamera::hdr`; `TEXTURE_FORMAT_HDR` to
`ExtractedView::target_format`; `ViewTarget::out_texture` becomes optional (R41); `shadows_enabled` becomes
`shadow_maps_enabled` (`terrain.rs:4029`, `:4031`, `:4057`; `lib.rs:1660`); `AlphaMode` moves to `bevy_material`; `Hdr` to
`bevy_camera`; `Frustum` wraps `ViewFrustum`; `Assets::get_mut` returns `AssetMut` (UB13). Our three `init_pipeline`
systems read no mesh pipeline, so they need no new edge (review B).

#### 4.8.2 The GPU clock's redesign

Its method stays if S0.2 is green: a stamp is the edge of an EMPTY PASS of the clock's own, because this GPU samples its
clock only at a pass's edges (DOC, the code's comment on the clock's first try: stamps outside a pass read nought and the frame took 275
ms, `gpu_clock.rs:11-21`). It keeps one query set of 8 slots, each resolved three frames later (memory: the clock lied
until that cure, 2026-09-29).

1. **The frame's two stamps.** `begin_stamp` before `camera_driver`; `end_stamp` after it and after the copies. The end
   stamp's pass loads and stores each camera's output picture; a camera with no output picture gets no end stamp.
2. **The per-pass stamps (`VD_GPU_CLOCK=passes`).** Stamp systems in `Core3d`, each stated after one NAMED pass system and
   before the next, for a list of the pass systems in their order (each name read from the source at the phase). A stamp's
   stated edges hold its place, whatever access the render context declares (that access is UNMEASURED).
3. **A census test** lists every system of the built `Core3d` schedule that takes a render context; each one is on the
   stamped list or on a named "no stamp" list, so a new upstream pass makes the test red.
4. **G-CLOCK-SUM** (T0's gate) stays: the passes' sum inside the frame's span.
5. **The same span on both clocks.** The begin stamp stands before all of a frame's GPU work and the end stamp after all
   of it, on both clocks. At UP0 (old clock) and at UP5 (new clock) the clock's frame span and the shared instrument (§2)
   read the same stands; the new clock counts only where the two agree within the noise rule.

The road refused: walking the built schedule at the first frame to insert stamps (it brings back the first-frame build that
hung the capture twice, MEASURED 2026-09-29, `gpu_clock.rs:420-429`). Bevy's own render diagnostics stay behind
`VD_RENDER_DIAG`: they lost the device on this Mac, and their GPU times read nought there (MEASURED 2026-09-23/24,
`lib.rs:16-28`).

#### 4.8.3 The atmosphere becomes an entity (if UQ7 (a))

Bevy 0.19 places the planet by the `Atmosphere` entity's `GlobalTransform` (#23651); from 0.19.1 the entity's rotation is
ignored and a Z-up world has a sound basis at its poles (#24998, in 0.19.1: `bevy_pbr-0.19.1/src/atmosphere/mod.rs:242-255`,
DOC, review B). Bevy chooses, per camera, the atmosphere whose ORIGIN is nearest; our client chooses the body by the largest
shell angle (`sky.rs:163-164`).
- `sync_sky` keeps its choice and spawns ONE `Atmosphere` entity at the centre that it already computes (`sky.rs:211-253`),
  scale 1, moved each frame; with one entity the two choices cannot disagree (R21); a test holds "at most one".
- `bottom_radius`/`top_radius` → `inner_radius`/`outer_radius` (`sky.rs:248-249`); `planet_center` leaves the patch.
  `Atmosphere` now lives in `bevy_light`, so the `below` field of change 6 cannot stay on it without a second vendored
  crate; this design's choice: `AtmosphereBelow` becomes its own component on the atmosphere entity, read by the vendored
  extraction. `sun_disk_on_sky` stays a vendored field of `AtmosphereSettings`.
- The f32 rounding of `world_to_atmosphere · eye` at 6.4 million metres equals today's: the f32 spacing there is 0.5 m
  (ARITH; PATCH.md's seventh change states the same half metre).
- G-SKY-CONTROL and `VD_SKY_FLAT` retire with change 1 (`docs/design/DEFERRED.md:8703-8704`): nothing of ours remains to
  control. G-AIR-UNDER and the coast stand hold the air (review C's M13). Under UQ7 (b) they stay.

#### 4.8.4 The points pass

It becomes a system after our reconstruction and before `bloom`. Its group-0 layout copies the atmosphere's binding numbers
0, 1, 3, 4, 8, 12 and adds 20–22 (`points.rs:151-161`, `:434-443`); the numbers stay the same at 0.19.1 and 0.20.0 (DOC,
review B). At 0.19.1 the uniform gains `world_to_atmosphere`, the radii are renamed, and `AtmosphereData` is removed; at 0.20
it returns inside the per-camera `AtmosphereBuffer` (`bevy_pbr-0.20.0/src/atmosphere/resources.rs:807-815`), while the
points pass's `ComponentUniforms<GpuAtmosphere>` stays. G-LAYOUT reads the form of each version.

#### 4.8.5 wgpu 29, the self-check and the rust-gpu road

- The direct `wgpu` dependency follows Bevy's exactly: the 0.19.1 lock takes wgpu and naga 29.0.4; the 0.20.0 lock takes
  wgpu 30.0.1 (DOC, review B). The lock record names them.
- The direct surface is 253 sites: `vd-client-render` 167 (`gpu_check.rs` 117), `crates/bins/examples/integer_bench.rs`
  45, `crates/bins/examples/gpu_spike.rs` 34, `crates/bins/tests/gpu_no_drift.rs` 7 (MEASURED 2026-10-09, review C, grep).
  `lint-combos` compiles the examples with `render`. UQ16 asks whether the two examples retire.
- `gpu_check.rs`'s own wgpu surface is small: buffers, compute pipelines, `map_async`, `PollType::wait_indefinitely` (still
  in wgpu 30) and `dispatch_workgroups` (review B); the new item is the error scope as a guard (UP3 item 11).
- The rust-gpu SPIR-V goes through `wgpu::util::make_spirv` and naga's SPIR-V front end into Metal (`gpu_check.rs:37`,
  `:171`, `:337`); wgpu 30 keeps that road (`wgpu-30.0.1/src/util/spirv.rs:25-27`; the `spirv` feature still enables
  `naga/spv-in`). S0.1 settles naga's verdict before any phase.
- The gate: G-SELF-CHECK (the verdict PASSED on the M4 Pro, and the REFUSED control), gpu-drift, gpu-seam,
  terrain-pictures-card. A FAILED verdict here (naga accepts the module, but the M4 Pro's words differ from the CPU's)
  keeps the game correct, because the CPU path builds, but it breaks F7; UQ17 holds the prepared roads.

#### 4.8.6 The HUD (bevy_egui 0.42, egui 0.33 → 0.36)

bevy_egui 0.42.0 pairs with Bevy 0.19 and egui 0.36 (DOC). egui moves three versions (0.34 with bevy_egui 0.40.0, 0.35 with
0.41.0, 0.36 with 0.42.0); our HUD calls `egui::Area`, `egui::Align2`, `egui::TextWrapMode` and the style's `wrap_mode`
(`lib.rs:2934-3025`), and each egui version's notes are read at the phase. Our multipass names exist in both 0.42.0 and
0.43.0-rc.1 (DOC, review B). From 0.40 the scale follows `native_pixels_per_point`, so the HUD's size can change; the
picture gate leaves out the HUD's own rectangle, so the HUD's size and scale at each stand join the pairs that you see
(review A-20).

#### 4.8.7 A GPU error

Bevy 0.19.1 and 0.20.0 install an uncaptured-error handler (`bevy_render-0.19.1/src/error_handler.rs:131-132`); their
default `RenderErrorHandler` logs "Quitting the application due to … RenderError", asks for `AppExit::error()` (exit code
1) and stops rendering, for every error type (`:69-84`, the same in 0.20.0; DOC, review B). The release post's sentence
"Vulkan validation errors are ignored" does not match that code. With UP3 item 12 the exit code reaches the process, so
Bevy's default policy is enough; a handler of our own is needed only for another policy (UB15, UQ9).

#### 4.8.8 Bevy's sky motion, the fast window build, and the switches

- From 0.19 a camera with `MotionVectorPrepass` writes the eye's turn into every sky pixel (#23187). The temporal cameras
  take `NoBackgroundMotionVectors`, so today's motion stays (UQ8); motion-truth (+control) proves it.
- The owner's fast window build (`scripts/client.sh --fast`: `dev-control,render-dylib`, Bevy as one dynamic library,
  `scripts/client.sh:108-127`) is built once at UP5 and once at UP8 (review C's M8).
- The switches that the move touches survive with a proof (a capture in that mode at the stop and at the close; for the
  window-only `VD_WINDOW_*` switches, the checklist session on your word), or go on your word (§4.13).

### 4.9 The sky patch, re-based

The patch today: 9 files changed and 1 new file, +660 / −93 lines over `bevy_pbr` 0.18.1; its shader part is +380 / −32,
or +370 / −29 without the `sky_view_lut` hunk (MEASURED 2026-10-09, reader 3 and review B, `diff` against the registry
copy). PATCH.md says that "the patch dies at the Bevy upgrade gate (0.19)" (`PATCH.md:115`); it does not: at least six
changes remain (2–7), and eight under UQ7 (b). We re-base it twice: onto 0.19.1 at UP5 (the old shader dialect, the passes as systems) and onto 0.20.0 at
UP8 (WESL; the per-camera buffer; about eight upstream atmosphere changes between, #23113, #24420, #24725, #24884, #25088,
#25398, #25440, #25699).

| # | The change (its date) | Its ruling or measurement | Upstream 0.19.1 / 0.20.0 (DOC) | Its fate | Its gate |
|---|---|---|---|---|---|
| 1 | The planet's centre stated (2026-09-18) | Slice 8s, owner-agreed | The entity's `GlobalTransform` (#23651) | UQ7: retired, or kept as a component that the vendored extraction reads | The pairs; G-AIR-UNDER |
| 2 | The sky per sample under MSAA (2026-09-19) | A stair on the horizon MEASURED | Still sample 0 (`render_sky.wgsl:33`, `.wesl:34`) | KEPT | The ground stand's horizon; picture-census |
| 3 | The march reads only its own path; no aerial table (2026-09-24) | The window's heat | Still reads, then overwrites | KEPT, in `render_sky` and the `atmosphere_luts` system | The table pass's time; the pictures |
| 4 | The air over a pixel marched once (`sky_air`, 2026-09-24) | Unpaced berth 30.0 → 11.3 ms (MEASURED 2026-09-24, af21326) | Absent | KEPT: a system between `atmosphere_luts` and `render_sky` | The frame |
| 5 | `sun_disk_on_sky` and the public labels (2026-09-29) | B24 | No setting; `render_sky` is `pub` in a private module | KEPT; the labels become a re-export of `render_sky` (R22) | motion-truth; the temporal disc |
| 6 | The air under its ground (2026-09-30) | B25, B26 | Still lifts the eye; still ends rays at the sphere | KEPT, re-based on `world_to_atmosphere` and the new phase mapping; `AtmosphereBelow` its own component | G-AIR-UNDER; the dry sea floor; the coast stand |
| 7 | The sun at the sphere (2026-10-01) | 541 dark specks MEASURED | No guard on the root | KEPT (one line) | The walk's stand: no speck |
| 8 | The table's eye (2026-10-02) | B31, B32: the sky light on foot, accepted in the window | `world_pos = cam_pos` (#22938) | UQ7: retired, or re-based on the new table frame | The on-foot sky light; G-CROSSING; your window look |
| 9 | Unlisted: the `environment_pass` span (2026-09-24, af21326; `environment.rs:21`, `:303`, `:313`, `:332`) | — | 0.19.1 records no span in any air pass; 0.20.0 records `atmosphere_environment`, `atmosphere_luts` and `render_sky` again (#25398) | UQ7: kept at 0.19.1; upstream's span from 0.20, and the reading's parse follows the name (`terrain.rs:2671-2689`) | — |
| — | The `.single()` limit (`PATCH.md:117-118`) | — | Fixed in 0.20 (#23113) | Retires at UP8 | — |
| — | The haze (B25 law 4) | B25 | 0.20's `ScatteringMedium::earth` still holds Mie absorption 3.996e-6 and scattering 0.444e-6 | KEPT in the game's own medium (`crates/client/src/sky.rs:21-42`) | The haze reading on the dry floor |

Every hunk keeps its `VOXELDUST PATCH (…)` marker. The upstream pull request that DEFERRED owes
(`docs/design/DEFERRED.md:8691-8695`) carries six changes, or eight under UQ7 (b) (§8, the later question).

### 4.10 UP6 — THE STOP at 0.19.1

- **The gate set.** The closing list's items 1–5 and 8 at 0.19.1 (§4.15), with the headless frame readings by the frame
  rule against UP0's kept build.
- **The pictures by B71.** Each stand's pair, with each difference named by its cause (§4.14). BETTER: re-frozen, and you
  see the pairs in ONE look. WORSE: cured at 0.19.1, or a road that you rule. Mixed or unclear: your look.
- **The window checklist flight**, on your word at that moment (§4.14).
- **UP8's reference.** UP6 keeps its own pictures as a run folder; UP8 compares against that folder, not against the
  frozen references (review C's I1), so a 0.19 change that you did not freeze cannot turn UP8 red by construction.
- **A rest at 0.19.1.** It AMENDS B70 item 1, so it is your ruling (UQ1, UQ2). 0.19.1 is the last 0.19 release: the 0.19.2
  milestone closed with no release, so known fixes such as #25427 ("Prevent windows from closing when a monitor is
  disconnected") never ship for it (DOC, review B). Before any commit of that state as the game's state: the clean window
  reading by the frame rule, and your in-game look.

### 4.11 UP8 — the window on Bevy 0.20

**The new shader language (WESL).** Bevy 0.20 writes its shaders in WESL and removes the naga_oil preprocessor (#25088);
WGSL with no preprocessor line keeps working; there is no conversion tool (DOC; a negative finding). Our eight shaders
(1 708 lines) hold 102 preprocessor lines (51 in `ladder_fade_prepass.wgsl`, 16 in `ladder_fade.wgsl`, 11 each in
`points.wgsl` and `probe.wgsl`) plus their import paths (MEASURED 2026-10-09, review C, grep). The rest is plain WGSL. The
risk is in the meaning (flags on struct members, engine names), not in the volume.

| Shader (lines) | What moves | Its gate |
|---|---|---|
| `star_law.wgsl` (88) | `#define_import_path` goes; the embedded path names the module; `load_shader_library!` still exists (`bevy_shader-0.20.0/src/lib.rs:28`) | star_sky_pixels; the points pass |
| `star_sky.wgsl` (134) | `bevy_pbr::mesh_functions` → `bevy_pbr::render::mesh_functions`; `bevy_render::view::View` keeps its path; `#{MATERIAL_BIND_GROUP}` → `constants::MATERIAL_BIND_GROUP`; the View bound by hand (R8) | star_sky_pixels |
| `points.wgsl` (176) | `bevy_core_pipeline::fullscreen_vertex_shader` keeps its path (only the import syntax changes); the atmosphere imports under `@if(AIR)`, with our patched functions | the temporal disc; motion-truth |
| `ladder_fade.wgsl` (240) | the `bevy_pbr` modules → `bevy_pbr::render::…`; `OCT_NORMAL`, `SPLAT`, `LIGHT_CASTER`, `MOTION_CONTROL_ENGINE` → `@if` | terrain-pictures; terrain-moving-eye; gpu-seam |
| `ladder_fade_prepass.wgsl` (376) | `bevy_pbr::prepass::{bindings, io}`; Bevy's flags on struct members (`:171-185`); `@location(7) instance_index: u32` needs `@interpolate(flat)` (naga 30, `:184`; R29); Bevy 0.20's own prepass output carries it | the far shadows; motion-truth |
| `probe.wgsl` (269) | the three `bevy_pbr` modules → `bevy_pbr::render::…` | the probe |
| `reconstruction.wgsl` (408), `render_picture.wgsl` (17) | the fullscreen import's syntax only | B29's rectification; motion-truth; render-scale; the temporal pictures |

The sky patch's own hunks move the same way (`functions`, `render_sky`, `types`, the change-6 part of `pbr_lighting`, and
`sky_air`, 85 new lines).

**The proof of the translation.** G-PIPELINES-OK on every stand and path; the flag table's gates draw every flag (§2 item
8; review C's M11); the pictures within one level of UP6's run folder, and the probes equal; the GPU clock's per-pass times
within the noise rule. 0.20 brings few quiet pixel changes of its own (R25); Bevy's glam 0.33 brings fused multiply-add and
other f32 changes to Bevy's transforms (§4.5), which may move a pixel by less than a level (ESTIMATE). A larger difference
is a translation defect until it has another cause, and the time box of §4.15 applies.

**The other 0.20 changes in the window** (DOC, reader 2 and review B): wgpu 30.0.1 for the direct surface (`Queue::present`;
`BufferBinding` from a slice; `TRANSIENT_ATTACHMENT`); `ViewDepthTexture` → `ViewDepthStencilTexture` and the depth as a
depth-stencil attachment (`reconstruction.rs:706`, `:727`, `:756`; `motion_capture.rs:35`, `:138`, `:202-219`; R31);
extraction names its world (`#[extract_app(RenderApp)]`, `bevy_extract`) at the six derives, the manual impl and the
`ExtractResource` derives; `MeshTag(x)` → `MeshTag::new(x)` (`terrain.rs:3445`, `:3447`, `:3803`, `:3833`, `:3856`);
`Tonemapping::None` is a full passthrough and `Linear` keeps the old meaning, so the probe camera (`lib.rs:3399`) takes
`Linear` unless the record shows `None` identical; the atmosphere keeps one buffer per camera (UB3); Bevy's built-in render
sets order weakly (#25128), and UP3 item 10's stated waits and schedule tests hold ours (R27); the HUD library moves to 0.43
(UQ2); the glam glue goes; the jitter canary's `Mat4::perspective_infinite_reverse_rh` moves to glam's camera module.

### 4.12 The Tier-A copies of Bevy behaviour (R10)

These copies link no Bevy, so a changed engine does not break their build; they read wrong numbers in silence. Each one is
re-read in the 0.19.1 and 0.20.0 sources, and each one has a canary in `vd-client-render` (Tier-B): first on 0.18 (UP3),
then at UP5 and UP8. If a copy changes, `coverage-fast` runs for its crate (HR5).

| The copy | Where | What it copies | Its canary |
|---|---|---|---|
| The screen dither | `crates/client-harness/src/eye.rs:101-111` | `screen_space_dither`'s constants (171, 231; 103, 71, 97); identical in 0.18.1, 0.19.1 and 0.20.0 (DOC, review B) | An in-process test `App` renders a flat grey with the dither on; each pixel's offset equals the copy's prediction (a local GPU test, not a process gate) |
| One level of pixel noise; the sentinel threshold | `crates/client-harness/src/edge.rs:106-111`; `crates/client-harness/src/picture_census.rs:73-78` | `core_3d/mod.rs:145`, `tonemapping.wgsl:25-30` | The same test |
| The f32 `looking_at` failure; the infinite reverse-Z depth | `crates/client-harness/src/camera.rs:216-219`, `:323`, `:988`; `crates/bins/tests/render_scale.rs:10-` | Bevy's transform and projection; glam's constructors | A CPU test: Bevy's own `looking_at` and clip matrix against the copy |
| The Earth medium and its swap | `crates/client/src/sky.rs:21-42` | `ScatteringMedium::earthlike` (0.18.1); `earth` from 0.19 (`bevy_light-0.19.1/src/atmosphere.rs:196`) | A CPU test reads the preset's terms against the copy |
| The jitter sequence | `crates/client/src/reconstruction.rs:152`, `:168` | Bevy's `TemporalJitter` | The canary at `reconstruction.rs:843`, extended |

### 4.13 The harness: vdctl, the readback, the probes, the switches

- **vdctl** feeds the same `InputAction` mailbox as the window's keys (`lib.rs:3-9`); only the input resources sit on its
  path. The 0.19 `InputFocus` change does not touch it.
- **The readback** (Screenshot, ScreenshotStill, Record; `crates/devproto/src/dispatch.rs:42`, `:51`, `:59`): the copy
  driver becomes a system after `camera_driver`, in the same frame's command stream.
- **G-CAPTURE-SAME-FRAME tests the pixel copy's own cure** (`lib.rs:3979-3986`: "THE COPY RIDES THE FRAME'S OWN COMMANDS
  … MEASURED on the turning leg (2026-09-29): in all 30 pairs of consecutive frames, the SECOND frame's probe matched the
  FIRST frame's depth exactly … and missed its own frame's depth by 25–121 ground pixels along every ridge"). The gate:
  on the turning leg, each frame's probe matches its OWN frame's depth (0 ground pixels off), not the frame before. A
  second, separate check keeps the stamp's cure (`lib.rs:3924-3929`): the overlay's rectangle reads non-zero on every
  recorded frame.
- **The motion copy and its sentinel** (`motion_capture.rs:50-125`): the name and the attachment's form are read at UP5
  and UP8 (R8, R31).
- **The probes**: the probe camera, `ProbeMaterial` and `probe.wgsl` (UP8's table); the probe kinds and the stamp rows live
  in `vd-devproto`, with no Bevy.
- **The switches that the move touches** (each survives with a capture in its mode at the stop and at the close, or goes
  on your word; no new switch; a capture opens no window, so the window-only `VD_WINDOW_*` switches are proven in the
  checklist session of §4.14, on your word at that moment): `VD_SKY`, `VD_SKY_MODE` (the lookup mode keeps upstream's lifted eye, DEFERRED:8680),
  `VD_SKY_ENV_PX`, `VD_SKY_PICK`, `VD_SKY_FLAT` (retires with change 1 under UQ7 (a)); `VD_RENDER_DIAG`; `VD_GPU_CLOCK`;
  `VD_WINDOW_MSAA`, `VD_WINDOW_BLOOM`, `VD_WINDOW_DIRECT_DRAWS`, `VD_WINDOW_FULLSCREEN`, `VD_WINDOW_SCALE`,
  `VD_WINDOW_CONTINUOUS`, `VD_WINDOW_FREE_CURSOR`, `VD_WINDOW_ON_TOP`; `VD_FRAME_CAP`, `VD_CLIENT_UNPACED`; `VD_TEMPORAL`,
  `VD_RENDER_SCALE`, `VD_PREPASS`, `VD_SHARPEN`, `VD_CLEAR_SENTINEL`, `VD_PRICE_G2_TARGETS`, `VD_MOTION_CONTROL`;
  `VD_TERRAIN_LIGHT_CASTER` and the `VD_TERRAIN_SHADOW_*` group (read in today's tree, `lib.rs`, `sky.rs`, `measure.rs`,
  `gpu_clock.rs`, `terrain.rs`).

### 4.14 UP9 — the pictures, the frame, the window

**BETTER, defined (B71; review A-9).** A pair is BETTER when (1) it is better in at least one stated reading, AND (2) it
is worse in no stated reading and in no region (the sky, the sky-lit ground, the bright areas, the HUD's size and scale),
AND (3) the gates find no new defect (no hole, no seam, no pop or flicker by the eye's rule, B27). The stated readings:
the picture census (G-GRID; the sky's bands above the dither's level; the sentinel); the probe beside the reference probe
(no tone mapping, no bloom), so a change of light separates from a change of shape; the on-foot sky light (B31's reading);
the specks under the sphere (nought); the stair on the horizon (none); the haze line at the eye's level (G-AIR-UNDER).
- BETTER: the new reference; you see the pair.
- WORSE (any reading or region): a defect. Its roads: our cure; a Bevy setting; a vendored change (a new patched crate needs
  your word, the library rule); or the previous version (the stop). Never "a matter of taste".
- MIXED, or a pair that no reading separates (for example a dimmer bloom with every reading equal): your look (B71 item 1).

**Example.** At UP5 the dusk stand's sky shows 6 bands where it showed 14 (better), and the dune's lee side under the sky
light reads 3 levels darker through the white-furnace fix (worse in one region). The pair goes to your look; it does not
re-freeze as "better".

**The frame.** The protocol of §2 and the frame rule of §4.15: five runs a mode, the 0.18 build kept from UP0 and the new
build in turn in ONE session, on the same running cluster (the wire and the world identity are equal by UP2's and UP4's
gates).

**The window checklist flight** (at UP6 and UP9, each on your word at that moment; Docker stopped). Every GPU gate uses the
capture, which turns the window plugin off (`lib.rs:3245`), and no test starts the window (review C, grep). So the session
flies: each key that the window binds; mouse look; cursor grab and release; full screen and back; a quick shrink (B34's
cure); an app switch and return; the HUD (its size, its text, its place); one minute of flight at the berth; and the
window-only switches (`VD_WINDOW_MSAA`, `VD_WINDOW_BLOOM`, `VD_WINDOW_DIRECT_DRAWS`, `VD_WINDOW_FULLSCREEN`,
`VD_WINDOW_SCALE`, `VD_WINDOW_CONTINUOUS`, `VD_WINDOW_FREE_CURSOR`, `VD_WINDOW_ON_TOP`), each set once. 0.19 and 0.20
change this path: input dispatch and the UI widgets join the default plugins (#23346); macOS activation follows
`Window::focused` (#24702); the borderless game mode on macOS (#22853); the exit systems move to `Last`, where
`cap_frame_rate` runs (#23624); bevy_egui's zoom keys. A cheaper, repeatable addition: an in-process test that builds the
client's app in memory and sends window events through Bevy's input (no game switch); it loads no window system and
opens no window, as the capture does.

**The flights.** flights, acceptance-flight, warp-pixels, two-ships, world-from-inside, boarding-storm, terrain-moving-eye:
the eye's rule on every leg; B70's example flight in the window on your word at that moment.

### 4.15 THE MOVE CLOSES BY A FIXED LIST

The move is done when every item below is green. Only a defect that a gate on this list finds, or your look, reopens it
(B69's pattern). Every gate of §4.17 names the item that governs it; no gate stands outside this list.

1. **S0**: green, or the road that you ruled for a red (UQ3) or for a FAILED verdict (UQ17).
2. **The record's gates**: `just gate` in full, the gates of §2 item 3, and every new gate once the record holds it, with
   no new red. Each red of the record keeps its exact failure, or we fix it (UQ5).
3. **HR5**: `coverage-fast` green at the end of every phase that touched a Tier-A crate; the io-prod floors.
4. **The pins and the wire**: unchanged. A moved pin takes item 9's time box; we re-record it only on your word.
5. **The orders**: G-ORDER-RECORD equal; no conflicting pair without a stated order.
6. **The pictures**: every stand's pair judged by §4.14 at the stop and at the close, with ONE look of yours at each.
7. **THE FRAME RULE.** Five runs a mode; the old and the new build in turn in one session; the median of the run medians.
   WORSE means: the new median exceeds the old by more than the larger of 2 % and the old build's own spread (the highest
   minus the lowest of its five run medians), the project's noise rule in the form of B69 item 2. The same form, with the
   same threshold, governs every reading that the record holds as a number, never as an absolute gate:
   - the frame's GPU span and the shared instrument (§2);
   - the stall readings at each stand: the share of seconds with a frame over 33 ms, and the count of pipelines compiled
     after loading (G-NO-STALL, G-PIPELINES-OK's count); B34 measured both red on today's build and gave their cure to T4;
   - G-TICK-TIME: the 12k-tick harness gate's wall time, the previous green state's test binary kept and run in turn with
     the new one.
   Worse: the per-pass clock (or the reading itself) names where, and you rule (a cure, or acceptance with a registered
   cause). Not worse: green, and the number goes to T4 as a registered reading.
8. **The window checklist**: every item passes, at the stop and at the close. A failing item with a known cause and no
   cure inside the move: you rule a road (a cure, the stop at 0.19.1, or acceptance with a registered cause), as in item 7.
9. **THE TIME BOX.** Its classes: a stand's pixels; a pass's time; an order; a pin's bit or byte; a gate's red. A timing
   gate that is red once (a deadline) first runs a second time under the load check (`uptime` under 1.5); a second red
   enters the box. Each case gets ONE working day of search for its cause. Then:
   - a cause with a cure inside the move: we cure it;
   - a known cause with no cure inside the move: you rule a road (a cure later, the stop at 0.19.1, or acceptance with a
     registered cause);
   - no cause after the day: we REGISTER it with what we know, and you rule whether it holds the move;
   - a difference that is not a red and not WORSE by §4.14 or item 7: we REGISTER it, and it does not stop the move.
10. **Anything else** that appears: we REGISTER it in DEFERRED, and it does not stop the move.

**The end of a red.** So every red on this list ends after at most one working day of search: in a cure, or in your
ruling. No red stays open as a hunt.

**Example.** On 0.20 the dusk stand reads 0.4 ms slower over five runs a build, inside the old build's own spread. The move
closes, and T4 receives the reading as a registered item.

### 4.16 UP10 and UP11 — the chosen features, and the close

- **UP10.** Each feature that you chose from §5 is its own change, with its own measurement and `coverage-fast` where it
  touches a Tier-A crate, after UP9, so the move's "after" record holds no feature change.
- **UP11.** DEFERRED: the sky patch's entry re-written (changes 2–7 kept; 1, 8 and the ninth by UQ7; the pull request);
  the glam glue gone; the HUD library's pin registered if it is a test release; the k3d proof registered; the wrong
  statement on the capture's MSAA at `docs/design/DEFERRED.md:8700` corrected. PATCH.md re-written for 0.20. The living
  status file changes in the same change as each re-freeze (its items name the references). `docs/design/spikes/bevy-readback`
  (Bevy 0.18, bevy_egui 0.39, not a member) retires or is registered on your word. The session's memory notes (the upgrade
  gate; slice 3's "Bevy 0.18" lock; the gates' times) are updated.
- **Abandoned work.** If a phase stops unfinished, its work goes to its own branch on your word, never out by a checkout.

### 4.17 The gates

Each gate names the item of §4.15 that governs it, and so the end of its red (§4.15, "the end of a red").

| Gate | What goes red | Phases | Governed by (§4.15) |
|---|---|---|---|
| S0's spikes | naga refuses the module; the stamps do not read; the scope does not hold the error; the real module's words differ | S0 | Item 1 (UQ3, UQ17) |
| `just gate` in full (fmt-check, lint, lint-combos, the pins it names, the suite, its process and GPU recipes, coverage with the io-prod floors) | A red that is not on the record's list with its exact failure | UP0, UP1–UP3, UP4 (or its server half), UP6, UP7, UP9 | Item 2; its pins item 4; its coverage item 3 |
| `coverage-fast` (after `cargo llvm-cov clean --workspace`) | A Tier-A miss | every phase that touches a Tier-A crate, and before every commit | Item 3 |
| The gates of §2 item 3 (the GPU gates beyond `just gate`, the other pins) | As today | UP0, UP1–UP3, UP5, UP6, UP8, UP9 | Item 2; the pins item 4 |
| `just terrain-legs` (SL10) | Any digest off on aarch64 Linux or x86-64 (emulated) | UP0, UP1–UP3 | Item 4 |
| G-WIRE-BYTES (new) | Any wire byte of a glam value | UP0, UP1–UP3, UP4, UP7 | Item 4 |
| The ephemeris pin | Any bit | UP0, UP1–UP3 | Item 4 |
| G-ORDER-RECORD (new; order and conflicts) | A server schedule's order or conflict list off its committed list | UP3, UP4, UP7 | Item 5 |
| G-TICK-TIME | The 12k-tick gate's wall time WORSE by the frame rule's form (five runs of each build in turn; the median; the same threshold) | UP4, UP7 | Item 7 |
| G-SERVER-TREE (new) | The server build links the full `bevy`, `wgpu` or `winit` | UP4, UP7 | Item 2 |
| G-SELF-CHECK + its REFUSED control + gpu-drift + gpu-seam + terrain-pictures-card | The verdict not PASSED on the M4 Pro; the control not REFUSED with a running process; a card byte off | UP3, UP5, UP8 | Item 2 (a FAILED verdict: UQ17) |
| G-EXIT (new; a unit test of the mapping, and from 0.19.1 an in-process test `App`; no client switch) | A GPU error ends a run with a success code | UP3, UP5, UP8 | Item 2 |
| G-PIPELINES-OK (new) + the flag table | A pipeline in Err (as in the record: none); a flag that no gate draws and is not marked; the count compiled after loading WORSE by the frame rule's form against the record | UP3, UP5, UP6, UP8 | Item 2; the count item 7 |
| G-LAYOUT (new) | The points pass's copy off the vendored layout | UP3, UP5, UP8 | Item 2 |
| The schedule canaries and the clock census | An order of ours that the schedules do not keep; a pass with no stamp and no name | UP3, UP5, UP8 | Item 5 |
| G-CLOCK-SUM; the two clocks' span | The passes' sum outside the span; the new clock off the shared instrument | UP5, UP8 | Item 7 |
| G-CANARY (new) | A Tier-A copy off the engine | UP3, UP5, UP8 | Item 2 |
| G-AIR-UNDER; the coast and dry-floor stands | The line at the eye's level; the air's look off | UP5, UP6, UP8 | Item 2; the stands' pictures item 6 |
| G-CROSSING (B31) | Over half a level at a boarding | UP6, UP9 | Item 6 |
| motion-truth (+control), render-scale, picture-census (+control) | As today | UP5, UP6, UP8 | Item 2 |
| G-CAPTURE-SAME-FRAME (new) | A probe that matches the frame before; the overlay's rectangle at zero | UP3, UP5, UP8 | Item 2 |
| The pairs (§4.14) | A worse pair; a new hole, seam, pop or flicker | UP5, UP6, UP9; identical at UP1–UP3; within one level of UP6's run folder at UP8 | Item 6 |
| The eye's rule on the flight legs | A pop that the eye sees | UP6, UP9 | Item 6 |
| The frame rule (§4.15) | WORSE by the rule | UP6 and UP8 (headless), UP9 (window) | Item 7 |
| G-NO-STALL | The share of seconds with a frame over 33 ms, or the count of pipelines compiled after loading, WORSE than the record's by the frame rule's form (never "any stall": B34 measured stalls on today's build, MEASURED 2026-10-02) | UP6, UP8, UP9 | Item 7 |
| The window checklist | An item that fails | UP6, UP9 | Item 8 |
| The Docker images; `just image-smoke` | An image does not build or start | UP1–UP3, UP7 | Item 2 |
| The fast window build | `dev-control,render-dylib` does not build | UP5, UP8 | Item 2 |

A control that goes red when it should proves each new gate before any phase trusts it.

---

## 5. EVERY BENEFIT

Each row: what it is; its source (DOC unless marked); what it would change here; its price; the measurement; the
recommendation. Nothing is adopted without your word (B70 item 3; the library rule). The marks: **RECOMMENDED, YOUR CHOICE
BEFORE UP5** (UQ7); **COMES WITH THE MOVE** (no switch exists to refuse it; we measure it, and the pairs judge any pixel it
moves); **BENCH, THEN YOU DECIDE**; **LATER, AT ITS STEP**; **NOT NOW**; **NO**.

| # | The benefit | Source | What it changes here | Price | The measurement | Recommendation |
|---|---|---|---|---|---|---|
| UB1 | The atmosphere as an entity at the planet's centre | 0.19, #23651 | Retires sky change 1 | Small code; the sky gates again | The pairs; G-AIR-UNDER; the coast and dry-floor stands | RECOMMENDED, YOUR CHOICE BEFORE UP5 (UQ7) |
| UB2 | The sky table marched from the true eye | 0.19, #22938 | Retires sky change 8 (B31, B32) | The table's azimuth frame moves some pixels | The on-foot sky light; G-CROSSING; your window look | RECOMMENDED, YOUR CHOICE BEFORE UP5 (UQ7) |
| UB3 | One atmosphere buffer per camera | 0.20, #23113 | Retires the `.single()` limit | None | — (ready for a cockpit screen that draws the air) | COMES WITH THE MOVE |
| UB4 | Clean removal of the atmosphere's components | 0.20, #24884, #25699 | A boarding that turns the sky off and on leaves no stale light | None | G-CROSSING; boarding-storm | COMES WITH THE MOVE |
| UB5 | Rotated and Z-up worlds | 0.19.1, #24998 (review B) | A planet frame with any rotation draws the same sky | Pixels can move at UP5 | The dusk and orbit pairs | COMES WITH THE MOVE |
| UB6 | The phase table's non-linear mapping | 0.19 | Fewer bands in the sky | Pixels move | The dusk sky's bands | COMES WITH THE MOVE; judged by §4.14 |
| UB7 | The white-furnace fixes | 0.19 | Truer sky light on partly metallic surfaces; no seams in generated environment maps | Pixels move | The sky-lit ground's pairs | COMES WITH THE MOVE; judged by §4.14 |
| UB8 | GPU-driven drawing (batched unpacking and depth passes, sparse uploads, parallel mesh collection, change lists) | 0.19 (`many_cubes` 49.47 → 18.77 ms on a laptop RTX 4090) | Gains where little changes a frame; in flight nearly every chunk's transform changes each frame (`terrain.rs:4869-4888`), so the gain can turn into a cost there | A risk to the frame in flight | The frame rule in flight and at a still stand; the GPU preprocessing line | COMES WITH THE MOVE; measured in both |
| UB9 | Per-column change ticks | 0.20, #25157 (Bevy's mesh extraction 132× faster on 1.6 M instances; a mutation about 1.14× slower in parallel) | Bevy's extraction of our chunks; slower writes of our chunks' transforms in flight | A risk to the frame in flight | As UB8 | COMES WITH THE MOVE; measured in both |
| UB10 | `NoCpuCulling` on the chunks | 0.19 | Skips Bevy's CPU visibility work for chunks that already skip its frustum test (`NoFrustumCulling`, 3 sites) | Small code; the shadow casters' lists UNMEASURED | The visibility systems' CPU time; the pictures identical | BENCH, THEN YOU DECIDE |
| UB11 | Occlusion culling, no longer experimental | 0.19, #22631 | Chunks behind a range not drawn | It reads the last frame's depth while our rungs morph: a pop is a seam (B16, B27). The Metal limits pass (31 storage buffers a stage from wgpu-hal 29.0.1; review B) | The eye's rule on every leg | NOT NOW; registered for T4 |
| UB12 | Partial bindless on Metal | 0.19, #23436 (+15 % frame, −57 MB on an M2 Max) | Very likely nothing for the ground: an extended material is bindless only when both its parts are, and `LadderFade` declares no bindless (0.18.1 `extended_material.rs:178-200`; re-read in 0.19.1) | None | The main pass's time | COMES WITH THE MOVE; very likely none |
| UB13 | `Assets::get_mut` marks a change only on a real write | 0.19, #22460 | Per-frame material writes that change no value stop their re-preparation | Small | The render world's prepare time | COMES WITH THE MOVE |
| UB14 | wgpu 29 and 30 on Metal: generated bindings ("resolve leaks and unsoundness"), the queue mutex removed, the one-second delay at a return to the front fixed | wgpu 29.0.0, 30.0.0 | The window after an app switch; memory over a long flight | A rewrite of wgpu's Metal layer under every frame on the only target machine (R54) | G-NO-STALL at an app switch; the frame rule; memory | COMES WITH THE MOVE; a risk as much as a fix |
| UB15 | Render recovery (`RenderErrorPolicy::Recover`) | 0.19 | A lost device rebuilds the renderer instead of ending the flight | Medium: the card builder, the clock's query set and the readback re-attach to a new device | A device-loss drill in an in-process test (that wgpu can drill it on Metal is UNMEASURED) | LATER, on your word (UQ9) |
| UB16 | Bevy's conflict report on built schedules (an existing setting); the schedule shuffle (0.20, `debug` feature) | 0.18 for the report; 0.20, #25094 | The report proves every order we depend on is stated (UP3) | None for the report; the shuffle needs the `debug` feature | G-ORDER-RECORD | The report: ADOPT AS A TEST TOOL (UQ4). The shuffle: NOT NEEDED |
| UB17 | Custom executors (`SystemExecutor` public) | 0.19, #23414 | A shard executor that runs our stated order whatever Bevy's sort does | Medium; G-ORDER-RECORD already guards the order | — | NOT NOW |
| UB18 | Contact shadows | 0.19 | Small rocks and dune ripples shade the ground near an occupant; a hull's interior | A per-pixel march per light; nothing for the far shadows | The GPU clock; your look | LATER, WITH THE LIGHT (B35) |
| UB19 | `KhronosPbrNeutral` tone mapping | 0.19 | A tone curve that keeps base colours | Every picture changes | Your look | LATER, AT G-LOOK |
| UB20 | Contiguous query iteration | 0.19 (5.58 → 4.88 µs over 10 000 entities without AVX2; the M4 has no AVX2) | The per-frame chunk placement; the shards hold no queries | Small | The system's CPU time | BENCH ONLY IF THE CPU FRAME SHOWS IT |
| UB21 | Faster bulk despawn | 0.20 (1.30–1.70×) | A far body's chunks dropped at once | Small | G-NO-STALL at a body's drop | BENCH ONLY IF THE STALL LIST SHOWS IT |
| UB22 | Mesh shaders | 0.20 + wgpu 30 on Metal | A road for the far rungs and the forest | Large research | — | LATER, A RESEARCH ITEM |
| UB23 | Texture compression (`ctt`) | 0.20 | The biome art and the cover (B35) | An asset step | — | LATER, WITH THE ART |
| UB24 | Physically based screen-space reflections; rectangular area lights | 0.19 | The water fill; a hull's lit panels | Look and frame | — | LATER, AT THEIR STEPS |
| UB25 | Chromatic phase functions | 0.19 | A dust's own colour law on the desert world (B23) | A law of the grain first | — | LATER, WITH THE DUST |
| UB26 | Bevy's sky motion vectors | 0.19, #23187 | Could replace our own sky-turn handling | Double motion today | motion-truth | TURNED OFF IN THE MOVE; benched later (UQ8) |
| UB27 | Remote entity reservation | 0.19 | Entities named off the ECS thread | None today | — | LATER, AT SLICE 9 IF NEEDED |
| UB28 | Transmission no longer required on every 3-D camera | 0.20, #25201 | A possible small saving | None | The GPU clock | COMES WITH THE MOVE |
| UB29 | The diagnostics overlay; the render debug overlay; more Tracy spans; the remote schedule graph | 0.19, 0.20 | In-window meters | A second overlay beside the HUD | — | NO |
| UB30 | The crates that Bevy brings: `wesl` 0.6, `bevy_extract`, `bevy_shape`, `bevy_curve`, `parley`, wgpu 30, naga 30, accesskit 0.24 | 0.19, 0.20 | Bevy's own dependencies; our code names only wgpu | Build size and time | The build time; G-SERVER-TREE | LISTED FOR YOUR KNOWLEDGE |

**Considered, with no use today:** Mars presets (a reading only, B38); Solari on macOS (no denoiser); DLSS (NVIDIA only);
vignette and lens distortion; BSN scenes, the new text and UI widgets, feathers (the HUD is egui); observers' run
conditions and hooks on resources; delayed commands (a clock outside the seam); weak ordering for our own systems; catching
panics as errors (its default re-panics); asset saving; GPU light clustering (one sun); HDR display output (not in 0.20).

---

## 6. THE RISKS

R1–R20 are reader 3's, with their numbers; R21–R56 come from the readers and the reviews.

| # | The risk | Where | The guard |
|---|---|---|---|
| R1 | The shard's tick order is Bevy's tie-break; a new Bevy moves it with no error, and two runs of one build stay equal | `register.rs:175-257`; `follower.rs:36-39`; DOC, the undated measurement in `frame_conversion_e2e.rs:3319-3330` | G-ORDER-RECORD with the conflict list; the total order stated on 0.18 (UQ4) |
| R2 | The test rigs run two executors | 10 rig sites | One constructor in vd-sim (UP3) |
| R3 | The 0.19 resource derive emits code that coverage can count at 68 Tier-A sites | the server crates | `coverage-fast` on the first 0.19.1 build; UQ15 |
| R4 | glam rides along: the wire's bytes, the ephemeris bits | `pose.rs:918-924`; the ephemeris pin | UP2 alone at an exact patch; G-WIRE-BYTES |
| R5 | A refused rust-gpu module ends the client (a panic today; Bevy quits it on 0.19/0.20) instead of giving the CPU path | `gpu_check.rs:169-173` | S0.1, S0.3; the error scope and the REFUSED verdict (UP3); G-SELF-CHECK's control |
| R6 | 64-bit shader integers only by Bevy's default priority | `lib.rs:37-48` | The priority stated (UP3); the startup line |
| R7 | The render-graph surface (10 nodes, the graph walk) | §4.8.1 | The node-to-system table; schedule tests |
| R8 | Engine data we change or copy | `motion_capture.rs:93-125`; `points.rs:151-161`; `star_sky.wgsl:19`; `ladder_fade_prepass.wgsl:171-185` | G-LAYOUT; G-PIPELINES-OK; motion-truth; the probe |
| R9 | Shaders fail only when a pipeline is built at run time | 7 of 8 shaders import engine modules | G-PIPELINES-OK on every stand and path |
| R10 | Tier-A copies read wrong numbers in silence | §4.12 | G-CANARY |
| R11 | The default path depends on Bevy's MSAA default | `lib.rs:1484-1490`, `:3433-3448` | MSAA stated (UP3) |
| R12 | The sky patch: 660 changed lines | `vendor/bevy_pbr` | The re-base by change, each with its gate (§4.9) |
| R13 | The window's own order | `lib.rs:1434-1461` | The conflict report; the order stated (UP3) |
| R14 | Plugin order and lifecycle | `lib.rs:1412-1415`; `reconstruction.rs:165-177`; `gpu_clock.rs:420-429` | Orders stated on systems at build; named stamps |
| R15 | Bevy's render diagnostics crashed this Mac | `lib.rs:16-28` | Kept behind `VD_RENDER_DIAG`; headless before any window |
| R16 | The HUD library for 0.20; egui 0.33 → 0.36 | `Cargo.toml:66`; `lib.rs:2934-3025` | UQ2; UP5's egui work; the HUD in the pairs |
| R17 | `-D warnings`: deprecations cannot wait | `justfile:76-77` | Each step moves its deprecated calls in the same phase |
| R18 | The toolchains, the images and the legs' script | `rust-toolchain.toml:5`; both Dockerfiles; `scripts/terrain_legs.sh:18`, `:26`; `justfile:7` | UP1 |
| R19 | The dev-profile names of Bevy's crates | `Cargo.toml:112-123` | G-TICK-TIME; the names re-read |
| R20 | The baseline is not green | §2 | UP0 |
| R21 | Bevy picks the atmosphere with the nearest origin; we pick the largest shell angle | `sky.rs:163-164` | One atmosphere entity at a time; a test holds it |
| R22 | `render_sky` is in a private module | the vendored `mod.rs` | The patch re-exports it; a schedule test |
| R23 | The air passes' spans: none at 0.19.1, upstream's own from 0.20 | `environment.rs` | UQ7's ninth; the parse follows the name |
| R24 | GPU culling on the Mac depends on the storage-buffer limit | `bevy_render-0.19.1/src/batching/gpu_preprocessing.rs:1345-1347` | The limits pass (31 from wgpu-hal 29.0.1; 30.0.1 the usable buffers); the startup line recorded |
| R25 | The quiet pixel changes: bloom's luma linear, the white furnace, the phase mapping, the table's frame, `Tonemapping::None`, the HUD's scale | §4.14 | The pairs by §4.14 — not a blocker by themselves; a worse one is a defect |
| R26 | A GPU error asks for exit code 1, but `run` drops the `AppExit`, so the process exits 0 | `lib.rs:1313-1322`, `:1467`, `:3328`; `client.rs:415-436` | The exit code carried (UP3); G-EXIT |
| R27 | Bevy 0.20 orders its render sets weakly; our waits through atomics, mutexes and channels lose their order | `gpu_clock.rs:315-320`; `pipeline_meter.rs:60`; `points.rs:106`; `lib.rs:1235` | The waits stated (UP3); schedule tests on the built render schedules |
| R28 | The HUD library's test release is built for 0.20.0-rc.1 | `bevy_egui-0.43.0-rc.1/Cargo.toml:160-161` | UQ2; an exact pin only on your word |
| R29 | naga 30 no longer makes integer outputs flat | `ladder_fade_prepass.wgsl:184` | `@interpolate(flat)`; G-PIPELINES-OK |
| R30 | The shader translation is by hand | 102 directive lines + the sky hunks | Within one level of UP6's run folder; the flag table; G-PIPELINES-OK |
| R31 | The depth-stencil renames touch the temporal path's depth | `reconstruction.rs:706-756`; `motion_capture.rs` | motion-truth (+control); render-scale; G-CROSSING on the temporal path |
| R32 | Bevy's sky motion doubles our sky turn | `reconstruction.rs:121-128` | Turned off (UQ8); motion-truth |
| R33 | The capture's same-frame copy | `lib.rs:3979-3986` | G-CAPTURE-SAME-FRAME on the pixel cure |
| R34 | The server images' dependencies | `docker/server.Dockerfile:10-11,31` | G-SERVER-TREE |
| R35 | glam's result changes up to 0.33.12 | the six glam crates | UP2 at an exact patch; the rule for a moved pin |
| R36 | 0.20.0 is new | — | Exact versions in the lock; the 0.20.1 milestone (5 items, none on our path on 2026-10-09) re-read before UP7 |
| R37 | The target directory's link-count wall stalls rustc (memory) | `target/debug/deps` | A fresh target directory for the move's worktree |
| R38 | A reading beside a build is noise | the machine | One job at a time; Docker stopped; load under 1.5 |
| R39 | The local registry holds only 0.18.1 sources | `~/.cargo/registry` | A network fetch at each step, with a neutral identity |
| R40 | 0.19's extraction keeps the render entity when the main component goes | `reconstruction.rs:121-128` | G-PIPELINES-OK; the capture's camera roles checked |
| R41 | `ViewTarget::out_texture` is optional in 0.19 | `gpu_clock.rs:719` | The end stamp skips a camera without one |
| R42 | Resources become entities: a broad query sees them | the window's 30 `Query<` sites | Each broad query checked; `Without<IsResource>` where needed |
| R43 | A newer Bevy runs our window's systems in another order where none is stated | the window's `Update` | The conflict report; the order stated (UP3) |
| R44 | Two glam versions in the window's build from UP2 to UP8 (Bevy's 0.30, then 0.32 from UP5) | `vd-client-render` | The lossless glue, re-checked at UP5, deleted at UP8 (UQ11) |
| R45 | The gates' first solves set the calendar (about an hour a run on generator 26: the home planet's first solve 3 514 s in its shard, MEASURED 2026-10-07) | `pixel.rs:320-325`; `solved.rs:114-127` | The stored worlds and `solve_home` on B64's cores (UQ13); B69's clean timing |
| R46 | naga 29 or 30 refuses the rust-gpu SPIR-V's 64-bit integers, or accepts it while the M4 Pro computes other words (F7) | `gpu_check.rs` | S0.1 first, with the game's validator settings and the real module's words against the CPU; UQ3; UQ17 |
| R47 | The GPU clock's stamps change on wgpu 29's or 30's Metal layer | `gpu_clock.rs` | S0.2 first; the shared instrument |
| R48 | The window and its input path have no gate | `lib.rs:3245` | The checklist flight (UP6, UP9); the in-process input test |
| R49 | A frame regression with no budget | the frame | The frame rule (§4.15); UQ12 |
| R50 | Shader variants that no stand draws compile but can draw wrong | `MOTION_CONTROL_ENGINE`, `LIGHT_CASTER`, `SPLAT`, `AIR` | The flag table; a control for each undrawn flag |
| R51 | 0.20's first month: a new shader language and wgpu 30 | — | Road (b) meets 0.20 later; the 0.20.x notes read before UP7 |
| R52 | Your time: looks, sessions, commit requests | §4.1 | Stated ahead; each prepared before it is asked |
| R53 | Timing gates beside a background first solve (a spawned planet's shard solves on half the cores, `shard.rs:323-328`, for about an hour: 3 514 s, MEASURED 2026-10-07) | `shard.rs:323-328` | The stored worlds; the load check before each timing gate |
| R54 | UB14: wgpu 29 rewrites its Metal layer on the only target machine | wgpu 29.0.0 | The frame rule; G-NO-STALL; memory on a long flight |
| R55 | 0.19.1 is the last 0.19 release | the 0.19.2 milestone, closed with no release | A rest at the stop lives with its known defects (UQ1, UQ2) |
| R56 | The mixed lock at UP4 and UP7 does not build | §3 | The proof build first; else one commit unit with the next window phase |

---

## 7. THE TIMING

**The facts** (DOC, readers 1–2 and reviews B–C, 2026-10-09).
- Bevy 0.20.0 was published on 2026-10-08. No 0.20.1 exists; its milestone holds 5 items, none on the sky, the terrain or
  the window.
- The HUD library has only `bevy_egui` 0.43.0-rc.1 (2026-09-28). It is built for Bevy 0.20.0-rc.1
  (`bevy_egui-0.43.0-rc.1/Cargo.toml:160-161`); that it builds on 0.20.0 is UNMEASURED. Its last stable release, 0.42.0, is
  for Bevy 0.19, and egui moves 0.33 → 0.36 on the way (three versions).
- In the past, bevy_egui shipped its stable release within 0–1 day of four Bevy releases (0.31.0, 0.34.0, 0.37.0 and
  0.40.0; DOC, crates.io, review C). This time it is late so far.
- 0.19.1 is the last 0.19 release (R55).
- The move starts only after S0, after generator 26 closes, and on your word (B70 item 1). On every road, 8.5–12 working
  days of Bevy-0.18 work (S0–UP3) come first: about 1.2–1.7 calendar weeks at the every-day pace (ARITH on §4.1's rows).

**So the HUD library does not separate the roads.** It matters only on the day that the first 0.20 phase starts. The real
timing reason for road (b) is that it meets 0.20 later: UP7 starts 21–31.5 working days (about 3–4.5 weeks, ARITH) after
the move starts, and the move starts on the date that generator 26 closes, which is UNMEASURED; so road (b) meets 0.20 at
least 3–4.5 weeks after its release (ESTIMATE), when 0.20.x can hold its first fixes for a new shader language and for
wgpu 30.

**The options** (for the first 0.20 phase; the start itself is the same on every road: after generator 26 closes and on
your word, with no wait for 0.20.1).
- (a) **A stable `bevy_egui` 0.43.** If it does not exist on that day, the game waits at the stop. Price: an open wait,
  green on 0.19.1, with its known defects (R55).
- (b) **The test release, pinned exactly, on your word.** Price: a release candidate in the game; that it builds on 0.20.0
  is UNMEASURED; a later swap to the stable release (about a quarter of a day, ESTIMATE).
- (c) **Our own copy of the HUD library, fitted to 0.20.** Price: a second patched crate to carry, and a fork that nobody
  chose (the library rule). We recommend against it.

**Recommendation: (a), with (b) as your choice if the wait grows longer than you accept.**

**Example.** Generator 26 closes, and you give the word. S0, the record, the compiler, glam and the written-down orders run
on Bevy 0.18; the shards and the window then move to 0.19.1 and reach the stop. If a stable HUD library for 0.20 exists on
that day, the 0.20 step starts; if not, the game stays green on 0.19.1 until it does, or until you choose the test release.

---

## 8. THE OWNER'S QUESTIONS

Each question has its options, a recommendation and its price. NOW: the answer changes the first phases. AT ITS PHASE: it
can wait. B71 answers the question "do you accept the changed pictures?": the pairs decide by §4.14, and you see them.

1. **UQ1 — THE PATH** (NOW; §3).
   - The days and weeks below are ESTIMATE (review C's bottom-up count with the levers of UQ13 and UQ16, under UQ7 (a);
     UQ7 (b) adds about 2–4 days to each road; §3).
   - (a) Directly to 0.20: 27–43 working days, about 4–6 weeks. (b) Two steps with a green stop at 0.19.1: 31–45 days,
     about 4.5–6.5 weeks. (b2) The shards straight to 0.20, the window through 0.19.1: 30–44 days, a mixed stop. (c') 0.19.1
     now, R2's bake next, 0.20 later: 23–34 days now and 10–14 later; it AMENDS B70 item 1.
   - **Recommendation: (b).** Price: about 2–4 days more than (a) in expectation (ESTIMATE). It halves the suspects of each
     defect and gives a safe place to stop. Choose (c') only if the terrain's start matters more to you than B70's target.
2. **UQ2 — THE HUD LIBRARY AT THE FIRST 0.20 PHASE** (NOW; §7).
   - (a) A stable `bevy_egui` 0.43, or a wait at the stop. (b) The test release pinned exactly, on your word. (c) Our own
     copy of the library.
   - **Recommendation: (a), then (b) if the wait grows too long for you; not (c).** Price of (a): an open wait on a green
     0.19.1 (a rest AMENDS B70 item 1).
3. **UQ3 — IF S0 IS RED** (prepared NOW; asked only on a red S0.1).
   - (a) Stay on 0.18 until naga or rust-gpu changes (measured again at each release). (b) A change on the rust-gpu side,
     so that the same source emits SPIR-V that naga accepts (F7 kept). (c) The card builder off and the CPU path of F8 as a
     stated, temporary state: it AMENDS F7. (d) A WGSL copy of the recipe: refused, because two sources break F7's one
     source and SL10's one generator. (e) A patched naga: a second vendored crate.
   - **Recommendation: (b), measured first (about a day, ESTIMATE); (a) if (b) is large.** Price of (c): the chunk work on
     the CPU's share of the cores only (B20, F6), with its cost UNMEASURED.
4. **UQ4 — THE SHARD'S CLOCK ORDER, AND THE ORDER TOOLS** (NOW, for UP3; R1).
   - Today Bevy runs the clock follower after the shard writes its realm rows, so each row carries the clock value of the
     tick before: one tick of world age on every realm row (DOC: a measurement recorded with no date in the test's
     comment, `frame_conversion_e2e.rs:3319-3330`). The code's comments say the opposite.
   - (a) Keep today's order and write it down; correct the comments; register the other order as its own later change. (b)
     Put the follower first now: every realm row one tick younger; chain-latency's numbers move inside the move.
   - The tools: Bevy's conflict report on the built schedules, in tests only (an existing setting); no schedule shuffle, so
     no build carries the `debug` feature.
   - **Recommendation: (a), and the tools as stated.** Price: none now. Example: the home planet's shard gets its first
     clock sync in tick 40; its rows carry that value from tick 41, on 0.18 and on 0.20 alike.
5. **UQ5 — THE SEVEN KNOWN REDS** (AT UP0).
   - (a) Fix each one before UP1, as its own change outside the move. (b) Keep each exact failure (the assertion, the
     values, the message) in the record; the same failure must stand after each phase.
   - **Recommendation: name them at UP0; then (a) for each red with a small, known cure, (b) for the rest.** Price of (a):
     UNMEASURED until UP0 names them. Price of (b): a second cause can hide behind the same message (small).
6. **UQ6 — THE TEMPORAL PATH'S FIRST REFERENCES** (AT UP0).
   - (a) Frozen after you accept them (B28: a first reference needs your acceptance; B71 compares with a reference). (b) A
     run folder only, compared through the move.
   - **Recommendation: (a).** Price: one more set of pictures for your look.
7. **UQ7 — THE SKY CHANGES THAT BEVY NOW MAKES** (BEFORE UP5; UB1, UB2; review A-3).
   - Change 1 (the stated planet centre) against Bevy's atmosphere entity; change 8 (the sky table's eye: the sky light on
     foot that you accepted in the window, B31, B32) against Bevy's march from the eye; the ninth (our `environment_pass`
     span) against Bevy 0.20's own `atmosphere_environment` span.
   - (a) Try Bevy's form first at UP5. Keep it only if its pairs (§4.14) are not worse on the coast, the dry floor, the dusk
     and the ground stands, and if your window look at the ground stand (on your word at that moment) accepts the sky light
     on foot. A worse pair: our change re-based instead. The ninth: our span at 0.19.1, Bevy's from 0.20. (b) Re-base our
     changes from the start.
   - **Recommendation: (a).** Price of (a): a second port of a change if its pair is worse (about 0.5–1 day each,
     ESTIMATE). Price of (b): about 2–4 working days (ESTIMATE: about 0.5–1 day for each of the two extra changes at each
     of the two re-bases, UP5 and UP8), so road (b)'s totals become 33–49 working days with the levers (36–54 without),
     about 4.7–7 calendar weeks, against 31–45 under (a) (ARITH, §4.1); the patch keeps eight changes and the pull request
     carries eight; G-SKY-CONTROL and `VD_SKY_FLAT` stay.
8. **UQ8 — BEVY'S SKY MOTION ON THE TEMPORAL PATH** (BEFORE UP5; UB26).
   - (a) Turn it off in the move (today's motion stays); bench it later as a replacement for our own sky turn. (b) Adopt it
     now.
   - **Recommendation: (a).** Price of (b): a change of the temporal path's motion inside the move.
9. **UQ9 — A REFUSED GPU MODULE AND A GPU ERROR** (BEFORE UP3; reviews B's B1 and B2).
   - (a) A refused module gives the REFUSED verdict and the CPU builder, and the client runs on; a GPU error keeps Bevy's
     default (it logs, stops rendering and asks for exit code 1), and `run` now carries that code to the process. (b) Also
     Recover: rebuild the renderer after a lost device (UB15). (c) A handler of our own that ignores GPU errors.
   - **Recommendation: (a) now; (b) later on your word; not (c)** (a hidden GPU error; Bevy's own code warns of strobing).
     Price of (a): small (UP3). Price of (b): the card builder, the clock and the readback re-attach to a new device.
10. **UQ10 — THE TOOLCHAIN AND THE CROSS-TARGET LEGS** (AT UP1).
    - Rust 1.97.1 in every place that names the toolchain, `scripts/terrain_legs.sh` included; the legs on both platforms
      with Docker up as their own job; a new rustfmt as its own change; the coverage nightly moves only if it fails.
    - **Recommendation: yes.** Price: the new lints fixed; the images rebuilt; one Docker job.
11. **UQ11 — glam** (AT UP2).
    - (a) glam 0.33.N pinned exactly (0.33.12 today), with the lossless glue in the window until UP8. (b) glam 0.32 at UP2
      (one glam at the stop), then 0.33.N alone before UP7.
    - The rule for both: a moved pin is a finding; we name its cause; we re-record it only on your word.
    - **Recommendation: (a) on road (b); (b) if a rest at 0.19.1 or road (c') is likely.** Price of (b): a second glam
      phase (about 0.5–1 day and part of a campaign, ESTIMATE).
12. **UQ12 — THE FRAME RULE AND THE WINDOW SESSIONS** (AT UP0 and UP9).
    - (a) The frame rule of §4.15 and four sessions (UP0's protocol run; the ground-stand look at UP5; the checklist at
      UP6; the runs in turn, the checklist and B70's flight at UP9), each opened only on your word at that moment. (b)
      Headless readings only: it AMENDS B70 item 4.
    - **Recommendation: (a).** Price: four short sessions of your machine, with you away from the screen during the
      readings.
13. **UQ13 — THE GATES' STORED WORLDS** (NOW, for UP0; R45).
    - (a) A gate's fixture takes realm stores that hold only the artifact rows of one solve, after a digest check: the
      production path of a planet's second boot. One fresh-fixture run a campaign keeps the first-solve path under test, and
      the artifact pin keeps the solve itself. `solve_home` runs on B64's cores (the same bytes on any number). (b) As
      today: every run solves again, on one thread.
    - **Recommendation: (a).** Price: harness code in `vd-bins` (Tier-B), about half a day to a day (ESTIMATE); no game
      switch (the harness places files and picks threads). It saves about 75–140 machine hours (ESTIMATE, review C).
14. **UQ14 — THE BENEFITS** (AT UP9; §5).
    - Accept §5's marks for the opt-in rows (UB10, UB11, UB15, UB17–UB25, UB27, UB29), or change any row.
    - **Recommendation: accept.** Price: the benches of UB10, UB20 and UB21 when their readings call for them.
15. **UQ15 — AN HR5 ROAD, IF NEEDED** (AT UP4; R3).
    - If the 0.19 resource derive puts code at the 68 Tier-A sites that `coverage-fast` reports: (1) tests that reach it
      (each crate's own tests insert each resource once: HR5 discipline (b)); (2) only if it is unreachable by construction,
      the in-source attribute on a dedicated item, measured; (3) a row in `coverage-exemptions.toml` only for a file that
      holds generated code alone, with the recipe wired to read that file (a reviewed gate change: today `coverage-fast`
      does not read the file, `justfile:30-41`).
    - **Recommendation: measure first; then road 1.** Price: tests only.
16. **UQ16 — THE TWO OLD wgpu EXAMPLES** (AT UP5).
    - (a) Retire `crates/bins/examples/integer_bench.rs` and `gpu_spike.rs` (79 direct wgpu sites); their measurements live
      in rulings V17 and F7; DEFERRED records the commit that holds them. (b) Port them at UP5 and again at UP8.
    - **Recommendation: (a).** Price: a restore from history to run them again.
17. **UQ17 — IF THE SELF-CHECK FAILS** (prepared NOW; asked only on a FAILED verdict in S0.1's real run, at UP5 or at UP8;
    review N4).
    - FAILED means: naga accepts the module, but the M4 Pro computes other words than the CPU. The game stays correct,
      because the self-check falls back to the CPU path of the same source (F8); F7's "the GPU does the client's chunk work
      to the maximum" is broken.
    - (a) Hold at the last green state (0.18 before UP5; the stop at 0.19.1 before UP8) until a cure: a change on the
      rust-gpu side (the same source), or a fix in naga or wgpu (a report upstream is an outbound action, on your word
      only). (b) Proceed with the CPU path as a stated, temporary state: it AMENDS F7. (c) A WGSL copy of the recipe:
      refused, because two sources break F7's one source and SL10's one generator.
    - **Recommendation: (a), after §4.15's one-day time box finds the cause; (b) only on your word if the cure is far.**
      Price of (a): the move waits (its length UNMEASURED). Price of (b): the chunk work on the CPU's share of the cores
      only (B20, F6), its cost UNMEASURED.

**A later question, after UP11:** the upstream pull request for the remaining sky changes (DEFERRED owes it). The session
prepares it; you decide whether and how it is sent (an outbound action, on your word only).

---

## 9. WHAT IS UNMEASURED

1. The seven known reds' names and failures.
2. S0's results: naga 29 and 30 on the rust-gpu SPIR-V with the game's validator capabilities and Metal language
   version; the real module's words on the M4 Pro against the CPU's; the clock's stamps on wgpu 29 and 30; the error
   scope on Metal with a cut module that keeps the magic word.
3. That Rust 1.97.1 builds the workspace on 0.18, 0.19.1 and 0.20 with our lints clean, and whether rustfmt changes files.
4. That the coverage nightly (rustc 1.98.0-nightly of 2026-06-06) builds `bevy_ecs` 0.19.1 and 0.20.
5. The cross-target legs on Rust 1.97.1.
6. glam 0.33.N's effect on the wire, the ephemeris bits, the 21 `.inverse()` and 4 `.lerp(` calls of the Tier-A crates,
   and the client's own float paths.
7. That the mixed lock builds at UP4 and UP7.
8. The shards' order and conflicts on 0.19.1 and 0.20 against 0.18's record; the tick cost with resources as entities.
9. Whether the 0.19 resource derive adds coverage regions.
10. The machine hours of a campaign on generator 26 (B69's clean first-solve timing).
11. The clean frame before and after; the gain or cost of GPU-driven drawing and per-column ticks in flight and at a still
    stand; partial bindless on our materials (very likely none).
12. The size of each quiet pixel change on each stand; the probe under `Tonemapping::None`.
13. The atmosphere entity's sky against ours (the pairs); the on-foot sky light with Bevy's table.
14. Each Tier-A copy against 0.19.1 and 0.20 (the dither is identical by source).
15. The HUD under egui 0.36: its size, scale and text.
16. The capture's same-frame copy after the copy becomes a system.
17. Whether the shader translation is exact (within a level of UP6's run folder).
18. Whether wgpu 29's Metal layer changes the window's stalls or memory.
19. The dates of Bevy 0.20.1 and of a stable `bevy_egui` 0.43.0; whether the test release builds on 0.20.0.
20. Whether the server images stay free of the full `bevy`, `wgpu` and `winit`.
21. Whether a refused module can be caught on Metal (S0.3); whether wgpu can drill a lost device on Metal (UB15).
22. Whether `NoCpuCulling` keeps the chunks' shadow casters correct (UB10).
23. The temporal path's pictures (no frozen references exist).
24. The window's input path under 0.19's and 0.20's changes (the checklist).
25. The time that each phase takes (§4.1 is ESTIMATE).

---

## 10. The marks and the sources

### 10.1 The marks

| Mark | Meaning |
|---|---|
| `path:line` | Read in today's tree (`7b756dd` plus its uncommitted changes), by reader 3, by a reviewer, or by this synthesis. This synthesis re-read for revision 2: `lib.rs:1308-1324`, `:1462-1469`, `:3324-3330`, `:3976-3990`; `client.rs:405-440`; `gpu_check.rs:150-176`, `:1296-1345`; `terrain.rs:4866-4890`; `scripts/terrain_legs.sh:14-30`; `justfile:60-72`, `:490-496`; `pixel.rs:318-328`; `solved.rs:110-128`; `artifact_worker.rs:36-42`; `par.rs:18-25`; `shard.rs:320-330`; the switch names; the status file's 2026-10-07 entries |
| MEASURED | A tool printed it, in an earlier run (the record and its date named), or in a read-only grep or diff of 2026-10-09 by a reader, a reviewer or this synthesis. No build, test or instrument ran |
| DOC | A ruling, a design document, a code comment, or a published source that a reader, a reviewer or this synthesis read at first hand on 2026-10-09 (review B read the published crates themselves) |
| ARITH | Arithmetic on marked inputs |
| ESTIMATE | A judgement; the day and hour estimates are review C's bottom-up count |
| UNMEASURED | Nobody printed it (§9) |

### 10.2 The sources

- Reader 1: the release post "Bevy 0.19"; the migration guide 0.18 to 0.19 (103 entries); the crate sources of 0.18.1 and
  0.19.1; the cited pull requests; the bevy_egui, glam and wgpu change logs.
- Reader 2: the release post "Bevy 0.20" (2026-10-08); the migration guide 0.19 to 0.20 (75 entries; reader 2 covered all
  of them, revision 1 miscounted 71); the v0.20.0 tree; crates.io; the wgpu release pages; the 0.20.1 milestone.
- Reader 3: the tree, `Cargo.lock`, `cargo tree --offline`, `diff -r` against the registry's `bevy_pbr` 0.18.1.
- Review B: bevy, bevy_ecs, bevy_render, bevy_pbr, bevy_light, bevy_core_pipeline, bevy_shader, bevy_math, bevy_camera,
  bevy_anti_alias, bevy_post_process, bevy_app, bevy_internal, bevy_image, bevy_extract, bevy_ecs_macros and
  bevy_render_macros at 0.19.0, 0.19.1 and 0.20.0; bevy_egui 0.42.0 and 0.43.0-rc.1; glam 0.33.2 and 0.33.12; wgpu 29.0.4
  and 30.0.1; wgpu-hal 29.0.0–29.0.4 and 30.0.1; naga 30.0.1 — downloaded from crates.io with a neutral user agent; the
  migration guides and release posts as raw markdown; the GitHub milestones.
- Review C: crates.io (bevy_egui's versions), Docker Hub (the 1.97.1 image tag), the bevy and wgpu repositories.
- **The contradiction that revision 1 settled.** Reader 1 read `rust-version = "1.97.1"` for `bevy` 0.20.0; reader 2 read
  `"1.94.0"` for `bevy_math` 0.20. This synthesis fetched https://raw.githubusercontent.com/bevyengine/bevy/v0.20.0/Cargo.toml
  (`"1.97.1"`) and https://raw.githubusercontent.com/bevyengine/bevy/v0.20.0/crates/bevy_ecs/Cargo.toml (`"1.95.0"`; no
  `multi_threaded` by default). The `bevy` crate decides, so the toolchain moves (UP1). Review B confirmed both.

---

## 11. THE REVIEWS' ANSWERS

Every finding of the three reviews, one line each. FIXED: the design changed (where). ASKED: an owner question carries
the decision (which one). ANSWERED: no change, with the reason.

**Review A — the rules and the rulings (3 blocking, 13 important, 11 minor).**

| Finding | Status | The answer |
|---|---|---|
| A-1 (B) SL10: the compiler's move skips the cross-target legs | FIXED | `scripts/terrain_legs.sh:18`, `:26` move at UP1; `just terrain-legs` in the record and the UP1 gate, Docker as its own job; the emulation caveat (§1, §2, §4.4, R18) |
| A-2 (B) a test seam: the shuffle on the GPU gates; the test scenes | FIXED | No shuffle: Bevy's conflict report on built schedules in tests; if wanted, the shuffle only in a crate outside the workspace; every test scene an in-process `App` (§4.6, §4.7, §4.12, UB16, UQ4) |
| A-3 (B) B70 item 3: two benefits built before the owner chooses | ASKED | UQ7 before UP5, with the pairs and a window look; the mark renamed (§5) |
| A-4 HR5 at the phases where Tier-A code changes | FIXED | `coverage-fast` at every phase that touches a Tier-A crate and before every commit (§1, §4.17) |
| A-5 UQ13's exemption road cannot work | FIXED | UQ15 restated with the three roads that HR5 and the exemption file allow |
| A-6 the frame readings not clean by B34; two instruments | FIXED | B34's three causes, load under 1.5, the shared instrument, the runs in turn (§2, §4.8.2, §4.14) |
| A-7 a window opens only on the owner's word at that moment | FIXED | Stated at every session (§0, §1, §2, §4.1, §4.14, UQ12) |
| A-8 the record's pictures: a conditional re-freeze; a first freeze with no acceptance | FIXED | The generator-26 re-freeze in B69's close with its three outcomes; UQ6 by acceptance (§2) |
| A-9 "better" has no definition | FIXED | BETTER defined; the roads for a worse region (§4.14) |
| A-10 the stop as a rest; the glam glue's life | FIXED | A rest AMENDS B70 item 1 (UQ1, UQ2); two glams from UP2 to UP8; the rest's conditions; glam 0.32 offered (UQ11) |
| A-11 UP4 and UP7 leave a crate that does not build | FIXED | The mixed lock stated, with its proof build; else one commit unit (§3, §4.7) |
| A-12 the record is not the pre-merge gate; known reds hide breaks | FIXED | `just gate` in full plus the other gates; the reds named with exact failures (UQ5); the process gates in the UP1–UP3 campaign (§2, §4.17) |
| A-13 G-WIRE-BYTES does not exist | FIXED | Built new at UP0 with a control (§2 item 5) |
| A-14 the lock takes a later glam patch | FIXED | An exact patch; the change list to 0.33.12; the moved-pin rule (§4.5, UQ11) |
| A-15 one constructor in vd-node cannot serve the rigs | FIXED | The constructor in vd-sim (§4.6 item 4) |
| A-16 F7/F8: the self-check measured too late | FIXED | S0.1 before UP0; UQ3 for a red (§4.2) |
| A-17 the order record copies the binaries' order | FIXED | The assembly lifted into library functions (§4.6 item 1) |
| A-18 the switches the move touches | FIXED | Listed, each with its proof (§4.13) |
| A-19 B31: the sky light on foot judged in the window | FIXED | A window look at the ground stand in UQ7 |
| A-20 the HUD's look has no judge | FIXED | The HUD's size and scale join the pairs (§4.8.6, §4.14) |
| A-21 identical or within a level at UP1–UP3 | FIXED | Identical, with the two-run proof at UP0 (§2 item 6) |
| A-22 a pin that a library moves lawfully | FIXED | A finding; its cause named; re-recorded only on your word (§4.5) |
| A-23 citations and statuses | FIXED | The commit rule cited; B69's commit a precondition; B34 AMENDED by B70; UQ12 (b) AMENDS; §7 (c) offered with its price |
| A-24 abandoning a phase | FIXED | Its own branch on your word, never a checkout (§4.16) |
| A-25 "at once" reads as "now" | FIXED | "After generator 26 closes and on your word, with no wait for 0.20.1" (§0, §7) |
| A-26 `rust-toolchain.toml:6` | FIXED | Line 5 |
| A-27 passive sentences | FIXED | The four sentences re-written in the active voice; this revision checked for more |

**Review B — the code and Bevy facts (2 blocking, 5 important, 16 minor).**

| Finding | Status | The answer |
|---|---|---|
| B1 (B) a refused SPIR-V ends the client | FIXED | The error scope, the REFUSED verdict and the CPU path, with a control (UP3 item 11); S0.3; R5 |
| B2 (B) the GPU error default asks for exit 1; our `run` drops it | FIXED | The true default stated (§4.8.7); `run` returns the `AppExit` and `client.rs` maps it (UP3 item 12); UQ9 restated |
| I1 glam resolves to 0.33.12 | FIXED | §4.5, R35, UQ11 |
| I2 the Metal limit is 31 from wgpu-hal 29.0.1 | FIXED | UB8, UB11, R24 corrected; §9 item closed |
| I3 one Bevy version in the lock is false | FIXED | §3, §4.7, R56 |
| I4 G-CAPTURE-SAME-FRAME measures the stamp's cure | FIXED | The gate tests the pixel cure (`lib.rs:3979-3986`); the rectangle a second check (§4.13) |
| I5 the HUD library's test release is for rc.1; egui three versions | FIXED | §7, UQ2, §4.8.6, R16, R28, §9 |
| M1 `ExtractComponent` stays; one manual impl | FIXED | §4.8.1 |
| M2 #24998 is in 0.19.1 | FIXED | UB5, §4.8.3 |
| M3 `earth` is a 0.19 rename | FIXED | §4.12 |
| M4 Bevy's inner glam is 0.32 at UP5 | FIXED | §4.5, R44 |
| M5 the WGSL diff is +380 / −32 | FIXED | §4.9 |
| M6 the 0.19 → 0.20 guide has 75 entries | FIXED | Header, §10.2 |
| M7 the resolved patch versions | FIXED | wgpu and naga 29.0.4; wgpu 30.0.1 (§4.8.5) |
| M8 no `init_pipeline` reads a mesh pipeline | FIXED | The edge removed (§4.8.1) |
| M9 two view systems, with edges already | FIXED | §4.6 item 10, §4.8.1 |
| M10 `gpu_check.rs`'s real wgpu surface | FIXED | §4.8.5 |
| M11 0.20 records the air spans again | FIXED | §4.9 row 9, UQ7, R23 |
| M12 four process gates are debug builds | FIXED | §2 item 4 |
| M13 the fullscreen and View paths do not move | FIXED | §4.11 table |
| M14 0.19.1 is the last 0.19 release | FIXED | §4.10, §7, R55 |
| M15 `AtmosphereData` returns at 0.20 | FIXED | §4.8.4 |
| M16 the canary's `System::type_id`; Rust 1.97.1 and its image exist | FIXED | §4.8.1; §4.4; §9 item closed |

**Review C — the plan and its estimate (3 blocking, 11 important, 15 minor).**

| Finding | Status | The answer |
|---|---|---|
| C-B1 (B) machine time hidden | FIXED | Hours by phase and campaign; the stored worlds and B64's cores (§4.1, UQ13, R45) |
| C-B2 (B) no closing list; gates with no end | FIXED | THE MOVE CLOSES BY A FIXED LIST, the frame rule and the time box (§0, §4.15; UQ12 for the rule) |
| C-B3 (B) the fatal unknowns come late | FIXED | S0 first, with each outcome's effect (§4.2; UQ3) |
| C-I1 UP8's test contradicts B71 | FIXED | UP8 against UP6's run folder (§4.10, §4.11) |
| C-I2 no gate runs the window | FIXED | The checklist flight at UP6 and UP9; an in-process input test (§4.14) |
| C-I3 the timing argument | FIXED | §7 re-written with bevy_egui's history and the real reason |
| C-I4 phases mis-sized | FIXED | Review C's bottom-up rows adopted (§4.1) |
| C-I5 G-SHUFFLE unpriced and against the no-seams rule | FIXED | Replaced by the conflict report; the shuffle not needed (§4.7, UB16) |
| C-I6 two cheap levers on the order | FIXED | UP1–UP3 as one campaign with a bisect rule; a second worktree (§4.1, §4.6) |
| C-I7 the calendar and your time | FIXED | Weeks, looks and sessions stated (§0, §3, §4.1) |
| C-I8 the frame across sessions weeks apart | FIXED | UP0's build kept; the runs in turn in one session; the shared instrument (§2, §4.14) |
| C-I9 benefits' hidden costs | FIXED | UB8, UB9, UB14 marked as risks; UB12 very likely none; R54 |
| C-I10 a cheaper road: 0.19.1, R2's bake, 0.20 later | ASKED | Road (c') in UQ1 with its price (§3) |
| C-I11 large risks missing | FIXED | R45–R56 |
| C-M1 the lockstep sentence; the cheaper server road | ASKED | Road (b2) in UQ1; §3 corrected |
| C-M2 the double work understated | FIXED | §3's table (the Rust hunks too; half porting) |
| C-M3 the direct road counts no bisect | FIXED | 2–5 days of bisect in road (a)'s price (§3) |
| C-M4 the shader work is 102 directive lines | FIXED | §4.11 |
| C-M5 UP1 misses three items | FIXED | The legs' script, fmt-check, the io-prod runs; the tag exists (§4.4) |
| C-M6 the wgpu port misses two examples and a test | ASKED | The 253 sites counted (§4.8.5); UQ16 asks whether two examples retire |
| C-M7 k3d and the image smoke | FIXED | `image-smoke` at UP1–UP3 and UP7; the k3d proof registered (§2, §4.16) |
| C-M8 the fast window build has no check | FIXED | Built at UP5 and UP8 (§4.8.8, §4.17) |
| C-M9 UP11 misses four documents | FIXED | CLAUDE.md (at UP1), the memory notes, the status file, the old spike (§4.4, §4.16) |
| C-M10 `VD_PICTURE_IDENTICAL` not proven on two runs | FIXED | Two runs at UP0 (§2 item 6) |
| C-M11 G-PIPELINES-OK proves compiles only | FIXED | The flag table with a gate for each flag (§2 item 8, §4.11) |
| C-M12 UB5's version | FIXED | 0.19.1, by review B (UB5) |
| C-M13 G-SKY-CONTROL loses its purpose | FIXED | Retired with change 1 under UQ7 (a) (§4.8.3) |
| C-M14 UP0's gate cannot fail usefully | FIXED | B69's rule for a red with no explanation (§2) |
| C-M15 `look_pixels` is no longer slow | ANSWERED | No change to the design; the memory note is updated at UP11 |

**The count.** 79 findings: 74 FIXED, 4 ASKED (A-3 → UQ7; C-I10 and C-M1 → UQ1; C-M6 → UQ16), 1 ANSWERED (C-M15). Every
blocking finding is FIXED (A-1, A-2, B1, B2, C-B1, C-B2, C-B3) or ASKED (A-3).

**The verifier's check of revision 2 (its verdicts: A-1, A-2, B1, B2, C-B1 and C-B3 HOLD; A-3 and C-B2 PARTLY; four new
items N1–N4; and its notes).** Revision 3 answers each item.

| Item | Status | The answer |
|---|---|---|
| V-1 N1: G-NO-STALL and G-PIPELINES-OK as absolute gates, red on today's build by B34 (C-B2's class) | FIXED | Both compare with the record by the frame rule's form (five runs in turn, the same threshold); a pipeline in Err stays a red as in the record (§2, §4.6 item 15, §4.15 item 7, §4.17) |
| V-2 C-B2: G-TICK-TIME has no run count and no statistic | FIXED | Five runs of each build in turn, the median, the same threshold, the previous test binary kept (§4.15 item 7, §4.17) |
| V-3 C-B2: a moved pin and a gate's red have no time box; a timing red has no second run | FIXED | Item 9's classes now hold a pin's bit or byte and a gate's red; a timing red first runs again under the load check; each case ends in a cure or in your ruling (§4.5, §4.15 item 9, "the end of a red") |
| V-4 C-B2: §4.17 and §4.15 do not match | FIXED | Every gate of §4.17 names the §4.15 item that governs it, in one table |
| V-5 C-B2: a checklist item with a known cause and no cure has no road | FIXED | Item 8 ends as item 7 does: you rule a road (§4.15 item 8) |
| V-6 A-3 / N2: UQ7 (b) has no price in days; "six changes remain" stated as a fact | FIXED | UQ7 (b): about 2–4 days, the totals 33–49 beside 31–45 (§0, §3, §4.1, UQ1, UQ7); "at least six, eight under UQ7 (b)" (§4.9) |
| V-7 N3: a truncated SPIR-V makes `make_spirv` panic before any error scope | FIXED | The cut falls on a word boundary and keeps the magic word, or one instruction word is corrupted (§4.2 S0.3, §4.6 item 11) |
| V-8 N4: S0.1 does not state the game's validator settings or prove equal words; no question for FAILED | FIXED | S0.1 states the capabilities (at least `SHADER_INT64`) and the Metal language version, and runs the real module against the CPU's words; UQ17 prepared (§4.2, §4.8.5, R46) |
| V-9 G-EXIT's form | FIXED | A unit test of the mapping and an in-process test `App`; no client switch makes a GPU error (§4.6 item 12, §4.17) |
| V-10 the §2 example opens a window without the words | FIXED | "On your word at that moment" (§2) |
| V-11 the in-process input test and the window-order test | FIXED | Both build in memory, load no window system and open no window (§4.6 item 9, §4.14) |
| V-12 the window-only switches cannot be proven by a capture | FIXED | The checklist session proves the `VD_WINDOW_*` switches, on your word (§4.8.8, §4.13, §4.14) |
| V-13 unmarked time counts | FIXED | S0's hours (ESTIMATE); an hour a run (MEASURED 2026-10-07); the 0.20 meeting time (ARITH and ESTIMATE, tied to generator 26's unknown close); 8.5–12 working days of 0.18 work (ARITH); R45, R53; UQ1's days (ESTIMATE) |
| V-14 MEASURED marks with no date | FIXED | The seven reds: MEASURED 2026-10-04; the clock order: DOC (an undated measurement in a test's comment); the clock's first try: DOC |
| V-15 `wgpu-30.0.1/src/lib.rs:325` is the other branch | FIXED | `:319`, wgpu-core's re-export on native targets (§4.2) |

**The verifier's count.** 15 items: 15 FIXED (V-8 with the new prepared question UQ17).
