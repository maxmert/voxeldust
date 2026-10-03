# The belts' law and the landscape's time — the design (phase 0; ruling B37)

> **Status:** REVISION 2, after two adversarial reviews (a physics and calibration review, findings P1–P14; a rulings
> and code review, findings R1–R17; read-only agents, 2026-10-02). Not yet for building: §6 holds the owner's
> decisions, and §5's bench comes first. Written on `game` at `1caca11` (generator 22).
> **Ruling:** B37 — *"Design now, build in phase 2"*.
> **Marks:** `path:line` = read in the code. [V] = a published number read in its source text in this phase (the
> research agent's extracted texts: Artemieva 2006, Montgomery & Brandon 2002, Hilley et al. 2019, Whipple & Meade 2006,
> and the abstracts with DOIs). [R] = a published number from memory or from an abstract not extracted. MEASURED = printed
> by our own instrument. ARITH = hand arithmetic, shown. UNMEASURED = nobody printed it.

---

## 0. For the owner

**The cause stands (MEASURED).** On the home planet's land, away from closed basins, the mountains are one fifth of
Earth's and the relief about half. The belts barely rise (769 m at most over five billion years), and the rivers turn
them into the lowest land.

**The reviews broke the first draft.** The cure is larger than the two laws B37 named. Four facts decide it:
1. **THE LIMITS MUST GO IN THE SAME CHANGE.** Today the land's height is clamped to 8 276 m over the grid's own radius
   (a drawn share of the law's 8 846 m), which is only 4 685 m over the sea. Two more clamps hold each node's TOTAL
   uplift over the age under that clamp. A belt that rises at Earth's rates takes in tens of kilometres of rock over
   its life while its rivers remove most of it. The clamps would cut that total before any river runs. The bench
   would then measure the clamps, not the laws.
2. **THE EROSION LAW ITSELF IS PART OF THE PROBLEM AT THE 8.2 km GRID (ARITH, P3).** At Earth's uplift rates today's
   law steepens a belt's head nodes to cliffs (about 156 % at one node, held only by the scree law). The steepness
   bound measured on Earth would hold the same nodes near 8 %, exactly at the relief rule's line, and the mountains
   would vanish. A law measured on Earth AT THIS SCALE exists: in a 10 km window, relief grows with the erosion rate
   and stops near 1 500 m (Montgomery & Brandon 2002 [V]).
3. **THE HIGH LAND COMES FROM THE PLATFORMS, NOT THE BELTS (ARITH, P3).** Each plate's crust thickness carries a drawn
   scatter of 0.7–1.4 with no calibration (`land.rs:85-87`). Today a platform's height follows its thickness by Airy's
   law, so plate platforms stand up to 4.9 km apart, and 61 % of the land stands over 1 000 m (Earth 20.5 %). On
   Earth thick old cratons are NOT higher: their cold, dense roots offset the thicker crust (Jordan's isopycnic balance
   [R]).
4. **THE WORLD ALREADY HAS A PLATE SPEED AND AN OCEAN-FLOOR CLOCK** (the craters' law: Earth's mean 50 mm/yr,
   `craters.rs:133-138`, `:311-328`). The belts must read the same ones. Two numbers for one fact are two truths.

**The plan.** One bench: a copy of the macro solve (the existing `router_bench` pattern; nothing in the game changes),
in which each law and each limit below can be varied in the copy only. It reads Earth's numbers, split by cause (belt,
platform, basin wall, crater). Then each decision comes to you one by one, with its numbers.

**Example.** Today a pilot who follows a folded belt of the home planet flies over a lowland 478 m over the sea. The
platforms beside the belt stand a kilometre higher, and the next plate's platform stands up to 4.9 km higher still. The
bench answers which combination of laws puts a range where the belt is, keeps the platforms near each other in height,
and keeps the ranges' relief at Earth's 10 km scale.

---

## 1. The rules this design obeys

| Rule | What it requires here |
|---|---|
| B37 | The belts' law and the landscape's time designed now with benches and Earth's numbers; built in phase 2's macro change before the first 1 km bake; the smooth rebound in phase 1. **The world changes once** — so every limit the new laws would hit must go in the same change (§3 L0). |
| B23 | No caps. The only ceilings are LAWS with a calibration body. A gate's band is a reading. |
| T9 | Every physical number from a law with a calibration body; a seed draw only for an identity choice, an unrecoverable history, or a NAMED scatter around a computed value — and the scatter's RANGE needs its own calibration. |
| B21 | One model, no second truth: one plate speed, one ocean-floor clock, one erosion law per level unless the owner rules otherwise. |
| B23 item 3; B35 item 3 | The same laws on every body; results read on the home planet, the home moon and the desert world. |
| SL5 | No variant world: the bench is a COPY of the passes it varies, an instrument, never a second generator in the game. |
| SL10; ruling F7 | The macro solve runs once in the planet's own shard under the fenced float `Gf` (fixed operations, fenced exp and ln) and ships its artifact; the artifact-pin gate covers it. Every new term states its deterministic form: a fixed operation order, integer time steps that sum exactly to the age, a fixed iteration count where a law is non-linear. |
| HR5, B29 item 3 | `vd-terrain` is NOT at 100 % today (the status file's 111 real misses, 2026-09-25); every file a change touches (`land.rs`, `solve.rs`, `craters.rs`) reaches 100 % region and branch in that change. |
| Owner 2026-10-01 | No switch in the game selects a law; the bench copies what it varies. |
| B6, B36 Q16 | Each step ends with the stands' numbers and pictures; later phases ADD stands, never move one; the living status file and DEFERRED change in the same change. |
| SL6 | Nothing new crosses a realm boundary unless a new charter word is chosen (Q-B1 (b)); that is an ask. |

---

## 2. The measured start (generator 22)

| Reading | Earth without Antarctica | Home planet | Source |
|---|---|---|---|
| mountains, land away from closed basins | 2.39 % | 0.47 % | `phase0_earth_relief.txt` |
| relief at 50 / 90 / 99 % of that land (24.6 km window) | 83 / 550 / 1 383 m | 60 / 258 / 704 m | same |
| land over 1 000 m, that land | 20.80 % | 63.17 % | same |
| the land at or under the sea level (closed basins) | 1.02 % | 3.04 % | same |
| the largest belt uplift over the age; the belts' room | — | 769 m; 2 323 m | `phase0_relief_cause.txt` |
| the belts (≥ 100 m of uplift): footprint, mean uplift, removal, height left | — | 4.5 % of the land; 300 m, 2 524 m, 478 m | same |
| the platforms' spread (Airy from the crust scatter 0.7–1.4 × the freeboard factor 1.3173) | — | up to 4 890 m between plates (ARITH: 6 986 m × 0.7) | `land.rs:85-87`, `:1095-1103` |
| the height clamp at the end: land nodes at its top; sea-floor nodes at its bottom | — | 14 500; 216 501 | `phase0_relief_cause.txt` |

**A link found on the way.** The 216 501 sea-floor nodes held at the clamp's bottom stand 8 276 m under the grid's
radius, so 11 867 m under the sea: the "11.9 km-deep pockets" of ruling B12 are that clamp's floor.

---

## 3. The laws (each item marks what the owner decides, §6)

### L0. The limits go in the same change (R1, P1; an owner decision: Q-B0)

- **Delete both uplift clamps** (`land.rs:1049-1057`; `solve.rs:1754-1756`): a node's total uplift over the age is not
  a height; under the belts' law it is tens of kilometres, most of it eroded as it rises.
- **The relief draw goes** (caps-list row 1, `body.rs:950-966`: the relief set at a drawn 0.5–1.0 of the law's cap):
  the law's own value stands.
- **The peak law becomes a law on the STANDING height over a stated base, applied per node inside the pass loop and
  before the talus** — never one global clamp after the loop (`solve.rs:1592-1604`, `:1867`), which removes mass
  that nothing carries. The base is the owner's choice (Q-B0): the sea (the law's calibration: Everest over the sea), or
  the local lowland. A mountain over its law collapses where it stands, and the talus carries the debris.
- **The ladder band** (caps-list row 2's band term) is sized from the law, so the grid holds what the law allows.
- **Price:** caps-list rows 1–3, which the phase-0 design put in phase 4, move into phase 2's macro change. Without
  them, the world changes twice (B37 item 2).

### L1. One plate speed and one clock

- **The speed.** The world already gives the drift share a unit: Earth's mean plate speed, 50 mm/yr
  (`craters.rs:133-138`; calibration body Earth). The belts read the same: a plate's velocity is its drift (direction
  times share) times 50 mm/yr, and the convergence share `c` (`land.rs:375-389`) becomes `v_n = c · 50 mm/yr`.
- **The check (not a fit).** The bench prints the ridge-weighted mean opening of the home planet's own plates against
  Earth's 53.4 ± 5.9 mm/yr [R, Cogné & Humler 2006] and the distribution of `v_n` at convergent boundaries.
- **The clock.** The craters' per-node clock (`plate_clock_yr`: the distance to the boundary over the plate's speed)
  and Parsons's turnover `t_m = 2·A_oc/C0` [V, Parsons 1982] are one family. `A_oc` is the OCEANIC CRUST's area (P9),
  not the sea's. The bench states one clock that both the craters and the belts read; a belt's crater retention never
  exceeds its own window (R6).
- **Other bodies** (Q-B1): (a) the same calibration for every body with plates until the charter states the mantle's
  facts; (b) a convection law with new charter words (an SL6 ask).

### L2. The belt's influx

- **Mass balance.** The belt takes in crust at `F = v_n · T_in` per metre of boundary a year, `T_in` the INCOMING
  plate's crust (the plate that is consumed or pushed in) [R, the flux balance as in Willett & Brandon 2002 and Whipple
  & Meade 2004/2006].
- **Airy.** The surface rises by `(ρ_m − ρ_c)/ρ_m = 0.1515` of the thickening — the world's own densities
  (`land.rs:73-75`; the rebound's 0.848 is the same pair), so steady erosion balances the thickening exactly.
- **The profile.** Today's bump `(1 − x²)²` of HALF-width `W` integrates to `8/15·W` per side (P11). A collision belt
  (continental on both sides) spreads `F` over both sides; an Andean belt (ocean under a continent) is one-sided, so its
  crest is `15F/(8W)`.
- **The Andean share** (Q-B5): an ocean plate is mostly consumed, and only a share `s` of `v_n` shortens the upper plate
  (Earth: 10–15 of 60–80 mm/yr [V, Norabuena et al. 1998]). A ratio standing in for a law is a T9 matter: a law for
  `s` (the megathrust's shear stress, Lamb 2006 [R]) or Earth's ratio as a stated calibration.
- **The pair rules (R4).** One key per plate pair (the two indices, ordered); ONE crust classification per pair (both
  sides read each plate's affinity, never the node's own byte on one side and the plate's word on the other,
  `land.rs:1010-1023`); `T_in` from the incoming plate; the bench reads the steps where `boundary_of`'s second site
  switches (`land.rs:349-362`).
- **Replaces** the room (caps-list row 53) and the factor `(1 − age/2)` (its only reader, `land.rs:1016`).

### L3. The belt's width and its height (Q-B2: the belt's ceiling)

- **(a) As B37 ruled: only the crust's strength, per node (L0).** The belt's steady size comes from the balance of its
  influx and its erosion: the steady width, crest and thickness scale as powers of the influx over the erosional
  efficiency (Whipple & Meade 2004 [R]; Willett 1999 [R]). That law is the PRIMARY width law (P10).
- **(b) Plus a force balance on the belt's mean height** (Molnar & Lyon-Caen 1988 [R]; England & Houseman 1986
  [R]): `ΔF = ρ_c·g·h·[L + h·ρ_m/(2(ρ_m − ρ_c))]` against the boundary's force. This is a second ceiling, so it needs
  the owner's word (R2). If chosen:
  - the force needs its own calibration: `σ_y·T_e` is a coincidence, not a law (P7). Tibet's 5 023 m [V, Fielding et al.
    1994] gives `F_b ≈ 7.3 × 10¹² N/m` (ARITH), against the 2 × 10¹³ N/m of England & Houseman 1986 [R], which gives
    10.2 km;
  - the width grows only by the influx LESS the erosion's efflux: `dW/dt = (F − F_E)·(ρ_m − ρ_c)/(ρ_m · h_max)` (P2).
    Without `F_E` a median window leaves a plateau 2 900–3 800 km wide (ARITH, P2), against Tibet's ~1 000 km;
  - at the ceiling the profile is a PLATEAU, not the bump (the bump's crest is 15/8 of its mean, so a belt held at
    `h_max` would carry a 10 km crest into the peak law);
  - the lowland `L` and its height are read per plate pair (the home planet's continental crust is 46.1 km times the
    scatter, and two sides' lowlands differ by up to 4.9 km today).
- **The width is computed inside the solve from the eroded field** (R2), not once in `land.rs` from the uneroded one.

### L4. The belts' time

- **Every present convergent boundary is active now**; its belt rose over its own window `t_active` before the present.
- **The window's distribution** (Q-B6): a uniform share of the clock gives a mean near 95 Myr with 70–75 % of belts
  older than 52 Myr (ARITH, P9); Earth's two examples (Taiwan 3–5 Myr, India–Asia ~52 Myr [V]) are not a distribution.
  Options: (a) a law of collision duration (a collision slows and stops as the crust thickens; England & Houseman 1989:
  "about 50 m.y." [R]); (b) a distribution calibrated on a published list of Earth's active belts' ages (owed: none found
  yet).
- **One draw per pair** (R4): the pair's own share, drawn by the pair's key from the body's seed (an unrecoverable
  history) — never the mean of the two plates' shares (the mean of two uniform draws is not uniform).
- **Collision speeds** (P2): the speed share times 50 mm/yr gives relative speeds up to about 100 mm/yr; Earth's
  collisions slow as they proceed. The bench prints the distribution; a slowing law is part of Q-B6 (a).

### L5. The time axis of the passes (Q-B7)

- **The problem (ARITH, R8, P4).** A pass is 125 Myr today; one pass's implicit cut removes about 27/28 of a node's
  excess on a folded belt at 1 m/yr of rain (`solve.rs:1190-1196`; c ≈ 27.3 [P5]), so a belt whose window falls in one
  pass keeps about `window/125 Myr` of its steady relief.
- **The proposal.** 40 passes on a geometric axis from the present: the last pass 0.5 Myr, ratio `r = 1.2109`, the first
  **871 Myr**; the first five passes span 3 081 Myr and the last ten 13.7 Myr (ARITH, P4). The schedule's step counts
  were the 8c design's ask 5 ("part of the world tag", `solve.rs:65-67`), so the axis is the owner's decision.
- **Every cadence in YEARS, not passes** (R8): the climate and the running sea (today every 10 passes: on the new axis the
  first would hold 4.26 Gyr, the W7 defect back, `solve.rs:1774-1778`); the rebound (every 5 passes: the first after
  3.08 Gyr); the talus; the glacial epoch (`epoch_step`, `solve.rs:1771`).
- **Exact integers.** The `Δt` are whole years summing exactly to `age_yr`; each window splits over the passes by an
  exact cumulative share (today's `uplift_step` rule, `solve.rs:742-748`); no selector field on `Schedule` (R8).
- **A law for the last pass's length** (`t_min`): the shortest response time the solve must resolve (P4 suggests the
  belts' own response time; Taiwan's e-folding 1.2 Myr [V]). M-TIME-1 varies the pass count AND `t_min`.

### L6. The platforms (Q-B8, Q-B9)

- **Their height (Q-B8).** Today: Airy from a thickness scattered 0.7–1.4 with no calibration (`land.rs:85-87`), so
  platforms stand up to 4.9 km apart. Earth: cratons 27–40 km thick and Proterozoic crust 40–55 km [V, Durrheim & Mooney
  1994] stand at a mean of 0.3–0.6 km [R]; a melt-depleted, buoyant keel offsets the cold, dense one (Jordan 1978 [V];
  Lee, Luffi & Chin 2011 [V]). Options: (a) the isopycnic law — a platform's height reads its compensation, not its
  thickness alone, calibrated on Earth's continental hypsometry (our Earth instrument); (b) the scatter's range
  calibrated on Earth's platform heights; (c) as today.
- **Their time (Q-B9; B37 item 1 and the D-TERRAIN-10 entry).** Each landscape erodes since its ground was last raised:
  - a plate interior's time is its tectono-thermal age: the plate's drawn age share (an unrecoverable history) read
    through Earth's measured distribution (Artemieva 2006, Table 4 [V]: 30.0 % of continental area younger than
    0.35 Ga, 53.7 % younger than 0.75 Ga, 88.8 % younger than 2.95 Ga), never longer than the body's age;
  - what the time controls on a platform: the erosion's start and the craters' retention;
  - old ranges inside the continents (the Appalachians: 27 m/Myr today [V, Matmon et al. 2003]) need the plates' PAST
    layouts (the Wilson cycle, each layout's belts rising and decaying since): (a) in phase 2 too, or (b) later.
    P3's arithmetic: the present belts' footprint is 4.5 % of the land, so they alone cannot lift the land's 90th
    percentile to Earth's 550 m; old ranges or another source must.

### L7. The erosion law at the 8.2 km grid (Q-B10; decided on the bench's numbers)

- **(i) Today's stream power, `n = 1`, no threshold.** At steady state the rebound makes the erosion equal the
  thickening, and a node's slope is `S = 1.56/√(A in nodes)` at the code's folded-belt erodibility (ARITH, P3): 156 % at
  a head node, 16 % at 100 nodes — cliffs held only by the scree law. Today's law does NOT wear old belts fast: about
  22 Myr, and about 150 Myr of mean decay with the rebound (ARITH, P6).
- **(ii) A bounded steepness** (Hilley et al. 2019 [V]: `k_s < 112 m^0.8` at θ = 0.4 for 96 % of Earth's points, read on
  500 m channels with `A₀ = 10⁶ m²`): `S = 8.29 %` at one node, 6.28 % at two (ARITH, P3) — at the relief rule's 8 %
  line, so the mountains class disappears. A `k_s` read from 8.2 km node means is not Hilley's quantity (P5).
- **(iii) The relief law AT THIS SCALE** (Montgomery & Brandon 2002 [V]): erosion grows with the mean local relief in a
  10 km window, `E = E0 + K·Rz/[1 − (Rz/Rc)²]`, `E0 = 0.01 mm/yr`, `K = 2.5 × 10⁻⁴ mm/yr per m`, `Rc = 1 500 m`;
  active belts hold 1 000–2 000 m of relief; at most 2 % of the land holds `Rz > 1 500 m`. Calibrated on Earth at nearly
  the macro grid's own scale.
- **(iv) Threshold-stochastic incision** (DiBiase & Whipple 2011 [R]; Lague 2014 [R]): non-linear, so it needs a fixed
  iteration count as its deterministic form, and it collides with R2's operator, which is "exact in one ordered sweep"
  (R9; ground_step_design §5.2) — two incision laws at two levels unless R2 takes it too.
- **The comparison itself** (P3): Earth's node means get their mountains from ridge-and-valley relief inside and between
  nodes; a solve where every node is a channel is a different field. The bench compares both the macro field and, once
  R2 exists, the 1 km field averaged to 8.2 km (rule M) against Earth's node means.
- **Merged with F49** (R9): the erosion constant's calibration (`solve.rs:83-88`, whose cited range belongs to
  `m = 0.4`, not `m = ½` [V]) is phase 2's existing item F49, not a new question.

### L8. Trenches, arcs, ridges and rifts (Q-B11)

- All four are shares of today's amplitude (`land.rs:461-476`); with the room gone they have no law. A ridge has
  `v_n < 0`; the trench reaches the clamp's floor today (§2).
- Options: (a) their published laws now — the ridge and the abyssal floor by the half-space cooling of the plate clock,
  the trench by the plate's flexure (caps-list row 28's owed K7, K8); this moves part of B12's parked basins into phase 2;
  (b) a stated interim on today's numbers, registered in DEFERRED, until phase 4.

### L9. Dry bodies (Q-B12)

- Rivers run only on a body with air AND water (`climate.rs:730`): the desert world never erodes by rivers, so only the
  ceilings, the talus and the clamps stop its belts (R10); with L3 (b) and no efflux its plateaus would reach plate
  size.
- The plate law reads no water (`land.rs:258-264`). Whether a dry body has plates is a law question (water and plate
  tectonics, [R]); the owner decides whether the plate law reads the water.
- A one-plate body (the moon): no boundary, no clock (`plate_clock_yr`'s `u64::MAX`), no belt; its field must not move
  (a control).

### L10. The rebound as a smooth flexure (phase 1)

- Each node reads the smoothed coarse field at its own position: a bilinear read of the nearest coarse centres in the
  face's index space; across a face seam through the coarse lattice's neighbour table with the neighbour face's rotated
  axes; at the eight cube corners from the three coarse centres that exist (R16, P14).
- Its gate: the block-step reading (edge against inside) within the control's band, and the total lift conserved to the
  sixteenth (P14). L5 changes its cadence, so it is read again in phase 2.
- Known limit stated, not cured: the load sinks locally and the unload rises regionally; no flexural moat beside a belt.

---

## 4. Hand numbers before the bench (P3)

| Quantity | Value | How |
|---|---|---|
| Airy factor | 0.1515 | (3 300 − 2 800)/3 300 |
| bump area per side | 8/15·W | ∫₀¹ (1 − x²)² dx |
| the draft's example belt: F | 1 400 m²/yr | 40 mm/yr × 35 km (Earth's crust; the home planet's is 46.1 km × the scatter) |
| its thickening, both sides of a 252 km half-width | 5.2 mm/yr at the crest; 2.8 mm/yr mean | the crest 15F/(16W); the mean F/(2W) |
| its mean surface rise | 0.42 mm/yr | 2.8 × 0.1515 |
| the steady slope, `n = 1`, folded belt | 1.56/√(A in nodes) | P3 |
| Hilley's slope at one node | 8.29 % | 112 · (6.7 × 10⁷)^−0.4 |
| the first pass on the geometric axis | 871 Myr | 0.5 · 1.2109³⁹ |
| the platforms' spread | 4 890 m | 6 986 m × (1.4 − 0.7) |
| a belt's crest when its mean is held at 5.6 km | 10.5 km | 15/8 × 5.6 km |

---

## 5. The bench (before any build; nothing in the game changes)

**One copy of the macro solve** (`router_bench`'s pattern: `initial_land`, then `boundary_of` on a re-stated direction
read, `uplift_m`, `province_of`, the pub `Land` fields overwritten, `from_land`, the pass loop copied with
`routing_base` re-stated as `router_variants.rs` does), in which every item of §3 can be switched IN THE COPY:
L0 (the limits), L1–L4 (the belts and their time), L5 (the axis), L6 (the platforms' height and time), L7 (the four
erosion laws), L8 (the margins), L9 (dry bodies), L10 (the rebound).

**Controls (each can fail).**
- The copy with today's laws and limits reproduces the pinned artifact digest (the release gate's pin).
- The land copy reproduces `Land.uplift` and `Land.province` node for node before any change.
- The moon's field does not move under any belt law.

**Readings, each split by cause** (belt / platform / closed-basin wall / crater rim): Earth's set (mountains, relief at
50/90/99 %, land over 1 000 m, the closed basins), the belts' width and height against any ceiling, every limit's hits,
the steps at the plates' switch lines, the solve's time and memory (R14: the per-node pair, distance and rate words cost
35–100 MB over 8.87 M nodes, not kilobytes), and the time axis's convergence (40 against 80 passes, and `t_min` halved).

**Order.** The hand numbers (§4) first; then the copy's control; then one change at a time, in the order phase 2's
macro change already states (M2-2, beside D∞; B11, B12), so every number names its cause.

---

## 6. The owner's decisions (asked one by one, each with its bench numbers)

| # | Decision | The options | What the bench must show first |
|---|---|---|---|
| Q-B0 | The limits move into phase 2 with the belts | (a) yes: the uplift clamps go, the relief draw goes, the peak law becomes a per-node law on the standing height over a stated base (the sea or the local lowland) inside the loop; (b) no | the copy with the new belts and today's limits: how much the limits cut |
| Q-B1 | The plate speed on other bodies | (a) Earth's calibration for every body with plates; (b) a convection law with new charter words (SL6 ask) | the home planet's opening against Earth's |
| Q-B2 | A belt's ceiling | (a) only the per-node strength (B37 as ruled), the steady width law; (b) plus the force balance with efflux and a plateau profile | the belts' heights and widths under (a) |
| Q-B5 | The Andean share `s` | a law (the megathrust's stress) or Earth's ratio as a calibration | the Andean belts' heights under each |
| Q-B6 | The window's distribution and the collision's slowing | (a) a duration law; (b) a published age list | the belts' ages and heights |
| Q-B7 | The time axis | the geometric axis, the cadences in years, `t_min`'s law | M-TIME-1's convergence |
| Q-B8 | The platforms' height | (a) the isopycnic law; (b) the scatter calibrated; (c) as today | the hypsometry against Earth's |
| Q-B9 | The platforms' time and the old ranges | the tectono-thermal age now; the plates' past layouts in phase 2 or later | the relief's 90th percentile with the present belts alone |
| Q-B10 | The erosion law at 8.2 km | (i) today's; (ii) a steepness bound; (iii) Montgomery & Brandon's relief law; (iv) threshold-stochastic (and R2's operator) | Earth's set under each |
| Q-B11 | Trenches, arcs, ridges, rifts | (a) their published laws now (part of B12's basins into phase 2); (b) an interim on today's numbers | the sea floor's depths against Earth's |
| Q-B12 | Plates on dry bodies | the plate law reads the water, or not | the desert world under each |

---

## 7. What the change re-records (R12)

`ARTIFACT_KEY_GENERATION` (`tag.rs:276`, today 21) and the generator version; the release gate's digest and its copies
(`relief_cause.rs`, `router_bench.rs`); `HOME_PLANET_SEA_M`, `HOME_PLANET_OCEAN_SHARE_Q4`, `HOME_IDENTITY_MEASURED`
(`home.rs:31-40`); the unit tests that state numbers in `land.rs` and `solve.rs`; the moon-under-the-home-planet's-words
fixtures (`solve.rs:2552`, `lakes.rs:2049`, the chunk, height, digest and client tests — on that lattice the moon gets
plates and belts); the solved pictures (the owner's look). The stands keep their directions; phase 2 ADDS stands (a
"mountains / folded belt" class stand; a reference stand with a range 30–60 km away, B6 item 2).

---

## 8. Risks

1. The bench's copy drifts from the solve: its control (the pinned digest) must pass before any reading counts.
2. Every law of §3 reads the others: the belts' heights depend on the erosion law, the platforms on the clock, the
   craters on the time. One change at a time, in a stated order, names each cause.
3. The macro field and Earth's node means are different fields (P3); a match on the macro field alone may not be the
   look. The 1 km level (R2) is where the comparison closes.
4. Scope: §3 moves caps-list rows 1–3 and parts of B12 into phase 2 and adds the platforms' law; phase 2 grows.

---

## 9. The stale record found on the way

- `solve.rs:85`: `K₀`'s cited range belongs to `m = 0.4` (Stock & Montgomery 1999 [V]); F49.
- `solve.rs:347`: the envelope report says "scaled"; the code clamps.
- `land.rs:85-87`: the crust scatter's range has no calibration.
- The caps list missed the room (now row 53).
- B12's "11.9 km-deep pockets" are the height clamp's floor (§2).

## Appendix — the review findings and their answers

| Finding | Severity | Answer in this revision |
|---|---|---|
| R1 / P1 the clamps bind first | blocker | L0; Q-B0 |
| P2 widening with no efflux | blocker | L3 (b): `F − F_E`; a plateau profile; collision slowing (L4) |
| P3 no node-scale arithmetic | blocker | §4; L7's four laws; L6's platforms; Q-B9 |
| R2 a second ceiling | major | L3 (b) is Q-B2, not a decision |
| R3 the rock map frozen | major | the province read from the belt law's pair, kind and window where the width is decided (to L2/L3's build) |
| R4 / P11 the pair and the normalisation | major | L2's pair rules; 8/15·W; the one-sided Andean crest |
| R5 / P11 margins with no law | major | L8; Q-B11 |
| R6 two speeds and two clocks | major | L1: one speed, one clock |
| R7 the platforms' time | major | L6; Q-B9 |
| R8 / P4 the axis | major | L5: 871 Myr; cadences in years; integers; Q-B7 |
| R9 the threshold law and F49 | major | L7 (iv); F49 merged |
| R10 dry bodies | major | L9; Q-B12 |
| R11 the missing questions | major | §6 |
| R12 the re-recording list | major | §7 |
| P5 the steepness units | major | L7 (ii) restated; Hilley's quantity named |
| P6 the decay times | major | L7 (i): the code's own 22 / 150 Myr |
| P7 the force's calibration | major | L3 (b) |
| P8 the Airy factor | major | 0.1515 |
| P9 the window's check | major | L4; Q-B6 |
| P10 the circular Taiwan check | major | the steady width law as primary (L3 (a)) |
| R13 HR5 | minor | §1 |
| R14 memory | minor | §5 |
| R15 the bench's route | minor | §5 |
| R16 / P14 the rebound's corners | minor | L10 |
| R17 deterministic forms and registrations | minor | §1, L5 |
| P12 source tags | minor | the marks of this revision |
| P13 the table and the example | minor | §2, §4 |

---

## 10. The bench's numbers (ruling B38; `examples/belts_bench.rs` on `vd_bins::solve_copy`)

**The copy (MEASURED).** The one copy of the macro solve moved from the washboard bench into `crates/bins/src/solve_copy.rs`;
its controls: the washboard bench's `--planet d8` rebuilds the release gate's pinned artifact bit for bit (GREEN) and
differs from `solve_full` in 0 heights and 0 lake levels (GREEN); the belts' bench's `today` case matches the pin again
after every law hook (switched off) was added.

**Run 1 and run 2 (MEASURED, the home planet, 40 passes, the land away from closed basins; Earth 2.39 % mountains and
83 / 550 / 1 383 m of relief at 50 / 90 / 99 %):**

| Case | mountains | relief 50 / 90 / 99 % | the mountains by cause | the belts' land |
|---|---|---|---|---|
| today (even axis) | 0.47 % | 60 / 258 / 704 m | basin walls 74 %, craters 18 %, belts 0 % | median 395 m over the sea |
| the influx belts under today's limits | 2.02 % | 51 / 287 / 1 696 m | walls 41 %, platforms 27 %, craters 17 %, belts 14 % | EVERY belt node at 4 605 m: flat at the clamp |
| the limits as laws, today's belts | 0.97 % | 54 / 270 / 953 m | walls 57 %, craters 37 % | as today |
| the limits as laws + the influx belts (first collapse) | 9.82 % | 88 / 973 / 9 686 m | belts 50 % | median 10.2 km, the largest 185 km (the collapse did not settle) |
| today's laws on the GEOMETRIC axis | 9.60 % | 132 / 829 / 3 036 m | craters 80 % | median 415 m |
| the limits as laws + the influx belts, even axis (settling collapse) | 3.79 % | 49 / 339 / 5 031 m | platforms 30 %, walls 28 %, craters 25 %, belts 16 % | median 8 695 m: flat at the peak law's cap |
| the same on the geometric axis | 15.23 % | 118 / 1 420 / 4 775 m | craters 66 % | median 8 829 m: flat at the cap |

**The belts' law's own readings (MEASURED):** 236 968 belt nodes (104 141 in collisions, 132 827 at Andean margins);
convergence at 10 / 50 / 90 %: 4.7 / 14.2 / 38.0 mm/yr; the ridges 64 294 km long opening at a mean 21 mm/yr
(`C0` 1.35 km²/yr), so the plates' turnover is 449 Myr; the windows at 10 / 50 / 90 %: 19.9 / 74.4 / 266.5 Myr; the
surface uplift rate at 50 % 0.087 mm/yr, the largest 1.36 mm/yr; the uplift over a window: mean 19.8 km, the largest
219 km.

**What the numbers say.**
1. **The limits cut the belts flat** (Q-B0): under today's clamps every influx belt is a plateau at the clamp's height.
2. **Today's solve is not converged in time** (a NEW finding; M-TIME-1 runs at 80 and 160 passes): the same laws on
   another 40-pass axis give twenty times the mountains. The implicit cut of one long step removes less than many short
   steps over the same years, so transients — the craters, stamped at the solve's start and eroded for the whole age —
   survive or vanish by the step sizes. "The passes are a resolution, not a dial" does not hold at 40 passes.
3. **With Earth's uplift rates today's erosion is too weak to hold a belt**: the influx belts pile to the peak law's
   cap (8.8 km over the sea) over their whole width, and a per-node collapse cannot carry that mass away (it did not
   settle in 256 rounds). The mass must go sideways — the belt's widening, Q-B2 — and the erosion constant decides how
   much of it there is (F49).
4. **The world's plate speed is half its own calibration**: "Earth's mean plate speed, 50 mm/yr"
   (`craters.rs:133-138`) multiplies a uniform share whose mean is ½, so the mean plate speed is 25 mm/yr and the ridges
   open at 21 mm/yr against Earth's 53.4 ± 5.9 [R]; the turnover is 449 Myr against Earth's ~190. The craters' clock
   reads the same constant. A correction for the owner (Q-B1), never a knob.
5. **Today's mountains are basin walls and craters**, never belts.
6. **The craters' time**: every crater is stamped at the solve's start and eroded for the whole age, though the crater
   law already chose which craters survive; stamping each at its own age is the landscape's-time law (L6) applied to
   craters.

**Run 3, the time convergence of TODAY'S laws (MEASURED; every pass-counted cadence scaled so its spacing in years is
the solve's own; the land away from closed basins):**

| Passes | mountains | relief 50 / 90 / 99 % | land over 1 000 m | the mountains by cause |
|---|---|---|---|---|
| 40 even (the world today) | 0.47 % | 60 / 258 / 704 m | 63.17 % | walls 74 %, craters 18 % |
| 80 even | 0.11 % | 45 / 190 / 358 m | 54.54 % | walls 91 % |
| 160 even | 0.02 % | 31 / 149 / 312 m | 43.31 % | walls 98 % |
| 40 geometric | 9.60 % | 132 / 829 / 3 036 m | 66.47 % | craters 80 % |
| 80 geometric | 6.12 % | 149 / 624 / 2 316 m | 58.54 % | craters 78 % |

**What it says.** Neither axis converges, and on both the finer the steps the more the same laws erode: the implicit cut
of a long step removes less than the law over the same years. Integrated accurately, today's laws grade the home planet
nearly flat; the relief the world ships is partly a NUMERICAL ERROR of 40 long passes — a dial, against the solve's own
rule (03 §4.6) and ruling B38 ("no artificial changes"). So more passes alone are not the cure: the laws change first
(the belts at Earth's rates over their own windows, each landscape's own time — the craters' included — and the erosion
constant's calibration), and the integration must then converge, measured by a gate (the readings at N and 2N passes
within a stated reading's tolerance). The cost: 160 passes took 311 s against 150 s for 40.

**Run 4, ruling B40 in the copy (MEASURED; the control matches the pin again):**
- **The plates' corrected mean speed** (a unit share 100 mm/yr, so a plate's mean is Earth's 50): the ridges' production
  `C0` = 2.70 km²/yr — Earth's measured 2.7 ± 0.2 [R, Cogné & Humler 2006], a number the correction was NOT fitted to —
  the turnover 225 Myr (Earth ~190–220), convergence at 10 / 50 / 90 %: 9.4 / 28.4 / 76.0 mm/yr, windows 10.0 / 37.2 /
  133.3 Myr. A belt's total influx does not depend on the speed (the rate doubles, the window halves): mean 19.8 km of
  surface uplift, the largest 219 km.
- **The force balance and the widening** settle (no node over the peak law's cap after any pass), but with today's
  erosion the plateaus grow to continent size: half-widths 1 636 km at the median and 3 511 km at the largest (Tibet
  ~500 km), 29.9 million km² within 100 m of their ceilings, and 1.09 × 10¹⁷ m³ of rock still without room. Mountains on
  the land away from closed basins: 3.15 % (even axis), 16.54 % (geometric axis, craters 64 % of them).
- **So the erosion constant decides** (F49): on Earth the rivers remove most of what the plates push in (the India–Asia
  balance loses roughly half), and today's constant removes too little. Its calibration is the next item, by a published
  law with its calibration body (ruling B38), never a value turned toward these readings.

**Run 5, the craters' own time (MEASURED; the control matches the pin again):** stamping each crater in the pass where
its impact falls moves today's laws from 0.47 % to 0.77 % mountains on the land away from closed basins (craters 24 % of
them, from 18 %), and B40 from 3.15 % to 2.72 %; the plateaus do not change. Most of the population is billions of years
old, so it is stamped early either way. The crater law already chooses which craters survive by each province's
retention age, and the solve's erosion then wears them again: one of the two is a second truth for the same survival
(ruling B21), a question for the build.

---

## 11. The erosion law's calibration (F49): the research (2026-10-03; a read-only agent, its extracted sources and its
arithmetic `predict.py` in the session's scratchpad `sources/`)

**Facts that frame it.**
1. In the solve's convention a surface uplift of 0.5 mm/yr is the Airy rise of a thickening crust; the rebound gives back
   0.848 of each cut, so the STEADY EROSION is 0.5/0.1515 = 3.3 mm/yr — Taiwan's rate (ARITH). The bench's belts (0.1–
   1.4 mm/yr of surface uplift) need steady erosion of 0.66–9.2 mm/yr.
2. `K₀` is a rate at 1 m/yr of rain: the cut reads `E = K₀·q·√(P/1 m yr⁻¹)·√A·S` (ARITH, `solve.rs:1190-1196`).
3. **The cited calibration is misread** (`solve.rs:85`): Stock & Montgomery 1999's values are at `m = 0.4` [V]; converted
   to `m = ½` at one node's area (×0.165) granitoids read 7.3 × 10⁻⁸–7.1 × 10⁻⁷ /yr and mudstones 7.8 × 10⁻⁵–1.2 × 10⁻³
   /yr [V + ARITH]. Today's hard-rock value (1.8 × 10⁻⁷) sits in the granitoids' band; its shale value is 40–600 times too
   low.
4. **Every global compilation finds `n > 1`**: the median 2.43 (Harel, Mudd & Attal 2016, 33 zones) [V]; 2.2 (Adams et al.
   2020, 142 basins) [V]; Harel publish no `n = 1` constant. With `n > 1`, no single `n = 1` constant holds both a 3 mm/yr
   belt and a slow platform.
5. **At `m/n = ½` the relief depends on the grid** (Kwang & Parker 2017 [V]): each halving of the node adds `(U/K)·ln 2`;
   the macro relief is "U/K times a logarithm", not Earth's relief. With `θ = m/n < ½` it does not.
6. **The node is far coarser than a hillslope** (`√(D/K)` ≈ 100 m, Theodoratos et al. 2018 [V]): every hillslope and
   first-order channel lies inside one 8.2 km node. Earth's node means hold that inner relief; the macro channel heights
   do not.

**The options (each a published law with its own calibration body; the owner chooses — ruling B38):**

| # | Law | Calibration body | A belt like Taiwan (crest over its outlet) | An old platform | Weakness |
|---|---|---|---|---|---|
| F1 | today's stream power, `n = 1`, `m = ½`, the cited range read right | SE Australian rivers over ~20 Myr | 11–111 km (absurd; today's value 45 km — the bench's belts at the cap) | flat in 13–26 Myr | grid-dependent; contradicted by every compilation; NOT recommended |
| F2 | stream power from ONE joint calibration: `E = K_lp·(P·A)^m·S^n`, `m = 1`, `n = 2.2`, `θ = 0.45`, `K_lp = 2.2 × 10⁻⁹ m⁻²` (Adams et al. 2020) [V] | Bhutan and central-eastern Nepal, 142 basins, ¹⁰Be | about 3.8 km, in head-node steps of about 1.5 km (nearly free of the grid: `L^0.09`) | low relief survives: 1 000 → 100 m in 8.6 Myr, → 10 m in 144 Myr | one region; outside its data under 0.02 mm/yr and 0.7 m/yr of rain; strong rock, so the rock shares must be referenced to gneiss; `K ∝ P` makes deserts steep |
| F3 | each node's INNER relief from the 10 km law: `E = 0.01 + 2.5 × 10⁻⁴·Rz/[1 − (Rz/1 500)²]` (Montgomery & Brandon 2002) [V] | Earth's continents and the Olympics | about 1.0–1.4 km of ridges and valleys inside each belt node | 40–180 m | `Rz` is a 1 km-grid statistic inside a 10 km circle (Earth must be read the same way); scatter of one order; no rain term; it moves no sediment, so it rides beside a channel law |
| F4 | threshold-and-flood incision (Tucker 2004; Lague 2014; DiBiase & Whipple 2011) | San Gabriels, the Himalaya, Eastern Tibet, the Andes | `n ≈ 2` at steady state; no closed form | stalls under the threshold (about 20 times longer) | needs flood statistics the climate does not yet give; collides with R2's one-sweep operator |
| F5 | Hilley's steepness limit | 96 % of Earth's points | — | — | an observation, so under B23 and B38 a CAP, not a law |

**The research's reading (not a decision):** drop F1; for the routed channels between nodes take `K`, `m`, `n` from ONE
joint calibration with `θ < ½` (F2 is the cleanest published set); give each node its inner relief from the 10 km law
(F3), for R2 to draw and for the comparison with Earth's node means; let F4 replace F2 when the weather step supplies flood
variability. The `n > 1` sweep needs a fixed Newton count as its deterministic form, and R2's operator must take the same
law (two incision laws at two levels would be a second truth, B21).

**Runs 6 and 7, F2 and F3 (MEASURED; the control matches the pin; the land away from closed basins; Earth's own 10 km
relief, Montgomery & Brandon's `Rz` read on the ETOPO 2022 60 arc-second grid, Antarctica left out: 69 / 562 / 1 377 m at
50 / 90 / 99 %, 2.93 % of the land over 1 000 m, 0.68 % over 1 500 m — within the paper's own "under 5 % and 2 %"):**

| Case | mountains | relief 50 / 90 / 99 % | inner relief `Rz` 50 / 90 / 99 % | the plateaus |
|---|---|---|---|---|
| today's laws, even axis | 0.47 % | 60 / 258 / 704 m | 0 / 0 / 0 m (the 125 Myr last pass averages today's rates away) | — |
| today's laws + F2, even axis | 0.89 % | 105 / 327 / 981 m | 0 / 0 / 0 m | — |
| B40 + the craters' time + F2, even axis | 3.73 % | 104 / 576 / 2 501 m | 0 / 0 / 64 m | 30.4 million km² at their ceilings |
| today's laws + F2, geometric axis | 10.82 % (craters 81 %) | 161 / 960 / 2 850 m | 0 / 53 / 435 m | — |
| B40 + the craters' time + F2, geometric axis | 20.68 % (craters 59 %) | 252 / 1 536 / 3 406 m | 0 / 187 / 1 248 m; 1.94 % over 1 000 m | 27.2 million km² at their ceilings |

**What they say.**
1. **F3 works where a pass is short**: in the belts the inner relief reaches Earth's top percentile (1 248 m against
   1 377 m), but half the land holds none, because most of the planet barely erodes today.
2. **Neither 40-pass axis converges** (ruling B39): the geometric axis's first pass (871 Myr) under-erodes the old craters,
   so craters still make most mountains there; the converged answer needs more passes on both axes.
3. **The plateaus stay continent-sized under every erosion law**: the bench keeps ALL the crust a collision takes in for
   windows up to 133 Myr. Two laws are missing — the share of the incoming crust that goes down into the mantle or out
   sideways (the India–Asia mass balance, RECALLED as roughly half, NOT in the extracted sources, unverified), and how a
   collision slows and stops (the design's Q-B6).

**Run 8, the time convergence of the current best law set (B40 + the craters' own time + F2; MEASURED; the land away
from closed basins):**

| Passes | mountains | relief 50 / 90 / 99 % | land over 1 000 m |
|---|---|---|---|
| 40 even | 3.73 % | 104 / 576 / 2 501 m | 76.08 % |
| 80 even | 4.05 % | 116 / 598 / 2 908 m | 71.61 % |
| 160 even | 3.60 % | 113 / 555 / 2 329 m | 65.21 % |
| 40 geometric | 20.68 % | 252 / 1 536 / 3 406 m | 78.18 % |
| 80 geometric | 19.71 % | 278 / 1 529 / 3 411 m | 73.48 % |

**What it says.** With the joint law the mountains and the typical relief nearly settle on the even axis (unlike today's
`n = 1` laws, which lost most of their relief per doubling), but the land's height still falls with each doubling, and
the two axes settle toward DIFFERENT answers: the geometric axis is fine near the present but its first passes are still
~390 Myr at 80 passes (ARITH), where old craters escape erosion (craters make 56 % of its mountains); the even axis is fine
in the past but its last pass is long (31 Myr at 160), so it can neither resolve a young belt nor read today's erosion
rates (its inner relief stays near zero). **The gate (ruling B39) needs short passes everywhere**: a mixed axis —
geometric from 0.5 Myr at the present up to a longest pass of a few tens of Myr, then even passes back to the start —
about 200–250 passes (ARITH), a solve of roughly 8 minutes against today's 2.5.

**Run 9, the mixed axis (MEASURED; 214 passes: 19 from 0.5 Myr at the present, each 1.25 times the next, up to 25 Myr,
then even 25 Myr passes back to the start; the land away from closed basins):**

| Laws | mountains | relief 50 / 90 / 99 % | land over 1 000 m | inner relief Rz 50 / 90 / 99 % | solve |
|---|---|---|---|---|---|
| Earth (the instrument) | 2.39 % | 83 / 550 / 1 383 m | 20.8 % | 69 / 562 / 1 377 m (Earth's own Rz) | — |
| today's | 0.75 % | 69 / 229 / 675 m | 35.72 % | 0 / 6 / 71 m | 560 s |
| B40 + the craters' own time + F2 | 16.21 % | 202 / 1 274 / 2 905 m | 62.71 % | 0 / 99 / 1 084 m | 638 s |

The mountains by cause (B40 + F2): craters 47.9 %, belts 24.2 %, the platform 20.3 %, basins' walls 7.6 %. The plateaus
stay continent-sized: 14 convergent pairs, the half-width at 50 % 1 847 km and the largest 3 509 km, 24.81 million km² of
land within 100 m of its ceiling; the belts' land 13.29 million km² at a median 5 508 m (Tibet: about 2.5 million km² at
5 023 m). Each belt node's uplift over its window: a mean of 19 833 m, the largest 219 001 m.

**What it says.** On the mixed axis the current best law set sits between the two axes of run 8 (16.2 % against 3.6 %
even and 19.7 % geometric). Its inner relief is Earth's kind: 1.23 % of the land with Rz over 1 000 m and none over
1 500 m (Earth: under 5 % and under 2 %). The plateaus are the defect left: every collision keeps all its crust and closes
at its full speed for up to 133 Myr. The two missing laws (§12) are the next bench (run 10), and the gate (ruling B39)
compares the mixed axis with a twice-as-fine one (0.25 Myr up to 12.5 Myr, 414 passes).

---

## 12. The belts' mass budget: the research (2026-10-03; a read-only agent; its sources in the scratchpad's `sources2/`)

**What a collision keeps (A).** India–Asia is the only collision with a full budget: about 50 % of the pre-collision
crust is missing (Ingalls et al. 2016 [V abstract]); India kept about 19 of its 38 km, 40–50 % went into the mantle below
a decoupling level 15–19 km down, while Asia kept 97 % (Replumaz et al. 2010 [V abstract]); upper-crust shortening is only
30–50 % of the 2 860–3 600 km of convergence since 52 Ma (van Hinsbergen et al. 2012 [V]); the Alps kept "only a very
small percentage" (Schmid et al. 1996 [V]), the Pyrenees lost their lower crust and mantle lithosphere (Beaumont et al.
2000 [V]), the Zagros shortened less than a third of their convergence (Hatzfeld & Molnar 2010 [V]). The physical form:
the cold mantle lithosphere carries down about 10 km of crust (Molnar & Gray 1979 [V]); crust thicker than ~15 km jams a
trench (Cloos 1993 [V]).

**How a collision slows (B).** India 110–137 → 32–50 mm/yr over 50 Myr; Arabia's e-folding 46–68 Myr (ARITH from [V]
tables). Two laws on record: the plate force balance — the plateau's gravitational potential energy resists the plate
against a linear basal drag (Copley, Avouac & Royer 2010 [V]) — and a constant strain rate, `v = v₀·e^(−t/45 Myr)`, Tibet
the calibration body (Clark 2012 [V abstract]).

**The ages of Earth's active belts (C).** Onsets 6–52 Ma, a median near 30 Myr (a list built from single sources [V]; no
published compilation found); the bench's 90 % window (133 Myr) is 2.5 times Earth's oldest active collision.

**Ocean under a continent (D).** Only 15 % of 62 segments have an upper plate in compression (Heuret et al. 2011 [V]);
back-arc shortening only where the slab dips under ~31° (Lallemand et al. 2005 [V]); a kinematic law,
`s = max(0, V_up − V_t)/v_n` with `V_t = 0.5·V_sub − 2.3 cm/yr` (Lallemand et al. 2008 [V abstract]); a force law, the
megathrust's shear stress holding a belt of at most ~2.5 km at 7–15 MPa (Lamb 2006 [V]; Dielforder et al. 2020 [V]). One
share on every margin overstates the Andean belts about six times on average (ARITH).

**The options (E; each the owner's):**

| # | Law | Calibration | Effect (ARITH, India–Asia over 52 Myr; Tibet ~2.5 million km²) | Weakness |
|---|---|---|---|---|
| P1 | the buoyancy budget: the incoming plate's cold root carries down `ρ_m·α·ΔT·L/(2Δρ_c)` ≈ 6–11 km of crust | rock densities, thermal expansion, lithosphere thickness | the plateau 3.4–7.7 million km² | needs each body's lithosphere facts |
| P2 | the decoupling level: the belt keeps the crust above a weak horizon (India: 15–19 km, k ≈ 0.4–0.5) | India–Asia | 3.7–4.6 million km² | one collision |
| S1 | the plate force balance (the plateau resists, the drag is linear) | India before and after | 2 825 km of convergence in 50 Myr (Earth 2 860–3 600) | plate speeds become solved, not drawn |
| S2 | the constant strain rate, `v₀·e^(−t/45 Myr)` | Tibet | a window's influx 0.68 / 0.59 / 0.32 of the constant-speed influx at 37 / 52 / 133 Myr | one collision; `v_n` becomes the speed at onset |
| W2 | a collision's lifetime (~50 Myr; slab break-off at 10–25 Myr in models) | models; England & Houseman 1989 | the 90 % window 133 → ~50 Myr | — |
| A1 | the kinematic margin law | 159 transects | `s = 0` on most margins | needs a mantle frame and each slab's width |
| A2 | the megathrust force balance | Northern Chile, global heights | a margin's ceiling ≤ 2.5 km at 7–15 MPa | the stress reads the trench's sediment |

**The research's reading (not a decision):** P1 the most physical, P2 the simplest; S1 the right slowing law, S2 a
one-constant interim; W2 before a list; A1 for the rate and A2 for the ceiling. **With P2 (k = 0.45), S2 and W2, the
median belt keeps about 31 % of its crust and its plateau's half-width falls from 1 636 km to about 500 km — Tibet's own
(ARITH).**

**Run 10, the two laws benched (MEASURED; `examples/belts_bench.rs`, the copy's `Laws::crust_partition` and
`Laws::collision_slowing`; the control `today` MATCHES the release gate's pin; the land away from closed basins; the
mixed axis of 214 passes unless stated).** P2 keeps the incoming crust above India's decoupling level, 19 km carried by
the elastic thickness (20.0 km on the home planet), so a collision keeps 31 / 35 / 42 % of its crust at 10 / 50 / 90 % of
its nodes — less than India's 50 %, because the freeboard law makes the home planet's crust thicker than Earth's. S2
closes a collision's window at `τ·(1 − e^(−W/τ))` years of its starting speed, `τ` = 45.1 Myr: 0.32 of the constant speed
on the oldest windows (133 Myr) and 0.90 on the youngest (10 Myr).

| Laws (with F2, the joint law) | collisions' rock | Andean margins' rock | mountains | relief 50 / 90 / 99 % | land over 1 000 m | plateau land |
|---|---|---|---|---|---|---|
| neither (run 9) | about 1.5 × 10¹⁷ m³ (ARITH) | 1.25 × 10¹⁷ m³ | 16.21 % | 202 / 1 274 / 2 905 m | 62.71 % | 24.81 million km² |
| P2 only | 5.56 × 10¹⁶ m³ | 1.25 × 10¹⁷ m³ | 14.21 % | 197 / 1 199 / 2 753 m | 60.24 % | 17.51 million km² |
| S2 only | 7.91 × 10¹⁶ m³ | 1.25 × 10¹⁷ m³ | 15.75 % | 199 / 1 261 / 2 970 m | 61.30 % | 21.68 million km² |
| P2 and S2 | 2.91 × 10¹⁶ m³ | 1.25 × 10¹⁷ m³ | 14.09 % | 196 / 1 212 / 2 771 m | 58.67 % | 13.81 million km² |
| P2 and S2, the solve's own erosion (`n = 1`) | 2.91 × 10¹⁶ m³ | 1.25 × 10¹⁷ m³ | 9.65 % | 128 / 895 / 2 720 m | 54.93 % | 16.54 million km² |

**The gate (ruling B39), P2 and S2 on 214 and 414 passes:**

| Erosion | passes | mountains | relief 50 / 90 / 99 % | land over 1 000 m | plateau land | Rz at 90 / 99 % | crater mountains |
|---|---|---|---|---|---|---|---|
| F2, the joint law | 214 | 14.09 % | 196 / 1 212 / 2 771 m | 58.67 % | 13.81 million km² | 115 / 1 039 m | 48.0 % of all |
| F2, the joint law | 414 | 13.10 % | 176 / 1 160 / 2 657 m | 50.40 % | 11.95 million km² | 196 / 1 183 m | 35.8 % of all |
| the solve's own | 214 | 9.65 % | 128 / 895 / 2 720 m | 54.93 % | 16.54 million km² | 71 / 612 m | 41.7 % of all |
| the solve's own | 414 | 9.93 % | 102 / 901 / 2 687 m | 45.01 % | 13.84 million km² | 88 / 815 m | 23.4 % of all |

**What it says.**
1. **The two laws act as their sources say**: together they cut the collisions' rock to a fifth (2.9 × 10¹⁶ against
   about 1.5 × 10¹⁷ m³) and the plateau land by almost a half. The mountains barely move (16.2 → 14.1 %).
2. **The Andean margins are now the belts' largest source**: 81 % of the belts' rock (1.25 × 10¹⁷ m³), one shortening
   share on every margin for up to 133 Myr — the research's (D) found this share about six times too large on average.
   Their land, 7.2 million km² at a median 4.3 km, is larger than the collisions' (5.9 million km² at 3.6 km).
3. **THE GATE FAILS ON THE LAND'S HEIGHT**: with twice the passes the mountains move by 0.3–1.0 point, but the land over
   1 000 m falls by 8–10 points under both erosion laws, the plateaus by 13–16 %, the crater rims' mountains by a quarter
   to a half, and the inner relief's rate reading moves by up to 70 %. The even axis drifted the same way (run 3: 63 /
   55 / 43 % at 40 / 80 / 160 passes, the steps growing). So a finer step still erodes more, and the land's height is not
   yet a property of the laws. The cause is not measured; it comes before any decision that reads these numbers.
4. **The craters**: 42 % of the land lies within one diameter of a crater, and the crater cause takes 23–48 % of all
   mountains. The cause order counts a mountain near a crater as the crater's before the belt's, so this share holds
   belt mountains too; G-CRATER (ruling B3, about 190 structures on Earth) is the reading to check.

---

## 13. The Andean margins' law: the research (2026-10-03; ruling B41 item 1; a read-only agent, its sources in the
scratchpad's `sources3/`, its index `sources3/INDEX.txt`)

[V] read in full; [A] abstract only; ARITH the agent's own arithmetic.

**A1, the rate (Lallemand et al. 2008 [V], 166 oceanic transects).** The neutral line in the HS3 frame is
`Vup = 0.5·Vsub − 2.3` cm/yr (slope fitted 0.46, R² 0.37); compression lies below it, extension above. Its signs: every
speed "counted positive landward", `vd = Vup − Vt` positive for EXTENSION, so the shortening rate is `Vt − Vup` — **the
design's §12 form `s = max(0, V_up − V_t)/v_n` holds only in the older papers' signs (Lallemand 2005, Heuret & Lallemand
2005) and gives `s = 0` at Peru with the 2008 numbers (ARITH); the 2008 form is `s = max(0, 0.5·Vsub − 2.3 − Vup)/vc`.**
The line depends on the frame (NNR `Vup = 0.32·Vsub − 1.9`, R² 0.23; SB04 `0.24·Vsub − 1.4`). **It is a sign rule, not a
rate law**: of 152 transects re-read (ARITH), all 36 compressional ones lie below the line, but so do 48 of the 76
neutral ones — 58 % of the transects stand on the compressive side and 24 % compress; the distance from the line
explains R² 0.15 of the geodetic rates; northern Chile's line gives 3.5–4.0 cm/yr against a measured 0.5–0.9. A bound,
not a rate: `0 ≤ Vd ≤ Vup` at 60 % of 80 segments (Heuret & Lallemand 2005 [V]). Slab edges: ranges grow only at the
centre of subduction zones about 4 000 km wide or more (Schellart 2008 [A]; Schellart et al. 2007 [A]).

**A2, the height (Dielforder & Hampel 2021 [V], Eqs. 1a–9, after Dielforder et al. 2020 and Lamb 2006).** With one
density the force balance reads `h = 2·(Fs − F_SMT)/(ρ̄·g·L·cosθ)` (ARITH from Eq. 1a), `Fs` the megathrust's
integrated shear force, `L` ≈ 60–75 km the depth where it ends, `θ` the dip, `F_SMT` the part that holds the submarine
slope (Eq. 9). It gives back the published heights (Andes 23°S 5.10 km against 5.12; the Himalaya 5.97 against 5.98),
and ten margins' mean heights stand within ±10 MPa of it (Dielforder et al. 2020 [V], Extended Data, Fig. 3): of nine
subduction margins only the three Andean ones exceed 2.5 km, only 23°S exceeds 4 km. **It is not the bench's form**:
the bench's B40 column `ΔF = ρc·g·h·[L + h·ρm/(2Δρ)]` is quadratic in `h` with `L` the 35 km crust, Dielforder's is
linear; the bench's Tibet force is 7.12 TN/m (ARITH) where Dielforder's Himalaya `Fs` is 4.80 ± 1.24 TN/m — each form is
calibrated in its own terms, so neither number moves the other, but the two are not one law. **Its inputs have no law
from the world's facts yet**: the effective friction `μ′` (0.02–0.13; set by the pore pressure, and by the trench's
sediment, Lamb 2006 [A], Lamb & Davis 2003 [A] — no source gives `μ′` as a function of the fill), the depth of the
frictional–viscous transition (a thermal law), and the dip.

**The window.** No closed-form law. The central Andes began to shorten about 50–45 Ma, when the slab entered the lower
mantle (Faccenna et al. 2017 [V]); in Schellart's 2017 model [V] the upper plate extends while the slab is in the upper
mantle and shortens from the slab's entry into the lower mantle. The world does not track slabs.

**Earth's readings (ARITH from ETOPO5, which is partly 1° coarse in the Andes).** The Altiplano–Puna 0.61 million km²
over 3 km at a mean 4.0 km; the Andes 2.16 / 1.35 / 0.93 million km² over 1 / 2 / 3 km. The home planet's Andean belts in
run 10: 5.37 / 4.84 / 4.16 million km² over 1 / 2 / 3 km.

| Option | Needs from the world | Calibration | Effect on the home planet (ARITH) | Weakness |
|---|---|---|---|---|
| A1 alone (the line as a switch, today's share as the rate) | the drift vectors, the normal, the frame's net rotation, the margin's length | 166 transects | about 60 % of the margins pass, about 80 % of today's Andean rock stays | a sign rule, not a rate; frame-dependent |
| A2 alone (each margin's force balance as its ceiling) | the trench depth, the dip, the crust, `L`, `μ′`, the transition depth | 10 margins within ±10 MPa | Earth's median 1.3 km against the bench's 4.3 km | `μ′`, the transition depth and the dip have no law yet; 2-D, says nothing of the width |
| A1 + A2 | both | both | about 0.8 × A2's | both sets of gaps; still no rate law |
| today's share | `v_c` only | the central Andes today (Norabuena et al. 1998) | 81 % of the belts' rock | one compressive margin's rate on every margin for 10–133 Myr |

---

## 14. The gate's cause (ruling B41 item 1; MEASURED, runs 11–13; the census's runs 11–13)

**The instrument.** The copy's PASS LEDGER (`Laws`-free; `solve_copy(…, ledger_on)`): each process's change of volume on
the land and on the sea floor in every pass (the craters, the talus, the ice, the sweep's uplift, cut and fill, the
rebound, the peak law) and the land's readings after it; `--ledger=DIR` writes it as CSV, and two small scripts in the
scratchpad (`ledger/summary.py`, `ledger/compare.py`) read it. THE CONTROL: `today` at 40 passes with the ledger on
MATCHES the release gate's pinned digest, so the ledger changes nothing.

**What it measured (today's laws; the land's volumes in 10¹⁵ m³ over the whole age):**

| Case | routings | the rivers' cut | the rebound | the ice | land over 1 000 m (away from closed basins) |
|---|---|---|---|---|---|
| 40 even passes | 40 | 196 | 120 | −12.5 | (63 % of all land) |
| the mixed axis, 214 passes | 214 | 588 | 425 | −0.4 | 35.7 % |
| 214 passes, 2 / 4 / 8 sweeps each on one routing | 214 | 608 / 615 / 619 | 436 / 440 / 441 | ~0 | 33.2 / 34.3 / 34.1 % |
| 214 passes, 2 sweeps each, the drains found again before the second | 428 | 802 | — | ~0 | 21.0 % |
| the twice-as-fine axis, 414 passes | 428 | 797 | 585 | ~0 | 21.1 % |

1. **The cut is the term that moves.** The ice is not the cause (under today's laws it removes almost nothing on the
   mixed axes); the deposit, the talus and the uplift move by a few units.
2. **The sweep's own step is NOT the cause.** Eight sweeps a pass on one routing change the cut by 5 % and the land
   over 1 000 m by under 2 points, and the series settles (588, 608, 615, 619).
3. **THE NUMBER OF ROUTINGS IS THE CAUSE.** Two sweeps a pass with the drains found again between them give the
   twice-as-fine axis's numbers within 1 % (802 against 797; 21.0 % against 21.1 %). The rivers' cut, and so the land's
   height, depends on how often the drains are found again, not on the laws. A pass cuts nearly the same volume whatever
   its length (4.4 × 10¹⁵ m³ in 25 Myr, 3.7 × 10¹⁵ m³ in 12.5 Myr). THE MECHANISM, A READING NOT YET MEASURED: a node
   beside a valley drains into it only after the next routing finds the deeper valley, so a valley widens into the plain
   beside it by about one node per routing — the plains wear down at a speed the routing's cadence sets.
4. **Three defects found on the way, separate from the cause:** (a) the ice's cut per call stops at the ice's own
   thickness — a cap standing in for a missing law (the glacier shrinks as its bed falls under the line), so the ice's
   total grows with the passes in the last 2.6 Myr and, on the even axis, with all passes; (b) the ice's cut is never
   booked for the rebound, so the crust never rises under ice-cut valleys; (c) the rebound runs every five PASSES, so its
   spacing in years changes with the pass count (625 Myr at 40 passes, 125 Myr on the mixed axis).
5. **The cost:** a sweep costs about 0.1 s and a routing about 1.2–3 s (the priority flood over 8.9 million nodes; the
   lower number on a quiet machine), so a converged solve needs either a faster routing or many more of them.
6. **THE ROUTING DOES NOT CONVERGE EITHER (run 13, 8 routings a pass):** the cut on the land 588 / 802 / 991 / 1 103 ×
   10¹⁵ m³ at 1 / 2 / 4 / 8 routings a pass; the land over 1 000 m (away from closed basins) 35.7 / 21.0 / 8.4 / 0.9 %;
   the land's mean height at the end 874 / 540 / 269 / 101 m. The end settles only because the land runs out of height.
   The HISTORY does not settle: at 1 Gyr the land over 1 000 m reads 64.9 / 57.7 / 48.9 / 39.3 % and the mean 1 729 /
   1 510 / 1 273 / 1 036 m — the steps do not shrink. So each routing unlocks about the same cut whatever the time
   between routings, even at one routing every 3 Myr: a defect in how the scheme carries a lowered node's news to its
   neighbours between routings, not only a coarse cadence. The candidates, each a measurement: the flats (a level
   stretch cuts nothing until a routing finds its new outlet), the lakes (a lake's level falls only once a routing, by
   its spill's last cut), and lateral capture (a node drains into a deeper neighbour only after the next routing).
   **With the routing converged, today's laws grade the home planet flat** (ruling B39's prediction, now measured).

**THE MECHANISM, FOUND (run 14, the cut's anatomy; ruling B42 item 1).** The ledger splits each sweep's cut on the land
by the node's state at the routing its sweep read: a lake node, a node HELD by a lake (its path to the outlet passes a
lake), a node REROUTED since the previous routing, and the REST. Today's laws over 5 Gyr, 1 against 2 routings a pass:

| Class | the cut, 1 routing | the cut, 2 routings | change |
|---|---|---|---|
| lake (skipped) | 0 | 0 | 0 |
| held by a lake | 76.3 | 103.2 | +27 |
| rerouted | 43.8 | 42.1 | −2 |
| the rest | 468.4 | 656.7 | +188 |

Side capture is refuted (the rerouted class does not grow); true lakes carry about 13 % of the growth; the REST carries
88 %. THE CODE SAYS WHY (`vd_terrain::solve::MacroSolve::routing_base`, `solve.rs:870`, and its copy in
`router_variants.rs:331`): the base reads a receiver as a lake wherever `z_flood > z`. The routing sets `z_flood = z` for
every ordinary node; the sweep then CUTS that node, its ground falls under its stale flood level, and the rule reads it
as a lake — so its upstream neighbour cuts toward the receiver's height AT THE ROUTING, never toward its new height. A
lowered node's news climbs one node per routing, and the Braun & Willett implicit sweep, which needs the new height,
becomes explicit in its receiver: more sweeps on one routing barely help (each relaxes toward the same old heights),
and each routing adds about the same cut, whatever the time between routings. **THE CURE (in the copy, `+base`):** read
a receiver as a lake only where the routing found one (`pit > 0`), every other receiver at its ground as it stands now
(`solve_copy::sweep_base`, `sweep_cured`; the joint law's sweep takes the same base). The game's solve carries the
defect today; the copy measures the cure first.

**THE CURE MEASURED (run 15; the control, cure off, MATCHES the pin).**

| Today's laws, cured | the cut by 0.25 Gyr | by 1 Gyr | over 5 Gyr | the mean land at 1 Gyr | the relief at the end (50 / 90 / 99 %) |
|---|---|---|---|---|---|
| 1 routing a pass | 296 | 817 | 1 162 | 560 m | 3 / 6 / 9 m |
| 2 routings | 334 | 878 | 1 161 | 462 m | 1 / 4 / 7 m |
| 4 routings | 352 | 895 | 1 161 | 430 m | 1 / 5 / 8 m |
| the 414-pass axis | 328 | 867 | 1 163 | 481 m | 1 / 4 / 8 m |

1. **The routing now converges**: each doubling halves the change (by 0.25 Gyr +38, then +18; by 1 Gyr +61, then +17),
   where the defect gave +214, +189, +112 over the whole age; what is left is first order — the true lakes, whose level
   still follows its spill only at a routing.
2. **With a correct sweep, today's laws grade the home planet completely flat** (the relief under 10 m at the end): the
   world's relief today is the defect's under-erosion (ruling B39's prediction, now with a correct scheme).
3. **The erosion law decides whether the old continents survive** (the full law sets, cured; the land's share of the
   globe at 0.5 / 4.5 / 5 Gyr, its mean height and its share over 1 000 m at the end):

| Full law set, cured | passes | land % of the globe | mean land at the end | land over 1 000 m (away from closed basins) | mountains | plateau land |
|---|---|---|---|---|---|---|
| B40 + P2 + S2, the solve's own erosion (`n = 1`) | 214 | 24.6 / 18.0 / 6.8 % | 2 717 m | 53.5 % | 28.3 % | 15.2 million km² |
| the same | 414 | 24.5 / 17.7 / 7.0 % | 2 718 m | 54.5 % | 33.9 % | 12.3 million km² |
| B40 + P2 + S2, the joint law (F2, `n = 2.2`) | 214 | 24.9 / 22.8 / 17.5 % | 994 m | 19.9 % | 15.2 % | 9.3 million km² |
| the same | 414 | 25.0 / 22.8 / 19.0 % | 958 m | 21.1 % | 14.6 % | 8.4 million km² |
| Earth | — | 29 % | 840 m | 20.8 % | 2.39 % | Tibet ~2.5 million km² |

   With `n = 1` the flat lowlands keep eroding and drown (a land share of 6.8 %: only the young belts stand over the sea);
   with the joint law a gentle slope erodes far more slowly, so the plains keep about 23 % of the globe until the belts'
   own time and end at 17.5–19 %, the land over 1 000 m near Earth's. **The gate with the joint law nearly passes**: the
   mountains 15.2 → 14.6 % and the land over 1 000 m 19.9 → 21.1 % from 214 to 414 passes; the plateau land −10 %;
   the inner relief's rate reading (F3) moves most (Rz over 1 000 m on 3.1 → 5.6 % of the land), because it reads the
   last pass's rate (0.5 against 0.25 Myr). The mountains stay six times Earth's: the belts' plateaus (the Andean margins
   parked, ruling B42 item 3) and the craters.

**The three defect cures (ruling B42 item 2; runs 16–17; the joint law on the cured base).** The first stepped ice was
INVALID (run 16): it read a node as water wherever its flood level stood over its ground — the same stale-flood rule as
the sweep's base — so it chopped every glacier at its first cut node and stopped after a step. With the water read from
the routing's lakes (run 17):

| Cures | passes | the ice on the land | mountains | relief 50 / 90 / 99 % | land over 1 000 m | plateau land |
|---|---|---|---|---|---|---|
| none (the solve's capped ice) | 214 → 414 | 3.5 × 10¹⁵ m³ | 15.2 → 14.6 % | 78 / 1 452 / 3 129 → 69 / 1 370 / 3 328 m | 19.9 → 21.1 % | 9.3 → 8.4 million km² |
| the ice stepped, no cap | 214 → 414 | 4.8 | 14.1 → 13.4 % | 78 / 1 384 / 3 692 → 68 / 1 273 / 3 975 m | 19.5 → 20.7 % | 8.4 → 7.4 |
| the rebound every pass | 214 → 414 | 3.6 | 17.2 → 20.3 % | 106 / 1 470 / 3 136 → 122 / 1 632 / 3 646 m | 25.6 → 28.8 % | 9.4 → 6.5 |
| all three | 214 → 414 | 6.1 | 18.0 → 20.1 % | 106 / 1 627 / 4 065 → 121 / 1 603 / 4 565 m | 27.1 → 30.7 % | 8.5 → 5.9 |

1. **The ice with no cap works**: it cuts more than the capped ice (4.8 against 3.5 × 10¹⁵ m³), deepens the deepest
   valleys (the relief at 99 % 3 129 → 3 692 m), keeps the gate's agreement, and costs little (1 921 steps; the solve
   829 s against 834 s).
2. **The ice's cut in the rebound** acts only where a rebound follows the ice (the ice works in the last 2.6 Myr; the
   rebound every five passes last ran before it).
3. **THE REBOUND'S SPACING HAS NOT CONVERGED**: the land over 1 000 m reads about 20 % with a rebound every 125 Myr, 25.6 %
   every 25 Myr and 28.8 % every 12.5 Myr, so the rebound is a resolution of its own, and "every pass" ties it to the pass
   count — the gate fails with it (+3.1 points). Its limit (the crust answers within ~10 kyr) needs its own measurement.

**THE REBOUND'S SPACING CONVERGES (run 19; ruling B44 item 2).** The solve's own rebound floors four times — about
−0.125 m a node per rebound, negligible over 40–214 rebounds but some 430 m over 3 400 — so the series runs with the exact
rebound (`+rebx`: the same lift without the floors, each node's fraction carried). The joint law, the cured base, the ice
cure, the mixed axis, the rebound after each of K sweeps:

| The rebound every | land over 1 000 m | mountains | relief 50 / 90 / 99 % |
|---|---|---|---|
| 125 Myr (every five passes) | ~20 % | ~14–15 % | 78 / 1 384 / 3 692 m |
| 25 Myr (every pass) | 26.40 % | 17.40 % | 105 / 1 593 / 4 035 m |
| 12.5 Myr | 29.09 % | 19.86 % | 119 / 1 840 / 4 758 m |
| 6.25 Myr | 28.65 % | 19.27 % | 123 / 1 772 / 4 719 m |
| 3.1 Myr | 28.86 % | 19.02 % | 130 / 1 734 / 4 668 m |
| 1.6 Myr | 30.01 % | 19.76 % | 150 / 1 768 / 4 689 m |
| the 414-pass axis, every pass (12.5 Myr) | 29.06 % | 19.05 % | 110 / 1 552 / 4 522 m |

1. **From 12.5 Myr down the shares stay within about 1.4 points**: the rebound settles at a spacing of 12.5 Myr or finer
   when it carries its fractions; the crust's own ~10 kyr answer is the limit it approaches.
2. **The gate passes with a matched rebound**: 214 passes with two sweeps each and 414 passes, both rebounding every
   12.5 Myr, read 29.09 / 29.06 % over 1 000 m and 19.9 / 19.1 % mountains.
3. **The sweeps' own step** (K = 8 sweeps, the rebound every five passes): the land over 1 000 m 19.5 → 21.2 %, the
   mountains 14.1 → 15.4 % — a smaller resolution of its own, settled by the same sub-sweeps.
4. **The converged world**: the land over 1 000 m about 29 % (Earth 20.8 %), mountains about 19–20 % of the land (Earth
   2.39 %), the relief at 50 / 90 / 99 % about 120–150 / 1 750 / 4 700 m (Earth 83 / 550 / 1 383 m). A prompt rebound
   keeps worn ranges high; the excess has three named causes left — the belts' plateaus (the Andean margins, parked),
   the craters, and the old plains' own time (the research).

---

## 15. The old plains: how Earth's old continents keep their height (2026-10-03; ruling B44 item 2; a read-only agent,
its sources and scripts in the scratchpad's `sources4/`, its index `sources4/INDEX.txt`)

[V] read in full; [A] abstract only; [R] recalled; ARITH the agent's own arithmetic.

**Earth's old plains (ARITH, ETOPO 2022 averaged into 7.4–9.3 km blocks; CRUST 2.0's Archean, Proterozoic and platform
types; Antarctica and the ice types left out; the stands' relief rule).** 86.7 million km² — 65 % of the land counted,
17.0 % of the globe; the mean height 485 m, the median 349 m (outside Africa–Arabia 408 / 262 m); 13 % over 1 000 m; the
relief at 50 / 90 % 56–62 / 299–325 m; by region: the Canadian Shield 305 m, the Baltic Shield 257 m, the Siberian
platform 448 m, the West African craton 304 m, Yilgarn–Pilbara 363 m, the Amazonian craton 210 m, the East European
platform 153 m, Congo 597 m, Kalahari 1 106 m. (Control: all counted land 655 m against the repository's own Earth
instrument's 661 m.)

**The home planet before its belts rise (ARITH, from run 15's ledgers):** the mean land 105–114 m over the sea, under
0.01 % of it over 1 000 m. **Its plains do not erode too fast**: they lose 0.2–0.8 m/Myr of rock in the solve's last
2.8 Gyr, against cratons' 2–20 m/Myr in the Cenozoic (Beauvais & Chardon 2013 [V]) and ≤ 2.5 m/Myr over 1.7 Gyr on the
western Canadian Shield (Flowers et al. 2006 [A]). **They are low because nothing raises them again for 4.8 Gyr**: the
solve's mean falls under Earth's 485 m at 1.5–1.7 Gyr, so Earth's old plains look like 1.5–1.7 Gyr of this solve; the
land loses 1.3 × 10¹⁸ m³ (0.27 km³/yr) and nothing returns it.

**Why Earth's old continents keep their height (each with its size):**
- the decay of ranges with isostasy and thresholds keeps relief for hundreds of Myr (Baldwin et al. 2003 [A]; Egholm
  et al. 2013 [A]); Ahnert's law `d = 0.0001535·h − 0.01088` m/kyr (`h` the mean relief in 20 km squares) stops at
  `h` ≈ 71 m [V], where Earth's plains read 56–62 m (ARITH);
- the freeboard stays within ~200 m through the Phanerozoic: the sea follows the ocean basins' volume (Korenaga et al.
  2017, Eq. 3.1 [V]; Parsons 1982 [A]); the Late Cretaceous sea stood 170 (85–270) m above today's (Müller et al.
  2008 [A]);
- the rock cycle balances: arcs add 2.7 km³/yr, subduction and sutures remove 3.2 km³/yr (Scholl & von Huene 2009 [A]);
- **RE-THICKENING**: 30.0 % of the continental area has a tectono-thermal age under 0.35 Ga, 53.7 % under 0.75 Ga,
  88.8 % under 2.95 Ga (Artemieva 2006, Table 4 [V]) — most continents were last raised far more recently than the
  planet's age;
- dynamic topography ±500 m at 10⁴ km (Hoggard et al. 2016 [V]) — the Kalahari's 1.1 km against 153–448 m elsewhere;
- a craton's depleted root does not lift it: the cold root sinks the column more than depletion lifts it (Lee et al.
  2017, Eq. 2 [V], ARITH: +1.52 km against a fertile root, −1.43 km against no root; Mooney & Vidale 2003 [A]).

**The candidate laws (each the owner's):**

| Law | Needs from the world | Calibration | Effect on the old plains (ARITH) | Weakness |
|---|---|---|---|---|
| **The landscape's time** (ruling B37): a plain erodes only since its ground was last raised | a raise age per plate or province (an unrecoverable history, T9's lawful draw) from a reworking law; the height after a raise | Earth's tectono-thermal ages (Artemieva 2006) | 477 m (414 passes) / 581 m (214 passes) for the 70 % older than 0.35 Ga, against 105–114 m today | Earth's ages carry a preservation bias; the height after a raise is assumed |
| a generated plate history (Wilson cycles) | past layouts, collisions, rifts | Earth's assembly intervals [R]; Artemieva as the check | as above, plus old ranges | the largest build; ends "today's plates for the whole age" |
| the rock cycle (a marine deposit, subduction's return, the arcs) | a deposit; the margins' lengths and speeds (has) | Scholl & von Huene 2009 | ~0 on the plains; without a return the sea would rise ~1.5 km | must not count the freeboard law's effect twice |
| the sea from the ocean basins' volume | the sea floor's ages through time | Earth's area–age and depth–age | 0 with fixed plates | needs the plate history |
| a low-relief threshold (Ahnert) | the relief in ~20 km windows (has) | 20 mid-latitude basins | decay stops near 71 m of relief | extrapolated; may repeat `n = 2.2`'s effect |
| dynamic topography, thermal isostasy | a mantle flow field; root thickness | Earth | ±0.5 km by region; not a cure | no mantle model; a drawn field is not lawful |

**The research's reading:** the plains' low height is a missing history, not a fast erosion; the landscape's time is the
strongest law that the world can state today (each plate's raise age from a reworking law calibrated on Earth's
tectono-thermal ages; the plates already carry a uniform age share, whose median of 2.5 Gyr is far older than Earth's
~0.7 Gyr, so the share needs its own law), and a generated plate history is its full form.

**THE LANDSCAPE'S TIME BENCHED (runs 20–21; ruling B45 item 2).** Each plate's ground erodes only since its raise,
`T = −ln(1 − u)/λ` from the plate's own age share `u`, `λ` = 1.019 per Gyr (Artemieva 2006's 30.0 % under 0.35 Ga; the
law gives 53.4 % under 0.75 Ga against the measured 53.7 %); a belt node from its window's start at the latest; a crater
older than its node's raise is not stamped. The home planet's plates come out raised 9 Myr to 2.8 Gyr ago; nothing erodes
in the first 2.2 Gyr. The readout now splits the WIDENED PLATEAU (the nodes the force balance's widening raised) from the
OLD PLAINS (the land no wall, crater, belt or plateau claims). The joint law, the cured base, the ice cure, the exact
rebound every 12.5 Myr:

| Case | old plains mean / median | old plains over 1 000 m | land over 1 000 m | mountains | plains / tablelands |
|---|---|---|---|---|---|
| without the law (214 passes, 2 sweeps) | 310 / 93 m | 8.6 % | 29.09 % | 19.86 % | 43.3 / 36.8 % |
| the law, 1 routing a pass (2 sweeps) | 586 / 392 m | 22.1 % | 28.67 % | 11.62 % | 19.5 / 68.9 % |
| the law, 2 routings a pass | 651 / 437 m | 26.4 % | 32.83 % | 10.90 % | 20.9 / 68.2 % |
| the law, 414 passes | 680 / 467 m | 28.7 % | 34.70 % | 11.63 % | 22.1 / 66.3 % |
| Earth | 485 / 349 m | 13 % | 20.8 % | 2.39 % | 52.1 / 45.5 % |

1. **The law brings the old plains to Earth's scale**: 586–680 m (median 392–467 m) against Earth's 485 m (349 m), where
   without it they read 310 m (93 m). The plates differ by age as on Earth (a plate raised 1.8 Gyr ago has worn its
   plains to about 40 m; one raised 223 Myr ago keeps them near 870 m).
2. **The mountains fall from 19.9 to 11.6 %** (Earth 2.39 %); the land turns to TABLELANDS (66–69 % against Earth's 45.5 %):
   the young plates stand high and flat at the height the freeboard law gave them, eroding only since their raise.
3. **THE GATE FAILS BY THE LAKES**: from 1 to 2 routings a pass the land over 1 000 m moves 28.7 → 32.8 % (the 414-pass
   axis 34.7 %); the cut's anatomy puts the difference in the nodes HELD BY A LAKE (397 against 214 × 10¹⁵ m³; the other
   classes barely move). A lake's level is set at a routing and stays fixed while the sweeps cut around it, and the law
   releases plates that stood frozen for billions of years full of hollows — so the lakes' handling is the remaining
   routing dependence.

**THE LAKES' CURE (run 22; ruling B46 item 2).** A lake's level follows its outlet inside the sweep: the outlet is
visited first, each lake node's level becomes the level its receiver stands at now (never under its floor), a node whose
level falls to its floor is dry and cut in the same sweep, and the deposit fills each lake only up to its level now. The
joint law, the cured base, the ice cure, the exact rebound every 12.5 Myr and the landscape's time:

| Routing | without the lakes' cure: land over 1 000 m | with it: land over 1 000 m | mountains | old plains mean / median |
|---|---|---|---|---|
| 1 routing a 25 Myr pass | 28.67 % | 27.83 % | 11.68 % | 564 / 378 m |
| 2 routings a pass | 32.83 % | 27.28 % | 11.45 % | 554 / 376 m |
| 4 routings a pass | — | 26.50 % | 10.69 % | 553 / 376 m |
| the 414-pass axis | 34.70 % | 29.14 % | 12.71 % | 584 / 407 m |

1. **The routing dependence is gone**: 1 → 2 routings move the land over 1 000 m by −0.55 points (it was +4.2), and the
   414-pass axis stands within 1.3 points (it was 6.0); the old plains settle at 553–584 m (median 376–407 m).
2. **THE GATE PASSES** for the whole law set within about 2 points on every share and 5 % on the old plains' height, from
   1 to 4 routings and from 214 to 414 passes. (Ruling B39 still asks for the stated tolerance.)
3. **The converged world**: land over 1 000 m 26.5–29 % (Earth 20.8 %); mountains 10.7–12.7 % (Earth 2.39 %); the old
   plains 553–584 m, median 376–407 m (Earth 485 / 349 m); plains / tablelands about 21 / 67 % (Earth 52 / 45.5 %). The
   excess left: the mountains (the belts, the widened plateaus and the craters) and the tablelands (the height a young
   plate stands at after its raise, today the freeboard law's starting height — an assumption).

---

## 16. The mountains: which cause carries them (ruling B47 item 3; MEASURED, run 23 and the Earth instrument's run 4)

**Earth, by CRUST 2.0's crust type** (Bassin, Laske & Masters 2000; the same grid and relief rule; the land whose window
stays over the sea level, no Antarctica, no ice type; mountains 2.42 % of it): ORogens 12.2 % of the land, 11.8 % of it
mountains, 59.3 % of all mountains, a mean of 1 841 m; arcs and forearcs 3.9 %, 8.1 %, 12.9 %, 810 m; extended and rift
6.2 %, 2.2 %, 5.7 %; the shields and platforms about 65 % of the land, 0.2–0.8 % of it mountains, about 13 % of all.

**The home planet's converged law set** (the joint law, the cured base, the ice cure, the exact rebound, the landscape's
time, the lakes' cure; the crater readout now counts only the craters the solve STAMPED — the earlier 43 % of the land
"near a crater" counted craters a plate's raise had erased):

| Cause | share of the land | mountains in it | share of all mountains | mean / median height |
|---|---|---|---|---|
| collision belt | 5.4 % | 68.2 % | 31.7 % | 2 745 / 2 392 m |
| Andean belt | 6.6 % | 49.1 % | 27.7 % | 2 885 / 2 147 m |
| widened plateau | 5.7 % | 58.1 % | 28.5 % | 3 635 / 2 542 m |
| old plains | 76.7 % | 1.1 % | 7.2 % | 516 / 318 m |
| craters (398 stamped; Earth knows about 190; 24 / 57 / 677 km at 50 / 90 / 100 %) | 1.8 % | 7.8 % | 1.2 % | 615 / 283 m |
| closed basins' walls | 3.8 % | 11.5 % | 3.8 % | — |

1. **The old plains match Earth's**: 516 / 318 m against Earth's shields and platforms at 485 / 349 m; about 1 % of them
   mountains against Earth's 0.2–0.8 %.
2. **THE BELTS AND THEIR PLATEAUS CARRY 88 % OF THE MOUNTAINS**: they cover 17.7 % of the land (Earth's orogens and arcs
   16 %), but 49–68 % of them are mountains against Earth's 8–12 %, and they stand 1–1.8 km higher (2.7–3.6 km against
   Earth's orogens' 1.8 km). The excess is the belts' ROUGHNESS and HEIGHT, not their area.
3. The craters are a small cause; their count (398 against Earth's ~190 known) and their largest (677 km) are G-CRATER's
   own reading, for the craters' law.
