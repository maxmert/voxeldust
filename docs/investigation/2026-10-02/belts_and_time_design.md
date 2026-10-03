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
