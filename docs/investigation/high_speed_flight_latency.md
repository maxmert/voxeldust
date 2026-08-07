> ### ⚠ INVESTIGATION BASE — NOT A DECISION RECORD
>
> **This document is input to an investigation, not the output of one.** It is analysis produced to
> explore a problem space, and it is deliberately more confident in tone than its status warrants —
> that was useful for finding defects and is misleading for planning.
>
> **Nothing here is committed.** Every ruling, recommendation, number and "settled" verdict is a
> *proposal to be re-validated by the pre-implementation investigation for its phase*, including the
> owner rulings recorded at the top of `block_system_design.md`, which record the owner's direction at
> the time rather than a frozen commitment.
>
> **The binding specs are elsewhere:** `docs/design/PLAN.md`, `docs/design/roadmap.json`,
> `docs/design/integration.json`, `docs/design/DEFERRED.md` and the hardened subsystem designs in
> `docs/design/`. Where this document and a binding spec disagree, **the binding spec wins** until an
> investigation says otherwise.
>
> **What this document IS good for:** the prior art it collected, the arithmetic it did, the failure
> modes it found, the one-way doors it named, and the questions it framed. Reuse those. Re-derive the
> conclusions.
>
> See `docs/investigation/README.md`.

# High-speed flight, latency, and the no-prediction law

**Status:** ruling, awaiting owner decisions. Not yet binding.
**Date:** 2026-08-03.
**Scope:** whether the standing "no client-side prediction, 100–150 ms interpolation buffer" law
survives high-speed atmospheric flight over one-metre voxel terrain, and what to do about it.
**Code state examined:** HEAD `cbfe573` (`worktree-new-system`).

---

## 1. What the owner observed

Discussing collision for ground vehicles, it was noted that at 30 m/s a vehicle is 3–4.5 m behind
its authoritative position under the render buffer. The owner replied:

> *"we will have ships that flying way faster than 30m/s over the surface and might collide into it"*

He is right that this is a bigger question than the ground-vehicle case, for four reasons that are
already committed in the repository and not up for debate:

1. **Ships are committed.** P8 is explicit: *"A player walks inside a ship interior while the ship
   flies under Newtonian physics"* (`docs/design/roadmap.json`, P8 definition of done).
2. **Atmospheric flight over voxel terrain is the pitch.** "Star Citizen meets Minecraft" is the
   project's own one-line description, and P4's demo is *"Open the client and fly/walk over a
   procedurally generated spherical planet with real greedy-meshed voxel terrain"*.
3. **Terrain is a one-metre solid you can fly into.** The block ruling fixed the block at 1 m and the
   starter world at R = 161,671 m, and physics is only ever handed one-metre cells
   (`docs/investigation/block_system_design.md`, the tier-0-only collision rule).
4. **The law was written for a different problem.** It was set in the context of players walking and
   colliding with each other, at 5–15 m/s, where every number in this report is sub-metre and
   invisible. Nothing about it was ever tested at 240 m/s.

So the question deserves a proper answer rather than a deferral. Here it is.

---

## 2. The answer in one page — plain language

**You do not have the problem you think you have, and the reason is uncomfortable: the delay you are
worried about is not actually happening today.**

The rule says the picture should be drawn about an eighth of a second behind the real world, so that
motion can be smoothed out. The code does not do that. Because of a mismatch between two settings —
one of them written when the server ran at a different speed — the picture is drawn only about a
twentieth of a second behind, and it is not smoothed at all. It jumps from one position to the next
fifty times a second. I proved this by reproducing the exact arithmetic the client uses and running
it over a second of play: out of 121 drawn frames, the smoothing step ran **zero** times.

Two things follow, and they point in opposite directions.

**The good news.** The total delay from your hand moving to seeing the result is currently about a
sixth of a second on a normal home connection. At 30 m/s that is 5 metres; at 240 m/s it is 40
metres. That is not the thing that stops you flying fast. Four separate things limit your speed, and
the delay is the *loosest* of them. The tightest is how fast the graphics card can swallow new
terrain: about 240 m/s at today's terrain format — that is 864 km/h, a low-flying strike fighter,
faster than any helicopter and faster than a WWII fighter. The delay does not start to bite until
around 320 m/s, and even there it is worth about a tenth of the total. The rest is your own reaction
time and, overwhelmingly, the simple physics of how long a heavy thing takes to turn. Even with a
magic zero-delay connection you would gain about 8% more speed — and you cannot spend it, because
you run out of graphics bandwidth 80 m/s earlier.

**The bad news, and it is real.** Your instinct that speed makes it worse is half right and half
backwards. For "will I hit that mountain", the delay is a fixed percentage of the problem at every
speed — going faster does not make it relatively worse, because the mountain and your stopping
distance both scale together. Where it genuinely gets worse is **slow, precise flying**: landing,
docking, hovering, flying through a gap. At 10 m/s the delay is a third of your turning circle; at
500 m/s it is under one percent. Latency hurts most when you are being careful, not when you are
being fast.

And there are three actual bugs, all of which get worse with speed, none of which has anything to do
with prediction:

- **The picture jumps instead of gliding.** Because the smoothing never runs, everything moves in
  discrete hops. At walking pace the hop is 60 cm and invisible. At 240 m/s it is 4.8 m, fifty times
  a second. That is the juddering you will actually see when you first fly fast, and if nobody knows
  about this finding it will be blamed on the buffer.
- **You move slower than the settings say, and slower still on a bad connection.** The server only
  moves you when a command packet arrives, and moves you by exactly one server-step's worth. The
  client sends 20 command packets a second; the server runs 50 steps a second. So you actually travel
  at 40% of the configured speed, and if 20% of your packets are lost you travel 20% slower again.
  For a walking avatar this is merely wrong. For a ship under thrust it means **your engines stop
  firing whenever the network hiccups**, and your top speed becomes a property of your internet
  connection. This is the mechanism that will genuinely feel like rubber-banding at speed, and it
  exists today with no prediction anywhere in the system.
- **You cannot change the buffer without recompiling.** The design document says this value must be a
  configuration setting so that "2am tuning is a config change, not a recompile". It is a constant
  compiled into the program with exactly one place that uses it. So nobody has ever flown with a
  different value, and this entire argument has never had any evidence behind it.

**A trap you must not walk into.** If someone fixes the smoothing without also reducing the buffer,
the game will suddenly start paying the eighth-second delay it has never paid, and the total will go
from about a sixth of a second to about a quarter — roughly 55% worse. It will feel awful, and the
law will be blamed for it. Fix both in the same change: smooth the motion *and* set the buffer to
two or three steps' worth (40–60 milliseconds) instead of six. Do that and also send commands 50
times a second instead of 20, and the total delay comes out at about a sixth of a second — the same
as today — while the motion becomes properly smooth for the first time.

**On the prediction question, and the distinction that matters most.**

The word "prediction" covers four different mechanisms that share a name, and the bad experiences
people have belong to specific ones:

1. **Guessing where *other* people are** — the client draws other players slightly into the future
   based on where they were heading. When they change direction, they visibly warp. This is the
   classic teleporting-traffic problem. **We do not do this, and we should never do this.** The code
   makes it structurally impossible, and that should stay.
2. **Letting the client decide where it is** and having the server accept it. Feels perfect, and it
   is how people cheat: speed hacks, teleporting, walking through walls. **We do not do this either.**
3. **Rewinding the world to judge a shot** so the shooter's aim counts. This is what produces "I was
   behind cover and still died". It is a separate mechanism entirely, it is already owed for combat
   whatever is decided here, and nothing in this document changes it.
4. **Running your own vehicle forward from commands you have already sent**, then snapping it back
   when the server disagrees. This is the only one that would help flying. When it goes wrong — and
   it goes wrong most often for a walking person, who is constantly touching steps, ledges, doors and
   other players — it produces your character being yanked backwards. **That is almost certainly what
   hurt you before**, because on foot the server disagrees with you constantly.

So: the thing that hurt you and the thing that would help flying are genuinely different mechanisms,
but they are the *same family*, and I will not pretend otherwise. The honest distinction is not
"prediction versus not-prediction", it is **contact versus free flight**. A person walking is in
contact with the ground every single moment, and contact is where the two machines disagree. A ship
in the open air is touching nothing, so the client and the server would compute the identical answer
from the identical commands, and there would be nothing to snap back. That is a real and load-bearing
difference — but it is a difference of *circumstance*, not of mechanism, and if it were ever switched
on for a ship in contact with the ground it would behave exactly like the thing you hated.

**My recommendation: do not build it.** Not because it would feel bad — in free flight it would feel
excellent — but because it buys about 8% more speed that the graphics budget forbids you from using,
it costs an exception to the rule that the physics engine is never re-run on a second machine, and it
converts the combat fairness question from one clean setting into two clocks and an argument about
whose gun barrel is real. Fix the three bugs first, fly it, and only then decide. If flying still
feels wrong afterwards, the next lever is not prediction — it is a proper flight computer that runs
on the server, where there is no delay at all, plus a speed limit that falls out of air density
rather than being a number somebody picked.

---

## 3. The arithmetic

### 3.1 The three delays, separated

All parameters below are read from code, not from design documents. Where they differ from the
design documents, the code wins and the discrepancy is flagged.

| Quantity | Value | Where |
|---|---|---|
| Server tick | **20 ms (50 Hz)** | `crates/bins/src/lib.rs` `DEV.tick_hz = 50`; `deploy/k3d/10-configmap.yaml` `VD_TICK_HZ: "50"` |
| Snapshot emit cadence | **every tick — 20 ms**, no divider | `crates/sim/src/stub.rs:1553-1565` — `emit_frames` sits in an unconditional per-tick `.chain()` |
| Client command cadence | **50 ms (20 Hz)** | `crates/bins/src/bin/client.rs:61` `DEFAULT_STEP_HZ = 20`, paced by `TickPacer` |
| Render buffer, nominal | **120 ms** | `crates/client/src/tuning.rs:21-24` |
| Render buffer, effective | **inert — see §3.2** | proved by exact replication of `RenderClock::cursor` + `EntityTrack::sample` |
| Gateway hop, intra-cluster | 0.3–1.0 ms one-way | `docs/design/connection_plane.md:282`; corroborated by the SPIKE-3a p99 of ~0.7 ms |
| Gateway store-and-forward | **tick-quantised, 0–20 ms each way** | `crates/node/src/app.rs:163` — `step_tick` = drain schedule, *then* flush outbox |

Three discrepancies with the design documents, all verified:

- Every design document and the wire doc-comment say **20 Hz snapshots**. The code emits at the shard
  tick rate, which is **50 Hz** in both the dev profile and the cloud ConfigMap. Nothing in the
  documents acknowledges the change. Every bandwidth and buffer figure written against 20 Hz is
  calibrated to a rate that does not ship.
- `docs/design/connection_plane.md:346` requires `interp_buffer_ms` to live in `TransportTuning`,
  "loaded from config/env, validated at startup". The shipped `TransportTuning`
  (`crates/connection-plane/src/gateway.rs:57-75`) has exactly two fields, `max_sessions` and
  `max_buffered_inputs`. The buffer is a compile-time constant with one call site
  (`crates/bins/src/bin/client.rs:108`), no CLI flag and no environment variable.
- The tick *rate* IS learned from the wire (`ServerControlMsg::UniverseRate` →
  `set_tick_hz_from_wire`, `crates/client/src/net.rs:385-390`, one-shot). The *buffer* adapts to
  nothing.

**(b) Input-to-effect** — the player's command reaching the simulation:

| Term | Mean | Worst |
|---|---|---|
| Wait for the next command assembly (20 Hz) | 25 ms | 50 ms |
| Uplink | RTT/2 | RTT/2 |
| Gateway drain wait (tick-quantised) | 10 ms | 20 ms |
| Gateway → shard hop | 1 ms | 1 ms |
| Shard drain wait (tick-quantised) | 10 ms | 20 ms |
| Integrate + emit (same tick chain) | 0 | 0 |
| **Total** | **46 + RTT/2** | **91 + RTT/2** |

**(a) Render lag** — how old the drawn world is, *as the code actually runs*:

| Term | Mean | Note |
|---|---|---|
| The drawn pose is one tick older than the freshest delivered | 20 ms | §3.2 — the clamp always fires |
| Gateway drain wait | 10 ms | |
| Gateway → client hop | 1 ms | |
| Downlink | RTT/2 | |
| Hold until the next arrival | 10 ms | 0–20 ms |
| **Total** | **41 + RTT/2** | |

If the buffer were actually being paid, the first and last terms are replaced by the buffer itself:
**131 + RTT/2**.

**(c) The full control loop** = (a) + (b):

| Round-trip time | Today (as it runs) | If the 120 ms buffer were paid | Fixed: 40 ms buffer + 50 Hz commands |
|---|---|---|---|
| 10 ms (LAN / same city, wired) | **97 ms** | 187 ms | **92 ms** |
| 30 ms (same city) | **117 ms** | 207 ms | **112 ms** |
| 80 ms (same continent) | **167 ms** | 257 ms | **162 ms** |
| 160 ms (cross-continent) | **247 ms** | 337 ms | **242 ms** |

Add a display and peripheral chain — roughly 25–40 ms at 60 Hz, 15–25 ms at 144 Hz — for
photon-to-photon figures. That is platform, not architecture, and it is excluded from every number
below.

**The planning figure used throughout this document is 167 ms, at 80 ms RTT.** The last column is the
key result: *fixing the smoothing, resizing the buffer to two snapshot intervals, and raising the
command rate to match the server tick gives smooth motion for the first time at a loop that is
5 ms **shorter** than today's.* There is no trade to make there. It is free.

### 3.2 The finding that reframes everything: the buffer is inert

`EntityTrack` holds exactly two poses — `prev` and `current` (`crates/client/src/interp.rs:72-76`).
`emit_frames` runs every tick with no divider, so those two poses are **one tick apart**: the
interpolation window is 20 ms wide.

`RenderClock::cursor` returns `anchored_tick + elapsed·tick_hz − buffer_ticks`
(`crates/client/src/render_clock.rs:78-83`). After the client learns the cluster rate of 50 Hz,
`buffer_ticks` = 120/1000 × 50 = **6.0**. So the cursor sits six ticks behind the freshest delivered
tick, while the window is one tick wide. The cursor can never be inside the window.

`lerp_at_game_time` therefore takes its `if target <= prev_time { return prev; }` branch
(`crates/client/src/interp.rs:43-45`) on **every frame**, and the blend closure is never called.

I replicated `RenderClock::cursor` and `lerp_at_game_time` exactly and simulated one second of play at
120 fps:

| Configuration | Frames | Returned `prev` | Returned `current` | **Blended** |
|---|---|---|---|---|
| Production (50 Hz learned, emit every tick) | 121 | 118 | 3 (start-up) | **0** |
| Client never learns the rate (2.4-tick buffer) | 121 | 118 | 3 | **0** |
| Production + 30% burst loss | 113 | 111 | 2 | **0** |
| Production + 70% burst loss (140 ms gaps) | 104 | 61 | 21 | 22 |
| Production with a 40 ms buffer | 121 | 118 | 3 | **0** |

**The renderer outputs the second-freshest delivered pose, held, and steps to the next one when it
arrives.** An entity must miss more than six consecutive ticks — over 120 ms of loss — before the
interpolator ever runs.

Consequences that must be carried through the rest of this document:

- **Every in-game validation to date ran at roughly 40 ms of client-side lag, not 141 ms.** No
  evidence exists on either side of the "does 100–150 ms feel bad" argument, and none can be
  gathered without a rebuild.
- **Motion is stepped, not smooth.** The rendered position is constant between arrivals and then
  jumps by `v × 20 ms`: 0.20 m at 10 m/s, 0.60 m at 30 m/s, **4.80 m at 240 m/s**, 7.86 m at
  393 m/s, 10.56 m at 528 m/s. Invisible at walking pace; the dominant visual defect in flight.
- **The documented cursor-rewind stutter cannot happen today.** `crates/client/src/render_clock.rs:11-18`
  warns that a re-anchor after loss "can step the cursor BACK … (a visible micro-stutter)" and names
  the unimplemented fix (a cursor slew). Since the cursor never reaches the window, that stutter is
  currently latent. **It becomes real the moment the interpolator is fixed**, so the slew must land in
  the same slice.
- **The entity-lane / realm-lane differential is one tick, not 120 ms.** Entities go through
  `sample()` (which clamps to `prev`); realm boxes go through `current_render_pose()` — the leading
  edge (`crates/client/src/realm_view.rs:107` vs `:118`). The gap is therefore 20 ms, worth 4.8 m of
  intra-ship slide at 240 m/s, not the 28.8 m a real 120 ms differential would give. It still must be
  closed before P8 (SPIKE-10a's exit criterion is "composite interior pose through hull pose at one
  render tick with no jitter"), but it is a smaller problem than it looks.

### 3.3 The speed table

At the block ruling's 1 m block, **metres and blocks are the same number**. All figures in metres.

| Speed | 10 | 30 | 50 | 100 | 250 | 500 | 1000 | 3000 |
|---|---|---|---|---|---|---|---|---|
| **Control loop, today, 80 ms RTT (167 ms)** | **1.67** | **5.01** | **8.35** | **16.70** | **41.75** | **83.50** | **167.0** | **501.0** |
| — of which render lag (81 ms) | 0.81 | 2.43 | 4.05 | 8.10 | 20.25 | 40.50 | 81.0 | 243.0 |
| — of which input-to-effect (86 ms) | 0.86 | 2.58 | 4.30 | 8.60 | 21.50 | 43.00 | 86.0 | 258.0 |
| Control loop, 10 ms RTT (97 ms) | 0.97 | 2.91 | 4.85 | 9.70 | 24.25 | 48.50 | 97.0 | 291.0 |
| Control loop, 160 ms RTT (247 ms) | 2.47 | 7.41 | 12.35 | 24.70 | 61.75 | 123.5 | 247.0 | 741.0 |
| If the 120 ms buffer were paid, 80 ms RTT (257 ms) | 2.57 | 7.71 | 12.85 | 25.70 | 64.25 | 128.5 | 257.0 | 771.0 |
| **Fixed: 40 ms buffer + 50 Hz commands, 80 ms RTT (162 ms)** | 1.62 | 4.86 | 8.10 | 16.20 | 40.50 | 81.00 | 162.0 | 486.0 |
| One rendered step (20 ms) — the judder amplitude | 0.20 | 0.60 | 1.00 | 2.00 | 5.00 | 10.00 | 20.0 | 60.0 |

**On the 3–4.5 m figure quoted to the owner.** That was the buffer term alone at 100–150 ms. Today's
true render lag at 30 m/s is **2.43 m** and the true full control loop is **5.01 m**. The number given
to him was in the right neighbourhood for the loop and wrong in attribution — the buffer is not where
it comes from.

### 3.4 The three thresholds

**(i) The loop exceeds the ship's own length.** `v = L / t_c`, at 167 ms:

| Hull length | Crossover speed |
|---|---|
| 10 m fighter | 59.9 m/s |
| 20 m | 119.8 m/s |
| 40 m hauler | 239.5 m/s |
| 100 m | 598.8 m/s |
| 200 m capital | 1,197.6 m/s |

**This threshold is arithmetically correct and operationally irrelevant, and I flag it as such rather
than let it look like a finding.** "The loop exceeds my hull length" only bites when the ship must be
positioned to within its own length — which is docking, landing and hangar entry, and those happen at
2–10 m/s where the loop distance is 0.33–1.67 m. Nothing at 250 m/s requires hull-length accuracy.
No hull dimensions exist anywhere in the repository; keep the formula, recompute when they do.

**(ii) The loop against the turning radius — the comparison inverts.** Minimum turn radius
`r = v²/a` grows as `v²`; loop distance `v·t` grows as `v`. So the ratio `d/r = a·t/v` **falls** with
speed:

| Speed | 10 | 30 | 50 | 100 | 250 | 500 | 1000 | 3000 |
|---|---|---|---|---|---|---|---|---|
| d/r at 2 g | 32.8% | 10.9% | 6.6% | 3.3% | 1.31% | 0.66% | 0.33% | 0.11% |
| d/r at 6 g | 98.3% | 32.8% | 19.7% | 9.8% | 3.93% | 1.97% | 0.98% | 0.33% |

The crossing where the loop distance equals the turning radius is at `v = a·t`: **3.28 m/s at 2 g,
9.83 m/s at 6 g** — at or below walking pace. **The owner's inference is backwards in this
dimension.** Latency is most costly for slow, precise, agile flying — hovering, landing, docking,
formation, threading a gap — and least costly for the fast dash. This is a control-bandwidth problem,
and control bandwidth does not care how fast you are going.

**(iii) The loop against the distance at which terrain becomes visible.** The required avoidance
distance is `v × (control loop + human reaction + manoeuvre)`. Taking a 250 ms human reaction and a
pull-up clearing 50 m at 3 g net:

| Budget term | Time | Share |
|---|---|---|
| Manoeuvre (3 g pull-up, 50 m clearance) | 1.843 s | **81.5%** |
| Human reaction | 0.250 s | 11.1% |
| **Machine control loop (today, 80 ms RTT)** | **0.167 s** | **7.4%** |
| Total | 2.260 s | |

Ground horizon on the starter world is **741 m** at eye height
(`docs/investigation/block_system_design.md:3357`, `block_system_design_addendum_1.md` C.3 — I recomputed
`√(2Rh + h²)` at R = 161,671 m and h = 1.7 m and got 741.4 m). Setting `741 = v × 2.260`:

| Configuration | Loop | Total budget | Loop share | Eye-height speed limit |
|---|---|---|---|---|
| **Today, 80 ms RTT** | 167 ms | 2.260 s | 7.4% | **327.8 m/s** |
| If the 120 ms buffer were paid | 257 ms | 2.350 s | 10.9% | 315.3 m/s |
| Zero machine loop (impossible) | 0 | 2.093 s | 0% | 354.0 m/s |

Sensitivity: at 2 g the three figures are 277 / 268 / 296 m/s; at 6 g they are 431 / 409 / 477 m/s.
The **ordering** is robust across the whole plausible band; the exact crossovers are not.

Above eye height the horizon opens fast — 5.7 km at 100 m altitude, 18.0 km at 1 km, 56.9 km at
10 km (`block_system_design_addendum_1.md` C.3) — so this limit is a ground-level limit only. At 100 m
altitude even 528 m/s needs 1,193 m of the 5,687 m available.

### 3.5 The four speed ceilings, in binding order

| # | Ceiling | Speed | Source |
|---|---|---|---|
| 1 | **Terrain upload bandwidth at the V1 vertex format** | **~240 m/s** | `docs/investigation/block_system_design.md:7937` — 100 m/s already consumes 26% of the 480 MB/s budget; 528 m/s is a 136% breach |
| 2 | Latency + reaction + 3 g pull-up, at eye height | **~328 m/s** today; 315 m/s if the buffer were paid; 354 m/s at zero latency | §3.4 |
| 3 | AoI warm ring (`R₀ / t_warm`) | 393 m/s | `docs/investigation/block_system_design.md:7650` |
| 4 | Terrain generation + meshing, 2 cores, 100 km view | 528 m/s pessimistic, 2,653 m/s typical | `docs/investigation/block_system_design.md:7671` |

**Bandwidth binds 88 m/s below anything latency does.** The entire span the prediction question is
worth is 315 → 354 m/s — **12.3%** — in a regime the graphics budget already forbids. At the packed V2
vertex format the same 528 m/s costs 12% of the budget instead of 136%, at which point ceiling 2 (the
control loop) becomes the binding one at ground level and ceiling 3 (AoI) at altitude.

For scale, 240 m/s is 864 km/h: a low-level strike fighter, faster than an A-10 (~130 m/s), a WWII
fighter (~150 m/s) or any helicopter (~70–80 m/s), and slower than a Mach-1 sea-level dash (340 m/s).
It circles the 1,015.8 km starter world in 70.5 minutes.

---

## 4. Is it control latency or visual inconsistency? — settled

**It is control latency. The picture is internally consistent and truthful, with one bounded,
speed-independent 3.6% optimism about clearance.**

The decisive structural fact is that **terrain does not travel on the wire at all.** Roadmap P4:
*"deterministic-from-seed (no chunk streaming, no networked terrain — only the seed crosses the
wire)"*, with a definition of done of *"same seed ⇒ byte-identical terrain on server AND client AND
across two target-cpu builds"*. The client generates the mountain locally and the mountain does not
move, so a static object's position 81 ms ago is identical to its position now. There is no
terrain-versus-ship desynchronisation to have. The whole view is simply a true photograph of the world
81 ms ago.

### 4.1 The frame-by-frame trace

Setup: 200 m/s, 80 ms RTT, cliff face at x = 1000 m, ship at x = 0 at t = 0, 50 Hz server.

| Wall clock | Server truth | What the client draws | Comment |
|---|---|---|---|
| t = 2.740 s | ship at x = 548 m; 452 m to go | ship at x = 531.8 m; cliff *looks* 468.2 m away | The last moment a pull-up can succeed. See §4.2. |
| t = 4.500 s | ship at x = 900 m | ship at x = 883.8 m | Steady 16.2 m lag, no drift, no correction |
| **t = 5.000 s** | **impact — server resolves the collision** | ship at x = 983.8 m, still flying | 16.2 m short. The impact snapshot has just left the shard. |
| t = 5.011 s | wreck | ship at x = 986 m | The snapshot is in the gateway's inbox, waiting for its tick |
| t = 5.051 s | wreck | ship at x = 994 m | The impact snapshot has arrived and become `current` |
| **t = 5.071 s** | wreck | **ship at x = 1000 m, stopped at the cliff face** | The impact pose becomes `prev` and is drawn |

**The player sees a crash at the correct place, against the correct cliff, 71 ms late, with no
teleport, no rubber-band and no correction.** The pixels never lie about the geometry. What is late is
the *knowledge*.

### 4.2 The one real perceptual error, quantified

At the last moment a pull-up can succeed (true remaining distance 452 m at 200 m/s), the client draws
the ship 16.2 m behind where it is, so the cliff **looks 468.2 m away when it is truly 452 m away** —
a systematic **3.6% over-estimate of clearance**.

That fraction equals `render lag / (control loop + reaction + manoeuvre)` = 0.081 / 2.260, and it is
**the same at every speed**, because every term scales with `v`. It is a fixed, small, one-directional
optimism. It is well within the error of eyeballing a cliff face.

### 4.3 Where the "the picture is consistent" claim FAILS

Three places, and they are the honest counterweight:

1. **Other ships and other players are genuinely stale.** A remote ship is drawn 81 ms behind. Two
   ships closing at a combined 400 m/s see each other 32.4 m out of position. This is the one true
   visual inconsistency, no netcode choice here removes it (only extrapolation would, and
   extrapolation is exactly the mechanism that produces warping), and it is what the owed combat
   rewind (`docs/design/DEFERRED.md` D-42) exists for.
2. **The motion is stepped, not smooth** (§3.2). 4.8 m hops at 240 m/s. This is a rendering defect and
   it is the artefact the owner will actually see first.
3. **Block edits, at P6, could become a real inconsistency.** Edits DO stream (reliable discrete
   actions, per the P6 deliverable). If an edit is applied to the client's local voxel store **on
   arrival** rather than **at its stamped tick**, a block will vanish `render lag × v` before the
   drawn ship reaches where it was — 19.4 m at 240 m/s. That is a genuine terrain-in-the-wrong-place
   error, it is free to prevent by writing the requirement down now, and it is a rework of the edit
   ingest path to discover later.

---

## 5. What shipped games do

**Sourcing warning.** The survey below was gathered by a research pass with web access earlier in this
run; my own web budget was exhausted before I could independently re-verify any of it. Every
repository fact and every arithmetic result in this document stands on its own. Treat this section as
separately sourced and separately fallible, and re-source anything that becomes decision-critical.

### 5.1 The character-versus-vehicle distinction, front and centre

This is the load-bearing structure of the whole survey, and it decomposes into three tiers, not two:

| Tier | What it is | How predictable | Who does it |
|---|---|---|---|
| 1 | **A pure inertial body with no controller** — a ball, a projectile, a coasting hull | Almost perfectly. No input to guess. | Rocket League predicts the ball; Psyonix's own slide says it "works well with ball (predictable)" |
| 2a | **Your own vehicle in free flight** | Near-perfectly. You already hold the input; the only error source is simulation divergence. | War Thunder, Space Engineers, iRacing, Star Citizen (inferred) |
| 2b | **Your own character on foot** | Poorly. Steps, ledges, doors and other players' collision volumes generate constant server-side surprises. | Every FPS — and this is where correction snaps come from |
| 3 | **Someone else's controlled body** | Genuinely unpredictable. | iRacing's documented "phantom" double-scored contacts; War Thunder's extrapolated aircraft |

Rocket League's slide is the trap: "not as well with cars (unpredictable)" refers to *other players'*
cars, whose future input is unknown — tier 3, not tier 2a. Read correctly it is the strongest form of
the thesis, not a counter-example.

Gaijin states the vehicle thesis verbatim about aircraft specifically: *"since vehicles are highly
predictive and inertial it is fine most of the time"*, and rejects the alternative outright — a thin
client *"is not acceptable for high-speed vehicles with critical unstable states, such as aircraft…
as it makes controls sluggish and less responsive."*

### 5.2 The survey result

In the entire survey no shipped title was found that flies a fast, directly-steered *local* vehicle
without simulating that vehicle on the pilot's own machine.

| Title | Authority model | Own vehicle | Notes |
|---|---|---|---|
| Rocket League | 100% server-authoritative, 120 Hz | Predicted (everything, including the ball) | "Input delay is not an option." 200 ms ping = 24 correction frames |
| War Thunder | Server-authoritative | Own aircraft local; remotes **extrapolated** | Explicit rejection of thin-client for aircraft |
| **Space Engineers** | Server-authoritative | Own controlled entity is a dynamic body client-side; all others static/animated | **The closest analog that exists** — see §5.3 |
| Overwatch | Server-authoritative, 16 ms command frames | Predicted + rollback replay | Disables hit prediction above ~220 ms RTT |
| Battlefield 2042 | Server-authoritative | Client ticks ~6 network ticks ahead | Ahead-ness grows with ping |
| iRacing | Server-authoritative | Own car local; rivals estimated | Documented "phantom" contacts scored to both drivers |
| Roblox | Server-authoritative | Physics **ownership handed to the client** for nearby parts | "no latency from communication with the server" |
| Star Citizen | Server-authoritative, entity-authority-swapping | Inferred local + "network correction" | **No primary source found. Do not lean on this.** |
| DCS | Unclear | Community account only | **No primary source found. Do not lean on this.** |

Three categories genuinely do without it, and every one of them **removed the need** rather than
tolerating the lag:

- **EVE Online** — fully server-authoritative, ships move at hundreds of m/s, but *the player never
  steers*. "Approach", "orbit", "align" are commands. A quarter-second loop on a command interface is
  imperceptible.
- **Sea of Thieves** — server-side ship and wave physics, and its ships top out near 10 m/s, where the
  loop costs 2–3 m.
- **Racing titles that ghost remote cars** — sidestep the disputed-collision problem rather than
  solving it.

And one pattern that is the *opposite* trade and must not be confused with prediction: **World of
Warcraft and Minecraft are client-authoritative for movement.** The client asserts its position and
the server validates loosely. That feels perfect and needs no reconciliation at all — but it is not
prediction, it surrenders server authority, and HR1 plus the standing "server does all math" law
forbid it.

### 5.3 Space Engineers, in detail — because it is the same game

Voxel planets, player-built ships you walk inside, authoritative server. Keen's reported numbers:

- Clients interpolate on the last **60 ms**, "effectively operating 64 ms plus half ping time late" —
  very close to this project's nominal design.
- Server holds a 4-packet (66 ms) playout buffer so late input does not stall the simulation.
- **Without prediction: 231 ms input-to-photon at 50 ms ping.** With it: "shy of 100 ms."
  (Our 167 ms at 80 ms RTT is *better* than their no-prediction figure, because the buffer is not
  being paid.)
- Corrections are threshold-gated and smoothed: "correct only when necessary… the correction should be
  applied over time with small doses."
- They hit **exactly the moving-frame problem this project already has**, and named it the "time
  paradox": a character flying alongside a 50 m/s ship sees a hull "2.5 m late to the server's state"
  and boards through a door that has already moved. Their fix is **relative prediction** — parent the
  predicted entity to the ship, deliver and correct in the ship's local space. That maps one-to-one
  onto this codebase's `FrameRef`, and `EntityTrack::observe` already collapses the interpolation
  window on a frame change for exactly this reason.
- Their failure catalogue is the most useful part: they **disable** prediction for rotor/piston
  constraint chains ("the prediction error accumulates and corrections breaks the physics
  simulation"), for rotating frames (a ship spinning at 25°/s gives ~40 m/s of centrifugal velocity
  server-side and zero client-side), and when too many contacts occur against bodies that are dynamic
  server-side but static client-side. Their standing rule: *"when client simulation outcomes are
  sufficiently different to server's, we switch off prediction and resort to animation. The animation
  has worse latency, but without any desyncs."*

**Every item on that failure list is something this project is building**: P9 functional blocks
(pistons, rotors), spinning stations and rotating realms, and P6 voxel edits creating client/server
terrain disagreement. Expect to ship the same escape hatch and to keep extending it during QA.

### 5.4 What the survey actually proves, and what it does not

It proves that *if you want a hand-flown fast vehicle to feel like a hand-flown fast vehicle, every
shipped precedent runs it locally.* It does **not** prove that this project needs to, because this
project's binding speed ceiling is a graphics-bandwidth number well below the band where the
difference is felt, and none of the surveyed titles has a 1 m voxel terrain streaming budget.

---

## 6. Option A — the law holds absolutely

The complete design if no departure is taken. It is coherent, it is buildable, and it is what I
recommend.

### 6.1 Fix the three defects first (all law-preserving, all owed regardless)

**A1 — the command rate.** Raise client command assembly from 20 Hz to the server tick rate the client
has already learned from the wire. Removes 15 ms of mean loop, and — far more importantly — fixes the
speed defect below. One line plus its coverage.

**A2 — motion must not be slaved to the command rate.** `integrate`
(`crates/sim/src/stub.rs:2270`) is called from exactly one site — line 2264, inside `apply_input` —
and `apply_input` is called from exactly one site, the `SessionInput` arm at line 2055. There is no
held-input resource and no per-tick fallback. Each arriving datagram advances the pose by
`move_speed × tick_dt × time_multiplier`. With the shipped `move_speed: 15.0`, `tick_dt: 0.02` and a
20 Hz client, **actual speed is 15.0 × 0.02 × 20 = 6.0 m/s — 40% of configured**, and 20% packet loss
makes it 4.8 m/s. Input rides the unreliable datagram class by design, whose stated behaviour for a
lost frame is "skip a tick, apply the next fresh frame" — correct for a latest-wins absolute-state
avatar, **catastrophic for a body under continuous acceleration**, because a ship's engines stop
firing for the duration of every gap and its top speed becomes a function of its connection. The fix
is a held-input resource plus a per-tick integrate system. No wire change, no prediction. *This is the
real rubber-band-at-speed mechanism in this codebase and it must land before P8.*

**A3 — make the interpolator interpolate, and resize the buffer in the same slice.** Deepen
`EntityTrack` to hold `ceil(buffer / interval) + 2` samples, add the cursor slew that
`crates/client/src/render_clock.rs:11-18` already names as owed, and **simultaneously** set the
buffer to 2–3 snapshot intervals. Doing the first without the second takes the loop from 167 ms to
257 ms — a 54% regression at 80 ms RTT and a 93% regression on a LAN — which will arrive as "flying
suddenly feels awful" and be blamed on the law.

**A4 — make the buffer configurable and adaptive**, as `docs/design/connection_plane.md:346` already
mandates. Measure inter-arrival deltas, keep a windowed p99 over ~10 s (≈500 samples at 50 Hz), set
target = `interval + p99 jitter + max_consecutive_loss × interval`, clamp to
`[1×interval, 250 ms]`, apply increases within one frame and decreases at ≤5 ms/s (asymmetric, as
every VoIP jitter buffer and Source's `cl_interp_ratio` are). Apply changes as a cursor *slew*, never
a step. All bounds as fields on the one tuning struct. Hard floor is one snapshot interval — below
that the cursor runs past the freshest sample every frame and entities freeze and jump, which is
exactly the feel the law exists to prevent.

Result after A1–A4: **162 ms at 80 ms RTT, 92 ms on a good wired line, with genuinely smooth motion
for the first time.**

### 6.2 The speed envelope — derived, not authored

No magic number. Use dynamic pressure with a per-hull limit: `q = ½ρ(h)v²` with
`ρ(h) = ρ₀·e^(−h/H)`, where ρ₀ and the scale height H are per-body authored fields on the addendum-1
planet ladder and `q_max` is a per-hull field. Setting `v_max(0) = 250 m/s` gives `q_max = 38.28 kPa`
and the envelope 250 m/s at sea level, 335 at 5 km, 450 at 10 km, 810 at 20 km. Continuous, mode-free
(the seamless law), and tightest exactly where terrain is closest.

Aerodynamic heating is an independent and stronger limiter: Sutton-Graves at a 0.5 m nose radius gives
4.26 kW/m² at 250 m/s, 34.1 at 500 and 272.5 at 1000, whose radiative-equilibrium skin temperatures are
523 K, 880 K and 1,481 K — an unprotected steel hull is near melting at 1000 m/s at sea level. So
sustained sea-level flight caps naturally around 500 m/s with a dash to ~1000.

Control authority collapses independently: at 6 g the turn rate is `a/v` = 58.86/v rad/s = **33.7°/s at
100 m/s but only 3.4°/s at 1000 m/s**. **At high near-ground speed you cannot dodge anything anyway**, so
a low-altitude speed limit removes a capability that does not exist.

Context: on R = 161,671 m at 1 g surface gravity, orbital velocity is `√(gR)` = **1,259 m/s** and
escape is 1,781 m/s. 3000 m/s is a *space* speed with no atmospheric justification at all.

### 6.3 Assisted flight — move the fast loop to the server, where latency is zero

Split it in two, because the two halves belong in different places:

**(a) The hard envelope limiter is a property of applying thrust in a realm**, not a property of the
ship. It runs on the physics-authority shard — which per the standing motion law already integrates
the hull's input signals plus realm ambient physics, and already owns the terrain colliders. It reads
pose, velocity, mass and available thrust, plus a swept cast along the ~2 s predicted path (500 m at
250 m/s, fanned to ~5 rays at ±10°; microseconds against a parry `Voxels` BVH, per ship, at 50 Hz).
It clamps commanded g to the q-limited maximum, clamps speed to `min(q-limit, clearance-limit)`, and
refuses a commanded attitude that intersects terrain inside the minimum time. **HR4-generic**: a realm
with no atmosphere and no terrain clamps nothing, so the identical code runs on a ship-interior shard
and a planet shard and the `assert_feature_anywhere` fixture is straightforward. **HR3-clean**: it
never matches on a shard kind.

**(b) The terrain-following autopilot is a functional block on the ship grid (P9).** Pure dataflow, no
priority machinery: the seat publishes `pilot-pitch-demand` / `pilot-roll-demand` / `pilot-yaw-demand`
/ `pilot-throttle-demand` plus setpoints `agl-hold`, `heading`, `speed-hold`; the flight computer
consumes those and republishes the plain `pitch` / `roll` / `yaw` / `throttle` names the thrusters
already listen for. **With no computer fitted, the seat publishes the plain names directly and the
ship is fully manual — a configuration difference, not a code branch.** It emits advisory signals
`time-to-terrain`, `terrain-pullup`, `envelope-limit`, `agl` for any block-cover HUD widget.

Assist is a **continuous 0..1 scalar signal, never a mode toggle** (seamless law): 1 = full
terrain-following and collision refusal, mid = envelope protection with warnings but obedience, 0 =
raw, and the player who wants to kill himself may. Real precedent: the F-111 and B-1B flew 60 m AGL at
Mach 0.85 fully automatically, precisely because a human cannot close that loop.

**Why this is the right lever and prediction is not:** the server-side limiter queries the terrain
*function*, so it sees infinitely far and its safety loop is never visibility-limited or
latency-limited. It is the only mechanism in this entire document that removes the delay rather than
shrinking it.

### 6.4 Altitude is a discount on both axes

Terrain collision is impossible above the planet's maximum terrain height, so cruising above it makes
the whole question moot. Below it: at 10 km altitude the finest resident detail rung is
`⌈log₂(10000/786)⌉` = 16 m cells, column ingest collapses to `0.16·v` per second — "16× cheaper than
the same speed at eye height" — and the one-metre collision sphere around the hull contains no ground
at all, so the server holds nothing. The cost appears on descent, covered by the warm ring of width
`v·t_warm` at `t_warm = 2.0 s`, which is where the 393 m/s AoI ceiling comes from.

### 6.5 Survivable collisions — make the tax a repair bill, not a wall

A voxel ship is already a per-block damage model (block LIFE per block, per-substance toughness,
rim-only edge crumble). A terrain impact should delete or damage blocks in the contact region in
proportion to delivered kinetic energy, with a per-substance destruction energy (J/m³) as a substance-
table field. For a 15 t, ~12 m fighter at a 5 MJ/m³ crush energy: 75 MJ at 100 m/s destroys ~15 blocks
(survivable scrape); 469 MJ at 250 m/s destroys ~94 (a serious wound); 1.9 GJ at 500 m/s destroys ~375
(destruction). **The survivable/destroyed crossover lands at 100–250 m/s, which is exactly the
q-limited band — the physics is self-consistent without tuning.** Precedents: Elite Dangerous (hull
integrity + module damage + rebuy), Star Citizen (soft-death then hard-death), Kerbal (per-part impact
tolerance), Space Engineers (grid deformation).

Pair it with a ground-proximity warning: `time-to-terrain` and `terrain-pullup` emitted by the
physics-authority shard at a threshold of `loop + reaction + manoeuvre + margin` ≈ 2.3 s, so the
warning already accounts for the loop.

### 6.6 What Option A permits and what it forbids

**Permits:** orbital and interplanetary flight at any speed; 240 m/s atmospheric cruise at V1 and
250–450 m/s once the packed vertex format lands, with terrain 3–126 seconds away; assisted
terrain-following at 50–150 m AGL at full envelope speed; 100–250 m/s dogfighting where the loop is
17–42 m against ships with 150–3,200 m turning radii; landings and dockings at any speed with assist,
or up to ~10 m/s by hand where the loop costs under 2 m.

**Forbids, honestly:** hand-flown nap-of-the-earth above ~150 m/s with assist off; threading a 30 m
canyon at 300 m/s by hand; manual hangar entry above ~10–16 m/s; and any stick feel resembling a local
flight simulator. The aircraft will feel like a heavy jet with actuator lag — which is what heavy jets
feel like, and why fly-by-wire exists. None of those are available in Star Citizen or Elite Dangerous
either.

---

## 7. Option B — the narrowest scoped departure

Presented in full so the owner can judge it, not because I recommend it.

### 7.1 The mechanism, exactly

Scope: **exactly one body** — the entity named by `ServerControlMsg::OwnEntity` while the local player
is its input source — and **only while the server reports that body in contactless free flight.**

The client keeps a ring of the input frames it has sent (≤15 frames at 50 Hz covering buffer + RTT +
jitter ≈ 300 ms; ~41 bytes each, ~615 B total) and integrates them forward from the last authoritative
state using **the same shared code the server runs**. On each authoritative own-entity snapshot it
rewinds to that state and re-applies every input the server has not yet applied — ≤15 integration
steps per snapshot, ~750/s, microseconds.

**Wire delta: one trailing field.** `own_applied_seq: Option<u64>` appended to `SnapshotDatagram`
(postcard-additive per the existing rule, PROTO_MINOR 7→8, ≤9 bytes/datagram = ≤3.6 kbps at 50 Hz).
The server-side datum it echoes already exists: `Dot.last_applied_seq`
(`crates/sim/src/stub.rs:253`, maintained at `:2259-2265`).

Correction is two-layered: state snaps to authoritative truth immediately; the *rendered* pose carries
a decaying error offset with a smoothing time constant and a `hard_resync_m` threshold above which it
snaps and logs. Both new fields on `ClientInterpTuning`, never inline literals.

### 7.2 Why the safe boundary is CONTACT, not ships-versus-characters

Free-flight integration has no feedback from state into the derivative: `v += a·dt; x += v·dt`. Float
differences stay linear at machine epsilon and never compound. The current integrator is exactly that
shape — documented "Pure f64 closed-form per tick" — and it already runs on `vd_core::kinematics`, a
module written specifically so the input producer and consumer "cannot drift". Expected divergence over
one round trip: **~1e-13 m**, bounded by f64 epsilon over ≤10 ticks even if the one transcendental on
the path (`DQuat::from_euler`) differs by 1 ULP between builds — and it cannot compound, because the
angles come from inputs, not from integrated state.

Contact solving is the opposite. Contact-island ordering and discrete branch choices (step-up vs
slide, which face) make outcomes chaotic — **one branch flip is ~0.5 m immediately, and it then
compounds.** This is why the project's own P5 definition of done says "physics within tolerance" and
runs `physics_determinism` as "a soft warn-only diagnostic".

For comparison, the mechanism we are refusing — extrapolating a *remote* 5 m/s walker who reverses
direction over a 167 ms loop — is `2 × 5 × 0.167` = **1.67 m of visible warp on someone else's body**,
with no proprioceptive expectation to hide it. Ship free-flight replay versus remote-character
extrapolation is **1e-13 m versus 1.7 m: twelve orders of magnitude.** That gap, not a general claim
that prediction is good, is the entire argument.

Gating on contact also means the amendment **never names an entity kind** (HR3/HR4 clean) and leaves
the situation the law was written for — players walking and colliding, permanently in contact —
literally untouched. It also means **it does not fix the ground-vehicle case that started this
discussion**: a wheeled vehicle at 30 m/s is in permanent terrain contact and keeps the full loop.

### 7.3 The amendment sentence

> **The client never predicts anything it does not command.** It may integrate — locally, at present
> time, from inputs it has already sent — exactly one body: the one the local player is currently
> piloting, and only while the server reports that body in contactless free flight. It runs the
> identical shared integrator the server runs, it sends no new message, it decides no gameplay fact,
> and it yields to the server's authoritative state the instant they disagree. Everything else in the
> world stays on the interpolation buffer with no extrapolation, ever.

### 7.4 Collision: the client must NOT test it

**Tier 1 (the only one recommended if Option B is taken).** The client runs no collision test at all.
The authoritative own-entity state carries a contact flag; the instant it says "in contact", local
integration switches off and the craft reverts to the interpolation buffer until free flight resumes.
The client never decides a gameplay fact, the chaotic solver is never replayed, and the departure is
active only where it is provably exact.

The residual artefact is real and must be stated: between the hull clipping a ridge and the contact
flag arriving, the client has flown through solid rock for up to one round trip — **20 m at 500 m/s and
40 ms RTT, 80 m at 160 ms RTT** — then reverts to the impact point. That is a genuine
fly-through-then-snap. Its only defence is that it happens inside a crash, where a discontinuity reads
as an impact.

**Tier 2 (a render-only non-penetration clamp against the client's own terrain) does not survive
scrutiny, and this is where a claim made elsewhere in this analysis must be rejected.** The argument
was that client and server hold bit-identical terrain, so a local clamp is exact. It fails three ways:

1. **Resolution.** Detail rung L is used out to `785.7 × 2^L` m. At 500 m/s the required look-ahead is
   ~1,130 m — rung 1, i.e. 2 m cells. At 1000 m/s it is ~2,260 m — rung 2, 4 m cells. At 10 km
   altitude the finest resident rung is 16 m cells.
2. **The coarse record is a lossy summary, not downsampled truth.** A coarse cell stores 8 octant
   occupancy bits plus a mean fill and a dominant substance. Reconstructable vertical resolution is
   `2^(L−1)` m: ±1.0 m at rung 1, ±8.0 m at rung 4.
3. **Epoch.** Player edits stream as chunk deltas with an edit epoch, so the two sides agree only up
   to the epoch the client holds.

Concretely: a player digs a 1 m tunnel; a ship approaches from 800 m at 240 m/s. Server-side the edit
is a one-metre write, the collider is built from one-metre cells, and the hull correctly enters or
hits. Client-side at 800 m that chunk is past tier 0's 786 m reach, so it is one octant bit inside a
2 m cell — the client **cannot distinguish a 1 m hole from a 2 m hole** or say where inside the octant
it sits. Closing the gap requires streaming one-metre voxels ahead of the ship, which contradicts P4's
binding "only the seed crosses the wire" and reintroduces the very bandwidth ceiling that already
binds at 240 m/s.

The honest narrow statement that survives: **base terrain is bit-identical at one-metre resolution
within 786 m of the camera, at the epoch the client holds.** That is enough for the *server* to be
right. It is not enough for the client to adjudicate anything.

### 7.5 The anti-cheat rule, and why the surface delta is zero

**The rule: the client may compute a position for its own eyes and may never transmit one.**

The client keeps sending exactly `InputDatagram { seq, is_cut_marker, client_tick, movement, look,
action_bits }` and nothing else. A modified client can only lie to its own screen. The existing
validation is already the complete required set:

- `InputDatagram::is_finite()` rejects forged NaN/Inf before it can stick in an authoritative pose
  (`crates/wire/src/channels.rs:187-197`).
- The monotone-seq gate drops replays (`crates/sim/src/stub.rs:2259`).
- **Movement axes are re-clamped to [−1, 1] on the *consuming* side**, inside
  `vd_core::kinematics::local_axes_from_movement` (`crates/core/src/kinematics.rs:76-82`) — so a forged
  movement of 1e6 cannot speed-hack. The client-side clamp is convenience, not the defence.

Marginal validation cost of the departure: **zero**. One pre-existing gap, unrelated to this proposal
but worth logging: the *per-frame* look delta is unbounded — only accumulated pitch is clamped
(`crates/core/src/kinematics.rs:22-28`) and yaw wraps freely — so a forged look can snap aim
arbitrarily in one tick. Harmless for movement; directly relevant to P11 aim.

### 7.6 What Option B costs

**Cost 1 — co-moving flight gets WORSE, and misleadingly so.** Today everything including your own
craft is drawn at `t − τ`, so the picture is a self-consistent photograph and the *relative* vector
between you and another ship is exactly right for that instant. Under the departure your craft is
drawn at `t` and everything else at `t − τ`, so the rendered relative vector carries an error of (the
other body's velocity) × τ.

- Approach-to-static — landing, canyon flying, docking at an anchored station: the other body's
  velocity is zero, the error is **zero**, and the departure is a strict improvement.
- Co-moving — formation at 500 m/s, boarding another player's moving ship: the error is
  `500 × 0.081` = **40 m**, and it is worse than merely wrong: your wingman appears 40 m behind where
  he is, you throttle back to match, and you actually fall behind. A stable misleading equilibrium.
- Inside a ship interior (P8) the ShipLocal frame makes everything near-stationary relative to you,
  so the error collapses to near zero for free.

**Cost 2 — the packet-loss interaction is specific to this codebase and severe.** Per §6.1-A2 the
server does not integrate held input, so during a 200 ms input gap the predicting client flies 48 m at
240 m/s while the server's ship does not move at all. The divergence is not a rounding error, it is
the full `v × gap`, and the correction is a snap. **Option B is unsafe until A2 is fixed.**

**Cost 3 — it requires a carve-out in the determinism law, which is a bigger door than the prediction
question.** `docs/design/PLAN.md:122`: rapier state is Category C, "checkpoint-carried, NEVER
re-simulated on another host". A predicting client is precisely a second host re-simulating it. Either
the predicted body uses a shared closed-form free-flight path that never touches the solver — a
constraint currently written down nowhere — or the rule needs an explicit exception. SPIKE-6a is where
that gets decided.

**Cost 4 — it complicates the combat fairness question.** Today the shooter's render instant is
exactly computable server-side (the freshest tick sent, minus one), so the owed rewind takes a single
well-defined parameter and the shooter's muzzle is authoritative and is exactly what he saw. Under the
departure the shooter's origin is at present time while every target is at `t − τ`, and the server must
choose between honouring a client-supplied muzzle position (a cheat surface that does not exist today)
or using its own (felt as "my shots come from the wrong place"). Neither is free, and the choice is a
one-way door in the fairness model.

**Cost 5 — coverage.** The replay loop and the reconciler are branchy Tier-A client code at 100%
region+branch, and per HR5's generic-code discipline this forces careful monomorphic-helper factoring.
Space Engineers' talk lists at least six distinct disable conditions accumulated over time; each is a
branch pair.

**Cost 6 — it is a one-way door in feel.** Once players have flown with 8–17 ms of control response,
reverting to 162 ms will read as a regression regardless of what the law says. Turning it off later is
politically harder than never turning it on.

---

## 8. The recommendation, with the reasoning exposed

**Do not take Option B. Take Option A, in this order, and re-open the question only after flying it.**

The reasoning, stated so it can be disagreed with:

**R1 — The premise is not established, so the debate is premature.** The 120 ms buffer has never been
paid. Every hour of in-game validation ran at ~40 ms of client lag. Ruling on feel before the code does
what the law says is ruling on argument. (Disagree with this if you believe the inertness finding is
wrong — see §11.1 for how to falsify it in one throwaway test.)

**R2 — Prediction buys 12.3% of a speed band the graphics budget forbids you from entering.** The
latency-limited eye-height ceiling is 315 m/s if the buffer were paid, 328 m/s as it runs, and 354 m/s
at physically impossible zero latency. The terrain upload budget binds at **240 m/s**. You cannot spend
what prediction returns. (Disagree if you intend to make the packed vertex format a P4 blocker and
target 450+ m/s hand-flown at low altitude — in which case the arithmetic changes and Option B becomes
arguable.)

**R3 — The delay is a third-order term in the calculation that decides whether you crash.** 81.5%
manoeuvre physics, 11.1% human reaction, 7.4% machine loop. Even a perfect network leaves 92.6% of the
budget untouched. (Disagree if the 3 g / 50 m clearance assumption is wrong for your ships; at 6 g the
loop's share rises to 9.7% — still third-order.)

**R4 — The symptoms the owner associates with prediction are present today WITHOUT it**, produced by
three specific defects that scale linearly with speed. Fixing them is cheap, law-preserving, and owed
regardless. Building prediction on top of an un-fixed §6.1-A2 would be actively worse than not
building it. (This is the finding I am most confident of, and the one with the highest ratio of value
to effort.)

**R5 — The one lever that removes the delay rather than shrinking it is server-side.** The envelope
limiter and the terrain-following autopilot run where the terrain lives, at zero latency, with
unlimited look-ahead. Prediction shrinks 167 ms to ~17 ms for one body; the flight computer removes it
entirely for the safety-critical loop, and it is a buildable, upgradeable, tradeable ship component
that the player-driven economy can price. (Disagree if you believe a computer that refuses to crash you
removes the skill expression — that is a taste judgement the arithmetic cannot settle, and it is why
the assist scalar must reach a genuine 0.)

**R6 — The honest caveat, stated against my own recommendation.** The machine loop at 167 ms is
roughly the same size as human reaction time, so it **roughly doubles** the pilot's effective reaction.
My 7.4%-of-budget figure is true for one-shot terrain avoidance and false for continuous closed-loop
precision control, where the flying-qualities literature puts ~100 ms of added transport delay as the
onset of degraded compensatory control and ~250 ms as the point a pilot is forced into pulsed
move-and-wait, with pilot-induced oscillation likely. **At 162–247 ms this game sits on that
boundary for precision tasks.** That cost is real, it is speed-independent, it is worst at docking and
landing, and no arithmetic in this document settles it. Only flying it will — which the vdctl capture
harness can already script.

---

## 9. What must be reserved now, so either ruling stays cheap

Reserving is free; retrofitting is not. All six of these are worth doing whichever way the ruling goes.

| # | Reserve | Where | Why it is free now and expensive later |
|---|---|---|---|
| **RS-1** | `interp_buffer_ms` moves out of the compile-time const into the one externally-tunable struct, with an adaptive control law behind it | `crates/client/src/tuning.rs` → the `TransportTuning` contract at `connection_plane.md:346` | Already mandated by the spec and unimplemented. Without it no evidence can ever be gathered. |
| **RS-2** | A held-input resource + per-tick integrate on the shard | `crates/sim/src/stub.rs` around `apply_input`/`integrate` | Fixes a live 40%-speed defect; and Option B is *unsafe* without it. Retrofitting under a predictor is far harder. |
| **RS-3** | `own_applied_seq: Option<u64>` as a trailing field on the snapshot datagram | `crates/wire/src/channels.rs` | Postcard-additive, ≤9 bytes. Reserving the SHAPE now costs one PROTO_MINOR bump; adding it after P8 clients ship costs a negotiated migration. The server-side value already exists. |
| **RS-4** | A contact/free-flight flag on the authoritative own-entity state | the same snapshot path | It is the gate for Option B, the input to the ground-proximity warning, and the trigger for survivable-collision damage. Three consumers, one bit. |
| **RS-5** | Written constraint: **own-craft free-flight integration must be a shared closed-form path that never enters the physics solver** | `docs/design/PLAN.md` determinism section + the P5/P8 entries | Currently written down nowhere. If P5/P8 route free flight through the solver, Option B becomes impossible *and* the Category-C rule is quietly at risk. Costs a sentence now. |
| **RS-6** | Written requirement: **client block edits are applied at their stamped tick, not on arrival** | the P6 deliverable | Prevents terrain that changes ahead of the drawn ship (19.4 m at 240 m/s). A design constraint now; a rework of the edit ingest path later. |

Two more that are not "reservations" but must be scheduled before P8:

- **Close the entity-lane / realm-lane differential** (one tick, 4.8 m at 240 m/s) — SPIKE-10a's
  stated exit criterion, currently unmet, tracked only as a smoothness refinement in D-45.
- **Decide the cockpit camera model before P8 starts** — see the decision register.

---

## 10. The decision register

Ordered by cost of lateness.

| # | Decision | Options | Recommendation | Cost of deferring |
|---|---|---|---|---|
| **D-1** | **Confirm or refute the buffer-inertness finding**, then fix the interpolator, resize the buffer and raise the command rate — **in one slice** | (a) all three together; (b) interpolator only; (c) nothing | **(a)** | (b) is the trap: it takes the loop from 167 ms to 257 ms and the law gets blamed. (c) means flight ships with 4.8 m judder at 240 m/s. **Highest cost of lateness in the register.** |
| **D-2** | **Fix motion being slaved to the command rate** (held-input + per-tick integrate) | (a) fix now; (b) at P8 | **(a)** | Today it is a 40%-speed bug on an avatar. At P8 it is "my engines cut out when the network hiccups", and it will be diagnosed as a netcode-feel problem rather than a missing resource. Also blocks any future Option B. |
| **D-3** | **Is the snapshot rate meant to be 50 Hz?** Code emits every tick; every document says 20 Hz | (a) 50 Hz is correct, fix the docs; (b) 20 Hz is correct, add a divider | Settle it before D-1 | Changes the correct buffer depth by 2.5×. Every bandwidth figure written against 20 Hz is wrong by the same factor. |
| **D-4** | **Cockpit camera: bolted to delivered hull attitude, or free-look?** | (a) free-look (the local camera turns immediately, as mouse-look already does); (b) rigidly attached | **(a)** | **The single largest feel lever in this analysis and it costs nothing up front.** If a P8 cockpit rigidly inherits delivered hull attitude, the pilot's *view* rotation inherits the full 162 ms loop and flying will feel dramatically worse than walking does now. Decided after P8 starts, it is a camera-rig rewrite. |
| **D-5** | **Target flight speed band** — and therefore whether the packed vertex format is a P4 blocker | (a) ≤240 m/s, V1 is fine; (b) 450+ m/s, V2 becomes a P4 blocker | Owner's call; it is a gameplay one-way door, **not** a latency decision | Ship performance gets tuned around it. Deciding late means retuning every hull. |
| **D-6** | **Adopt the derived speed envelope** (`q = ½ρ(h)v²` per hull) + a continuous 0..1 assist scalar with 0 meaning genuinely raw | (a) adopt; (b) author a flat speed cap | **(a)** | (b) is a magic number and violates the seamless law if it manifests as a mode. Late adoption means new per-body fields (ρ₀, scale height) and per-substance thermal limits retrofitted across P4/P6 tables. |
| **D-7** | **Where the flight computer lives** | (a) split — hard limiter as a realm property on the physics-authority shard, terrain-following autopilot as a P9 functional block; (b) all of it as an engine feature | **(a)** | (b) loses the buildable/upgradeable/economy hook and the HR4 fixture. Also, (a) keeps the safety-critical half where the terrain already is, so it does not need the P9 cross-shard signal arm to exist first. |
| **D-8** | **Collision model: energy-based per-block destruction** reusing block LIFE, with a per-substance destruction energy (J/m³) | (a) adopt; (b) a ship hitpoint bar | **(a)** | (b) is a magic number and loses the salvage/repair economy hook. The substance-table field is cheap now and a schema migration later. |
| **D-9** | **Whether to take Option B at all** | (a) no; (b) yes, Tier 1 only; (c) Tier 1 + Tier 2 | **(a)**, and re-open only after D-1/D-2 are measured in-game | Deciding it late is *cheap* if RS-3/RS-4/RS-5 are reserved and *expensive* if they are not. Deciding it early and wrongly costs a determinism carve-out and a combat-fairness one-way door. |
| **D-10** | **Minimum drawn terrain radius ≥ `v_max × 4 s`** as a binding requirement on the detail ladder | (a) bind it; (b) leave it to the rendering budget | **(a)** | This is the one rendering-budget decision that can make the no-prediction law *unsafe*: draw less than ~1 km of terrain and the human loop becomes visibility-starved regardless of latency. |
| **D-11** | **Amend the law's wording** to state that the view transform is client-local while all simulated state is server-authored | (a) amend; (b) leave implicit | **(a)** | The client already turns the camera locally on mouse-look and labels it "NOT prediction" in two files. Writing the boundary down blesses what ships and fences the precedent so it cannot later be stretched into real extrapolation. |

---

## 11. Adjudicated objections

**11.1 — "The inertness finding is derived from a Python replication, not a test against the real
crates."** Sustained as a caveat, and it is the single most important thing to confirm before acting.
The finding is exact arithmetic over two short functions I read in full, and it reproduces under both
tick rates and under 30% loss. **Falsify it in one throwaway test**: build a two-sample `EntityTrack`
at one-tick spacing, drive `RenderClock` with consecutive ticks, and assert the sampled pose equals
`prev` on every frame. If that test fails, several conclusions in this document need revising, because
they all assume the buffer is not being paid. I could not add the test (no repository modifications).

**11.2 — "Speed makes latency worse, so fast ships are the problem."** Half sustained, half overruled.
Sustained for absolute error: everything scales linearly with `v`, so 4.8 m of judder at 240 m/s is
real and 0.6 m at 30 m/s is not. Overruled for *relative* error in the two dimensions that matter:
the loop's share of the terrain-avoidance budget is **speed-independent** (every term scales with `v`),
and its share of the turning radius **falls** as `1/v` — 32.8% at 10 m/s, 0.66% at 500 m/s. The
degraded cases are docking, landing, hovering and canyon work, not the dash.

**11.3 — "The buffer is what makes you 3–4.5 m behind at 30 m/s."** Overruled on attribution, sustained
on magnitude. The buffer is inert; today's render lag at 30 m/s is 2.43 m and the full control loop is
5.01 m. The figure quoted was roughly right for the loop and wrong about its source.

**11.4 — "Client and server hold bit-identical terrain, so a client-side collision test is exact."**
**Overruled, three ways** (§7.4): the detail ladder means the client draws 2 m cells at 1.1 km and 16 m
cells at 10 km altitude; a coarse cell is 8 octant bits plus a mean fill, so reconstructable resolution
is `2^(L−1)` m; and streamed edits mean agreement only holds to the epoch the client holds. Concretely,
a client 800 m from a 1 m tunnel cannot distinguish a 1 m hole from a 2 m one. The narrow surviving
statement — bit-identical at one metre *within 786 m*, at the client's epoch — is enough for the server
to be right and not enough for the client to adjudicate.

**11.5 — "Terrain flight is the case where lag hurts most, because terrain is a hard solid."**
Overruled, and the reason is structural: terrain never crosses the wire (P4: "only the seed crosses
the wire"), so it is generated locally and is timeless. A static object's position 81 ms ago equals its
position now. Terrain flight is the one collision case with **no** lag-compensation problem at all —
the picture is a true photograph of the past and the mountain is exactly where it is drawn. The genuine
staleness is other *players'* ships, which is D-42's territory.

**11.6 — "Predicting your own ship is the same thing that produced my bad experience."** Partly
sustained, and this is the distinction the run had to get right. Four mechanisms share the name.
Extrapolating *other* entities produces warping traffic — we do not do it and must never start.
Client-authoritative movement produces cheating — we do not do it. Server-side rewind produces "I was
behind cover" — a separate mechanism, owed for combat regardless. Own-body replay-and-reconcile is the
fourth, and it is genuinely the same *family* as the thing that hurt. **The load-bearing difference is
circumstance, not mechanism: contact versus free flight.** A walking character is in contact every
moment, contact solving is chaotic, and one branch flip is ~0.5 m — that is where correction snaps come
from. A ship in open air touches nothing, the integration is closed-form, and divergence is ~1e-13 m.
The difference is real (twelve orders of magnitude) but it is conditional, and if the same code were
ever allowed to run for a ship in contact it would behave exactly like the thing the owner hated.

**11.7 — "Every shipped game predicts the local vehicle, so we must."** Sustained as fact, overruled as
inference. No surveyed title flies a fast, directly-steered local vehicle without simulating it on the
pilot's machine — that appears to be genuinely universal. But no surveyed title has a 1 m voxel terrain
streaming budget that binds at 240 m/s, and the three titles that genuinely do without prediction
(EVE, Sea of Thieves, ghosting racers) all *removed the need* rather than tolerating the lag. Option
A's assisted-flight design is the same removal strategy, applied where it actually fits.

**11.8 — "A flight computer that refuses to crash you removes the skill."** **Sustained, and the
arithmetic cannot settle it.** This is a taste judgement. The shipped precedent everywhere (Elite's
FA-off, Star Citizen's SCM/NAV split) is a continuous scalar where 0 genuinely means you may kill
yourself, which is also what the no-modes rule demands. Ship it that way and the objection dissolves
into a player choice.

**11.9 — "167 ms is small against a 250 ms human reaction, so it does not matter."** **Overruled — this
is the argument I most want the owner NOT to accept.** The loop is 0.67× human reaction, so it roughly
*doubles* effective reaction time (417 ms total at 80 ms RTT). It is a small share of the *avoidance
budget* (7.4%) and a large share of the *reaction*. Both are true. The place it genuinely costs is
continuous closed-loop precision control, which is docking, landing on a moving pad and formation
flying — not the mountain.

**11.10 — "The entity-lane / realm-lane split means a 120 ms intra-ship slide at P8."** Overruled on
magnitude, sustained on substance. Because entities clamp to `prev` and realm boxes use the leading
edge, the differential is **one tick — 20 ms**, worth 4.8 m at 240 m/s, not 28.8 m. It still violates
SPIKE-10a's exit criterion and must be closed before P8, but it is a smaller problem than a 120 ms
differential would be — and note that **fixing the interpolator without also fixing the realm lane
would widen it to the full buffer.**

**11.11 — "Reserving the wire field commits us to Option B."** Overruled. A trailing
`Option<u64>` that is always `None` is postcard-additive, costs ≤9 bytes when populated, and is
inert when not. Reserving it plus the contact flag makes D-9 a cheap late decision instead of an
expensive one, which is exactly the property the greenfield rebuild exists to buy.

---

## 12. Residual risks and unverified inputs

- **The inertness finding is replicated, not tested in-crate** (§11.1). Confirm it first.
- **The shipped-games survey could not be re-verified in this pass** — my web budget was exhausted.
  Star Citizen and DCS in particular have **no primary source** and should carry no weight.
- **Ship dimensions, masses and thrust figures do not exist anywhere in the repository.** The 10–200 m
  hull lengths in §3.4(i) are genre-plausible assumptions. Keep the formula, recompute the crossovers
  when real hulls exist.
- **The 3 g / 50 m clearance / 250 ms reaction figures are mine, not the repository's.** At 6 g and
  20 m clearance the manoeuvre term nearly halves and the loop's share roughly doubles; the *ordering*
  of the four ceilings is robust across the plausible band, the exact crossover speeds are not.
- **No voxel, chunk, block or mesh code exists in the workspace**, and rapier3d, parry3d, noise and
  block-mesh are all absent from Cargo.lock. Every terrain-collision, residency and flight-speed number
  is design-only and unvalidated by a running system.
- **The human-factors thresholds** (~100 ms for degraded compensatory control, ~250 ms for forced
  move-and-wait) are a generic pilot model from the flying-qualities literature, not measured on this
  game.
- **The felt magnitude of the three defects is unmeasured.** No test measures cursor rewind under
  injected jitter, and none measures the frame-change window collapse at speed. Both are cheap to add
  to the existing harness and both should be measured before anyone concludes the visuals are fine —
  the arithmetic says they scale with speed, and walking-speed testing would not have surfaced either.
- **The display and peripheral chain (~25–40 ms at 60 Hz) is excluded** from every figure. It varies by
  a factor of two across hardware, and it is roughly the same size as a correctly-sized buffer.
