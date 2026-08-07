# Investigation base — block system, world, rendering, signals

**Status: NOT BINDING. These are inputs to investigations, not the outputs of decisions.**

Everything in this folder was produced in one intensive day (2026-08-03/04) of multi-agent research
and design exploration around the block system, the world model, rendering, and the signal system.
It is deliberately written in a confident, decided voice — that tone was useful for surfacing defects
and forcing arguments to conclusions, and it is **misleading if read as a plan of record**.

The owner's instruction, which is why this folder exists:

> *"For all those decisions about the world and blocks — we still should run the investigations before
> we start the implementations, so I'd mark all those documents as a base to investigate on, rather
> than a decisions."*

---

## What is binding, and what is not

| Location | Status |
|---|---|
| `docs/design/PLAN.md`, `roadmap.json`, `integration.json`, `DEFERRED.md`, and the hardened subsystem designs in `docs/design/` | **BINDING.** The approved plan, the six hard rules, the phase definitions of done, the deferred-work registry. |
| **`docs/investigation/` (this folder)** | **NOT BINDING.** Analysis, prior art, arithmetic, failure modes, one-way doors and framed questions. To be re-validated before implementation. |
| `scripts/*.md` | Working design docs from earlier arcs — same non-binding status, not yet reorganised. |

**Where a document here and a binding spec disagree, the binding spec wins** until an investigation
says otherwise and the change is made in `docs/design/`.

The **owner rulings** table at the top of `block_system_design.md` records the owner's stated direction
at the time each question was raised. It is a record of intent, not a frozen commitment — those rulings
are re-validated by the pre-implementation investigation for their phase like everything else here.

---

## What these documents are good for

Re-derive the conclusions; **reuse the inputs**:

- **Prior art**, sourced and evidence-graded — roughly forty shipped games and engines examined for
  what they actually do and what it cost them, with marketing claims separated from technique.
- **Arithmetic** — storage, bandwidth, frame budgets, crypto costs, chunk counts, and one benchmark
  that was actually run rather than estimated (see below).
- **Failure modes** found by adversarial review, several of which were live defects in the design or
  in shipped code.
- **One-way doors**, named with the moment each must be shut and what retrofitting would cost. These
  are the highest-value content here and should survive any re-investigation.
- **Questions framed properly**, with options and costs, so an investigation starts from a sharpened
  question rather than a blank page.

---

## The documents

| File | What it explores |
|---|---|
| `block_system_design.md` | The main body (§1–§8): the block model, the world and its grid, shapes, rendering, decoration, signals and functional blocks, and a delivery plan. Carries the owner-rulings table. |
| `block_system_design_addendum_1.md` | Planet size as per-body data on a ~39.5 m ladder; gravity derived from authored mass; seed-plus-configuration worlds; the curated starter world |
| `block_system_design_addendum_2.md` | Detail levels: the retraction of the "no LOD" ban, and why fractal terrain coarsens for free |
| `block_provenance_collapse.md` | Three-state provenance (terrain / biome feature / player-placed) and what may collapse under gravity |
| `concealed_resources.md` | Whether resource placement can be hidden from a client that generates its own terrain |
| `collidable_decoration.md` | What collides, what merely responds, and what vehicles demand of both |
| `stunning_look_plan.md` | What makes a blocky world beautiful, ranked by beauty per millisecond, against a frame budget |
| `storage_and_streaming.md` | Columnar formats, surface-only representations, and streaming player-built megastructures |
| `high_speed_flight_latency.md` | Fast flight under a no-prediction, server-authoritative model |
| `signal_authority_and_relays.md` | Channel ownership and the takeover attack; relays as built infrastructure; the force-to-parent handoff |

A tenth document, `decision_board.md`, reconciles all of the above into one register and slice order.
It lands in this folder when its run completes.

---

## The one thing here that IS a measurement

`scripts/noisebench/` is a standalone benchmark that was **built and run**, not estimated. Its numbers
(one noise evaluation at 12.19 ns; a surface chunk in 0.885 ms with a coarse-lattice cave field) are
real measurements on an Apple M4 Pro, reproducible with `cargo run --release`. It is its own workspace
root, so `cargo check --workspace` never sees it.

Treat those figures as data. Treat everything derived from them as argument.

---

## How to use this folder

1. **Before implementing a phase**, run its investigation. Start from the relevant document's framed
   questions and one-way doors; do not start from its conclusions.
2. **Re-derive anything load-bearing.** Several conclusions here were reached, contradicted by a later
   run on the same day, and re-decided. That is the process working, and it means no single statement
   should be trusted because it is written confidently.
3. **Promote what survives.** When an investigation confirms something, the result goes into
   `docs/design/` as a binding spec — that is the only place a decision becomes real.
4. **Watch the doors.** The one-way doors are the items where being wrong is expensive rather than
   annoying. They deserve investigation effort out of proportion to their size.
