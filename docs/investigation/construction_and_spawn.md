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

# Construction, insurance, and the shared starting point

**Status:** ruling, for owner decision. Adjudicates six owner asks recorded 2026-08-04, plus owner
ruling R18 (the dock construction mechanic), against five parallel investigations and one adversary
pass.
**Date:** 2026-08-04.
**Scope:** blueprint placement constraints; construction companies, build robots and materials;
insurance and non-destructibility; splitting the shared spawn into several Area realms; and the single
shared starting point itself.

---

## 1. The owner's six decisions

The owner's words, verbatim, split into the six separable things they contain.

> **1.** *"I thought that we have one starter point for everybody. That will allow to build economy
> emerging from one world to others."*

> **2.** *"Each blueprint should include some parameters, for example - buildable only on planets, or
> only in space, or only in docks. Space station can be built only in the planet's SOI, ship - only in
> DOCK, Space Port only on the planet's surface."*

> **3.** *"the only thing is that we have to pick a construction company, that will provide the build
> robots (can be emulated) and materials."*

> **4.** *"For big constructions you can pay additional cost to buy insurance - that means that your
> building will be undamageable."*

> **5.** *"Also for that place where all players will appear, I want to be able to place several Area
> realms, otherwise we might have overload issues when there are thousands of players in one space
> port."*

> **6.** *"Also I wanted to make some parts non-destructable."* — implied again by (4), and separated
> here because it is a different mechanism from insurance and a far cheaper one.

---

## 2. The answer in one page

*This page has no technical words in it. It is the one you read; everything after it is the working.*

**One starting point for everybody: keep it, but for a different reason than you gave.** You said it
lets the economy grow outward from one world. It does not do that on its own — what makes a thing worth
more far away is that somebody had to carry it there, and that is true whether or not everyone started
in the same place. What one shared start genuinely buys you is that on the first day of the game there
is a market with actual buyers and actual sellers in it, and everyone can see what a load of iron sold
for. In a game with no computer-run shops, that is the difference between a market that works and a
market that never gets going. Keep it for that. The bill: every new player walks through the same door,
so that door is the busiest and most attacked place in the world, and the land around it will be
stripped bare within about two weeks of a thousand people playing.

**Blueprints saying where they can be built: yes, and it is almost free.** Each blueprint carries a
short list of the kinds of place it is allowed to go and roughly what sort of ground it needs, and the
game checks it twice — when it quotes you a price, and again the moment it starts building. It never
checks again after that. What will bite you that you did not name: the check has to be done in a way
that also stops one player planting a station inside another player's station, and nothing in the game
does that today. Also: a dock should be a marked-out bay inside a ship or a building, not a place in
its own right — making it a place in its own right costs real work for nothing.

**Construction companies with robots and materials: the mechanic is right, the timing is wrong.** A
construction company should be nothing more than a name and an owner written on a shipyard, so that the
game's own starter shipyard and a shipyard a player builds later are literally the same thing with a
different name on the sign. The materials should come out of the ground — the player hauls them in, or
buys them from whoever hauled them in. The robots should not be real objects flying around; the ship
should simply grow, block by block, in front of you, and the robots should be decoration your own
machine draws around the growing edge. That is roughly a thousand times cheaper than real robots and it
looks better. **But none of this can be built yet**, because the game has no way to hold a stack of
anything — no boxes, no inventories, no items at all. That is a real piece of work nobody has scheduled,
and construction is stuck behind it. This is a delay, not a refusal.

**Paying to make a building undamageable: this is the one I want to argue with you about.** You asked
for two different things in the same sentence and only one of them is a good idea.

The first — *"some parts non-destructable"* — is right, cheap and should ship. The starter spaceport
should simply be unbreakable, by design, not for sale, and not something anyone can buy for anything
else. Every game does this. It costs about a dozen lines.

The second — *paying money to make your own building undamageable* — I recommend against, and here is
why in one breath. Once the strongest defence in the game is a payment rather than a position, fighting
over territory stops meaning anything, because the answer to every siege is a purchase. Worse, it cannot
be made to work: to stop someone digging out the ground beneath your protected building, the protection
must cover the ground too — which means selling protection is selling land. And worst, on the day the
part of the system that handles money is down, there is no correct behaviour. If protection stays on,
every protected building in the game is simultaneously unbreakable and cannot be cancelled. If it
switches off, every protected building in the game becomes vulnerable at the same moment, which any
attacker will arrange. **The version I recommend instead is insurance that pays out rather than
protects**: your ship is destroyed, and the insurance pays for it to be rebuilt at a dock from the plan
it was built from. That is what the real-world word means, it is nearly free because the dock already
does the rebuilding, and it creates a job somebody can do — underwriting other players. This can wait
until docks exist.

**Several separate areas at the spawn so a thousand people do not overload one machine: half right, and
the half that is wrong matters.** Splitting the spaceport into eight areas genuinely relieves each
machine, by about seven and a half times. But it only works if people in one area cannot see people in
another. If they can see across, the split makes things about five per cent **worse** than not splitting
at all — because everyone still receives everyone, and now in slightly more, slightly emptier packets.
So the boundaries have to be real walls, decks and airlocks in the architecture of the port, not
invisible lines across an open concourse. And you should know that the thing that would actually fix
crowding — only telling each player about the people near them, instead of about everyone in the
building — is a change nobody has made yet, and it is worth about thirty-three times what the split is
worth. Do that one first. Also, honestly: if everyone crowds into one plaza, the split does nothing at
all, because they are all in the same area again. The split is a way of building several rooms, not a
way of absorbing a crowd in one room.

**One thing you did not ask about that will happen anyway.** Nothing today stops twenty accounts
standing shoulder to shoulder in the door of the spaceport. Players physically push each other, which
you have asked for and I am not proposing to change, so no rule about damage or protection touches this.
The only cure is the shape of the building: many wide entrances, no single choke point, and new arrivals
appearing spread across a volume rather than on a spot. That has to be in the spaceport's design before
it is built, not after.

**And one thing in the existing plan that has to change.** Ruling R18 says you pay at a dock and the
payment is what causes your ship's world to be created. That has to be turned around: paying can be how
you get the parts, but paying must never be the thing that makes a piece of the game start running — or
the day the money system is unavailable, either nobody can get a ship, or everybody gets one free. The
brake should be a crafted item you consume, which the plan already has. You still pay for a ship. Paying
just stops being what launches a process.

---

## 3. Blueprint placement constraints

### 3.1 The predicate, and why a constraint language would be gold-plating

Every case the owner named is a pair of facts the containment machinery already computes each tick, so
the whole constraint is a fixed-width record in the blueprint header and two mask tests:

```rust
/// Blueprint header field. 12 bytes. Persisted, copied and re-sold data — see the door below.
pub struct SiteRequirement {
    pub host_kinds: u16,           // bitset over RealmKindTag (6 defined, 10 reserved)
    pub classes: u16,              // bitset over SiteClass   (6 defined, 10 reserved)
    pub min_altitude_m: u32,
    pub bound_half_blocks: [u16; 3],
}

pub enum SiteClass { Grounded, Suspended, Orbital, FreeSpace, Dock, ConstructInterior }
```

Admission is `((host_kinds >> host as u16) & 1) & ((classes >> class as u16) & 1)` — branchless, total,
monomorphic, so HR5's per-monomorphisation trap never applies. Reserved bits are **rejected on decode if
non-zero**, the same slack discipline owner ruling R1 applied to the eight-byte placement record.

The four cases the owner named are one row each:

| Blueprint | `host_kinds` | `classes` | `min_altitude_m` |
|---|---|---|---|
| Spaceport | `{Planet}` | `{Grounded}` | 0 |
| Space station | `{Planet}` | `{Orbital}` | authored |
| Ship | all | `{Dock}` | 0 |
| Anything | all | all | 0 |

The masks admit a cross-product rather than a list of exact `(kind, class)` pairs. That is lossy only for
a blueprint wanting two specific pairs but not their mix, which none of the owner's cases is, and the
reserved bits let an explicit pair list land later without a format change.

**A general constraint expression language is refused.** It adds an evaluator, a decode surface and an
adversarial input for zero cases the owner named. If it is ever adopted it must be bounded and
content-hashed into the connect handshake exactly as the per-body override table is (addendum 1 §D.3).

### 3.2 "In the planet's sphere of influence" costs nothing — verified

`crates/core/src/worldgen.rs:483` and `:655` construct the Planet body with `shape: shell(pl.planet_soi_r_m)`.
**A planet realm's containment region and its sphere of influence are the same volume.** `geometry::container`
(`crates/core/src/geometry.rs`, folded every tick) already returns the deepest containing `RealmId`, never an
`Option`. So `host_kind == Planet` *is* "inside that planet's SOI", and "somewhere under a planet" is a scan
of the `RealmPath` the RLM already carries. No distance query, no second spatial notion, nothing added to
the admission path.

### 3.3 The awkward case: a station "in the SOI" three metres above a mountain

The SOI shell reaches all the way down to the ground, so the naive predicate would let a space station be
placed on a hilltop. The fix is exact-integer and needs no terrain sample: a planet's writable grid is the
closed radial band `[floor_radius_m, ceiling_m)` with `ceiling = R + H_scale` (~8.4 km above datum on the
starter body at R = 161,671 m), and the generator carries `max_relief ≤ H_scale/2`, so the ceiling is
guaranteed at least 4.2 km above every generated peak (§3.6). Therefore:

> `SiteClass::Orbital` = inside the Planet region **AND** `addr_of(anchor)` returns `None`.

O(1), deterministic, identical for every observer, no float evaluation. `min_altitude_m` then pushes the
floor further out **as blueprint data**, so there is no magic number in the rule. Testing against the
analytic `atmosphere(seed, r)` field is available per blueprint later and has the identical comparison
shape; it loses on the admission path only because it costs a float for no stronger guarantee.

`SiteClass::Grounded` is one class shared by planet surface, station floor and ship floor. The
`host_kinds` mask already separates "only on a planet" from "only in a station", so three separate
grounded classes would encode the same distinction twice.

### 3.4 A dock is a declared box, never a realm — and this is forced, not preferred

`frame_for_realm` (`crates/core/src/pose.rs:132`) resolves `RealmId::Area` **only** with
`Some(RealmId::Planet(_))`; every other parent returns `None`. `worldgen.rs:218` calls it with
`.expect("roster realms have a canonical frame")`. So a dock bay modelled as an Area inside a station, or
inside a spaceport Area, **panics the forest build** — it is not a degraded path.

The box alternative costs nothing new. `Boundary::Obb { half, orient }` with `box_signed_distance` is
already in `crates/core/src/geometry.rs:229`, so a point-in-dock test is one existing call on a
centre-relative position. It composes identically for a dock on a ship, in a station, and on a surface
pad — HR4 satisfied by construction — and it spends zero realm depth. Deriving the dock's box from
dock-marker blocks means it is maintained by the ordinary edit path: destroy the markers and the dock
stops existing, with no background watcher.

**A dock must declare a capacity**, and this is load-bearing rather than a nicety. `child_fits_in_parent`
(`geometry.rs:1078`) is the fence that would otherwise catch an oversized ship minted at the rim of a
station box — but its first line is `if child.frame != parent.frame { return None; }`, and a ship's frame
is `ShipLocal` while a station's is `StationLocal`. The fence is skipped by construction. The site
predicate is therefore the **only** place a dimension check can happen: refuse a blueprint whose
`bound_half_blocks` exceeds the dock's `max_half`. Skipping it means discovering the problem at station
**boot**, days after the build.

### 3.5 What realm kind each construction becomes

| Construction | What it is | Verdict |
|---|---|---|
| **Spaceport** | Blocks in the planet grid, plus optional sibling Area realms for crowd load | Never its own realm. §3.6's detachment rule forbids re-homing a voxel volume across a Spherical→Identity grid boundary; `ShardProfile::satisfies()`'s exact geometry equality refuses it |
| **Space station** | A `Station` realm under the `System` (as the shipped forest nests it), or blocks in a construction realm | Existing kind, works today |
| **Ship** | A realm — **and it cannot be addressed by the lifecycle machinery at all** (§3.7) | Blocked |
| **Floating base** | A construction realm: Cartesian, Identity mapping, parented to the planet, minted by a `ConstructionAnchor` (§3.6) | Designed; same object as a ship without engines |

The Area realms at the spaceport are a **load** split, not a build container: the port's blocks stay in
the planet grid whichever Area a player stands in, which keeps one writer per store (§3.9.1) and keeps
the cube-sphere addressing unsplit. That is the recommended answer to "does an Area own its voxels" —
**no, the planet keeps them.** The alternative needs the Area to inherit Spherical geometry to satisfy
`satisfies()`, which the shipped Area region (a Cartesian box) does not.

### 3.6 Does the six-level realm cap bind? — No. Three vocabulary holes bind first, and one is worse than reported

`RealmPath` is a plain `Vec<RealmLevel>` with no length cap. Nothing counts to six. The owner's deepest
legal configuration — Universe / Galaxy / System / Planet / Station / Area — is six levels and the ladder
has room. **The count is not the problem.** Three other things are, and all three are verified in code:

1. **Area under Station is a panic, not a refusal.** `frame_for_realm(Area, Some(Station)) == None`,
   consumed by `.expect()` at `worldgen.rs:218`. The owner's item 5 (several Areas wherever thousands of
   players gather) is therefore impossible at a busy station, which is exactly where it is wanted.
   `FrameRef::AreaLocal { planet_seed, area_seed }` has no room for another parent kind.
2. **A ship has no realm kind.** `worldgen::level_of` returns `None` for `RealmId::Ship(_)`
   (`worldgen.rs:300`, explicitly: *"a ship is ENTITY-backed, NOT a seed-lineage realm"*). The entire RLM
   demand loop is keyed on `RealmPath` — `DemandLedger { cells: BTreeMap<RealmPath, LedgerCell> }`
   (`crates/sim/src/rlm.rs:234`) — so a ship realm cannot be spun up, kept alive or torn down. **R18's
   headline mechanic has no code path.**
3. **Appending `RealmKindTag::Ship = 6` does not fix (2), and every prior run missed this.**
   `RealmLevel { kind: RealmKindTag, seed: u64 }` carries a **u64** payload. `RealmId::Ship(EntityId)`
   carries a **u128** (`crates/core/src/ids.rs:157`, packed `kind:8 | mint_shard:32 | seq:64 | rand:24`).
   A ship's identity does not fit in a level. Making ships addressable by path therefore needs either a
   widened level payload (a wire change on a frozen postcard type) or a seed-lineage identity minted for
   the construct at birth. **The second is much cheaper**: a construction realm is already seed-keyed and
   already the same object as a ship without engines, so a ship should carry a `Construct(u64)` lineage
   identity minted at build time and keep its `EntityId` for physics.

Note also that `ProfileKind` (`crates/core/src/taxonomy.rs`) already has **eight** kinds — Galaxy, System,
Planet, **Ship**, **Asteroid**, Station, Area, Stub — against `RealmKindTag`'s six with neither Ship nor
Asteroid. The two vocabularies claim to mirror each other and already do not.

**Recommendation:** append **one** generic `Construct = 6` covering ship, station interior, construction
realm and dock host, carrying a minted `u64` lineage seed; and append **one** `FrameRef` arm carrying a
generic parent realm instead of `AreaLocal`'s hardwired planet seed. Both are append-only and free today.
Appending `Ship` alone repeats the mistake — `Room` is next, and `Asteroid` is already waiting in the
other enum.

### 3.7 Evaluate the predicate exactly twice, and never again

At quote, and at mint. Not per tick.

"The ground was mined away later" is answered by the structural-support mechanic that already exists:
§2.5.2's support flood plus owner ruling R10 (axial-free, lateral-limited, `MAX_SUPPORT_RADIUS = 32`)
means a port whose footing is mined is eaten progressively rather than vanishing, bounded at 45,825 cell
visits per edit, deferred and budgeted, and zero for an edit in undisturbed terrain. Re-running the site
predicate every tick would put per-realm work proportional to the number of standing constructions on the
tick loop, and would make a legal build illegal with nobody acting.

There is an existing precedent to copy verbatim: R19 re-orders the autopilot handshake to re-resolve the
effect set **at mint** and refuse if it changed while the dialog was open. §3.9.7's `expected: BlockState`
CAS guard is the same idiom one level down.

### 3.8 Nothing stops one player building on another's work — and the fence that looks like it would never fires

Edit admission today is reach, realm membership and rate (§3.9.7). Ownership is item 4: *"a stated but
deferred model"*, with a `ClaimId` TLV tag reserved on `ChunkRecord` and nothing behind it.

For realm-versus-realm overlap the situation is worse than "deferred", and this correction matters because
one prior run reported it wrongly:

- **The boot fence IS wired in production.** `crates/bins/src/bin/shard.rs:248` calls
  `guard_regions_nest(&regions, MAX_REGIONS)` before the infallible `RealmRegions::new`, and fails the
  boot loudly. The claim that it has no production caller is **false**.
- **But it never checks siblings.** `guard_children_fit_parents` compares each child to its *parent* only.
  Sibling-versus-sibling overlap is unchecked, so an authored overlap boots silently, and `container()`'s
  tiebreak (depth DESC, then `RealmId` ASC) hands the player to whichever sibling has the lower id
  regardless of where they physically are.
- **And the geometric half is a no-op on the shipped forest.** `child_fits_in_parent` short-circuits with
  `if child.frame != parent.frame { return None; }`, and `frame_for_realm` gives every realm a distinct
  `FrameRef` keyed on its own id — so **no parent/child pair in a generated forest shares a frame**. The
  check is exercised only by synthetic fixtures that all use `FrameRef::SystemSpace { system_seed: 0 }`.
- **And the count fence lives in the binary, not in the constructor.** `RealmRegions::new` is infallible
  with no length check, and both the in-process harness (`crates/harness/src/topology.rs:1513`) and
  `tests/src/lib.rs` construct it directly. `RegionMembership` is a `u64` and the detector's scan does
  `1u64 << ix` with `ix` from an unbounded `enumerate()`. In the shipped `[profile.release]` (which sets
  only `strip` and `lto`, leaving overflow checks off) a 65th region would alias onto region 0. The shard
  binary is protected; nothing else is, and the protection is one refactor away from being lost.

**The minimum sufficient answer** is two things, and neither is a claim system:

1. A **sibling-overlap refusal** added to `guard_regions_nest`, O(direct siblings) and bounded by
   `MAX_REGIONS = 64`, plus moving the count check into `RealmRegions::new` so it cannot be bypassed and
   bounding the bitset index. That gives "you cannot put your station inside mine" from pure geometry with
   zero new records.
2. For the surface, where there are no sibling regions: **one immutability flag on the starter port's
   blocks plus one declared no-build box around it**. Two fields, and it delivers the owner's items 1, 4
   and 6 together.

A per-chunk claim record carrying `owner: AccountId` is the next tier and can wait. When it lands, ruling
R6-9 already requires that a falling group's landing counts as an edit and passes the same check — or
dropping a tree on a protected build bypasses every permission check by construction.

---

## 4. Construction companies, build robots and materials

### 4.1 The mechanic: one field, not a subsystem

A construction company is **an owner field on a build site**. The site is the expensive thing and it is
entirely game state: a build volume, a materials hopper (a *game* container), a job queue, an assembly
rate and a fee rate. The company is metadata — `owner: OwnerRef`, a name, a fee rate.

`OwnerRef::World` versus `OwnerRef::Actor(..)` is the *only* difference between the game's starter
shipyard and a player-run business. Nothing forks, so HR3's "never match on a shard kind in feature code"
is satisfied trivially and there is no NPC-service subsystem to remove later. The starter yard is stamped
at world creation exactly as addendum 1 §D.4 stamps the spaceport, with `owner = OwnerRef::World`.

**Reserve `OwnerRef` with a `World` arm before the first site is persisted.** Adding the arm afterwards is
a store migration over every saved yard.

The owner's concession that robots *"can be emulated"* turns out to be unnecessary rather than exploited:
under the recommended answer there are no robot entities at all, so there is nothing to emulate.

### 4.2 The economy boundary — and the brief's prior is inverted

The shipped interface is `EconomyPort` (`scripts/dormant_world_simulation_design.md` §3.2) with exactly
two methods, both returning `()`. The document states the review rule explicitly: *"both methods return
`()`, so no method's absence can prevent a game action. If a third method ever returns a value, the trait
is wrong."*

"Construction emits a refusable command and the economy answers" requires precisely that third method,
because *answers* means the game waits for a value before mutating the world. Two consequences follow
immediately: with the economy compiled out no answer ever arrives, so construction never happens and the
three-arm `G-ECON-ABSENT` gate fails **by design**; and the same document already deleted a weaker version
of the same idea (`weights(SubjectId) -> PolicyWeights`, removed because *"a total value is still a value
a game decision reads"*).

**The correct direction is the one already built.** A build job has exactly **one** precondition and it is
physical: *are the materials in this yard's hopper*. Materials get there by hauling (works with the
economy compiled out, under total partition), by buying them from the yard as a barter bundle (existing
market machinery), or by paying the company — which decomposes into the second plus a `DeliverGoods`
`EconCommand`, a verb already in the enum (`MoveStack | GrantHull | DeliverGoods | SetLien | ClearLien`).
The economy's only influence is that refusable, idempotent, fenced command; the game's only output is
unconditional `WorldFact`s absorbed by a null sink.

**Two rules must be written down now because both are free today and a redesign later.**

- **The fee is never taken in materials out of the build's own bill.** `G-ECON-ABSENT` runs the accumulated
  scenario suite three ways and asserts byte-identical results. A build-a-ship scenario belongs in that
  suite. If the yard's cut is a percentage skimmed off delivered material, the economy-on arm consumes a
  different quantity of matter than the economy-off arm, the resulting construct differs, and the
  assertion fails — not from a bug but because an economic rate was put on the matter path. This rhymes
  with `money_and_markets_design.md` §5.1's already-adopted *"the cut is fixed at listing time — a design
  requirement, not an implementation detail"*, for the same class of reason.
- **A world-owned yard's fee is burned, not accumulated.** A world-owned yard building a treasury is a
  faucet with no counterparty, and `economy_research_20260726.md` §4.9 names the floor-price-buyer-with-a-purse
  as a top-ranked degeneracy. Before currency exists, charge nothing.

### 4.3 R18's payment brake is a live rule violation, and the substitute already exists

R18 consequence (b): *"the payment is the anti-abuse brake on realm spin-up, which is a real server cost
and needs one."* `decision_board.md` repeats it as a P8.1b deliverable.

`scripts/dormant_world_simulation_design.md` §3.4, LAW-WL-7: *"No economy value and no life-tier value may
enter `desired_alive`, `teardown_ready`, the `aoi_decide` OCCUPANT SET, or the `AoiMembership` map."* A
payment that gates a realm mint is an economy value deciding process topology — the deepest available
decoupling breach. It would be caught by `G-WL-LIFECYCLE-BLIND` (toggle the economy mid-scenario, assert
the RLM action trace is bit-identical) and by the economy-absent gate (with the economy off nobody can
pay, so nobody can ever get a ship). The same document already flags an identical proposal in
`economy_research_20260726.md` §6.4 as *"a defect"*.

**It also has no correct failure mode.** `money_and_markets_design.md` §7.1(1) lists what stops when the
account service is undeployed: deposits, withdrawals, transfers, wages, treasuries, note issue, balance
display, remote proceeds. Fail-closed: nobody can obtain a ship while an economy service is down — a P8
game feature unavailable because an economy service is unavailable. Fail-open: free ships for the duration
of an outage an attacker can induce. There is no third option.

**The substitute is already designed and costs nothing.** §3.6 makes minting a construction realm require
placing a crafted `ConstructionAnchor` item, capped by `max_anchors_per_account` in `BlockStoreTuning`,
refused past the cap with a typed error: *"One click never mints a realm; one crafted item does."* Move
the brake there. The economy may **sell** you an anchor; the anchor is what mints. R18's intent survives
intact — you still normally pay for a ship, because the anchor costs and the bill of materials costs far
more.

**R18 also smuggles in a computer trader.** *"The dock reads it, works out the materials and sets a
price."* An NPC dock computing a price from a bill of materials is a computer trader with a formula price,
which the recorded economy direction refuses outright, and it is an arbitrage surface either way (a stable
formula price is a bound players trade against; an unstable one is a random tax). The rule-compliant
version: **the dock states the bill of materials** — a pure function of the blueprint, derived from
substance rows that already exist, so there is no recipe table for ships at all — and the market prices
those materials.

### 4.4 Materials come out of the ground

Three independent constraints force this.

1. **Conservation.** The identity both the dormant-world and economy designs depend on is
   `Σ mint − Σ burn − Σ DECLARED_loss == Σ positions`. A construction that conjures matter is an undeclared
   mint. `economy_research_20260726.md` §4.9 records that this identity is what *"makes a dupe a failing
   test the tick it happens, not an economics observation"* — with the RuneScape 2003 party-hat dupe at
   *"well over 2,000,000"* copies as the cautionary figure. `block_provenance_collapse.md` §5 already
   traces conservation cell by cell for a felled tree (162 cells in; 11 back as blocks, 150 as stacks, 1 as
   a drop). Construction is the exact mirror and should assert the same invariant from the same test
   helper, in the opposite direction.
2. **Price formation.** In a purely player-driven economy with no formula prices, relative labour cost *is*
   the price. If a ship can be built from nothing, every material is worth nothing.
3. **Demand.** The shared starter point sits on a curated Earth-like planet (addendum 1 §D.4) — the most
   material-rich place in the game — and R11 already makes common ore seed-derived, visible and free. The
   first profession in the game is hauling ore to the yard, which is exactly the demand side the economy
   needs. CCP is recorded in the same research as naming the *absence* of negative feedback on extraction
   as the root cause of its Scarcity crisis.

**The starter yard should not sell materials at v1.** If it ever sells, it must sell from a finite stock
on a config replenishment budget: infinite-depth fixed-price NPC orders are named in the research as *"the
single most dangerous pattern"*, and the design already refuses a starting money grant, a convertible
starter kit, and the floor-price buyer (measured: one automated miner drains a venue purse in ~6 hours).

### 4.5 Build time: a real job that runs while you are elsewhere

Three options were considered. **(A) Instant on materials-complete** is cheapest but loses the
shipyard-as-a-place and removes the only natural brake on converting a stockpile into hulls in a combat
game. **(C) Rate purely proportional to equipment present** (Space Engineers' welder walls) is emergent
and playable, but uncapped it degenerates into (A) with extra steps. **(B) A durable job with a completion
tick, evaluated on touch, with (C)'s rate mechanic folded in** is the recommendation.

Nothing is authored:

```
job_ticks = Σ over cells (substance.build_work_dp) / site.assembly_rate_dp_per_tick
build_work_dp := integrity_dp     // building is the inverse of breaking; the table already exists
assembly_rate := Σ over the assembler blocks the owner actually built into the yard
```

Hard materials take longer for free; "a bigger yard builds faster" is something you build; and the brake
(C) lacks is that assembler blocks cost materials, take space and need power.

**Reference numbers to react to.** A 10,000-cell ship (the signal section's reference ship) at a metal-ish
50,000 dp/cell is 5 × 10⁸ dp. At a starter-yard rate of 4 × 10⁵ dp/s that is **1,250 s ≈ 21 minutes**; at
ten times the rate, about **2 minutes**; at hand-tool power (600 dp/s, from the 150 W axe row) it would be
231 hours — the absurd end that shows the dial has real range.

**The attached hard rule:** a build job is **never** a foreground wait. The claimant must be able to log
out, fly away or open a second job, and the job must complete correctly across a realm reap, a re-shard
and a `kill -9`. Progress is `placed = min(total, elapsed_ticks × rate)` — a closed form, the cheapest of
§2.5.2's three catch-up classes, needing no clock and no awake realm.

**One dormancy consequence must be designed in, not added after.** A job that completes during a long
dormancy owes its remaining edits in one budgeted batch at spin-up. That is bounded and computable at
job-open time, because the blueprint's cell count is known then. So **a yard must refuse a job whose bill
would exceed the realm's remaining delta budget AT OPEN**, rather than stalling mid-build. For a ship built
into a fresh construction realm with a fresh budget this rarely bites; for a spaceport or station written
onto a planet's surface the budget is real — `block_provenance_collapse.md` §5.4 shows clear-cutting
81.3 km² fills a heavily-played planet's entire ~1.5 GB budget, and §3.9.6 records that No Man's Sky's
unbudgeted equivalent lets visiting a heavily-edited base wipe a player's own edits.

### 4.6 The robots: the construct materialises in place, and the robots are decoration

Three options, priced against measurements rather than estimates.

**(A) Simulated flying welders.** DEFERRED D-9 is measured: snapshot emit is whole-realm broadcast with no
within-realm per-entity area of interest, at 128 sessions in one realm producing 357,248 gateway messages
and peaking at 1,920 per tick. Emit is O(entities × clients). Twenty visible robots at a yard with 200
sessions in the realm is 20 × 200 × ~125 B × 20 Hz ≈ **10 MB/s of extra gateway egress for one yard**.
`max_debris_entities` already exists because *"500 transient entities into a realm whose snapshot emit is
still a whole-realm broadcast is a self-inflicted denial of service."* Worse: robots would be NPC-kind
occupants, and §3.4 of the dormant-world design shows **one materialised NPC makes `empty_confirmed`
permanently false**, so the yard's realm is never reaped and the dormant-world pillar collapses in exactly
the realms that have content.

**(B) A progress bar.** Costs nothing and makes the yard indistinguishable from a menu, which is what the
seamless law exists to prevent.

**(C) The construct materialises cell by cell in place.** The job's cursor walks the blueprint in a
deterministic order and each increment is an ordinary block edit — so it rides the already-reserved
`BlockEdit` arm, enters `block_wal`, updates the edit pyramid for free, and is therefore visible at every
detail rung including from orbit. Cost: 10,000 cells over 1,250 s at 20 Hz is **0.4 cells/tick**; at R1's
8-byte durable record plus ~1.1 pyramid entries at 8 B, that is **~6.7 bytes per tick**, 168 KB one-off
for the whole ship. The dominant cost is re-meshing the one chunk the frontier is in, at the measured
65 µs per 62³ chunk = **0.13% of a 50 ms tick**. The robots themselves are then **derived decoration** in
§6's existing lane — *"derived, cosmetic, uncollidable, zero bytes on the wire, deliberately different per
client"* — drawn client-side around the visible frontier and seeded from the job id.

**Recommend (C).** No new subsystem, no relaxation of §6's promotion boundary, no R4-composite
entanglement, no dependency on the D-9 reshape, and the player actually watches the hull grow.

**One sub-item is a door and it is free now:** the deterministic build **order** must be folded into
`BlueprintHash`, or two servers build the same ship in different orders and the write-ahead log diverges.

### 4.7 The hopper: escrow at delivery, never per cell

`RULE WL-ITEM` (§8.2 of the dormant-world design) makes a stack an append-only **position**, never a
mutable count, with every `ItemId` backed by a durable monotone high-water persisted before each mint. If
the build debits the hopper per cell, a 0.4-cells/tick build mints ids at the tick rate and advances a
durable high-water eight times a second per job, forever.

**The rule: the job holds an escrowed reservation, filled by deliveries, and cells are produced from the
reservation.** Churn then equals the number of player delivery actions, not the number of cells. This is
also the escrow invariant already adopted verbatim (*"no offer, order, want-ad or contract may ever be
backed by value the settling server cannot see"*), and it gives progressive Foxhole-style delivery — the
frame is free, the hull grows as material arrives, anyone may contribute.

**The byte constraint is real and verified in code.** `KindDef::max_state_bytes` is 4096 for `PLAYER_DEF`,
8192 for `SHIP_DEF` and `NAMED_CONSTRUCTION_DEF` (`crates/core/src/entity_kind.rs:204, :213, :222`). At
the design's ~55 B per inventory row that is **~74 stacks for a player** and **~148 for a construction**.
A ship's bill spans maybe 8–20 distinct substances, so ~20 rows ≈ 1,100 B fits a yard comfortably — but a
player cannot carry a ship's bill in one trip by a wide margin. **Hauling is a vehicle activity from day
one, which means the profession is gated on P8**, not on economics.

### 4.8 Wire and capability surface: zero new arms

`InterShardFlow` has exactly **27** arms today (enumerated from `crates/wire/src/intershard.rs`: Ghost,
Transfer, Directory, Saga, SagaAck, DirectoryReply, FlushSource, TransferAck, Demote, Promote,
TransientRelease, TransientDrop, ReleaseComplete, TransientAbandon, ReHome, TransientDiscard,
ReSolicitBatch, CrossingRequest, TransientCrossingRequest, TransientCrossingGrant, CrossingAborted,
CrossingAbortedAck, RealmDemand, ShardPresence, OccupantInterest, ProxySceneSet, RealmCascade), with
`BlockEdit`, `Coupling` and `Signal` as reserved names — 30 with them, which `decision_board.md` states is
door 38's own review ceiling with *"no headroom left — everything further rides inside an existing arm as
an internal discriminator."*

A yard in realm A building into a fresh construction realm B needs cells written to B by B's owner — which
is exactly `InterShardFlow::BlockEdit(BlockEditForward)`, already specified. The realm mint is the existing
`RealmDemand` path plus the anchor. **No `Trade`, `Ward`, `BuildJob` or `EconCommand` arm may be added**,
and that constraint should be written into the S0.4 wire plant in prose before someone designs one.

One new `ShardProfile` capability is wanted: `build_site: bool`, derived in `ShardProfile::build`'s
existing validated lattice exactly as `signal_graph` is, coherent with `block_edit`. The HR4 fixture then
writes itself — a shipyard inside a station (Cartesian) and a surface pad on a planet (Spherical) run the
identical build fixture — with the placement class reading **capability values**, never a shard kind,
exactly as `SupportModel` does in §2.5.3.

### 4.9 Prior art

⚠ **Confidence capped.** This session's web-search budget was exhausted (200/200) before the survey could
be re-sourced. The EVE figures below come from the repo's own sourced research and are safe; the rest are
recalled and should be re-sourced before any of them is quoted in a binding spec.

- **EVE Online.** Manufacturing is an installed job at a facility, duration scaled by blueprint base time,
  a researched time-efficiency level and facility/rig bonuses, plus an install fee. The repo's own
  `economy_research_20260726.md` records the formula as `EIV × (system_cost_index − structure_role_bonus +
  facility_tax + 4% SCC)` with an NPC facility tax of 0.25% of EIV, and lists Manufacturing as the
  4th-largest ISK sink at 12.13 T in June 2026. **The load-bearing property is that the job runs while you
  are logged off.**
- **Foxhole.** Structures are built by placing a frame and physically delivering materials to it — the
  labour is the hauling. The money design already cites it as the decisive precedent (*"the cargo carries
  the risk, not the cash"*).
- **Space Engineers.** A projector holds ghost blocks and welders build them at a rate set by welder count
  and power, with no timer at all. Players build welder walls that assemble a ship in seconds — the
  cautionary result for any equipment-proportional rate not capped by cost.
- **Factorio / Satisfactory.** Blueprints stamp instantly or via construction drones; the wait is never the
  mechanic, the supply chain is.
- **Dual Universe / Empyrion.** Both build from a blueprint at an industry unit against a connected
  container, progressively.

The convergent rule across all of them: **waiting is only tolerable when it happens while you are
somewhere else.** The failure mode every one of them avoids is the middle — a synchronous wait long enough
to be a wait and short enough that you feel you ought to stand there, which is precisely what a naive
progress bar at the dock would ship.

---

## 5. Insurance and non-destructibility

### 5.1 Split the owner's two things, and ship only the first

The owner asked for two things in one sentence, and they are different mechanisms with different costs and
opposite consequences.

| | **(A) Civic non-destructibility** | **(B) Insurance as a player product** |
|---|---|---|
| What it is | Authored, permanent, not for sale | Purchased, per-construction |
| Where | Starter port, safe-zone infrastructure | Anywhere the owner allows |
| Ship it | **At P6.** It is what he asked for first, and it is what actually protects the shared start | **Defer entirely.** Free to defer: the player-facing product should be payout, payout needs a dock, docks need blueprints, blueprints are P8 |
| Cost | ~600 production lines with the admission seam it shares | 1,200–1,800 lines plus a persisted table, an expiry sweep, a fuel economy and an economy command surface |

**And a third thing nobody separated:** *undamageable* and *unremovable* are different. A build that
cannot be damaged by a weapon but can still be dismantled by its owner, reverted by an admin, or decayed
when unpaid, is far less dangerous than one that cannot be removed at all. Every exploit below except
deliberate lapse and payout fraud turns on **unremovability**, not on invulnerability.

### 5.2 The rule, in one sentence

> A cell inside an active ward volume is refused as the **target** of any state-changing intent whose
> `EditSource` is not in the ward's allow set; the refusal happens at **one** admission seam, before the
> mechanic runs, and `apply_hit` / `resolve` stay byte-identical.

### 5.3 It is not a fourth provenance state

`block_provenance_collapse.md` §3.2 fixes provenance at two bits (Terrain=0, Feature=1, Placed=2, value 3
reserved and rejected on decode) with the load-bearing rule **PROVENANCE NEVER GAINS PRIVILEGE**: every
write to an existing cell must satisfy `new >= old`, enforced by one comparison in a typed constructor.
That single `>=` is what structurally prevents a player laundering built matter into unbreakable terrain.
Protection expires; a `Protected` provenance value would have to **decrease** on expiry, which is the exact
write the type forbids.

The storage arithmetic is worse. A 400 m × 400 m × 100 m civic volume is 16,000,000 cells. Per-cell
protection needs an 8-byte durable record for every one, including the ~15.8 M cells of untouched rock and
air that have no record at all today: **128 MB**, or 7.5% of a planet's entire ~1.7 GB delta budget, for
one spaceport. The equivalent region record is 8 (id) + 2 (scope) + 1 (source) + 26 (prism) + 8 (owner) +
8 (expiry) + 4 (fuel) = 57 bytes, ~72 framed. **Well over a million to one.** Materialising records for
cells the generator already answers is the same category error §3.9.3 avoided when it ruled that a
`ChunkRecord` never embeds generator output.

Keep provenance value 3 for `NpcBuilt` or `Ruin`, which is what §3.2 reserved it for.

### 5.4 It does not belong inside the damage kernel either

§2.3 is right that a pickaxe, a cutter, a bullet, an explosion, a fire and a ship collision are all one
object resolved by `apply_hit`. But there are **fourteen** ways a cell changes and only eight go through
it. The other six: a player break/place `WorldAction` (an instant op is not a hit); structural collapse and
the falling group's settle writes; growth / spread / weathering / freeze writes; the §3.9.5 dormant-fold
WAL emission; realm reap and the §3.6 weld; and admin `RevertRegion`. **A ward at the damage site is
defeated in one tick by mining the ground and letting the support kernel do the work.**

Separately, `apply_hit` is a pure monomorphic fixed-8-iteration branchless loop with no world access,
called ~23,000 times per HE detonation. It cannot look a ward up, and adding a ward parameter just means
the caller did the lookup — which is the seam. Under HR5 a ward branch inside it adds a region that must be
covered at every call site.

```rust
EditAdmission::admit(EditIntent { realm, cell, source, op }) -> Admission
// Admission::Refused(RefusalReason::Warded(WardId) | ::Claimed(ClaimId) | ::Reach | ::Rate | ..)
```

Called from eight sites, implemented once. `resolve` and `apply_hit` stay byte-identical, so §2.3.6's
worked examples and their golden tests are untouched. **A grep-level tripwire test must assert that nothing
constructs a grid write outside the admission module**, in the style of the existing registry drift
tripwires — a mechanic added later that writes cells directly bypasses every ward with no error at all.

### 5.5 The ward closure invariant — why a purchasable shield is a purchasable land claim

Under `block_provenance_collapse.md` §4.1, `anchored(cell)` holds if the cell is Terrain, is in
`realm.anchor_cells`, or is reached by the support flood from an anchor with remaining load capacity.
Protect a tower's blocks but not the rock under it and a raider mines the rock: the tower is unanchored,
the collapse kernel dissolves it, §4.5's scripted topple gives it a closed-form final pose, and it lands.
Freeze the falling cells and you have a tower hanging in mid-air; refuse the landing hit and you have an
intact tower lying on its side. Both are incoherent.

> **WARD CLOSURE INVARIANT.** A ward's volume extends `foundation_depth_cells` below the lowest protected
> cell, so the ground is *inside* the ward and mining it is refused at the same seam. Enforced at ward
> **creation**: run the existing §2.5.1 flood restricted to the ward volume and assert every protected
> cell is anchored within it; a ward that does not close is refused with a typed error.

That collapses two proposed flags (COLLAPSE, UNDERMINE) into one property of the volume. On a
gravity-free profile (a ship, a construction realm) closure is trivially satisfied because nothing falls,
so the same predicate passes both HR4 fixtures **with no branch** — matching §2.5.3's `SupportModel`
data-row idiom.

**And it is the single strongest argument against selling protection.** An insured surface construct
necessarily takes the ground under it out of play. A purchasable ward is a land claim sold by the metre.

### 5.6 The exact immunity table — and fire needs no flag

| Scope | Civic ward | Insured ward (if ever shipped) | Payout insurance |
|---|---|---|---|
| Weapon hits | immune | immune | — |
| Tool hits | immune | immune | — |
| Other players' **break** | immune | immune | — |
| Other players' **place** | immune | — | — |
| Explosions | immune | immune | — |
| Ship impacts | immune | immune | — |
| Falling-group landing damage | immune | — | — |
| Natural growth / fluid / freeze | immune | — | — |
| Dormant NPC fold | immune | — | — |
| Realm reap | immune | — | — |
| Admin `RevertRegion` | **not** immune | **not** immune | — |
| Owner's own edits | always allowed | always allowed | — |
| Thermal | reserved bit, ship **unset** | reserved bit, ship **unset** | — |

**Fire looks like it needs a flag and does not.** §2.3.6's worked numbers: a wood fire gates at 1,100 K,
granite's thermal gate is 1,500 K, steel's is 1,800 K. **Fire never damages stone or steel** — with no rule
saying so, just a temperature comparison a player can read off a tooltip. A civic port built of granite and
structural steel is fireproof by material, and a griefer cannot import fuel because *placement* inside the
ward is refused.

Two more exact answers. A ship crashing into a civic structure is refused as damage to the ward, but the
ship still resolves its own kinetic damage and dies — one-sided, with debris bounded by the existing
`debris_merge_radius_blocks` / `max_debris_entities`. And a ward **never blocks motion**, only writes; a
motion-blocking bubble would be a perceptible instancing seam and the seamless law forbids it.

### 5.7 The tensions, stated without softening

**Insurance versus raiding.** If the strongest defence is a purchase rather than a position, position stops
mattering, and territory is only interesting when position matters. Dual Universe is the shipped
demonstration in the other direction: a large permanent safe zone plus a permanently PvP-immune Sanctuary
moon meant essentially all construction happened inside safety and territorial conflict nearly died.
(Recalled, not re-sourced this session.)

**The siege re-buy, which is the mechanic-killer and appears in none of the five investigations.** §2.3.8
states that constructed materials do not regenerate — only natural terrain heals on the random tick. So a
besieger who has ground a titanium hull block down through 138 rifle rounds or four 1 kg HE shells
(§2.3.6's numbers) loses nothing when the defender buys a ward, and gains nothing either: every subsequent
hit is refused, the damage sits frozen, and the siege resumes exactly there whenever the policy lapses. **A
siege whose outcome is decided by who can re-buy fastest is not gameplay.** The attacker's symmetric
version is deliberate lapse: insure a forward structure to establish it during the contested window, let it
lapse when defenders are offline. The only known cure is a **combat lock** — protection cannot be started or
extended while the structure has taken damage inside a window. EVE ships the equivalent as structure
reinforcement timers with an owner-declared vulnerability window rather than a purchase.

**The outage.** Covered in §4.3: fail-closed makes every insured structure simultaneously invulnerable and
un-cancellable; fail-open makes every insured structure in the game vulnerable on a schedule the attacker
chooses. This is the cleanest available proof that a protection flag may never be an economy value read on
the physics path. **The ward table must be game state the edit path reads directly, whose issuance the
economy may only influence through a refusable command** — and civic wards must be authored, so the starter
port is still protected with the economy compiled out, which is what the economy-absent gate asserts.

**Cheapest exploit, if a shield ships anyway.** The owner's own framing — *"for big constructions"* —
selects for exactly what a griefer wants. A hollow cube 22 m on a side is 22³ − 20³ = 2,648 blocks and
denies 8,000 m³ of interior plus a 10,648 m³ bounding box; at 10 edits/s that is 265 seconds of work, in
free dirt dug from beside the spawn. Price on material value and dirt is the cheapest denial per credit in
the game; price on bounding volume and it is honest but still purchasable. Cheaper still is the interface
attack: an insured slab over a cave mouth, a landing pad or a dock's approach vector needs to cover only the
interface. And the spite shell — an insured skin around someone else's base, leaving it intact and
unreachable — is available on day one, because nothing stops you building beside or around another player's
work.

### 5.8 Payout insurance is nearly free, because R18's dock is the machinery

R18: *"you bring a blueprint to a dock; the dock reads it, works out the materials and sets a price; you
pay; it spins up a realm belonging to you and the ship is fully reconstructed there, wiring intact."*

Every component of a payout claim is in that sentence. Insurance-as-payout is: capture the as-built
blueprint at policy time, and on destruction issue a rebuild order at a dock with the same bill, paid by the
insurer minus a deductible. No new pricing engine, no new reconstruction path, no new realm-minting path.

It also fits the standing economy ruling better. Purely player-driven, no computer traders, means insurance
should be **underwritten by players** — one account contracts to rebuild another's construct for a premium.
A contract between two accounts over an existing dock service, zero NPC pricing, and a genuine profession.
The one thing it needs that does not exist is a durable author/owner identity that survives blueprint
copying — which R18 already flags as owed for royalties, so it is a shared cost, not a new one.

**Author royalties should be out of the first version.** They require exactly that durable identity, and
they are the one part of R18 with no cheap shape. Whichever way it goes, one rule is not negotiable: a
royalty is a payment, so it is an economy object, and it may never gate the build — the ship gets built and
the royalty is a fact the economy observes, exactly like the yard's fee.

### 5.9 Permanence, and the bigger problem underneath it

The "ugly tower at the spawn forever" scenario is a **placement** problem, not a protection problem: even an
unprotected tower is a scar until somebody bothers to demolish it. So the civic ward's player-edit flag must
be symmetric — it refuses **place** as well as **break** — and `civic_ward_margin_m` makes the volume larger
than the port so the shared front door can never host a player structure. That does mean players cannot
build shops at the spaceport; the shipping answer is a build ring outside the civic volume, with a
plot-allowlist deferred.

What expires and who removes what: civic wards never expire but are authored, so the world author can retire
one in a patch and a player never can; every non-civic ward expires by term or fuel; an expired ward's
construct becomes ordinary matter, which nothing deletes — other players remove it with tools, needing no
garbage collection at all.

**The residual is genuinely undesigned and it is a bigger risk than anything in the ward design.** §3.4's
prune rule deletes a record only when a cell equals what the generator says, and a `Placed` cell can never
equal a `Terrain` cell. **Player builds are permanent delta forever unless somebody breaks them.** §3.6
reaps abandoned construction realms; there is no equivalent for planet-surface builds. Against §3.9.3's
~1.7 GB per-planet budget and §5.4's demonstration that 0.025% of the surface clear-cut exhausts it, that
gap matters more than wards do. The fix that costs nothing new is **weathering-as-decay**: an unvisited,
unclaimed, unwarded structure takes a slow Rate hit through the ordinary damage path and eventually prunes
back to terrain. Rust, Foxhole and Boundless all ship decay for precisely this reason.

### 5.10 Cost, performance and the feedback amplifier

**Storage.** §3.4 canonicalisation rule 2 already means a player-placed cell never prunes, warded or not, so
the ward changes nothing about the per-cell storage of the thing it protects. The record itself is 32 bytes
realm-scoped (a ship, a construction realm — no geometry), ~48 framed; ~72 for the prism form.
`max_wards_per_realm` in a `WardTuning` struct (sibling of `BlockStoreTuning` / `CollapseTuning`) bounds the
table absolutely. **If the starter port is generated rather than stamped, its cells have no records at all,
so the whole protected installation costs exactly 72 bytes.**

**Query cost.** Per realm keep `wards: BTreeMap<WardId, Ward>` plus an accelerator `BTreeSet<ChunkKey>` and
a three-valued per-chunk verdict `Open | Sealed(WardId) | Mixed`. A 400 × 400 × 100 m civic volume spans
⌈400/62⌉ = 7 tangential each way by ⌈100/62⌉ = 2 vertical = **98 chunks**; a `BTreeSet` lookup over 98 keys
is ~7 comparisons, cache-resident, **~20 ns**. Every mechanic that touches many cells already iterates chunk
by chunk, so the verdict is resolved once per chunk and cached in the cursor: an HE detonation's
1,352 rays × ~17 steps ≈ 23,000 queries touch at most ~27 chunks, so 27 × 20 ns = **540 ns against
~115 µs of marching — 0.47%**. `Mixed` chunks fall back to an integer AABB range test in index space, ~2 ns,
branchless, no floats. Aligning the civic volume to chunk boundaries removes the fallback entirely. One
geometry caveat: a prism is per cube-sphere face, so author the port away from the 12 face edges.

**Feedback must coalesce or it is a D-9 amplifier.** A refused hit needs feedback, but persisting damage on
a warded cell is wrong — it writes `damage_dp` into §2.4's sparse side table for every attacked cell, a
storage griefing vector against the dense-promotion threshold of 79,442 entries. So **refuse before
`resolve` is called, write nothing, and emit a transient ward shimmer** on §2.5.5's already-reserved
transient per-realm state-light lane, capped at **one effect per warded chunk per tick**. A thousand players
shooting the port then produce at most **98 messages per tick, against D-9's measured peak of 1,920 —
5.1%**. The alternative worth naming and rejecting is letting damage accumulate to a ward ceiling so the
attacker sees progress: §2.3.8 says constructed materials never heal, so a warded steel wall would sit
permanently at 75% damage and the storage bill is unbounded.

### 5.11 Ward upkeep collides with a standing ruling — denominate it in material

Every shipped protection system that expires does so via upkeep, and the owner's recorded economy decision
is a per-sale fee and **never clock-rent**. A ward deducting currency per day is clock-rent by another name.

Two compliant forms: **(a) fuel is a material the ward consumes** — you feed it ore or power cells, exactly
as Space Engineers' Safe Zone consumes zone chips and Boundless beacons consume fuel. No currency is
deducted, the cost is a physical supply chain, and it creates hauling demand. **(b) a fixed-term policy
re-bought at expiry**, which makes each renewal a per-sale event rather than a rent. Recommend (a), with (b)
available for the payout product. And set the rate low: Dual Universe's Demeter territory-tax patch is the
shipped demonstration that upkeep priced wrong causes mass abandonment rather than mass engagement.

**Ward expiry must be in universe ticks.** §2.3.2 already rules that the per-realm `time_multiplier`
dilates occupant movement only and no f64 may enter the damage path. A ward whose expiry ran on subjective
or wall-clock time would make protection depend on a per-realm float.

### 5.12 Generate the starter spaceport; do not stamp it

Addendum 1 §D.4 item 3 says the spaceport is *"stamped deterministically at a known place… written into the
planet's delta store at world creation"* and then that it is *"a normal player-visible build: persisted,
**damageable**, repairable"* — which directly contradicts the owner's non-destructible ruling.

Cost the stamp. A 300 m × 300 m port with ~30% building coverage at ~20 m average height is roughly 200,000
authored cells. Sparse that is 1.6 MB; the port spans about 5 × 5 × 1 = 25 chunks at 8,000 authored cells
each, which clears §3.9.3's masked-dense crossover of 4,256 cells, so the compactor picks masked-dense at
25 × (29,791 + palette + 8,000) ≈ 0.95 MB; plus the edit pyramid at ~1.1 entries per edit × 8 bytes =
1.76 MB. **About 2.7 MB of permanent delta, per port, for content that is entirely derivable.**

Make it **generator stage 9** instead — running after `block_provenance_collapse.md` §4.4's owed stage 8
feature placement — and it costs zero bytes, reproduces byte-identically on a fresh shard or after a
re-shard with no store read, satisfies P4's "only the seed crosses the wire" with no exception, and gains a
property the stamped version cannot have: if a player damages it and it is repaired to baseline, §3.4's
prune rule deletes the record and the storage comes back. Keep the addendum's mechanism (a relative walk
from an origin block, the same thing blueprints do) and delete only the write.

One thing to check: generator-authored port cells would be `Feature` (rigid, collapsible when cut) rather
than `Terrain`, which is correct for a building and is exactly what the ward makes moot.

### 5.13 The armistice, as a sixth item

Weapons simply do not fire inside the starter system. It is one call to the same admission seam from
`WorldAction::Fire`, which §3.9.7 has **already** reserved as the second consumer of the `WorldAction`
carrier — so it costs ~20 lines and stops griefing at the source rather than protecting each target.
**Alongside the ward, not instead of it: the armistice stops the shooting; the ward stops the digging.**

### 5.14 The cheapest version that ungriefs the starter spaceport

Five items, ~600 production lines, 0.8% of `decision_board.md` §8's ~77,000-line plan, zero new wire arms,
zero economy dependency:

1. **Enumerate `EditSource` fully in the S0.4 wire plant.** It is a declared field on a **binding** spec
   (`docs/design/sealed_shards.md:309`) with **zero defined arms anywhere** — verified. Proposed:
   `Player(SessionId) | Tool | Weapon | Hazard(HazardKind) | Impact | Collapse | Natural(MechanicId) |
   DormantFold | Admin(AdminOp) | Ward(WardId)`. **ONE-WAY DOOR** — widening it after
   `InterShardFlow::BlockEdit` ships is a PROTO_MINOR bump plus a client migration, and the arm is gated by
   four structural conformance tests.
2. **`WardScope` as a u16 bitmask with reserved bits**, in `vd-core`. Free now, a migration later.
3. **`EditAdmission::admit` as the one gate**, with `RefusalReason::Warded` and `::Claimed` arms, plus the
   bypass tripwire.
4. **One authored ward kind — `Civic { scope: ALL }`** — read from the realm definition (already
   content-hashed into the handshake per addendum 1 §D.3), volume a chunk-aligned prism in the planet's
   index space, validated at load by the closure invariant.
5. **Generate the starter spaceport** as generator stage 9 instead of stamping it.

Add the armistice as the sixth if the owner wants belt and braces.

The resulting world rule is one legible sentence: **"In the starter system nothing can be broken by anyone.
Everywhere else everything can be broken by anyone, and what you buy is a rebuild, not a shield."**

### 5.15 One number per realm dissolves the insurance-versus-raiding tension

If a shield tier is ever wanted, protection strength must be a property of the **place**, capped by the
realm's own authored definition:

| Realm class | `ward_ceiling` | Effect |
|---|---|---|
| Starter Area realms and the port | `Civic` | All scopes, permanent, not purchasable |
| Starter planet outside the port | `Claim` | PLAYER_EDIT only — nobody may break or place in your plot; weapons still work, the world still burns |
| Everywhere else | `Payout` | No write refusal at all; insurance restores value |

A player ward can then never be stronger than its realm allows, so "pay to make my base unraidable" is
structurally impossible outside the safe band — not a rules argument, a data row.

**The HR3 trap to name now:** it will be very tempting to write `if realm.kind == RealmKindTag::Area {
civic }`. That is a match on a realm kind in feature code and it is the exact G-NO-SHARD-FORK violation the
project bans. `ward_ceiling` must be an **authored field on the realm definition, consulted by value**.

---

## 6. The multi-Area spawn

### 6.1 The partition must be spatially disjoint, and overlapping copies are unexpressible

Containment is decided by `container()`: the deepest region whose hysteretic membership holds wins, ordered
by (depth DESC, `RealmId` ASC, slice index ASC). Two **overlapping** same-depth sibling Areas would both
pass membership and the tiebreak would hand the player to the lower `RealmId` no matter where they
physically are — so an overlap does not create two parallel copies, it silently collapses to one arbitrary
winner. There is no instance, layer or shard-copy axis anywhere in `RealmId`, `RealmPath`, `FrameRef` or
`RealmRegion`, so instancing would need a new dimension on the containment key and would break the seamless
law outright.

`ContainmentBand` hysteresis is per-region (inset 1 m, outset 2 m at walk scale) but the argmax between two
same-depth members is memoryless, so at a shared wall the crossing point is asymmetric by about a metre
depending on direction, and the arbiter is `RealmId` order rather than geometry. **Sibling Areas should
overlap by at least inset + outset so the dead zone actually functions**, and a `SiblingsOverlap` arm should
be added to `RegionNestError` (§3.8).

### 6.2 What a player experiences crossing a boundary today

**They vanish from each other.** `emit_frames` (`crates/sim/src/stub.rs`) collects dots from the shard's own
`Dots` resource filtered **only** by `emits()` — no observer filter of any kind — and nothing anywhere emits
an entity belonging to another realm. The only cross-realm content a client receives is the realm
**outline**: `ProxySceneSet` ships *"the FULL current set of the proxy's in-range sibling outlines (public
`RealmShape` geometry)"*. So player A sees the neighbouring Area's **box** and not one person inside it.

**And they pass through each other.** The ghost machinery all exists — `InterShardFlow::Ghost` with
Spawn/Delta/Despawn, `SourceGhostMirror`, ghost collider registration, `OverlapBand` — but the only producer
is `register_and_spawn_source_ghost`, called from `on_saga_promote`. Ghosts are **transfer-scoped, not
band-scoped**. There is no body across the seam and no collision, which violates the players-must-collide
law at exactly the busiest place in the game.

**And the crossing costs two transfer sagas.** `neighbourhood_scope` (`crates/core/src/worldgen.rs:774`)
admits a region only if it is an ancestor of, or a direct child of, a held realm — the test
`realm_neighbourhood_for_config_excludes_sibling_planets_over_the_visual_forest` pins the intent (*"a
SIBLING is NEVER in scope — the origin-stacking flap cure"*). So Area 1's shard cannot see Area 2's region
at all, the container fold falls to the shared parent the moment an occupant leaves Area 1's box, and the
crossing is Area1 → Planet → Area2: two fence-CAS sagas, two Frozen windows in which no input is applied,
two route swaps. At 1,000 players in 50 m cells wandering at 1 m/s the straight-line crossing interval is
about 50 s, so ~20 crossings/s and **~40 sagas/s**; halving the cell to 25 m doubles it to 80. **The
partition gets more expensive the finer it gets.** The client's 100–150 ms interpolation buffer absorbs one
short freeze; two back to back is at the edge, and this exact path produced a hard freeze in production
(commit `cbfe573`, a same-node re-home closing the player's own live subscription).

On arrival the old Area's occupants are evicted by the reliable `EventMsg::EntityRemoved`, so everyone the
player was standing beside a second earlier vanishes. **That is the perceptible seam, and it is not subtle.**

Of the four interactions worth asking about: **talking** has a designed answer (chat is a Protocol-scoped
signal routed up to the least common ancestor, one hop up and one down, P9). **Colliding** needs a new
band-driven ghost producer. **Shooting** has no cross-realm arm at all (P11). **Trading** has no
realm-crossing two-party handshake designed anywhere.

### 6.3 The arithmetic — and cross-visibility costs MORE than not splitting

From the measured D-9 baseline (128 sessions, one realm, 357,248 gateway messages, peak 1,920/tick): peak ÷
sessions = **15 datagrams per client per tick**; at the 1,100-byte datagram budget
(`crates/harness/src/topology.rs:1093`) that is ≤ 16,500 B per client per tick, i.e. **~120–129 bytes per
entity row**. The struct arithmetic agrees independently: `EntityId` is a u128 with the kind in the top byte
so postcard varints it to 19 B, `FrameRef` ~11, `LatticePos` 3 + 24, velocity 24, orientation 32, universe
tick ~2 — about 115 B. Use **125 B/entity** below.

⚠ **One correction every prior run repeated:** D-9's *"elapsed 7.6 s"* is **wall-clock under the virtual
clock**, not simulated seconds — 357,248 ÷ 1,920 implies at least 187 ticks, i.e. ~9–10 simulated seconds.
Any per-second rate derived from 7.6 s is ~25% too high.

At **N = 1,000 players**, MTU 1,100 B:

| Configuration | Datagrams per client per tick | Total msgs/tick | Verdict |
|---|---|---|---|
| One realm (today) | ⌈1000 × 125 / 1100⌉ = **114** | 114,000 | The baseline |
| 8 sealed Areas (no cross-visibility) | ⌈125 × 125 / 1100⌉ = **15** | 15,000 | **7.6× better** |
| 8 Areas, full cross-visibility | 8 × 15 = **120** | 120,000 | **5.3% WORSE** |

The payload is identical in the last two rows (each client still receives 1,000 entity rows); what changes
is that eight partly-empty datagram tails replace one. **Splitting the spaceport buys nothing at all if
people can see across the boundaries.**

**And the bad configuration is guaranteed rather than optional.** `VISIBILITY_THETA_MIN_RAD = 0.139626`
(8°, `crates/core/src/worldgen.rs:139`) gives `visibility_factor = cot(4°) = 14.301`, and a realm streams
in out to `finite_extent × 14.301` from its centre. Tile a floor with cells of half-extent *h*: the
in-range disc has area π(14.301h)² and each cell's footprint is (2h)², so the ratio is
**π × 14.301² / 4 = 160.6 siblings — and *h* cancels completely.** For any spaceport split into k ≤ 160
Areas, every Area is in every occupant's area of interest, always. A 50 m cube Area streams in out to
357 m, which is larger than a 200 m spaceport.

**Per-client downlink, for calibration.** 125 B × N × 20 Hz: **320 KB/s at N = 128**, 2.50 MB/s
(20 Mbit/s) at N = 1,000, with **2.50 GB/s of total gateway egress** at N = 1,000. The gateway is not the
first wall — 114,000 sends/tick is ~440 ns per send, roughly two batched cores — the **client downlink** is,
and it breaks at about 78 co-visible entities on a 1.5 Mbit/s entity-lane budget and about 16 on the
inferred 40 kB/s one.

**21% of every row is provably dead weight.** `crates/client/src/interp.rs:11` states plainly that nothing
on the render path reads `StampedPose::vel` — it is the no-prediction firewall — so 24 of those bytes are
shipped and discarded: 480 KB/s per client of pure waste at N = 1,000. And when D-41 activates non-zero
lattice cells at P4/P5 the row grows by up to 30 more bytes (three i64 varints), **about +26%, exactly when
the spaceport ships**.

### 6.4 The two changes that actually make a thousand-player spaceport affordable

| Change | Worth | Why |
|---|---|---|
| **S0.5, the within-realm per-entity area-of-interest reshape** | **~33×** | Per-client cost becomes proportional to *visible neighbours*, not realm population or Area count. At 1,000 players with ~30 visible neighbours that is ~30,000 entity-rows/tick instead of 1,000,000 |
| **A narrower client-facing snapshot row** | **~6×** | Drop `vel` (−24 B, never read); a per-subscription 16-bit entity handle instead of a 19-byte u128 (−17 B); millimetre-quantised i32 offsets instead of three f64 (−12 B); smallest-three 32-bit orientation instead of an f64 quaternion (−28 B). ~125 B → ~20 B |
| **Splitting into k Areas** | **k× at best, 0.95× at worst** | And only if the boundaries are opaque |

`decision_board.md` §8 already lists S0.5 as a **precondition**, not a follow-on. Do it first. The narrow
row must be a **different, narrower type** from the inter-shard pose — the ghost feed and the transfer flush
genuinely need velocity — which is the same slice as the emit reshape.

### 6.5 The costs the partition creates on the parent

The AoI loop (`crates/sim/src/stub.rs:5356-5415`) iterates every direct-child placement and, inside it,
every observer. Verified in the code: `was_demanded` does `membership.get(&(*obs, path.clone()))` per
observer; the inner body then does `let key = (*obs, path.clone())` and `live_keys.insert(key.clone())`.
**Three to four `RealmPath` Vec clones per (observer, child) pair per tick.** At 1,000 occupants and eight
Areas: 8,000 pairs × 4 = 32,000 allocations per tick = **640,000/s at 20 Hz**; at the 160.6-sibling figure
it is **12.8 million allocations per second** — on the planet shard, which is the shard the partition was
supposed to relieve. Separately, the containment detector computes `region_signed_distance` for every entity
against every region every tick: 1,000 × 64 = 64,000 frame re-expressions per tick.

Both are straightforward fixes (hoist the clone, key the map by an interned handle) but they must be
budgeted **before** Areas multiply.

**Credit where due, and no prior run said this:** the realm lane costs nothing for a static partition.
`emit_realm_frames` early-returns when `authored_realm_snaps` is empty and only carries **moving** children,
so a spaceport's static Areas ship zero per-tick realm bytes.

### 6.6 Rebalancing, and the worst case

**Nobody can rebalance a live spaceport.** The region forest is closed-form `f(seed, config)` recomputed
identically on every shard at boot; `guard_regions_nest` runs once at boot; and addendum 1 §D.3
content-hashes the authored override table into the connect handshake, so changing the partition changes the
hash and **refuses every connected client at the door**. Moving an Area realm between shard processes is a
realm re-home, and DEFERRED FA-6 records it as unbuilt and split-brain-prone. What *is* already supported is
**co-hosting several Areas on one process** (`realm_neighbourhood_for_held` takes a held set) — the cheap
half of the same lever, and it should be the default before any process-per-Area fleet.

**Worst case: everyone crowds one Area's plaza.** The design does literally nothing, and there is no
mechanism it could do anything with. Containment is coordinate-based, so 1,000 players in one box are all
owned by that Area's shard, which then carries the full 114,000 msgs/tick. There is no per-realm occupancy
cap anywhere in the codebase — the only session cap is `GatewayTuning::max_sessions`, which is per-gateway.
Moving a player between Areas without them walking is a teleport, which the seamless law bans. **The
partition's only defence against its own worst case is architectural**, which directly contradicts "one
shared starter point where everyone appears".

**What Areas genuinely buy, stated fairly:** the starter world can be authored large and cost nothing when
empty (the demand lifecycle never spins up an Area no occupant can see); each Area is a unit the
orchestrator can move between processes later; and process isolation means one Area's fault does not take
the whole port down.

### 6.7 The client half is ready; the gateway half is sized for four

`DeliveredView` (`crates/client/src/view.rs`) keys tracks per `EntityId` rather than per (SubId, EntityId)
and its own doc comment already states that multi-realm interest legitimately holds several subscriptions —
*"a player in a ship docked at a station on a planet holds Ship+Station+Planet+System subs at once"* — with
per-entity eviction driven by the reliable `EntityRemoved`. **Rendering people from two Areas at once needs
no client change whatsoever.**

The gateway supports multiple subscriptions keyed by shard `NodeId`, but `SubTable`
(`crates/connection-plane/src/gateway.rs:505`) is documented as *"a boxed sorted slice — ≤ ~4 entries"*, and
`publish_subs` republishes the whole table on every membership change. The binary search is fine (~8
comparisons at 161 entries); the **stated sizing assumption** would be violated by an order of magnitude.
The genuinely missing piece is the producer: `connection_plane.md:189` puts InterestSet computation at P6
and it does not exist. The clean source already exists though — the parent already computes each occupant's
in-range sibling set for the outline feed — and it must name **RealmIds, not NodeIds**, because
`ProxySceneSet` is documented as *"PUBLIC parent-authored geometry ONLY — no NodeId / gateway / session
(HR1)"*.

### 6.8 Honest verdict

**Areas help less than they appear, and only in the configuration the seamless law makes hardest.** They are
a shard-CPU, fault-isolation and lifecycle lever worth up to k×; they are worth **nothing** as a bandwidth
lever unless the boundaries are opaque; they are worth **negative** value if cross-visibility is added
first; they are worth **zero** against the crowd-in-one-plaza case; and they create new O(occupants × Areas)
work on the parent. **Recommend k = 1 or 2 until a measured per-shard occupant ceiling exists** — there is
exactly one data point (128 sessions in one realm; every invariant held) and it passed — and recommend that
every boundary be an authored bulkhead, deck or airlock. Land S0.5 first.

---

## 7. One shared starting point

### 7.1 What it buys, and what it does not

**It buys cold-start liquidity and a shared reference price.** A purely player-driven economy with no
computer traders and no formula prices cannot bootstrap a market with three sellers and no buyers. Putting
100% of the population through one venue guarantees the first trades clear and that "what is a tonne of iron
worth" gets an answer in week one. In a no-NPC-trader economy that is worth real money.

**It does not buy a price gradient.** Spatial price equilibrium (Samuelson 1952; Takayama–Judge 1964,
already logged in `economy_research_20260726.md`) says goods flow from *i* to *j* only when
`p_j − p_i ≥ c_ij`, and in equilibrium no price gap anywhere exceeds transport cost. **The gap is bounded by
`c_ij` and by nothing else — distance from a spawn point appears nowhere in the relation.** The four forces
that actually produce divergence are already named in the economy research §7.13: seed-derived per-realm
resource and demand profiles; transport cost as an additive integer term; tariffs as edge costs at the least
common ancestor; and a bounded parent-coupling weight that is low at the leaves. Two realms 100 m apart with
different endowments show a larger gap than two identical realms ten light-years apart.

What the single origin genuinely produces is a **demand sink**: in the first weeks 100% of the population
sits at radius zero, so demand density falls off with radius, and that *reads* like a gradient. The moment
players settle further out it resolves into local basins.

**Keep the origin. Retire the gradient justification.** Otherwise the owner will later be surprised that
prices converge at the hub while diverging at the frontier for reasons unrelated to the origin, and may
build gradient-defending machinery that was never needed.

### 7.2 The check that passes: consignment plus a per-sale fee is what makes distance matter

`money_and_markets_design.md` §5.1 makes a sale a genuine custody move — the seller flies to the venue and
moves cargo from their own container into the venue's consignment container. **Goods cannot move without
being flown**, which is the single property that keeps `c_ij` above zero. The per-sale fee compounds it: it
is an additive cost per hop, structurally identical to a tariff.

**Leak one is already open and is acceptable.** §5.4's 2026-07-27 revision moved every **currency**
operation onto the wire — deposit, withdrawal, account-paid sale, remote proceeds credit, note issue,
balance read — at ~80 bytes per sale. Money moving at zero distance does not arbitrage a goods price,
because you still have to move the goods. **State the asymmetry as a rule rather than leaving it as an
accident of a revision history: goods never move without being flown; money may.**

**Leak two must never open.** §5.3's want-ads are the buy side, and a want-ad fillable remotely is a teleport
for cargo. RuneScape is the shipped proof: Lumbridge was the single shared start for two decades and it did
not produce a gradient, because the 2007 Grand Exchange gave the game one global order book with instant
fulfilment and regional trade ended. **The seamless law — warp is a physical fly-by, no teleport, no map
mode — is therefore not only an aesthetic rule; it is the strongest economic protection the project owns.**

### 7.3 What shipped games learned

⚠ Recalled, not re-sourced this session (web-search budget exhausted at 200/200). Structural patterns
reliable; specific figures need a browser pass before quotation in a binding spec. The repo's own economy
research already flags several Albion and Dual Universe figures as unverified because their wikis are
bot-blocked.

| Game | Configuration | Outcome that matters here |
|---|---|---|
| **EVE Online** | ~7,800 systems, several racial starter systems | One dominant hub (Jita) emerged that CCP never designed. **The owner does not choose whether there is one dominant market — only whether he places it deliberately or discovers where it landed.** Insurance is per-hull, tiered, hull-only, expiring, voided when CONCORD kills you, and never prevents destruction. NPC stations are indestructible; player structures are destructible everywhere behind reinforcement timers |
| **Albion Online** | **Five** royal starter cities, markets not shared | The closest analogue to what the owner wants. Separated markets produced genuine price divergence and made hauling the arbitrage gameplay. Documented complaint of record: fragmented markets suppress volume, so per-realm books need a deliberate cross-realm affordance. **Multiple starts + separated markets produced the hauling economy; a single start produced a hub** |
| **Dual Universe** | **One arkship**, everyone spawns there | The literal analogue, and it hit four of the five failure modes in this brief within months: territory around the arkship claimed out immediately (forcing a retrofitted second start, the Sanctuary moon); the surface strip-mined into a crater field within weeks, eventually fixed by removing manual surface mining entirely in favour of Mining Units; commerce concentrated on one market district with the rest of the map empty; and a permanent safe zone that killed territorial PvP |
| **RuneScape** | Lumbridge spawn, Varrock Grand Exchange | Arrival point and market point **deliberately separated**. And the Grand Exchange is the shipped proof that remote fulfilment ends regional trade |
| **Star Citizen** | Effectively instant acquisition with a claim timer | Loss is time plus cargo; cargo never covered |
| **Classic WoW (2019)** | Layering — multiple copies of one zone | Players learned to layer-hop to re-farm nodes within days; guilds split across copies; removed after the opening weeks. **The shipped proof that players perceive and exploit instancing seams** |
| **Boundless** | Voxel MMO; protection is a beacon claim consuming fuel | An unfuelled claim is reclaimed by the world |
| **Minecraft** | `spawn-protection` server property (a radius) plus bedrock | The answer to spawn griefing is a **region flag**, never a block attribute |

### 7.4 Failure modes, with numbers and mitigations

**(a) The ring is stripped in days.** At R = 161,671 m the surface is 4πR² = 3.285 × 10¹¹ m² =
**328,460 km²**. A 2 km ring around the port is π(2000)² = 12.57 km² = 1.257 × 10⁷ surface cells at 1 m
blocks. At 1,000 players × 1,000 blocks/day the ring's visible surface is consumed in **12.6 days**; at a
dedicated 10,000 blocks/player/day, **1.3 days** (for calibration, §2.3.6's steel-pick-on-granite worked
example is 2.99 s/block, so a dedicated eight-hour miner reaches ~9,600). The storage bill:
`block_provenance_collapse.md` §5.4 states that clear-cutting 81.3 km² fills the planet's entire budget, and
12.57 / 81.3 = **15.5% of the whole ~1.7 GB planet budget concentrated in a 2 km disc**. *Mitigation:*
R7/R8 natural recovery drifts cleared land back to baseline and **prunes its delta as it succeeds** — which
is why R7/R8 are load-bearing rather than flavour. Stone does not regrow, so the resource policy is the real
lever.

**(b) Resource policy: tiered, not abundant or sterile.** Pure abundance produces the moonscape above and
gives hauling nothing to carry inward. Pure sterility means a brand-new player with no ship cannot leave and
the market has no local supply — killing the very cold-start liquidity that justified the shared start. The
resolution, which is Albion's shipped shape, is **abundance in the lowest tier and outright ABSENCE above
it**. Low-tier materials are exactly the ones R7/R8 prunes the delta for, so the storage bill for the
abundant materials self-heals and the bill for the scarce ones is never incurred. "Absent" beats a `Finite`
regeneration policy because absence removes the depletion race entirely. One row in the override table.

**(c) The delta budget is a shared, exhaustible resource with no per-account quota.** §3.9.6's response to
exhaustion is a **loud realm-wide refusal of all further edits**. A placed cell costs 8 B plus ~1.1 pyramid
entries at 8 B = 16.8 B, and §3.4's prune rule can never fire on a `Placed` cell. So **1.7 GB / 16.8 B =
101 million placements permanently bricks editing on the starter planet.** At a 10 edits/s cap that is 2,810
account-hours: **117 accounts for 24 hours, or 12 accounts for 10 days**. The material brake normally slows
this (harvesting granite at 2.99 s/block gives ~28,858 placements/account/day, pushing it to ~3,500
account-days) — but recommendation (b)'s abundant free low-tier material at the spawn ring is precisely what
removes the brake. *Mitigation:* **a per-account resource allowance** (delta bytes, debris entities, market
listings) at the same admission seam as reach and rate. One counter and one comparison, and it converts four
un-mitigated spawn attacks from "denial of a shared resource" into "denial of your own allowance".

**(d) Building is 15–30× cheaper than breaking, and every "other players can take it down" mitigation loses
by that factor.** Placement costs one admission check and one 8-byte record, bounded only by
`max_edits_per_second_per_session` — a field with no chosen value anywhere in the design. Removal costs
`integrity_dp / power`: 2.99 s/block for a steel pick on granite; 50 s/block for a 2 kW handheld plasma
cutter on titanium. At a plausible 10 edits/s cap that is **30:1 on granite and 500:1 on titanium**. A
3 m × 3 m doorway walled three blocks deep is 27 cells — **2.7 s for the griefer, 81 s for one defender**,
and defenders queue at a doorway so they cannot parallelise. *Mitigation:* a per-cell **placement time**
derived from the same `integrity_dp` the break path already uses. Free, because the table exists.

**(e) Body-blocking is structurally unstoppable under a standing law.** Players physically collide and
collision must not be removed for optimisation. A 3 m doorway holds ~9 bodies at one per square metre, so
twenty alt characters hold every entrance indefinitely. It is not damage, so a PvP flag does nothing; not an
edit, so the ward does nothing; not a discharge, so the armistice does nothing. *Mitigation:* architectural
only — many wide entrances, no single choke point, arrival positions chosen by a least-occupied argmax over
a spawn **volume** rather than a spawn point. **This must be in the port blueprint before it is stamped.**

**(f) Non-destructible + only market + per-sale fee = an unassailable monopoly venue.** Three individually
correct decisions combine badly: the port cannot be destroyed, has structurally guaranteed footfall, and
would earn a cut on every trade. That is an incumbent no player-built venue can ever compete with, and it
silently kills the venue-ownership profession the consignment design creates — at exactly the moment the
owner is trying to make the economy emerge outward. *Mitigation:* the starter port is **zero-fee and
unowned** — infrastructure, not a business. Zero fee also removes the wash-trading loop at spawn (selling to
your own second account at a venue you own, where the fee is a transfer to yourself).

**(g) One griefer fills the entire starter market.** `NAMED_CONSTRUCTION_DEF.max_state_bytes` is 8192
(verified) and at ~55 B per inventory row that is **~148 stacks total** for a consignment venue's container.
Because the owner has ruled out clock-rent, the venue charges a per-**sale** fee only — so listing 148 stacks
of dirt and never selling them costs the griefer nothing. *Mitigation:* a listing **deposit** refunded on
sale or withdrawal (an event, not a rent, so the ruling survives) plus a per-account listing cap.

**(h) Single point of failure, and the whole decision being wrong.** Today spawn selection is a constant, so
one failed spin-up blocks 100% of new logins. *Mitigation, and it is one comparator:* extend addendum 1
§D.4's *"picks the highest-scoring reachable body — one sort"* from (habitability) to
**(habitability, Area-realm health)** over a candidate set. Three problems close at once: a new arrival lands
in the least-loaded Area (the distribution mechanism item 5 needs, with no separate balancer); k Areas give
k−1 redundancy for free; and **as long as spawn selection is an argmax over a candidate set rather than a
constant, adding a second starter world later is one more row in the override table rather than a
migration.** That is the cheapest possible hedge against the whole decision being wrong.

Good news worth stating: the login-stampede risk after downtime is small, because returning players log in
where they left; only the new-player path funnels through the shared start.

**(i) Separate arrivals from the market.** RuneScape did it deliberately; EVE did it by accident. Under this
architecture it is free and is the same mechanism as the congestion fix: the arrivals hall is one Area realm
and the market hall a different one within walking distance. Arrivals then have a bounded population (new
logins per minute) while the market absorbs the unbounded one.

**(j) Hauling viability is decided two phases later by someone not thinking about the economy.** A hauler's
earnings are (price gap × units carried) / round-trip hours, minus fuel and fees. The only term the market
design controls is the price gap; units-carried is a ship blueprint field and round-trip hours are thruster
and fuel fields — both P8, both authored for ship-feel reasons. Against a mining baseline of ~1,200 blocks
per hour (from the 2.99 s/block figure), a hold too small makes hauling permanently unprofitable and the
outward-radiating loop dies quietly with no failing test anywhere. *Mitigation:* record the hauler-viability
inequality as a named gate alongside the existing `min_price_dispersion_bp` dispersion gate.

Foxhole is the caution on the profession itself: an economy where all logistics are physically hauled
concentrates that labour in a small minority who describe it as a second job. Expect hauling to be done by
few and consumed by many, and price the fees accordingly.

---

## 8. What must be reserved now, ranked by cost if retrofitted

| # | Reserve | Cost now | Cost if retrofitted | Why |
|---|---|---|---|---|
| **1** | **`EditSource`'s full arm list** in the S0.4 wire plant | ~40 lines | **A PROTO_MINOR bump plus a client migration** | A declared field on a **binding** spec (`sealed_shards.md:309`) with zero defined arms. It rides `InterShardFlow::BlockEdit`; postcard is positional; the arm is gated by four structural conformance tests. It is also what lets a realm owner re-run admission on the forward path, and what R6-9 needs when a falling group's landing counts as an edit |
| **2** | **The arm-ceiling ruling, in prose** | ~0 | A rejected design after it is built | 27 arms today + 3 reserved = 30, which `decision_board.md` calls door 38's own review ceiling with *"no headroom left"*. No `Trade`, `Ward`, `BuildJob` or `EconCommand` arm may ever be added |
| **3** | **One generic `RealmKindTag::Construct = 6`** + one `FrameRef` arm carrying a generic parent | ~150 lines | **A format migration over every saved world** | Discriminants are frozen append-only because `RealmCoord` rides the wire behind `RealmDemand`. Without it: Area-under-Station is a `.expect()` panic, and a ship realm has no path so the demand reconciler cannot address it — **both of the owner's headline mechanics are unimplementable**. Note `RealmLevel.seed` is a u64 and `EntityId` is a u128, so the construct needs a minted lineage seed, not the entity id |
| **4** | **`OwnerRef` with a `World` arm** | ~30 lines | A store migration over every saved yard | NPC-provided and player-run are one field. Reserving the enum is what makes the extension a value change |
| **5** | **`SiteRequirement`'s field widths + reject-non-zero-reserved-bits decode** | ~60 lines | A migration over every saved blueprint | Blueprints are persisted, copied and re-sold data — exactly the property `decision_board.md` door 49 prices for the scoped channel key |
| **6** | **The deterministic build ORDER folded into `BlueprintHash`** | ~0 | A diverged write-ahead log between two servers | Free today |
| **7** | **`WardScope` as a u16 bitmask** (not a bool) | ~20 lines | A migration | Free now |
| **8** | **Generated, not stamped, starter spaceport** | 0 bytes | A data migration over every saved planet | Shuts at first world creation. ~2.7 MB of permanent delta per port otherwise |
| **9** | **Spawn selection as an argmax over a candidate set** | ~150 lines | A migration when a second starter world is wanted | The cheapest hedge against the whole single-start decision being wrong |
| **10** | **`EditAdmission::admit` as the one gate + the bypass tripwire** | ~300 lines | **A silent hole, and an audit of every write site** | P6 owes the seam anyway; the `Warded`/`Claimed` arms are free today. A mechanic added later that writes cells directly bypasses every ward with no error |
| **11** | **The containment fences** — move the count check into `RealmRegions::new`, bound the bitset index, add a `SiblingsOverlap` arm | ~120 lines | A silently mis-contained player | The boot fence *is* wired in the shard binary; but the constructor is infallible, the harness bypasses it, the geometric half never fires on the shipped forest, and sibling overlap is unchecked entirely |
| **12** | **A per-account resource allowance** (delta bytes, debris, listings) | ~120 lines | A live denial-of-service on a shared world | Every limit today is per-**realm**, so one griefer denies a shared resource to everybody |
| **13** | **`ShardProfile::build_site`** in the validated lattice | ~20 lines | A lattice change | Free now, coherent with `block_edit` |

---

## 9. The decision register for this ruling

| # | Decision | Options | Recommendation | Cost of deferring |
|---|---|---|---|---|
| **C-1** | **Does a purchasable SHIELD exist at all?** | (a) payout only; (b) shield capped by `ward_ceiling` and fuelled; (c) freely purchasable anywhere | **(a) at launch, (b) reserved.** (c) is what Dual Universe accidentally shipped | **Low.** Payout needs a dock, which needs blueprints (P8). Deferring costs nothing |
| **C-2** | **Civic non-destructibility for the starter port** | (a) `ClaimId`/ward refusal at the admission seam; (b) provenance = Terrain; (c) via insurance | **(a).** ~12 lines in the admission path, zero new durable bytes, no wire change, and it works with the economy compiled out — which (c) cannot | **Medium.** Addendum 1 §D.4 currently specifies the port as **damageable**; if it stamps first this becomes a live-world migration |
| **C-3** | **Does the civic volume refuse PLACE as well as BREAK?** | yes / no | **Yes** (Minecraft's `spawn-protection`), plus a build ring outside `civic_ward_margin_m` for shops | **Medium.** Saying no reopens the permanent-scar-at-spawn problem and needs a plot allowlist that is not designed |
| **C-4** | **Generated or stamped starter spaceport?** | generated (stage 9) / stamped into the delta store | **Generated.** ~2.7 MB of permanent delta per port otherwise, and repairs to baseline reclaim storage | **HIGH — one-way.** Shuts at first world creation |
| **C-5** | **Armistice: do weapons fire in the starter system?** | yes / no | **No.** One call to the same seam from `WorldAction::Fire`, already reserved as the carrier's second consumer. ~20 lines | Low |
| **C-6** | **Ward upkeep denomination** | (a) consumed material; (b) fixed-term re-bought policy; (c) perpetual | **(a)**, with (b) for the payout product. (c) is the permanent-scar generator | Low; only bites once wards are purchasable |
| **C-7** | **Is admin `RevertRegion` exempt from wards?** | yes / no | **Yes** — `EditSource::Admin` in every allow set, so griefing is always recoverable through the auditable WAL path | Low, but saying no makes a mis-issued civic ward unfixable without a code change |
| **C-8** | **The exact `EditSource` arm list** | as proposed in §5.14 | Adopt; append-only afterwards | **HIGH — one-way** (PROTO_MINOR bump) |
| **C-9** | **BUILD TIME** | (a) instant; (b) durable job, on-touch, `integrity_dp`-derived duration, yard-derived rate; (c) rate purely by equipment | **(b).** Reference: 21 minutes for a 10,000-cell metal ship at a starter yard; ~2 minutes at 10× rate. Say which end feels right and the rate field is set | Low; the whole part is P8-blocked anyway |
| **C-10** | **THE ROBOTS** | (a) simulated welders; (b) progress bar; (c) materialise in place, robots as client decoration | **(c).** ~10 MB/s of extra egress for one yard under (a) versus ~6.7 B/tick under (c); (a) also makes the yard's realm permanently un-reapable | Low, **except** the build-order-in-`BlueprintHash` sub-item, which is free now |
| **C-11** | **Does the starter yard SELL materials?** | (a) no; (b) finite stock on a replenishment budget; (c) infinite at fixed price | **(a) for v1, (b) never before the economy exists, (c) never.** This decides whether construction is a sink, a shop, or a money printer | Low |
| **C-12** | **Where does a world-owned yard's fee go?** | (a) burned; (b) world treasury; (c) not charged until a player owns it | **(a) once currency exists, (c) before.** (b) is a faucet with no counterparty | Low now, expensive once balances are real |
| **C-13** | **R18's payment brake** | payment gates the mint / a crafted `ConstructionAnchor` gates the mint | **The anchor.** As written R18 violates LAW-WL-7 and has no correct failure mode during an account-service outage | **HIGH.** It is written into an owner ruling AND a P8.1b slice deliverable; by the time it ships it is entangled with realm minting |
| **C-14** | **R18's price** | the dock sets a price / the dock states the bill of materials | **The bill.** A dock computing a price is a computer trader with a formula price | Medium |
| **C-15** | **Author royalties (R18's third consequence)** | in / out of v1 | **Out.** Needs a durable author identity that survives copying. Non-negotiable rule either way: a royalty may never gate the build | Low |
| **C-16** | **Spaceport parenting** | Areas under the Planet / a Station with Areas inside | **Planet-parented Areas.** Free and matches addendum 1 §D.4. A Station with Areas inside **panics the forest build** today | **HIGH** if the Station shape is assumed and built against |
| **C-17** | **Realm vocabulary** | append `Ship = 6` / append one generic `Construct = 6` + a generic `FrameRef` parent arm | **The generic pair.** `Ship` alone repeats the mistake (`Room` next, `Asteroid` already in `ProfileKind`) and **does not even work** — `RealmLevel.seed` is a u64, `EntityId` a u128 | **HIGH — one-way** (format migration over every saved world) |
| **C-18** | **Relax the never-siblings neighbourhood scope for co-framed STATIC siblings?** | keep / relax by a typed condition | **Relax, narrowly** (same frame AND static placement), never by a shard-kind match. Keeping it costs two transfer sagas per boundary crossing (~40/s at 1,000 players in 50 m cells) | Medium |
| **C-19** | **May the client-facing snapshot row diverge from `StampedPose`?** | yes / no | **Yes**, landed in the same slice as the D-9 emit reshape. ~125 B → ~20 B, and 24 of the current bytes are a velocity field the render path provably never reads | Medium; touches a frozen wire type so it needs an explicit ruling |
| **C-20** | **Target per-client entity-lane bandwidth budget** | pin a number | **Pin it deliberately.** The 40 kB/s figure everything rides on is inferred and never measured. At 40 kB/s the current row supports 16 co-visible entities; at 1.5 Mbit/s it supports 78 | Medium — Area count, co-visible cap and row width are all consequences of this one number |
| **C-21** | **How many Areas (k), and are the boundaries walls?** | k and the geometry | **k = 1 or 2 until a measured ceiling exists**, and every boundary an authored bulkhead. If people can see across, the split is negative value | **HIGH.** The geometry fixes k at stamp time |
| **C-22** | **Does an Area own its voxels?** | Area / planet | **The planet keeps them.** The Area split then relieves entity and snapshot load only, one writer per store survives, and cube-sphere addressing is never split | Medium |
| **C-23** | **Order of work: S0.5 before the spawn carries population?** | yes / no | **Yes.** ~33× versus the split's ≤8×, and the only one of the two that relieves the gateway | **HIGH.** Landing sibling subscriptions before S0.5 makes the measured wall worse |
| **C-24** | **Separate arrivals from the market?** | yes / no | **Yes** — different Area realms in the same port. Free if decided before the stamp | Medium |
| **C-25** | **Starter-ring resource policy** | abundant / sterile / tiered | **Tiered:** lowest tier abundant and regrowing, everything above **absent** | Medium |
| **C-26** | **Does the starter port charge a per-sale fee, and who owns it?** | fee/owned vs zero-fee/unowned | **Zero-fee and unowned.** Infrastructure, not a business | Medium |
| **C-27** | **PvP at the spawn Areas** | on / off | **Off at the port, on outside**, expressed as a per-realm capability **value** on the Area's `ShardProfile` (like `signal_graph`), never a match on shard kind | Low |
| **C-28** | **Market listing deposit + per-account listing cap** | yes / no | **Yes.** A deposit is an event, not clock-rent, so the standing ruling survives. Without it 148 stacks of dirt fills the only market in the game at zero cost | Low |
| **C-29** | **Per-cell placement TIME derived from `integrity_dp`?** | yes / no | **Yes.** Removes the 30:1 build-vs-break asymmetry that generates most spawn griefing, and the table already exists | Medium |
| **C-30** | **Do abandoned player builds DECAY back to terrain?** | yes / no | **Yes**, as weathering through the ordinary damage path. Today every structure ever built is permanent delta against a ~1.7 GB budget with no reap path for surface builds | **HIGH.** This is a bigger risk than anything in the ward design |
| **C-31** | **Remote fulfilment of goods — reconfirm the refusal** | refuse / allow | **Refuse, in writing.** "Goods never move without being flown; money may." Want-ads local-fill only | **HIGHEST for the economy.** Reversing it ends distance-based trade entirely |

---

## 10. Scale honesty

`decision_board.md` §8 is explicit: the block-plus-signal plan alone is ~77,000 production lines against a
verified baseline of 115,284, with an honest total of 140,000–170,000 including HR5 test volume, and *"the
plan as scoped is not achievable for one developer with AI assistance inside any horizon you would accept."*
The load-bearing subset is ~34,000–38,000 lines.

**What this ruling's recommendations would add, if all of them were built:**

| Item | Production lines |
|---|---|
| `EditSource` enumeration + decode rule | 40 |
| `EditAdmission` seam + refusal enum + bypass tripwire | 300 |
| Containment fences (count into the constructor, bounded index, sibling-overlap arm) | 120 |
| Spawn-selection argmax over a candidate set | 150 |
| Per-account resource allowance | 120 |
| `OwnerRef` + `ShardProfile::build_site` + `WardScope` + `SiteRequirement` type | 130 |
| **Subtotal — owed regardless of whether any of the six asks ever ships** | **~860** |
| Civic ward record, closure invariant, the one authored ward, expiry | 600 |
| `Construct` realm kind + generic `FrameRef` parent + RLM keying | 500 |
| `SiteRequirement` predicate + dock-as-OBB + capacity | 450 |
| Construction site + hopper + job record + cursor + materialise driver | 1,400 |
| **Items / containers / `ItemId` durable substrate — nobody has scheduled this** | **2,000–3,000** |
| Cross-Area visibility (InterestSet producer, gateway sub-table, client) | 1,200 |
| Fixing the AoI loop's clones and the containment scan | 150 |
| **Total** | **~7,300–8,300 production, plus roughly the same again in Tier-A tests** |

That is **~15,000–16,000 lines, about 40% on top of the load-bearing core**. It is not affordable, and
saying so is the useful thing to say.

**Ship now (~860 lines):** items 1–6 above. Every one is owed whether or not the owner's six asks ever land,
and every one is either a one-way door or a live gap.

**Ship at P6 (~600 more):** the civic ward, because it is what the owner asked for first and it is what
actually protects the shared start.

**Defer explicitly — and tell the owner these are deferrals, not deletions:**

- **Insurance in every form.** The player-facing product should be payout; payout needs a dock; the dock
  needs a blueprint format; blueprints are P8.
- **Construction companies and build robots.** Blocked on an items-and-containers substrate that does not
  exist anywhere in `crates/` (verified: no `ItemStack`, `Inventory`, `Container` or `ItemId` type; no
  shard-side durable `Store`; the exactly-once journal is still a RAM `BTreeSet`, D-22 owed). **Construction
  cannot escrow what the game cannot hold**, and pricing the substrate inside "construction companies" would
  hide a 2,000–3,000 line slice.
- **Cross-Area visibility.** It costs more than the single-realm broadcast it replaces.
- **The multi-Area partition beyond k = 2.** There is exactly one measured data point and it passed.
- **A material shop, author royalties, parallel build slots, and any robot simulation.** All additive, none
  a door.

**Already correctly refused, and worth keeping refused:** a constraint expression language on blueprints; a
fourth provenance state for protection; per-cell protection records; a ward check inside the damage kernel; a
new `InterShardFlow` arm for anything in this ruling; instancing at the spawn; and a shield tier that is not
capped by the realm's own authored `ward_ceiling`.

---

## 11. Adjudicated objections

**"`guard_regions_nest` has no production caller — the 64-region bound is enforced by nothing."**
**REJECTED as stated, and corrected.** `crates/bins/src/bin/shard.rs:248` calls
`guard_regions_nest(&regions, vd_sim::stub::MAX_REGIONS)` before the infallible `RealmRegions::new` and
refuses the boot on error. What *is* true, and is what should be fixed: the constructor itself is infallible
and unguarded, so `crates/harness/src/topology.rs:1513` and `tests/src/lib.rs` bypass it; the detector's scan
does `1u64 << ix` from an unbounded `enumerate()`; the geometric half never fires on the shipped forest
because `child_fits_in_parent` short-circuits on differing frames and no two generated realms share a frame;
and sibling overlap is unchecked entirely. Three real gaps, one wrong headline.

**"Append `RealmKindTag::Ship = 6` and ships become addressable."**
**REJECTED.** `RealmLevel { kind, seed: u64 }` carries a u64; `RealmId::Ship(EntityId)` carries a u128
(`crates/core/src/ids.rs:157`). The level cannot hold the identity. A construct needs a minted seed-lineage
identity at birth, with the `EntityId` retained for physics — which also fixes the re-parenting churn
problem, because identity is then frozen at birth and the live containment parent is a separate field.

**"A mobile realm's path changes as it moves, so the reconciler sees a death and a birth on every SOI
crossing."**
**UPHELD, and it is the same fix.** `RealmCoord`'s own module docs name `path` as the collision-free key and
warn never to key a directory, saga or dedup on `lowered()`. Identity must be the **birth** lineage, frozen
at mint. FA-6 covers a realm changing owner **shard**, not changing **parent**, and explicitly says the
reparent detector is not built.

**"Splitting the spawn into Area realms fixes the crowding."**
**PARTIALLY REJECTED.** It fixes per-shard CPU by up to k× **only if the boundaries are opaque**. With
cross-visibility it is 5.3% worse than not splitting. Against everyone-in-one-plaza it does nothing. And it
creates O(occupants × Areas) work with four Vec clones per pair per tick on the parent shard. S0.5 is worth
~33× and relieves the gateway too.

**"D-9's measurement is 357,248 messages over 7.6 seconds."**
**CORRECTED.** The 7.6 s is wall-clock under the virtual clock, not simulated time; 357,248 ÷ 1,920 implies
at least 187 ticks, i.e. ~9–10 simulated seconds. Rates derived from 7.6 s are ~25% too high. Every prior run
repeated this.

**"A single origin creates the price gradient the owner wants."**
**REJECTED.** Spatial price equilibrium bounds the gap by transport cost alone; distance from a spawn point
appears nowhere. The origin creates a demand **sink** in the first weeks, which resolves into local basins as
soon as players settle outward. Keep the origin for cold-start liquidity and a shared reference price.

**"Insurance is what makes a building non-destructible."**
**REJECTED for the starter port.** Insurance is an economy overlay, and a spaceport that becomes destructible
when the economy is compiled out breaks the one-way dependency law and fails the economy-absent gate. Civic
non-destructibility must be authored game state.

**"Protection should be a fourth provenance value."**
**REJECTED.** The `new >= old` monotonicity invariant that closes the laundering exploit forbids a privilege
that can expire, and per-cell protection costs well over a million times more storage than one region record.

**"The ward check belongs in the damage kernel, since every destructive event is one object."**
**REJECTED, with the direction upheld.** Only 8 of 14 removal paths go through `apply_hit`; a ward there is
defeated in one tick by mining the ground. The seam is the caller — the edit-admission gate — and the
property is per-region rather than per-object.

**"The starter yard's cut can be taken as a percentage of delivered materials."**
**REJECTED.** It makes the economy-absent byte-identity gate unsatisfiable, because the two arms would consume
different quantities of matter. Free to forbid now; a redesign after the first yard ships.

**"R18's payment brake is the anti-abuse mechanism for realm spin-up."**
**REJECTED.** It is a LAW-WL-7 violation (an economy value entering `desired_alive`), it fails the
economy-absent gate, and it has no correct behaviour during an account-service outage. §3.6's crafted
`ConstructionAnchor`, capped per account, already exists and costs nothing new.

**"The dock sets the price."**
**REJECTED.** That is a computer trader with a formula price, which the recorded economy direction refuses.
The dock states the bill of materials; the market prices it.

**"The blueprint header should carry a constraint expression language."**
**REJECTED.** Two bitmasks and one scalar cover every case the owner named. A language adds an evaluator, a
decode surface and an adversarial input for zero benefit.

**"A dock should be a realm."**
**REJECTED.** `FrameRef::AreaLocal` is hardwired to a Planet parent and the call site is an `.expect()`, so a
dock-as-Area panics the forest build. `Boundary::Obb` is already written and covered, spends zero realm
depth, and composes identically on a ship, in a station and on a surface pad.

**"The site predicate should be re-evaluated as the world changes."**
**REJECTED.** Twice — at quote and at mint. The structural-support mechanic already answers "the ground was
mined away later", bounded at 45,825 cell visits per edit and zero in undisturbed terrain.

**"Prior art supports permanent purchasable indestructibility."**
**REJECTED, with a caveat on sourcing.** No shipped game ships unconditional, permanent, purchasable
indestructibility on a shared world; every one limits it by place, by time, or by replacing object-protection
with value-restoration. ⚠ These specifics are recalled and could not be re-sourced (web-search budget
exhausted at 200/200 before this run). Treat the structural pattern as reliable and every specific figure as
needing a browser pass before it is quoted in a binding spec.
