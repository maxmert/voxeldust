# THE LOOK HORIZON — a realm draws itself at depth, with a bound the world measures

Synthesis of three designs (local-bound, verdict-pull, first-principles) and three judge verdicts.
Spine: **first-principles**. Grafts: **verdict-pull**'s carrier discipline and presence floor,
**local-bound**'s measured bound. Worktree `new-system`, HEAD `ffc9af1`.

Every code anchor cited below was read in this pass unless it is explicitly labelled
**CLAIM (unverified)**. Every number is either taken from a shipped constant or labelled as an
estimate.

---

# ⚠ OPEN QUESTIONS FOR THE OWNER — READ FIRST

Four decisions are yours. Nothing in this design ships past slice 2 without answers to Q1 and Q2.
Full statements are in §7; here is the one-line form so nothing is buried.

| # | Question | Default if you say nothing | Blocks |
|---|---|---|---|
| **Q1** | May a realm be told that something outside it may be looking in? One byte. Two values. It decays to "nobody is watching" if the message stops. | NO ⇒ **the flown defect cannot be fixed by any mechanism** (proof in §3.4). | Slices 4, 5 |
| **Q2** | May a parent state its child's SIZE (one number it already holds for containment), so a sleeping or lapsed thing still draws as a correctly-sized point of light instead of vanishing? | NO ⇒ every non-glowing thing (a city, a station, a ship) **disappears** when its own picture lapses, instead of shrinking to a dot. Verified defect today, not a new risk. | Slice 1 |
| **Q3** | At the current compressed world scale, a built structure on a planet needs its picture to travel **three** steps, not two. Do you want: (a) raise the carrier to three steps now, (b) land the real astronomical scale first, or (c) let the game refuse to start until one of those happens? | (c) — the game refuses to start, loudly, before a wrong pixel. | Building (P6/P8) |
| **Q4** | May a child later tell its parent one number — "things inside me are visible from this far out" — so player-built interiors wake correctly? | Deferred. Not asked now. Seed-generated interiors need no message. | Player building |

**Verdict on new wire arms: ONE, and only if you approve Q1.** Everything else in this design is
an appended field, an appended enum variant, an appended tag, or a rename in place.

---

# 1. PLAIN-LANGUAGE OWNER SUMMARY

## 1.1 What you saw

You flew out of the star system. The planets became small points of light. The points moved
correctly. You expected the planets to keep looking like planets.

## 1.2 Why it happened

Three separate faults produce one picture. All three are confirmed in the code.

**Fault one — the planets were asleep.** A realm only starts when its own parent asks for it. The
star system is the planets' parent. The star system held nobody. A realm that holds nobody asks for
nothing. So the planets never started. A thing that is not running cannot draw itself.

**Fault two — even awake, a planet's own picture travels only one step.** It reaches the star
system. You stood in the galaxy. That is one step further out. The picture never arrived.

**Fault three — the galaxy cannot name a planet.** The galaxy only knows about its own direct
children. It can say "this star system matters". It cannot say "this planet matters".

The dots you saw moved correctly because the star system tells the galaxy where its planets are.
It does that from its own records. It does that whether the planets run or not. **Correct movement
was not proof that the planets were alive. They were not.**

## 1.3 What changes

**Change one — tell a realm that somebody outside may look in.** A parent sends one byte to a
child it cares about. The byte says "assume somebody may look inside you". The byte carries no
name, no position, no direction, and no count. The child then does the work it already knows how to
do: it decides which of its own children matter, and it starts them. This needs your approval. See
Q1.

**Change two — a picture travels two steps instead of one.** A realm sends its own picture to its
parent. The parent then passes that picture, sealed and unread, to the grandparent. The parent
cannot open it. The parent cannot change it. The parent can only pass it on or drop it.

**Change three — the picture cannot travel three steps.** The message that carries a child's
picture has no place to put a grandchild's picture. This is not a rule that somebody must remember.
There is no field. To go deeper, somebody must edit one reviewed file.

**Change four — measure the depth the world needs, at start-up.** The world generator already
walks every body and every ancestor. Today it gives a yes-or-no answer and refuses to start on a
"no". We change it to report a number: for each body, how many steps its picture must travel. The
game refuses to start only if that number is larger than the message can carry. Today the number is
two, with four metres of room to spare. Four metres is exactly the safety margin the world already
reserves. The live test and the world's own size calculation are the same equation.

**Change five — a sleeping thing still shows as a correctly sized point of light.** Today a point
of light is drawn from brightness only. A thing that does not glow gets no point of light at all.
It vanishes. We give the point of light a size. This needs your approval. See Q2.

## 1.4 What you will see afterwards

Fly out of the star system. The planets stay planets. They shrink smoothly with distance. They do
not collapse into identical dots. Fly back in. The planets grow. There is no jump at the boundary,
because a planet starts up before you can get far enough out to look back at it.

Later, fly down to a planet with a city on it. The same machinery shows you the city, then the
buildings. No new code is needed for that. One number may need to grow first. See Q3.

## 1.5 The honest cost

- **Traffic.** The message from a star system to a galaxy grows by about half. Measured per link,
  that is small. Measured across the whole cluster it is the number that matters, and it is written
  out in §5 with the multiplier the earlier drafts left out.
- **Running realms.** A lone player keeps a few more realms running. The rule stops exactly two
  levels below the player. It cannot cascade further.
- **One new message type**, if you approve Q1.
- **One risk you must know.** At today's compressed world scale, a large structure on a planet
  surface needs three steps, not two. The world will refuse to start rather than draw it wrongly.
  See Q3.

---

# 2. THE SL6 FORMAL ASK

Default is NO. Each item states: what data, from which realm to which, why the receiver cannot
compute it, and what doing without costs.

## ASK A — THE SEALED INTERIOR FORWARD (a grandchild's own picture reaches its grandparent)

- **Status: NEEDS-A-NEW-RULING** (low controversy — it extends the landed Q2 PARENT RELAY by one
  hop, with every Q2 property intact).
- **What data.** A direct child's own sealed statement batch, byte-for-byte, plus that child's own
  fence, carried in the relay its parent already sends upward.
- **From which realm to which.** Authored by realm S. Held sealed by S's parent (this already
  happens: `RelayHeld`, `crates/sim/src/stub.rs:517`). Now additionally forwarded, unopened, to
  S's grandparent, where it is again held sealed and handed to the gateway. **No realm ever opens
  it.** Production code in `vd-sim` calls no `open_relay_*` function; the only production caller is
  `crates/connection-plane/src/gateway.rs:4044`, and the gateway is not a realm.
- **Why the receiver cannot compute it.** A shard's world scope is its ancestors and its direct
  children (`crates/core/src/worldgen.rs:140-157`). A galaxy shard holds no planet region at all.
  Inventing a grandchild's appearance is forbidden by SL3 and by THE DRAW LAW ("no third source").
- **Cost of doing without.** Exactly the flown defect: planets that the world's own visibility law
  says subtend up to fifteen degrees draw as three-pixel dots.
- **Wire change.** `InterShardFlow::WindowRelay.statements` is renamed `own` (same type, same
  position) and one field is appended: `interior: Vec<InteriorRelay>`. `ShardToGateway::WindowRelayed`
  gains the same appended field. **No new arm. No new discriminant. Append-only.**

```rust
// crates/wire/src/intershard.rs — appended field on an existing arm
pub struct InteriorRelay {
    pub child: RealmId,      // the FORWARDER's direct child — the author of `own`
    pub child_fence: Fence,  // that author's OWN fence, OUTSIDE the seal (zombie guard)
    pub own: Vec<u8>,        // that author's sealed batch, VERBATIM. Never opened by a realm.
}
```

- **The depth bound is the type.** `InteriorRelay` has no `interior` field. There is nowhere to put
  a third level. Deepening requires an edit to `crates/wire/src/intershard.rs`, which is HR1's one
  reviewed file.

## ASK B — THE INTEREST BIT (parent tells a direct child that something outside may look in)

- **Status: NEEDS-A-NEW-RULING. THIS IS THE ONE THAT MATTERS. It creates ONE new wire arm.**
- **What data.** One byte with two lawful values. `1` = "assume somebody may look inside you".
  `0` = "do not". No account, no identity, no position, no direction, no distance, no count. Plus
  the routing coordinate, the sender's fence, and the tick — the same envelope every other lane
  carries.
- **From which realm to which.** From a parent to ONE of its direct children, direct to the child's
  head node, on the route the parent already resolves for attestation (`ChildRealmNodes`,
  `crates/sim/src/stub.rs:399-411`). Never further. Never sideways. Never through the orchestrator.
- **Why the receiver cannot compute it.** The child does not know its own position (SL1). It cannot
  see any occupant pose (SL2). Its world scope holds nothing outside its own boundary. It is
  structurally blind to its surroundings.
- **Cost of doing without.** The owner's law becomes unachievable — not expensive, **impossible**.
  The proof is in §3.4. Verified premises: a realm draws only while running; a realm is demanded
  only by its own parent (`push_demand` refuses anything but self-or-direct-child,
  `stub.rs:8188-8199`); a vacated realm's decision loop returns before it evaluates any child
  (verified this pass at `stub.rs:7873-7893` — it pushes an Empty verb, emits an empty membership,
  and returns; every spin-up emission sits after that return). Therefore a vacated star system can
  never wake its planets from anything it legitimately holds.
- **Alternatives priced, all worse.**
  - Wake every interior whenever the child runs: on THE world a star system runs out to
    11 458.47 m, while its planets are only visible from inside about 444 m. That is a radius ratio
    of about 26 and a volume ratio of about 1.8 × 10⁴ of pointless shard start-ups.
  - Let the gateway open a window directly on the deep realm: that is `WindowScope::Observed`,
    which you already refused, and it discloses strictly more (a named gateway, per observer, at
    unbounded depth).
  - Ride an existing arm: the only parent→child arm is `RealmCascade`, which is a tombstone
    (`crates/wire/src/intershard.rs:295-310`) and must not be revived. Riding the demand arm to a
    new receiver is a new lane wearing an old discriminant, and a judge correctly called that
    dishonest. **So it is declared as its own arm.**

```rust
// crates/wire/src/intershard.rs — ONE new arm, reliable/ReDriven
pub struct RealmInterest {
    pub child: RealmCoord,   // routing key + misroute guard
    pub parent_fence: Fence, // the sender's authority over this child
    pub at: UniverseTick,
    pub look_inside: u8,     // 1 or 0. Nothing else is lawful.
}
```

- **What it costs you in principle.** The landed Q2 rationale says "am I observed from outside stays
  unrepresentable in every realm". That clause ends. This is a ruling change and it is stated as
  one, not smuggled.
- **Fail-closed shape.** Misroute, unattested sender, stale fence — all refused and counted,
  mirroring `retain_child_live` (`stub.rs:8221-8267`). Held under the derived retain TTL. If the
  lane goes silent the byte decays to `0` — "nobody is watching" — which is the safe direction.

## ASK C — THE MARKER'S SIZE (a parent states its child's extent in the point-of-light bag)

- **Status: NEEDS-A-NEW-RULING. Crosses NO realm boundary. It is an SL3 question, not an SL6 one.**
- **What data.** One appended tag on the existing point-of-light bag, carrying the child's
  circumscribed extent — the number the parent already holds and already reads for its own proxy
  logic (`stub.rs:7840`).
- **From which realm to which.** None. It is derived inside the authoring realm from its own boot
  roster.
- **Why it is asked at all.** Verified this pass: `current_bodies` (`stub.rs:7339-7358`) emits a
  point-of-light marker **only for children the boot roster carries a brightness bag for**, and the
  bag codec (`crates/core/src/look.rs:25-51`) has exactly two tags — the realm's own outline and
  the brightness pair. **There is no lawful bag content for a non-glowing child.** So today a city,
  a station or a ship has no marker at all: when its own picture lapses, the bag selection returns
  empty and the thing is tracked but not drawn. It vanishes. Every failure table in all three input
  designs said "body → marker, never body → nothing". For non-glowing subjects that sentence is
  **false today**.
- **Cost of doing without.** (a) Non-glowing subjects vanish rather than shrink, at every hop death
  and every picture-lapse. (b) A planet's marker draws at the three-pixel floor while its true size
  at the handover is about eleven pixels, so the marker-to-body handover is a visible jump of about
  3.8×.
- **The SL3 tension, stated rather than decided.** SL3 says the realm authors how it looks. An
  extent is arguably part of how it looks. The argument for allowing it: the parent already authors
  the containment volume and the interest band from that same number; it is a bounding number, never
  a surface, a detail or a mesh; and the instant the realm runs, its own picture supersedes it
  entirely. **Default NO. Your call. Slice 1 is conditional on it.**

## ASK D — THE LIVE INTERIOR REACH (deferred, NOT asked now)

- **Status: LEDGERED, NOT REQUESTED.** For the seed-generated world the number is derived at boot
  from the forest the generator already holds, so **nothing crosses a boundary today**. Once player
  building lands, a realm whose interior grows will under-wake until its parent re-boots. That is
  the moment to ask. Registered as **D-LOOK-3**.

## APPROVED-BY-EXISTING-RULING (no new permission needed)

| item | ruling it rides |
|---|---|
| The one occupancy bit upward | SL7, landed |
| A realm's own picture reaching its direct parent | Q2 PARENT RELAY, landed |
| A parent authoring its children's placements and shipping them | SL1, landed |
| The realm-side interest verdict shared by every observer under it | owner ruling 2026-08-16 |
| Fixtures planting player-built regions on THE world | SL5, landed |

---

# 3. THE TECHNICAL DESIGN

## 3.1 The three locks, and the one mechanism that opens each

| lock | verified anchor | opened by |
|---|---|---|
| **WAKE** — a vacated realm demands nothing | `stub.rs:7873-7893` — zero observers ⇒ Empty verb, empty membership, `return`; every spin-up sits after it | §3.3 the interest bit + the down-proxy |
| **REACH** — a picture travels one hop | `emit_window_relays` forwards `entry.statements` verbatim; the ship site builds only `current_bodies` | §3.2 the sealed interior forward |
| **NAME** — a grandparent cannot name a grandchild | membership ships direct children only; `scene_bag`'s member gate is authors ∪ members (`window.rs:793-802`) | §3.5 the forward gate IS the membership gate |

## 3.2 PROPAGATION — what travels, how far, what stops it

**What travels.** Sealed statement batches, unchanged in content. A batch is one `Level` (the
author's own children's placements, in the author's own frame) plus one `Body` per statement (the
author's own picture, and one point-of-light marker per direct child).

**How far.** Two hops. A realm ships its batch to its parent (today). The parent forwards its own
batch **plus each held child batch** to the grandparent (new). The grandparent forwards **only its
own batch** upward. The forward rule reads `held[child].own` and never `held[child].interior`.

**What stops it — the type.** `InteriorRelay` carries no `interior` field. A third level is
unrepresentable. This is verdict-pull's contribution and it is strictly stronger than any check:
a buggy forwarder, a hostile shard, or a future feature cannot extend the climb, because there is
no field to put the bytes in.

**What is admitted at the gateway.** For each `InteriorRelay`:
1. the named child must appear in the relaying child's own attested roster
   (`relay_child_roster`, the existing vouch at `gateway.rs:4062-4071`); otherwise refuse and count
   `relay_interior_unvouched` — **a violation counter, gates assert 0**;
2. its fence orders against the existing per-realm fence map (`admit_relay_fence`,
   `window.rs:271-281`, already keyed by realm id, so it generalises unchanged);
3. from the opened batch, admit only the author's own picture (`SelfLook` whose subject is the
   author). Its `Level` and its markers describe depth-3 subjects, for which no row can exist, so
   they are dropped and counted `relay_interior_filtered` — **an expected-nonzero counter, never
   asserted zero.** (This splits verdict-pull's single counter, which its own gate contradicted.)

**Levels versus bodies — what each depth needs.** Depths are relative to the observer's realm.

| depth | placement comes from | hops | picture comes from | hops | status |
|---|---|---|---|---|---|
| 1 | the observer's realm's own level | 0 | the child's own batch | 1 | landed |
| 2 | the relayed child's own level (already arrives) | 1 | the grandchild's own batch | **2** | **this design** |
| 3 | would need 2 hops | 2 | would need 3 hops | 3 | **unrepresentable** |

So **levels do not change at all**. They already travel exactly as far as they must. Only pictures
travel one hop further. This asymmetry is forced: a parent authors placements (SL1) but a realm
authors its own appearance (SL3), so the appearance lane starts one level deeper and must therefore
be one hop longer.

**Does the composer's descent recurse? NO.** The fold performs exactly one downward hop
(`hop_book`, `window.rs:640-653`) on top of the chain's own descent, which is exactly two levels
below a chain author. That already matches the carrier's arity. Building recursion would build
machinery for rows that cannot exist.

## 3.3 THE TERMINATION OBLIGATION — measured, not argued

This is the part the earlier drafts got wrong in three different ways, so it is stated carefully.

### 3.3.1 What is NOT a proof

The world's boot fence (`guard_grandchildren_invisible_outside`, `crates/physics/src/worldgen.rs:816-824`)
refuses to boot a world in which any body two or more levels deep would still subtend the minimum
angle from just outside that ancestor. Two of the three input designs used it as their whole
termination proof. **It cannot serve as one**, for a reason verified this pass: it runs over
`generate_system_forest`, and that generator emits a universe, a galaxy, star systems and planets —
**nothing else**. Player-built regions never enter it. So it says nothing about the content the
owner's law was written for.

A second verified fact makes this sharper. The offence test is
`d_min = ancestor_extent − worst_distance − extent` against `required = extent × 76.39`. For
anything standing on a planet surface inside a star system, `worst_distance` is the planet's extent
plus its orbital excursion, about 146.0 m against a system extent of 150.0 m, so `d_min ≈ 4.0 − e`
while `required ≈ 77.4 × e`. **Any surface object above about 5 cm is an "offence" under the current
predicate.** A guard that refuses the boot on that basis would refuse the first city.

### 3.3.2 What IS a proof — the boot MEASUREMENT plus a carrier arity

Take local-bound's construction, keep the world solve untouched:

```rust
// crates/physics/src/worldgen.rs — replaces the boolean guard, same walk, same numbers
pub struct VisibilityClimb { pub body: RealmId, pub top: RealmId, pub levels: usize, pub slack_m: f64 }
pub fn measure_visibility_climb(seed: u64, cfg: &UniverseConfig) -> Vec<VisibilityClimb>;
pub fn guard_visibility_climb_bounded(seed, cfg, arity: usize) -> Result<(), VisibilityClimb>;
pub fn guard_candidate_climb_bounded(candidate, forest, arity) -> Result<(), VisibilityClimb>;
```

Same ancestor walk as `grandchild_visibility_pairs`, with the `hops >= 2` filter dropped. For each
body it reports the highest ancestor from outside which the body is still visible, and the slack at
the level where visibility stopped. The boot fence then refuses a world whose measured climb exceeds
what the carrier can carry — and it prints the offending body with its numbers, in today's
fail-loud shape.

**The measurement on THE world today** (all inputs are shipped constants):

```
minimum angle factor            = 76.389830658075
planet extent                   =      3.9541737529995578 m
planet visibility reach         =    302.05866338424295  m
planet worst excursion in system=    142.04582624700046  m
system extent                   =    150.0               m
galaxy ring                     = 12 031.398328646887    m
galaxy shell (FROZEN)           = 12 483.45699203113     m
two-level clearance (FROZEN)    =    452.058663384243    m
containment margin              =      4.0               m

budget of a planet at its system  = 3.954 + 302.059 + 142.046 = 448.058663384243 m
   system extent 150.0  ≤ 448.059  ⇒ still visible ⇒ climbs to the galaxy
budget of a planet at the galaxy   = 448.059 + 12 031.398    = 12 479.456992031130 m
   galaxy extent 12 483.457 > 12 479.457 ⇒ NOT visible ⇒ STOPS.   slack = 4.000000 m
```

**The slack equals the world's containment margin exactly**, and that is not a coincidence: the
galaxy shell is solved as `ring + two_level_clearance`, and `two_level_clearance` is
`worst_reach + extent × (1 + factor) + margin`, which is the planet's budget plus the margin. I
verified the identity arithmetically: `12 031.398328646887 + 452.058663384243 = 12 483.45699203113`,
the frozen shell. **The live bound and the world's own size calculation are one equation written
twice.** One formula lives in `vd-core` and has three consumers: the world solve, the boot
measurement, and the runtime tripwire.

**Measured max climb on THE world today: 2. Worst stopping slack: 4.000000 m.**

Note on which margin to quote: `FROZEN_TWO_LEVEL_WORST_MARGIN_M = 11.127605697744457` is the worst
**drawn-eccentricity** margin. The **reserved worst-case** margin the solve guarantees is
`VISUAL_SYSTEM_MARGIN_M = 4.0`. The engineering-relevant number is 4.0.

### 3.3.3 What is kept

`two_level_clearance_m`, `galaxy_shell_r_m`, and every frozen constant stay exactly as they are.
They now buy headroom rather than state a prohibition, and the headroom they buy is precisely what
makes today's climb stop at two. Retiring them would move `FROZEN_GALAXY_SHELL_R_M` and break four
pinned tests for no gain.

### 3.3.4 The three instruments

| instrument | measures | fails how |
|---|---|---|
| `guard_visibility_climb_bounded` at boot, in **every** process's fence (today only the shard's, `crates/bins/src/bin/shard.rs:331-337`) | the generated world's required climb ≤ carrier arity | the process refuses to start, printing the body and its numbers |
| `guard_candidate_climb_bounded` on the build-admission path | a candidate player-built region's required climb ≤ carrier arity | the placement is refused, never the boot |
| `relay_climb_levels_max` (shard gauge) + `relay_interior_unvouched` (gateway) | an implementation climb bug, world-independent | gates assert the gauge equals the boot number and the violation counter is 0 |
| **`G-NOTHING-OWED`** (§6, the law gate) | for the camera pose in the pixel gate, every generated subject whose true angular size exceeds the minimum has a drawn row with its own picture | fails if anything that should be a body is a dot, at any depth |

`G-NOTHING-OWED` is the only gate in this document that can fail for the reason the owner actually
cares about. It uses the generator out-of-band as its oracle, so it is independent of the carrier,
the bound, and the wake rule.

### 3.3.5 The honest consequence at today's scale — Q3

Run the measurement on the interim world with a 20 m structure planted on a planet: the structure's
visibility reach is about 1 528 m, and it remains visible from outside its star system, so its
required climb is **3**. Today's carrier serves 2. Three responses exist, and the choice is the
owner's (Q3):
(a) raise the carrier arity to 3 — a reviewed edit adding one more flat carrier level;
(b) land the near-real astronomical scale first, where a system's sphere of influence dwarfs a
    surface structure's reach and the required climb is expected to fall to 1 — **an expectation, to
    be settled by running `measure_visibility_climb` at that scale, not asserted here**;
(c) leave the boot fence to refuse, loudly, until (a) or (b).
This design ships (c) as the default, because a refusal is a measurement and a wrong pixel is not.

## 3.4 WHO DECIDES — including the vacated star system

### 3.4.1 The forced-disclosure result (first-principles' contribution, premises re-verified)

Compose four facts:
1. THE DRAW LAW: a realm draws itself only while running.
2. A realm is demanded only by its own parent (`push_demand` refuses anything else,
   `stub.rs:8188-8199`).
3. A realm knows neither its own position nor any occupant pose (SL1, SL2).
4. The owner's new law: a realm draws itself whenever visibility reaches it, at any depth.

Then for a planet to draw to a galaxy-standing observer, the star system must demand it; the star
system holds nobody; so the star system's demand must depend on information that originates outside
it. **Therefore "something outside may be looking in" must become representable inside a realm.**
There is no local formulation. This is why Ask B is put to the owner rather than engineered away.

### 3.4.2 The two decisions, both realm-side, both shared

| decision | decided by | from what | cadence |
|---|---|---|---|
| **coarse** — is this direct child's interior worth waking? | the realm holding the observer | its own authored placement for that child, and that child's own interior reach (boot-derived, no message) | the existing AoI beat |
| **fine** — which of my own children are in band? | that child itself | its existing decision loop, with one synthetic observer added | its existing AoI cadence |

Neither is per-observer. Neither is per-tick-per-row. The owner's ruling is honoured literally: the
per-row verdict stays with the realm that owns the rows, one set shared by every observer under that
scope, computed on the slow cadence. Nothing per-observer enters the gateway.

### 3.4.3 The down-proxy (SL7 read in the other direction)

> **An interested realm is its outside observers' proxy, at its own scale.** A realm that holds a
> live interest byte inserts ONE synthetic observer at its own centre, with reach equal to its own
> extent, into the observer list it already builds (`stub.rs:7857-7870`), and runs its existing fold
> unchanged.

This lands three lines from the existing upward proxy, in the same list, with the same hysteresis,
the same grace, and the same demand path. One machinery. No realm-kind branch anywhere.

**The error bound is SL7's own.** An occupied child stands in for its occupants because the child is
small relative to its parent. An interested realm stands in for the observers outside it, with the
error bounded by its own extent, for the identical reason.

**The cascade cap is structural, not a counter.** An observer entry carries a flag set at
construction: it is *occupancy-derived* (a real dot, a held transient, or an occupied-child proxy)
or *interest-derived* (the synthetic one). **Only occupancy-derived observers produce interest for
the next level down.** So an observer's realm wakes its children's interiors and stops. Live realms
below the observer: exactly two. No depth number and no hop count crosses any boundary, which is
what SL7 requires literally.

### 3.4.4 The interior band, and why there is no pop at the boundary

The coarse test uses the child's **interior reach**: the largest distance from the child's centre at
which something inside it is still visible. It is the maximum over the child's own direct children of
(that grandchild's worst excursion + that grandchild's visibility reach). On THE world:
`142.045826247 + 302.058663384 = 444.104489631 m` for a star system, against the system's own
band of `11 458.474598711 m`.

The tear-down radius adds the derived lead, giving about `469.104 m`. **Both bracket the 150 m
system shell.** The planets therefore start before the player can get far enough out to look back
at them. A latency race becomes a geometric impossibility.

**CLAIM (unverified in this pass):** the generator can stamp each region's interior reach at boot,
because it holds the whole forest before scoping. **The measurement that settles it:** a unit
asserting that a galaxy shard's boot roster row for a star system carries interior reach
`444.104489631` exactly. If that measurement fails, Ask D moves from deferred to required.

### 3.4.5 The membership question needs NO new data

`scene_bag` attaches a picture only to a member, and a grandchild is in nobody's membership set. The
obvious fix — ship the child's verdict upward — is unnecessary. **The forward gate IS the membership
gate**: a realm forwards a held child's batch only when that child is in its own in-band verdict. So
the arrival of a grandchild's picture is already conditioned on the middle realm's membership
decision. The gateway records which subjects arrived through that path and adds them as a third
disjunct in the member test. This is derived from bytes the gateway already opened. **One fewer
boundary crossing than any alternative**, and it is THE DRAW LAW's own enforcement style: absence of
data is the gate.

## 3.5 COMPOSER CHANGES

| # | site | change | why |
|---|---|---|---|
| C1 | `window.rs:449-460`, `:808-814` | `ComposedRow` carries `parent: Option<RealmId>` explicitly; `scene_parent` is deleted | **Live defect, verified this pass.** `scene_parent` derives a row's hierarchy parent from `row.stratum`, the CHAIN index, so every relayed interior row today ships the GRANDparent as its parent. The wire field already exists, so zero wire change and zero client change. |
| C2 | `window.rs:775-783` `body_tag` | **no change** | already depth-blind — the instant a deep picture lands in any chain ingest, presence lights up. This is why the fix is small. |
| C3 | `window.rs:793-802` `scene_bag` | one added disjunct: subjects admitted through an interior forward | §3.4.5 |
| C4 | `window.rs:132`, `:617-634` | `relay_levels` becomes a per-child RING at the existing span; the relay fold resolves at the tick the chain already composed at, falling back to the newest stamp at-or-before it | §3.7 |
| C5 | `window.rs:624-634` | split `relay_unplaceable` into `relay_descent_refused` / `relay_stamp_missing` / `relay_unrostered`; add `relay_stamp_skew_ticks` (max) and `relay_depth_max` | the shipped counter aggregates two arms and one of them is structurally dead, which is why the ledgered question is unanswerable |
| C6 | `gateway.rs:2145` | prune relayed state against the same head the ring trims on, not the gateway's own clock | a gateway a few ticks behind evicts a stamp before the prune drops it |
| C7 | `gateway.rs:4033-4080` `admit_relay` | iterate `interior`; vouch, fence, admit only the author's own picture | §3.2 |
| C8 | `window.rs:703-727` `descent_at` | **no change, no recursion** | §3.2 |
| C9 | client | **no change** | the parent field already exists on the wire |

### Shared composition survives — the proof

The fold is memoised on (origin realm, tick) (`gateway.rs:2170`) and takes no observer argument. It
still takes none. Every new store is per-window ingest state, shared by every session whose chain
contains that window. Nothing added is per-session.

**And the row COUNT does not grow.** Relayed interior rows already compose today
(`window.rs:610-694`); only their tag changes from marker to picture. That is the whole reason this
design's fold cost delta is zero rather than merely small — and it is the reason local-bound's
depth-3 admission was rejected: rows per fold is a per-**player** cost, because the memo hit clones
the composed result per session (`gateway.rs:2241`) and then runs a per-session shadow advance and
delta. §5 quantifies that.

## 3.6 FAILURE BEHAVIOUR

| failure | behaviour | law kept by |
|---|---|---|
| a hop dies (the middle realm's shard is killed) | the held entry expires on the derived retain TTL; the forward stops; the gateway prunes pictures on the picture TTL; **markers are never pruned** | body → marker, never body → nothing — **but only once Ask C lands**; without it a non-glowing subject vanishes (verified defect, §2 Ask C) |
| a realm re-homes mid-forward | the grandchild's OWN fence rides outside the seal, so a deposed incarnation's picture is rejected and counted; the relaying child's attestation is unchanged | this is the graft that fixes the fence hole two of the three designs had |
| a picture is lost | reliable, redriven, re-asserted on the beat; worst case one beat as a marker | bounded, counted |
| a level datagram is lost | at-or-before resolution costs one stamp of declared skew instead of a whole beat of refusals | §3.7 |
| the interest byte is lost | reliable and re-asserted every beat under a TTL of two beats plus one; two consecutive losses expire it, then teardown cooldown and grace still apply | one lost message never tears anything down |
| the interest byte never arrives (unattested head) | no interest ⇒ no demand ⇒ no shard ⇒ no picture ⇒ the marker | absence of data is the gate |
| a hostile or malformed interior batch | unvouched ⇒ refused and counted; a third level ⇒ **not representable** | fail-closed |

## 3.7 D-WINDOW-6(2) — ROOT CAUSE AND DISPOSITION: FIXED

**The zero is a topology fact, not a fold failure.** In the two-ships fixture the only live pair is
at chain index 0 with the planet as the relayed child. The planet is a leaf — the generator emits
universe, galaxy, systems and planets and stops (verified) — so its sealed level carries no rows,
and the counter counts rows, not children. Zero is arithmetically forced.

**The 50-87 refusals are all one arm, and it is not the arm the ledger names.** At chain index 0 the
descent loop is empty and returns unconditionally, so the descent arm is structurally dead there.
Every refusal is the placement lookup returning nothing: the relay path demands **exact tick
equality between stamps authored by two different follower clocks**, while the main chain solves the
identical problem with a tolerant prefix. A follower's clock is the last observed sync value and
does not free-run, so a coalesced sync makes a follower skip a stamp permanently.

**The fix: the ring plus resolve-at-T with at-or-before fallback (C4-C6).** Rejected alternatives,
with reasons:

- *Make the follower clock free-run between syncs* (verdict-pull's Slice 0). **Rejected on a verified
  fact:** the sync application structurally refuses backward slew — it returns a rejection outcome
  rather than regressing (`crates/node/src/universe_clock.rs:196-208`, read this pass), and `now()`
  simply returns the held value. A free-running follower that overshoots can never be corrected down.
  It would stamp ticks the orchestrator has not published, permanently, in the determinism seam.

**Why at-or-before costs ZERO positional error — structural, not an argument.** Verified this pass:
the relay ship trigger is a fingerprint over `(level rows, bodies)` — the row **values**, not their
identities (`stub.rs:7568-7572`). Therefore a stamp older than T can only be resolved when **no
fingerprint change occurred in between**, which means the rows are byte-identical to the newest rows.
A stale stamp with different rows cannot exist, because different rows would have shipped. The one
residual case is an in-flight loss, which is bounded by the reliable redrive and by the re-assert
beat.

Also fixed: the middle realm's placement (which frame the interior sits in) comes from the parent's
own level at T — exact, unaffected. Only the interior placements inside the middle realm's frame use
the child's stamp.

**The fixture must be repointed.** The two-ships relayed child is a leaf and can never exercise the
fold. The gates move to a topology whose relayed child has children.

**Disposition: D-WINDOW-6(2) is discharged by slice 0**, with the counter split first so the gate is
a real measurement (`relay_stamp_missing` goes to zero and stays there;
`window_relay_rows_composed > 0` in the non-leaf topology).

---

# 4. LAW-COMPLIANCE TABLE

| law | how it is satisfied STRUCTURALLY |
|---|---|
| **HR1 sealed shards** | The forwarder moves an opaque byte vector. Production code in `vd-sim` calls no open function; the only production caller is the gateway, which is not a realm. Authorship is enforced by the **absence of a function in a crate's vocabulary**, and slice 3 adds a structural test that pins it. |
| **HR2 generic transfer** | Untouched. No transfer machinery changes. |
| **HR3 one machinery** | One visibility formula in `vd-core`, three consumers. One decision loop — the down-proxy joins the existing observer list beside the up-proxy. One demand path. No realm-kind match anywhere on any path introduced here. |
| **HR4 features once, run anywhere** | The pixel gate runs on a star-system/planet pair **and** on a planted station/area pair (`G-IDENTICAL`). Same code, two realm kinds, identical assertions. |
| **HR5 100% Tier-A coverage** | All new branching lives in monomorphic helpers (the climb comparison, the admission predicate, the interest fold), following the canonical branchless-shim shape. Every refusal arm is driven by a named example, including the unvouched, misrouted, stale-fence and TTL-expiry arms. |
| **HR6 agent-operable E2E** | The flown-symptom gate is a headless GPU pixel test driven by the existing control harness, with run manifests. |
| **SL1 only the parent knows positions** | The interest byte has no field a position could ride in; the closed-wire absence test extends to the new arm and passes by construction of the type. The interior forward carries no placement a realm can read. No realm ever learns its own pose. |
| **SL2 no occupant pose crosses** | The byte is one bit of information — the exact mirror of the occupancy bit already licensed upward. It cannot be aggregated into anything about any specific observer. **Explicitly rejected:** the alternative continuous "nearest distance" scalar, because at one player the minimum over a set is that occupant's exact range and its time series is that occupant's closing speed. The tombstoned occupant-interest lane died for precisely that. |
| **SL3 a realm draws itself** | Only the realm authors its own picture. The forwarder cannot open or alter it. The one contested item (the marker's size) is **Ask C, default NO, its own conditional slice**. |
| **SL4 physics and re-home separate** | The interior reach reads a worst-instant excursion **in the generator at boot**, exactly where the existing interest configuration already reads a closing speed, and is consumed by the sim as a plain scalar. No orbit, gravity, thrust or drag symbol appears on any crossing path. Enforced by the existing crate dependency rule. |
| **SL5 ONE WORLD** | No variant, no preset, no scale knob, no second generator. The deep gates plant player-built regions on THE world, which SL5 licenses explicitly. |
| **SL6 ask before new data crosses** | Two asks (A, B), stated in the owner's required form with priced alternatives, plus one non-crossing ask (C) and one deferred (D). Exactly one new arm, declared plainly, not smuggled into an existing discriminant. |
| **SL7 an occupied realm is its own occupants' proxy** | Every realm still judges only its own direct children. Liveness stays self-and-one-level-down. Nothing reaches upward. The down-proxy is the same sentence with the roles swapped and the same error bound; its cascade cap is the observer-origin flag, not a counter. |
| **THE DRAW LAW** | The bag selection still returns exactly one of picture / marker / empty, chosen by presence of data, with no running check anywhere. The interest byte controls whether the shard exists; it never appears on the draw path. **The "never empty" half is FALSE today for non-glowing subjects and is fixed only by Ask C** — this is stated rather than assumed. |
| **No client prediction / pure renderer** | The client is unchanged. It never branches on realm kind. |
| **Positional wire discipline** | One rename in place, appended fields, one appended variant, one new arm. **No type substitution at an existing position and no field insertion in the middle of a struct** — local-bound's shape was rejected precisely because postcard is positional and both of its changes would silently misparse against an older peer instead of failing loudly. Minor bump, owner-citation build gate, tombstones untouched. |
| **No magic numbers** | The carrier arity is read from the boot measurement. The interior band is derived from a worst excursion plus a visibility reach, both already seed-derived. Every TTL comes from the existing cadence derivations. |
| **Never assume — measure it** | Every claim in this document is either a shipped constant, an arithmetic identity I recomputed, a labelled CLAIM with the measurement that settles it, or a labelled estimate. |

---

# 5. PERFORMANCE AND SCALABILITY

## 5.1 The multiplier the earlier drafts omitted

Relay egress is fanned **per open window**, and a window is deduplicated per (shard, scope) per
**gateway**. So a shard's outbound relay cost is `W × C × blob × rate`, where `W` is the number of
gateways holding a session under that realm and `C` is its held-child count. Every per-second figure
below carries `W`.

Reference model: 5 000 players, 50 gateways, 500 distinct origin realms, 50 Hz, chain height 3.

## 5.2 Bytes

A composed row is about **100 B** encoded (realm id, frame reference, and a stamped pose of cell
integers, offsets, velocity, orientation and tick). Note: two of the three input designs used
145 B, which double-counts an angular-velocity field the pose type does not have.

Per star-system-to-galaxy relay, five planets:

| part | today | after |
|---|---|---|
| own level, 5 rows | ~500 B | ~500 B |
| own picture + 5 markers | ~140 B | ~140 B (+~40 B with Ask C) |
| **five interior batches** | — | **~365 B** |
| **total per relay** | **~640 B** | **~1 045 B (+63%)** |

A system with orbiting planets re-ships every tick (its fingerprint changes every tick). So:

```
per-shard relay egress = W × C × blob × 50 Hz
  today  : 50 × 3 × 0.64 kB × 50 =  4.8 MB/s
  after  : 50 × 3 × 1.05 kB × 50 =  7.9 MB/s
```

The interest byte is negligible: about 40 B per in-band child at 2 Hz.

## 5.3 CPU

- **Parent, per beat:** one extra in-range test per direct child (about 10 ns each) plus one message
  per interested child. Effectively zero.
- **Child, per tick:** a vacated star system goes from an early return to five distance evaluations
  per tick — 250/s. Negligible.
- **Gateway, per fold:** row count unchanged, presence scan unchanged. One added set lookup per
  relayed row. At about 120 k relayed row-folds/s that is roughly **0.25 % of one core**,
  cluster-wide.
- **Gateway, per relay decode:** one open over a batch about 60 % larger, once per (window, child)
  change.

## 5.4 Two pre-existing costs this design must not make worse, and one it improves

- **The per-session clone is the real census wall.** On every memo hit the composed result is cloned
  per session, then a per-session shadow advance and delta run. At 5 000 players × 50 Hz × ~19 rows
  × ~160 B that is on the order of **760 MB/s of clone traffic cluster-wide today**, linear in rows
  per fold. **This design does not change rows per fold at all** — that is the quantitative reason
  the depth-3 admission was rejected (one city layer would take 19 rows to roughly 319, a 17×
  multiplier on a per-player cost). A gate pins rows per fold on the departure fixture.
- **The per-window blob compare.** The send-on-change baseline stores and memcmps the whole blob per
  (window, child) per tick — on the order of 10 MB/s of memcpy on one galaxy shard at W=50 today.
  **Improvement shipped here:** the baseline stores an 8-byte digest over `(fence, own, interior)`
  instead of the bytes. This both re-keys the baseline (without it a grandchild's picture change
  with the child's own statements unchanged would be **invisible to send-on-change** — a silent
  staleness bug on the load-bearing draw path, which two of the three input designs had) and removes
  the memcpy.
- **The relay ring's memory.** The ring holds up to the existing span of stamps per (window, child):
  about 51 × 5 rows × 160 B ≈ **40 kB per (window, child)**, roughly **6 MB per gateway** at W=50,
  C=3, chain height 3, plus a per-tick trim. Stated because the input design that proposed it did
  not price it.

## 5.5 What each cost scales with

- **NOT with players.** Everything added is authored once per realm per beat and consumed by a fold
  memoised per (origin, tick). No per-observer filtering enters the gateway anywhere. The owner's
  6 M-versus-120 k row-encode concern does not materialise.
- **NOT with world size.** A dormant realm runs nothing and is held by nobody. A shard's scope is
  its ancestors and its direct children regardless of galaxy size.
- **WITH live realms**, bounded per observer by `chain height + in-band children + interested
  children × their branching factor`, and **capped at two levels by the observer-origin flag**. On
  THE world that is 4 → 9 live realms per lone observer. Because the interior band is about 26×
  tighter in radius than the visibility band (444 m against 11 458 m), the interested-child count is
  0 or 1 in practice.
- **WITH gateways (W)** on the relay egress lane, linearly, as §5.2 shows.

## 5.6 The census worst case, honestly

The worst case is a dense cluster where a realm has many in-band children. Egress is capped by the
carrier arity — depth 3 is unrepresentable, so the exponent cannot grow. Rows per fold are
unchanged, so the per-session clone is unchanged. **The one unbounded-in-principle path in the
candidate set — a purely geometric climb whose height is set by player-built structure size — was
rejected for exactly this reason.** If the branching factor ever exceeds budget, the discharge is the
already-ledgered detail-tier seam: more tags in the same bag, a coarser tier for distant children,
selected by a rare and coarse decision at the gateway, which is where you ruled it belongs.

---

# 6. THE SLICE PLAN

Each slice lands green under `just gate`, at 100 % Tier-A region and branch coverage, with both arms
of every new predicate driven by named examples.

### SLICE 0 — COMPOSER TRUTH. No wire change. No new data. No owner ruling.
Carry the parent explicitly on a composed row and delete the chain-index derivation. Turn the
relayed level store into a ring; resolve at the composed tick with an at-or-before fallback; split
the refusal counters; align the prune clock; replace the send-on-change blob baseline with a digest
over fence, own and interior.
**Owner-visible outcome:** nothing changes on screen. Two live defects stop being wrong.
**Gates.**
- `G-PARENT-TRUE` — **declared RED before the slice**: in the existing departure test, every planet
  row's parent is the star system, not the galaxy.
- `G-RELAY-STAMP` (**discharges D-WINDOW-6(2)**): `relay_stamp_missing == 0` over a 60 s process
  flight in the non-leaf topology; `window_relay_rows_composed > 0`; `relay_descent_refused` driven
  by a named unit at chain index ≥ 1 so neither arm is dead; `relay_stamp_skew_ticks ≤ one beat`.
- Determinism suite byte-identical.

### SLICE 1 — THE PRESENCE FLOOR. **Conditional on Q2.** No boundary crossing.
Append the extent tag to the body bag. Emit a point-of-light marker for **every** direct child, not
only glowing ones. The client sizes a marker by the larger of its brightness radius and its angular
radius, against the existing pixel floor.
**Owner-visible outcome:** points of light grow smoothly with distance instead of sitting at a
three-pixel floor, and non-glowing things stop vanishing.
**Gates.**
- Pixel: a planet's point of light grows monotonically and strictly as the camera closes from 11 km
  to 200 m; it never pops and never blanks.
- A station whose picture is allowed to lapse degrades to a correctly-sized marker and **never to an
  empty frame** — this gate is RED today.

### SLICE 2 — THE BOUND, MEASURED. No wire change. No owner ruling.
One visibility formula in `vd-core` with three consumers. Replace the boolean guard with
`measure_visibility_climb` plus `guard_visibility_climb_bounded(arity)`. Add
`guard_candidate_climb_bounded` on the build-admission path. Wire the fence into **every** process's
boot, not only the shard's. Keep every frozen constant and the shell solve untouched.
**Owner-visible outcome:** the game prints how deep pictures must travel, and refuses to start rather
than draw something wrongly.
**Gates.**
- `G-CLIMB` — on THE world: max climb **== 2**, planet stopping slack **== 4.000000 m exactly**,
  cross-checked against the containment margin constant; the two-level clearance constant still
  equals its frozen value, proving the live formula and the world solve are one equation.
- A hand-built forest with a zero-margin level proves the monotone-slack arm.
- A planted 20 m surface structure at today's scale makes the boot **refuse**, printing the body and
  its numbers — the Q3 evidence, produced by a test rather than by an argument.

### SLICE 3 — THE SEALED INTERIOR FORWARD. **Ask A.** Dark until slice 4.
Append the interior field to both arms. Split the held entry. The forward reads only the child's own
half. Gateway vouches, fences, and admits only the author's own picture. Record the admitted set and
add the third disjunct to the bag's member test.
**Owner-visible outcome:** none yet by itself.
**Gates.**
- `G-VERBATIM` — the bytes a grandparent forwards are **byte-identical** to what the author sealed.
  A measurement that could fail.
- `G-STRUCTURAL-SEAL` — a test pinning that `vd-sim` contains no call to any open function. This is
  the entire load-bearing guarantee of HR1 across two hops and it becomes a test, not a claim.
- A hostile batch naming an unrostered grandchild is refused and counted; the violation counter is
  asserted 0 on a lawful flight while the lawful-filter counter is asserted **non-zero** (these are
  two different counters, on purpose).

### SLICE 4 — THE INTEREST BIT. **Ask B. REQUIRES Q1.**
The new arm; emit on the AoI beat from the observer's realm; fail-closed admission on the child;
TTL; the down-proxy observer; the observer-origin flag that caps the cascade.
**Owner-visible outcome:** the planets are running when you look at them.
**Gates.**
- With a galaxy-standing observer at a stated distance, the vacated star system spins up exactly the
  planets satisfying the derived interior test **asserted with the numbers, not the count**; at the
  band edge it spins up none.
- `G-NO-CASCADE` — planet shards run and **no area shard ever boots**; the cascade cap is asserted
  structurally (interest is derived only from occupancy-derived observers).
- `G-INTEREST-BAND` — the derived spin-up and tear-down radii equal `444.104489631` and about
  `469.104` on THE world, and both **bracket the 150 m system shell**, so interiors are awake before
  the crossing.
- Fail-closed units: misroute, unattested sender, stale fence, TTL expiry.
- The closed-wire absence test extends to the new arm and passes by construction of the type.

### SLICE 5 — THE FLOWN-SYMPTOM PIXEL GATE.
New process-tier test and `just` target, added to `just gate`. One demand cluster, one headless GPU
client. Park outside the 150 m system shell and inside the interior band, along the outer planet's
instantaneous radius vector, reusing the existing legal-park derivation.
**Owner-visible outcome:** the flight you flew, fixed, asserted in pixels.
**Gates.**
- Every planet's presence is its **own picture**, not a parent marker — RED today for all five.
- The outer planet's drawn radius is far above the three-pixel floor: at a 200 m park the derived
  expectation is tens of pixels against today's exactly 3.0. The assertion is derived from the
  camera model, never a literal.
- All five planets present with **distinct** radii — not five identical dots.
- Fly outward across the stop level: the marker handover happens once, with no blank frame and, with
  slice 1 landed, no radius step above the readback quantum. Fly back: the reverse handover within a
  derived budget that **states the extra relay hop as its own term**.
- Every planet row's parent is the star system throughout.
- `G-IDENTICAL` (HR4): the same fixture and the same assertions on a planted station/area pair.
- **`G-NOTHING-OWED`** (the law gate): from the same camera pose, enumerate the generated forest,
  compute each subject's true angular size out-of-band, and assert that **every** subject above the
  minimum angle has a drawn row carrying its own picture. This is the only gate that can fail for the
  reason the owner's law exists, at any depth, independent of the carrier.

### SLICE 6 — SCALE AND LOAD.
Pin rows per fold on the departure fixture. Measure relay egress per shard **with the gateway
multiplier** against the §5.2 formula and pin it. Measure the union over-draw (rows drawn for one
observer because another is closer) and record it rather than assuming it is small.
**Gates:** the pinned numbers, and `just coverage-fast` at 100 % Tier-A.

### LEDGER ON LANDING
- **D-LOOK-1 🟥** — the candidate build-admission predicate exists but its **policy at interim
  scale** is Q3. Blocking for P6/P8.
- **D-LOOK-2 🟥** — the union coarsening accepted by shared verdicts is **unmeasured at near-real
  scale**; slice 6 measures it at interim scale only.
- **D-LOOK-3 🟥** — the live child-to-parent interior reach (Ask D), owed when player building lands.
- **D-WINDOW-6(2) 🟩** — discharged by slice 0.
- **D-WINDOW-3** — the census-scale roster bound now also owes the interior forward; §5 gives today's
  numbers, the census derivation is still owed.
- **D-WINDOW-5** — the detail-tier seam is unchanged and better placed: the interior forward is its
  natural carrier.

---

# 7. OPEN QUESTIONS FOR THE OWNER

### Q1 — May a realm be told that something outside it may be looking in?

**The datum:** one byte, two lawful values, on a new parent-to-direct-child message. No identity, no
position, no direction, no distance, no count. It decays to "nobody is watching" if the lane goes
quiet.
**Why you must decide:** it ends a landed ruling's rationale clause ("am I observed from outside
stays unrepresentable in every realm"). I am not smuggling it.
**Why there is no alternative:** §3.4.1. A realm draws only while running; a realm is woken only by
its parent; a realm is blind to its surroundings. A vacated star system therefore cannot wake its
planets from anything it legitimately holds. The alternatives are: wake every interior always
(about 1.8 × 10⁴ times the volume of pointless start-ups), open a gateway window directly on the deep
realm (which you already refused and which discloses strictly more), or leave the defect.
**If you say no:** slices 0-3 still land and still improve the picture, but the planets stay dots.

### Q2 — May a parent state its child's size?

**The datum:** one number the parent already holds for containment, appended to the point-of-light
bag. It crosses no realm boundary.
**Why you must decide:** SL3 says the realm authors how it looks, and a size is arguably part of that.
**What saying no costs, measured:** the point-of-light bag today can carry only brightness, so a
non-glowing thing gets **no marker at all** and disappears when its own picture lapses. Verified in
code this pass. Saying no also leaves a roughly 3.8× jump at the marker-to-body handover.
**My recommendation:** yes, narrowly — the bounding number only, never a surface, a detail or a mesh,
and the realm's own picture supersedes it entirely the instant the realm runs.

### Q3 — At today's compressed scale, built content needs three hops. Which response?

**The fact:** the boot measurement, run over a world with a 20 m structure planted on a planet
surface, reports a required climb of **3**, because at the interim scale a star system's shell is only
150 m and a structure's visibility reach is about 1 528 m.
**The options:** (a) raise the carrier to three levels now — a reviewed edit in one file; (b) land
the near-real astronomical scale first, where the required climb is **expected** to fall (an
expectation, settled by running the measurement at that scale); (c) let the boot refuse until (a) or
(b).
**Default shipped:** (c). A refusal is a measurement; a wrong pixel is not.

### Q4 — May a child later tell its parent one number about its own interior?

Deferred, not requested. For the seed-generated world the number is derived at boot and nothing
crosses. The ask becomes real when players build.

---

# 8. THE JUDGE HOLES REGISTER

Every hole raised by the three judges, with its resolution. "Moot" means the winning synthesis does
not contain the mechanism the hole was about.

## From the law judge

| # | hole | resolution |
|---|---|---|
| H1 | local-bound never wakes the planets, so its pixel gate cannot pass | **Resolved by exclusion + replacement.** Local-bound is not the spine. The wake lock is opened explicitly (§3.4) and its own gate (`G-NO-CASCADE`, slice 4) asserts the spin-ups by number. |
| H2 | the termination oracle is arithmetically false for any built content | **Resolved (§3.3).** The boolean guard becomes a measurement plus a carrier arity. The guard's blindness to player content is verified this pass and surfaced as **Q3**, not buried. |
| H3 | local-bound's unit loses its fence; the zombie guard is defeated | **Resolved by graft.** The carrier carries the author's own fence **outside** the seal, feeding the existing per-realm fence map unchanged. |
| H4 | first-principles' card has the same defect | **Resolved by the same graft.** The card shape is dropped in favour of the fenced carrier. |
| H5 | local-bound's bound rests on every forwarder behaving | **Resolved by exclusion.** No forwarder rewrites anything. The bytes go out exactly as received, and the depth bound is a missing field, not a computed scalar. |
| H6 | verdict-pull's continuous distance scalar is an SL2 breach at one player | **Resolved by exclusion.** The downward datum is one byte, not a range. Explicitly recorded in §4. |
| H7 | local-bound's wire change is not append-safe | **Resolved by exclusion.** This design does one rename in place, appended fields, one appended variant, one new arm. No type substitution at an existing position, no mid-struct insertion. |
| H8 | first-principles calls a new parent-to-child control lane "no new arm" | **Resolved.** The interest byte is declared as its **own arm**, with its own admission, attestation, fence ordering and TTL. §2 Ask B says so in the first line. |
| H9 | local-bound widens SL7 without an ask | **Moot.** Subtree-closed membership is not used. Membership is the forward gate, which needs no new authority (§3.4.5). |
| H10 | local-bound decides the SL3 size question silently | **Resolved.** It is **Ask C / Q2**, default NO, its own conditional slice. |
| H11 | first-principles makes hop depth a routing input | **Resolved by graft.** The carrier's arity replaces the depth subtraction. No depth arithmetic on any routing path; no depth or hop count crosses anywhere. |
| H12 | verdict-pull creates two machineries deciding liveness | **Resolved.** The interest byte does **not** decide liveness. It adds one observer to the existing decision loop, which then uses the **existing** demand path. One machinery decides whether a realm runs. |
| H13 | a measurement-scoped number used as a universal proof | **Resolved (§3.3.1).** The guard's domain limit is stated, verified, and turned into Q3. |
| H14 | an unsettled authorship question at the first hop | **Moot.** No accumulated scalar exists on the wire. |
| H15 | the marker-versus-picture size choice lands in the client | **Partially resolved, ledgered.** The client still takes the larger of two radii against the pixel floor. Making the server compose a ready-to-draw radius is the correct end state and is registered under the existing streaming-contract work; it is not on the critical path for this defect. |
| H16 | `scene_parent` derives hierarchy from the chain index | **Resolved. Lifted to slice 0** with a red-before-green gate, independent of every pending ruling. Verified live this pass. |

## From the performance judge

| # | hole | resolution |
|---|---|---|
| LB-1..LB-8 | local-bound: no wake; unbounded climb; growth in rows per fold; row size inflated; unpriced blob loop; missing gateway multiplier; unbounded skew | **Moot by exclusion**, except: the row size is corrected to ~100 B (§5.2); the gateway multiplier is in every figure (§5.1-5.2); the blob loop is **improved**, not just re-keyed (§5.4); the skew is shown to cost zero positional error structurally (§3.7). Local-bound's genuine contributions — the boot measurement, the slack identity, and the "hops ≥ 2 need no cross-clock join" observation — are all grafted. |
| VP-1 | the termination proof is vacuous for built content and the hole is unnamed | **Resolved.** Named, verified, and escalated to Q3. |
| VP-2 | every bytes/s figure omits the gateway fan | **Resolved (§5.1-5.2).** |
| VP-3 | the interior half would be invisible to send-on-change | **Resolved (§5.4).** The baseline becomes a digest over fence, own and interior. This was a silent staleness bug on the load-bearing draw path in two of the three designs. |
| VP-4 | the follower free-run fix has the widest blast radius | **Rejected on a verified fact** (§3.7): sync application structurally refuses backward slew, so a free-running follower can never be corrected down. The ring plus resolve-at-T is used instead, contained to the composer. |
| VP-5 | a new downward shard-to-shard lane | **Accepted and declared.** Bytes are trivial; the route already exists. |
| VP-6 | quotes the looser margin | **Resolved.** 4.000 m is used and the distinction is stated (§3.3.2). |
| FP-1 | the headline cost is wrong by ~25× | **Resolved.** The relay message re-ships every tick for a realm with orbiting children, so nothing amortises. §5.2 states +63 % on the link and the per-shard figure with the gateway multiplier. |
| FP-2 | the card field would be invisible to send-on-change | **Resolved** — same fix as VP-3. |
| FP-3 | no visibility gate on the forward means wasted forwards at real scale | **Accepted, ledgered, not fixed now.** The waste is bounded and small at today's scale. The correct discharge is a visibility filter **inside** the arity cap, never as the cap; it is registered with the detail-tier seam and is a measurement away (`measure_visibility_climb` at near-real scale tells us how much is wasted). |
| FP-4 | the relay ring is an unpriced memory cost | **Resolved (§5.4):** ~40 kB per (window, child), ~6 MB per gateway at the reference model. |
| FP-5 | missing gateway multiplier | **Resolved (§5.1).** |
| FP-6 | a second per-row lookup on the hot path | **Resolved.** One added set lookup per relayed row, priced at about 0.25 % of one core (§5.3). |
| SH-1 | the per-session fold clone is the real census wall and no design models it | **Resolved (§5.4).** Modelled, quantified, pinned by a gate in slice 6, and used as the quantitative argument against depth-3 admission. |
| SH-2 | the per-window blob clone and memcmp | **Improved (§5.4):** an 8-byte digest replaces both. |
| SH-3 | the fence input is the generated forest, full stop | **Resolved.** Candidate predicate on the build-admission path lands in slice 2; the policy question is Q3. |
| SH-4 | mis-parenting today | **Resolved — slice 0.** |

## From the correctness / player-experience judge

| # | hole | resolution |
|---|---|---|
| H-LB-1..6 | local-bound cannot fire; wrong failure mode; false never-nothing claim; unbounded skew; vacuous gate; silent SL7 widening | **Moot by exclusion**, except the never-nothing claim, which is **general** and is handled below (H-X-1). |
| H-VP-1 | the follower free-run collides with the backward-slew clamp | **Rejected, verified** (§3.7). |
| H-VP-2 | the gate contradicts the admission rule | **Resolved.** Two distinct counters: a lawful-filter counter (expected non-zero, never asserted zero) and a violation counter (asserted zero). |
| H-VP-3 | the termination proof rests on a fence that never judged the fixture | **Resolved** — §3.3.1 and Q3. |
| H-VP-4 | the mid-crossing wake exception is unmeasured | **Resolved.** The cascade cap here is the observer-origin flag, not a distance premise, so the exception does not arise: a synthetic observer never produces interest. `G-NO-CASCADE` asserts it. |
| H-VP-5 | the SL1 argument for a continuous scalar is overstated | **Moot.** No continuous scalar crosses. |
| H-FP-1 | the presence floor is conditional, so a city can still vanish | **Resolved as far as it can be.** Escalated from a size-pop refinement to **Q2, a named owner question**, with the verified cause: the bag codec has no lawful content for a non-glowing child. The failure table in §3.6 states the dependency instead of asserting the law. |
| H-FP-2 | the horizon gate is self-fulfilling | **Resolved.** That gate is demoted to what it is — an implementation check — and the real law gate is `G-NOTHING-OWED`, which uses the generator out-of-band and can fail for the right reason at any depth. |
| H-FP-3 | the depth constant is asserted, not measured, with no headroom instrument | **Resolved.** `measure_visibility_climb` reports the climb **and the slack**, so the world cannot drift to the edge silently. `G-CLIMB` pins max climb 2 and slack 4.000000 m. |
| H-FP-4 | compose-at-T does not cure the stamp miss for a static child | **Resolved structurally (§3.7).** The re-ship trigger is a fingerprint over the row **values**, so a stale stamp implies byte-identical rows, which implies zero positional error. Verified this pass. The residual is an in-flight loss, bounded by the redrive and counted by the skew gauge. |
| H-FP-5 | the structural seal pin is prose, not a gate | **Resolved.** `G-STRUCTURAL-SEAL` in slice 3. |
| H-FP-6 | the dual-sourced interior reach is a drift surface | **Resolved by deferral.** Only the boot-derived value ships. The live field is Ask D / D-LOOK-3, with a disagreement counter owed when it lands. |
| H-X-1 | every failure table depends on a marker that may not exist | **Resolved as a named question (Q2)** and as an explicit caveat in §3.6. **Slice 1 lands before the picture lane widens**, so no non-glowing subject reaches depth without the floor. |
| H-X-2 | no design gates the owner's actual sentence at arbitrary depth | **Resolved.** `G-NOTHING-OWED` is exactly that instrument, and Q3 is the honest consequence of running it at today's scale. |
| H-X-3 | the mis-parenting defect is live and unledgered | **Resolved — slice 0, red-first gate.** |
| H-X-4 | nobody measures the union coarsening | **Resolved as far as honesty allows.** Slice 6 measures it at interim scale; **D-LOOK-2** records that it stays unmeasured at near-real scale. |

---

# 9. WHAT THIS DESIGN DOES NOT DO — stated so nothing becomes a silent claim

- It does not draw a subject three levels below the observer's realm. It cannot, and it says so at
  boot rather than at a pixel.
- It does not prove that two levels is enough forever. It **measures** the required depth per world
  and refuses to run when the measurement exceeds what the message can carry.
- It does not add a window scope, a gateway-to-distant-realm window, or any per-observer filter.
- It does not carry any occupant pose, entity set, verdict identity set, depth or hop count across a
  realm boundary.
- It does not make the server compose a ready-to-draw radius; the client still takes the larger of
  two radii. That is a known step away from full server authorship and is ledgered, not hidden.
- It does not measure the union coarsening at near-real scale.

---

# RULINGS ADDENDUM (owner, 2026-08-17)

All four open questions were answered by the owner on 2026-08-17. This addendum is the signed
record; the design text above is verbatim as put to the owner and is not edited.

- **Q1 — APPROVED.** The interest byte (Ask B) may cross: one byte, two lawful values, on its own
  new parent-to-direct-child arm, exactly as §2 Ask B states it. Implemented in LATER slices
  (slice 4), not in slices 0–2.
- **Q2 — APPROVED, with the one-radius law.** The parent may state its child's circumscribed
  extent in the point-of-light bag. The law line, to be written into the law text where the design
  says it: **"a bound is a promise about space; a look is a statement about appearance"** — the
  parent's bag may never carry more than the single radius; the realm's own picture supersedes it
  the instant it runs.
- **Q3 — RULED = (b)-first-then-measure.** The carrier arity STAYS 2. The build-admission
  predicate refuses at interim scale (the design's default (c)). The near-real-scale world
  re-solve is the scheduled cure, and its first gate run must include `measure_visibility_climb`;
  the carrier goes to arity 3 only if that measurement demands it. Recorded in the ledger row the
  design names (D-LOOK-1).
- **Q4 — DEFERRED.** Not asked now; registered as D-LOOK-3. The ask becomes real when player
  building lands.

---

# LANDED ADDENDUM (2026-08-17) — §6 IN LANDED TENSE

The design text above stays verbatim as put to the owner (the RULINGS ADDENDUM's clause); THIS
is the landed-tense record of §6. **All seven slices are LANDED and green.**

- **SLICE 0 — LANDED.** G-PARENT-TRUE red-then-green (the chain-index parent deleted; the row
  carries its parent); G-RELAY-STAMP green (`relay_stamp_missing` at the derived churn bound —
  `<= window_open_sent`, reformulated after a measured first-service transient; skew ≤ one
  beat; depth 2). D-WINDOW-6(2) discharged 🟩.
- **SLICE 1 — LANDED (Q2 approved, the one-radius law).** The extent tag; a marker for EVERY
  direct child; G-LOOK-GROWTH green (3.00 px at ~11.45 km growing strictly to ~20–23 px at
  ~170–190 m, no pop, no blank); the station-lapse law green.
- **SLICE 2 — LANDED.** One visibility formula in `vd-core`; `measure_visibility_climb` +
  the arity fences in BOTH world-deriving process boots; G-CLIMB green (max climb 2; planet
  stopping slack 4.000000 m == the containment margin — the one-equation identity); the Q3
  evidence produced by a test (a 20 m surface structure measures climb 3 and is REFUSED).
- **SLICE 3 — LANDED (Ask A, mesh minor 20).** The sealed interior forward; the depth bound is
  the TYPE; G-VERBATIM byte-identity across both hops; G-STRUCTURAL-SEAL as a test; the two
  counters split (violation asserted 0, lawful filter measured non-zero — tens of thousands on
  a real flight).
- **SLICE 4 — LANDED (Ask B, Q1 approved, mesh minor 21).** The interest byte on its own arm;
  fail-closed admission; the down-proxy; the observer-origin cascade cap. G-INTEREST-BAND green
  (444.104489631 / 469.104489631 m bracketing the 150 m shell); G-NO-CASCADE green (structural
  unit + the process running-set equality); the wake proven end-to-end in real processes.
- **SLICE 5 — LANDED.** THE FLOWN-SYMPTOM PIXEL GATE: `crates/bins/tests/look_pixels.rs`,
  `just look-pixels`, wired into `just gate`. One demand cluster on THE world PLUS the planted
  player-built station/area pair (the SL5 fixture-forest doctrine's process path —
  `vd_physics::worldgen::FixturePlant::StationArea`, ONE derivation
  (`station_area_plant`), booted by every process via `vd_bins::process_world_config` /
  `VD_FIXTURE_PLANT`, judged by the same boot fences: both plants measure climb 2 and are
  admission-accepted at the landed arity). Measured green, twice consecutively: every planet's
  presence is its OWN picture at the derived park (297.05 m — outside the 150 m shell, inside
  the 444.104489631 m interior band, along the outer planet's instantaneous radius vector);
  the outer planet 20.48 px at 169.5 m (camera-model-derived, far above the 3 px floor); all
  five planets DISTINCT (7 model-separated pairs, ordered); every planet row's parent the star
  system THROUGHOUT; the marker handover EXACTLY once each way, no blank frame, no radius step
  above the readback quantum; the reverse handover 26–36 ticks against a derived budget of
  123 + resolution whose EXTRA RELAY HOP is stated as its own term. G-IDENTICAL (HR4): the
  planted STATION passes the identical assertions in the same scene, and the planted AREA one
  level deeper (its own picture at the camera-model size; reverse handover 19–20 ticks against
  58 + resolution). **A deviation, stated:** the pair is planted in the walk-forest shape —
  the station under the SYSTEM and the area under a PLANET, not as parent/child of each other —
  because the frame law (`frame_for_realm`) requires an Area's parent to be a Planet, and
  because under any but the inner planet the area's measured climb is 3 (refused, the Q3
  posture). G-NOTHING-OWED (THE LAW GATE): the generator + plant spec enumerated OUT-OF-BAND
  at both parks; every subject above the minimum angle draws its OWN picture; the oracle sets
  are non-empty with depth-2 members (park A: 4 owed, 3 at depth 2; park B: 7 owed, the area
  at depth 2; returned: 7 owed, 6 at depth 2). A realm CONTAINING the eye is not a subject
  (the owner's ruling that a containment boundary is never drawn as an object).
- **SLICE 6 — LANDED.** ROWS PER FOLD pinned on the departure fixture as a derived-SET
  equality (8 rows on THE plain world: the in-band child + its 5-planet interior + 2 sibling
  markers; containing realms draw nothing) — the §5.4 promise held. RELAY EGRESS measured
  against §5.2's `W × C × blob × rate` with the gateway multiplier MEASURED: W=2 exactly
  doubles W=1; C=2 adds exactly the second child's blob; the departure-fixture blob PINNED at
  **1118 B** (the §5.2 model said ~1045 B — the model under-counted by 7 %; at the §5.1
  reference model that is 8.4 MB/s per shard against §5.2's 7.9). THE UNION OVER-DRAW measured
  at interim scale: a maximally-separated observer pair's shared verdict is exactly the union
  of their disjoint singletons, and each observer over-draws the OTHER's child — **6 drawn
  realm rows per fold per observer** — recorded on D-LOOK-2 (the near-real-scale number stays
  owed there, by design).
