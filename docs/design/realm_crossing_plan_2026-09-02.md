# THE RULER SWITCH — a realm crosses between realms (plan, 2026-09-02)

Status: PROPOSED. Written after the owner's request (*"can we then work on supporting the ruler
switch"*) and after three code readings: the occupant crossing arc, every site that holds a child
realm's parent, and the binding rulings. Nothing here is built. The owner decides the asks in §7
before slice 1 starts.

Written in ASD-STE100. Every example uses the game's words: a realm, a hull, a star system, the
galaxy, a shard, a window, the gateway, a crossing.

---

## 1. The problem, measured

A hull is a realm. Its star system authors its placement on a millimetre ruler. The ruler ends at
two to the fifty-third metres, about a third of a parsec. The owner flew a hull to that end twice
(2026-09-02). The coordinates froze, one axis at a time, and the sky stopped sliding. The hull never
left the star system's authority. The directory still listed the hull under the star system, and no
crossing appeared in any log.

The cause is structural, not a bug in a formula:

- The containment scan re-homes OCCUPANTS. A child REALM is an object the scan measures against,
  never a subject it measures.
- The parent's forest, its child index and its berths are boot-only. No mutator exists.
- A hull's parent is a launch argument (`VD_OWN_COORD`), frozen in the shard's config and in the
  orchestrator's launch ledger.
- The gateway derives a session's lineage at login and patches it only when the SESSION crosses.
  A hull that moves house leaves every session aboard with a stale chain.
- The saga's shard-side arms return no-op for a `Realm` or `Ship` subject, by design, with the
  note *"when P8 lands it consumes `Ship` explicitly"*.

Example. The hull leaves the home system's shell at two hundred million million metres per second.
The star system keeps writing millimetres. At a third of a parsec the millimetres run out. The
galaxy, whose ruler counts in two-metre steps and reaches six hundred parsecs, never hears of the
hull.

## 2. What the rulings already decide

These constraints bind the design. Each traces to a clause in `CLAUDE.md`, the owner's decisions,
or the ledger.

1. **Two crossings, never one.** Out into the shared parent, then in again. A hull leaving a star
   system becomes the galaxy's child. Entering another system, it becomes that system's child.
   (SL2; the worked example in the 2026-08-26 movement ruling; D-45 C-6.)
2. **One machinery.** The same saga, the same fence, the same directory CAS as an occupant. No
   second FSM. (HR2, HR3; D-37 slice 4 says "generalize the re-home from Entity to Realm/Ship".)
3. **The crossing code may not name the hull's motion or kind.** A ship, a station, a moon and a
   rock cross by identical code. The only question is "am I the parent of this thing". (SL4; V4.)
4. **The parent decides, on its own swept verdict.** It is the only party that holds both
   placements. The point answer runs first, the sweep can only turn false into true. (M-D; D-S5.)
5. **The hand-over starts EARLY.** The real limit is time, not detection: a saga needs ticks and
   the dwell is five ticks. Warm ahead, never slow down. (M-D; D-MOVE-3 piece 2.)
6. **A transfer never changes the speed.** No clamp, no refusal, no stop. (M-A, M-D.)
7. **A parent converts, in the direction SL1 states.** Going up, the pose is shipped in the old
   parent's own frame and the NEW parent adds the placement it authored for the old parent. Going
   down, the OLD parent subtracts the placement it authored for the destination before shipping.
   The child is carried and states nothing. (SL1 clauses 1 and 3.)
8. **The arithmetic runs at the destination book's rung.** Coarsening is total, refining refuses
   past the finer rung's reach. (D-S9.)
9. **The hull's shard keeps running.** What changes hands is the AUTHORSHIP of its placement and
   the lane its drive rides up. (D-MOVE-2 NEXT; transfer_protocol §8: exterior moves, interior
   never touched.)
10. **The acceptance test.** From the tick after the switch the hull sends the identical six
    numbers to its new parent, in its own frame, and does not know it moved house.
11. **The facts are restated once per connection.** Mass, cross-section, drag, later the reach.
    The new parent has forgotten everything. (D-MOVE-2.)
12. **Velocity never crosses upward.** The new parent receives the hull's pose on the transfer
    envelope, parent to parent, never as a child's assertion. (SL1 clause 3.)
13. **The galaxy walks no children.** The child test is a lookup. A newly arrived hull is a mover
    in a realm of 233,220 children. (SL9; Decision 3 of the reach ruling.)
14. **The destination star system must be awake before the hull arrives.** The only lawful means
    is the interest lead grown by closing speed. (M-B; SL7.)
15. **The client never participates.** The gateway recomputes the chain, bumps the epoch, ships one
    level composed at the same tick. (connection_plane C1; window_lane §2.7.)
16. **G-IDENTICAL on two parent kinds.** A hull inside a star system and a hull inside a planet.
    (HR4; D-37 slice 4.)
17. **Ask before new data crosses or a wire arm is added.** (SL6.)

## 3. The shape

### 3.1 Two keys, two authorities

The directory's key space already holds the answer. It has both `Realm(RealmId)` and
`Ship(EntityId)`, and the transfer design (`transfer_protocol.md` §8) already says what the second
one is for: *"What transfers is the ship's EXTERIOR authority, who owns the hull rigid body in the
host frame, via the standard saga on the `Ship(ShipId)` key. The interior world is owned by the
ship-shard throughout and untouched."*

| Key | Who holds it | What it means |
|---|---|---|
| `Realm(Ship(id))` | the hull's own shard | who RUNS the hull: its interior, its occupants, its drive |
| `Ship(id)` | the PARENT's shard | who AUTHORS the hull's placement: its exterior |

Today only the first key exists. The hull's parent is implied by a launch argument and by which
store file holds its berth row. The plan makes the parent a directory fact: **the authority of the
`Ship(id)` key is the parent's shard.** The ruler switch is then a standard saga on the `Ship(id)`
key from the old parent's node to the new parent's node. The hull's own key does not move. The
hull's process does not stop.

Example. The hull is berthed in System 7. The directory holds `Realm(Ship)` at the hull's node
and `Ship` at System 7's node. The hull leaves the shell. A saga moves `Ship` to the galaxy's node.
`Realm(Ship)` stays at the hull's node. The hull's crew feel nothing.

### 3.2 The subject of the saga is the hull's EXTERIOR

The occupant path moves a dot: its pose and its input authority. The realm path moves an exterior:
the placement state the parent holds (`DrivenState`: position, velocity, facing, spin), the bound
and the look the parent keeps as readings, and the berth row. All of these live on the PARENT
today, in `DrivenChildren`, the region row and the parent's store. So the source of the saga is
the old parent's shard, and the destination is the new parent's shard. The hull's shard is a third
party that is TOLD the result.

The `StubCrossing` payload already carries a `StampedPose` and an opaque `state` blob reserved for
the first compound kind. The exterior rides it:

| Field | Carries | Source |
|---|---|---|
| `pose` | position, velocity, facing, and the instant — a PLACEMENT the old parent authored | the old parent (§3.5 says in which frame) |
| `state` (TLV) | spin (the placement's angular velocity, which `StampedPose` lacks) and the berth fence | the old parent's own authorship |

Nothing the HULL stated about itself rides the blob. Bound, look, mass, cross-section, drag and
later the reach are the hull's own facts: the hull restates them to its new parent on the
on-change lane, once (constraint 11, and SL1 clause 4: what a parent holds about a child as a
reading is not the parent's to pass on; SL3: a realm authors how it looks). Until the hull
restates its bound, the new parent holds no bound for it — one head-read cadence, the same gap
the facts already have.

### 3.3 The trigger: the parent's swept verdict over its driven children

`evaluate_realm_boundaries` gains a third subject lane beside `Dots` and `OwnedTransients`: the
realm's DRIVEN CHILDREN. For each driven child the scan asks the same question it asks of a dot,
with the same swept lookup and the same container fold: which of my children holds this point, or
is it outside me altogether. The subject key is `DirectoryKey::Ship(id)`. Nothing in the fold reads
the child's kind, drive or rating (constraint 3).

Two verdicts exist:

- **Out.** The child's line leaves my bound. Destination: my parent. The request goes to the
  orchestrator as today, with `from_realm` = me and `to_realm` = my parent.
- **In.** The child's line enters a direct child of mine. Destination: that child. Same request.

Example. System 7 scans the hull's line from last tick to this tick. The line exits the system's
shell. System 7 emits a crossing request: subject `Ship(hull)`, from System 7, to Galaxy 1. The
orchestrator reads the three heads, starts the saga with source System 7's node and destination
the galaxy's node.

### 3.4 The early start

At the speeds the owner flies, a shell is crossed inside one tick and the saga needs several. The
parent authors the child's velocity, so it can see the exit coming. The scan tests the line from
this tick to this tick plus a LEAD: `lead = velocity × saga_ticks`, where `saga_ticks` is derived
from the saga's own tuning (the freeze budget plus the CAS round trip), never a literal. The
request goes out when the LED line crosses, and the pose the source ships at the flush is the
truth at the flush tick, so an early request never plants the hull anywhere it is not.

Hysteresis stays what it is: the boundary band's dead zone and the per-subject in-flight latch. A
hull sitting on a shell boundary does not ping-pong because the band decides membership, and a
crossing back needs the dwell.

Example. The hull closes on the shell at one tenth of a shell per tick. The saga takes eight
ticks. The lead is eight tenths of a shell. System 7 emits the request while the hull is still
inside, the saga commits as the hull reaches the shell, and the galaxy's first authored placement
is the hull's true position at that tick.

### 3.5 The conversion: who adds, who subtracts

SL1 clause 1 decides the direction of the arithmetic, and the occupant path already obeys it:

- **OUT (system → galaxy): the old parent ships the hull's pose VERBATIM, in its own frame.** The
  galaxy authored System 7's placement, so the galaxy ADDS it on arrival, through
  `place_arriving_pose`'s direct-child arm, at the galaxy's own rung (`transfer_frame` converts the
  millimetre count into two-metre steps; coarsening is total). System 7 never names its own
  placement in the galaxy. This is the same arm an occupant leaving a system takes today.
- **IN (galaxy → system 8): the old parent SUBTRACTS before shipping.** The galaxy authored System
  8's placement, so it expresses the hull's pose in System 8's frame at System 8's rung (refining;
  it refuses past the millimetre ruler's reach, which cannot happen for a point inside the
  system's shell) and ships it. System 8 accepts its parent's number as given.

Velocity gains or loses the container's velocity in the same fold. Facing composes with the
container's facing. No ceiling and no clamp touch it (constraint 6).

Then the destination splits the pose for its book: a berth cell on its rung plus a `pos_m`
residual, exactly the shape `placement_row` reads today.

### 3.6 The arrival at the new parent

The new parent adopts the exterior in one tick, so the placement book and the child roster never
disagree (today's `child_rows` expects them to agree, and panics otherwise):

1. Insert the hull's region row, incrementally: `ix_of`, `children_of`, its depth, its ancestor
   chain, one entry in `movers_of`, one entry in the child index. No rebuild over all children
   (constraint 13). The cost is measured on the galaxy.
2. Seed `DrivenChildren[hull]` from the shipped pose and blob: position, velocity, facing, spin,
   bound, look. The facts stay empty until the hull restates them.
3. Resolve `ChildRealmNodes[hull]` from the directory (`Realm(Ship)` head), so the up-lanes admit
   the hull's drive from its first tick.
4. Write the berth row into my own store, with the fence bumped. Durable before the source
   releases (the same order the berth tool uses: child row first, then parent row).
5. Author the first placement row for the hull at the crossing tick.

The old parent, on release: remove the region row, `DrivenChildren`, `ChildRealmNodes`,
`ChildLiveness`, `RelayHeld`, the interest latch and the AoI membership keys; delete the berth row
from its store; close the child windows it held.

### 3.7 The hull is told, one hop

The hull must learn three things: its new parent's node (to send its drive), its new lineage (to
build the `child` field of every up-lane message and its demand key), and that its facts are owed
again. It cannot compute the lineage: a lineage is the ancestors' identities, which it does not
hold (HR1). So the new parent states it, once, at adoption. This is the one new datum (§7, ask 2).
It rides the lane the parent already uses to speak to one direct child (`RealmInterest`), or a
sibling arm beside it.

On receipt the hull:

- sets `ParentRealmNode` to the sender, after checking the sender is the directory's `Ship(id)`
  authority (attestation, as every lane does);
- replaces `own_coord` with the stated coord, so `child.parent()` matches the new parent in every
  misroute guard from the next tick;
- clears `StatedFacts`, so the facts go out again on the next tick (constraint 11);
- re-arms `WasOccupied`, so the first occupancy bit fires off-cadence.

Example. The galaxy adopts the hull and tells it: *"you are Universe / Galaxy 1 / Ship hull".* The
hull's next drive datagram names that coord, the galaxy's misroute guard accepts it, and the six
numbers are the same six the hull sent System 7 one tick earlier. That is the acceptance test.

### 3.8 The orchestrator and the launch ledger

The orchestrator runs the saga, so it knows the commit. At commit it rewrites the hull's launch
intent with the new coord and re-keys its demand cell under the new path, retiring the old cell. A
restart then rehydrates the hull under its true parent. The peer book of the hull gains the new
parent's node, so the hull can reach it (today the book is computed once at launch from the
ancestor closure).

### 3.9 The gateway and the sessions aboard

The saga already speaks to the gateway per phase. For a `Ship` subject the gateway does not swap a
session route; it re-derives the LINEAGE of every session whose chain contains the hull: the new
parent's lineage plus the hull and everything below it. Then it bumps those sessions' origin epoch
and ships one full level composed at the same tick as the last old-epoch datagram (window_lane
§2.7 already does this for a session crossing). Old windows on the old parent close, new windows
on the new parent open, both chains held through the overlap.

The client sees an epoch bump with the SAME origin realm, so the camera keeps its facing (the
scene-swap rule keys on the origin). The sky anchor changes its lift path but not its value at the
swap tick, so the stars do not jump.

### 3.10 Waking the next system

While the hull is the galaxy's child, the galaxy's area-of-interest fold treats it as an occupied
child at the placement the galaxy authored, with the galaxy-authored velocity. The interest lead
`velocity × boot_ticks_p99 × tick` demands the star system ahead. The wake-up constant is the
ruling's own open item: it must be DERIVED from a measured boot time on THE world, never guessed
(§7, ask 4). Until it is, the dev cluster's six thousand ticks stand, and the report says so.

If the hull's line enters a system that is not yet running, the crossing request finds no head for
the destination, counts `crossing_unresolved`, and the source re-drives on its ttl. The hull keeps
flying in the galaxy until the system answers. That is the honest degrade, not a fake.

## 4. What is NOT in this plan

- **Passengers as `ChildOf` records with an N+1 CAS (D-33).** The occupants aboard the hull are
  keyed to the hull's interior, which does not move. The interior is not the subject. D-33 stays
  owed for the P8 host/dock case where the interior authority itself moves.
- **The versioned observer bundle (D-35).** A version field is not added here. What DOES keep an
  outside observer's picture whole through the hand-over is the ghost row of §7 ask 7: the old
  parent mirrors the hull's placement to the new parent across the boundary band, the same
  kinematic ghost an occupant has, so the new parent can row the hull before the commit and the
  old parent after it, and no tick lacks the hull.
- **The reach datum (R9 step 4)** and the marker deletion (R9 step 5). The hull is a mover, so it
  is rowed by whichever parent holds it; the pixel gate of §6 measures one drawing at one place.
- **Path keying of the directory (D-RLM-10).** The `Ship(id)` key is lineage-blind by construction
  and needs no path.
- **The tunnel between galaxies.** Its parent is the universe (Q5); it is four ordinary crossings
  and its topology has its own design pass first.

## 5. The slices, in order

Each slice lands green on its gates before the next starts. Each is a commit the owner may take.

| # | Slice | What lands | Gate |
|---|---|---|---|
| 0 | **The exterior key** | The parent leases `Ship(id)` for every berthed child at boot. The hull resolves its parent from the `Ship(id)` head instead of the launch coord. `ParentRealmNode` follows it. A test moves the head by hand and the hull's drive follows within one head-read cadence. | sim unit tests; byte-identity: a cluster with no hull is unchanged |
| 1 | **The driven-child subject** | The third subject lane in the containment scan, with the swept lookup and the early lead. The request carries `Ship(id)`. The keep-alive `expect` on a seed-lineage destination becomes a counted refusal. | a fixture where a hull's line exits a system's bound emits exactly one request; a hull sitting on the shell emits none (hysteresis) |
| 2 | **The exterior on the envelope** | The old parent's flush computes the hull's pose at the destination rung and the TLV exterior blob. The new parent's arrival adopts it in one tick (§3.6) with the incremental region insert. The old parent's release removes it. The berth rows move, child row first. | G-IDENTICAL: the identical fixture on a hull leaving a star system into the galaxy AND a hull leaving a planet into a star system; a measurement of the insert cost on the galaxy's 233,220 children |
| 3 | **The hull is told** | The adoption statement to the hull; `own_coord` replaced; facts restated once; misroute guards pass from the next tick. | THE ACCEPTANCE TEST: the six numbers the hull sends the tick after the commit equal the six it sent the tick before, and the new parent integrates them |
| 4 | **The orchestrator's ledger** | Launch intent rewritten at commit; the demand cell re-keyed; the old cell retired; the peer book grown. | kill and restart the orchestrator after a switch: the hull rehydrates under its new parent |
| 5 | **The gateway's chain** | Lineage re-derived for every session aboard; epoch bump; one level at the same tick; windows moved. | the crossing no-flicker gate on a pilot aboard: the sky anchor's value is continuous across the swap tick; the picture holds exactly one drawing of the hull |
| 6 | **The flight** | The dev cluster: the owner flies out of the home system, watches the panel's coordinates keep growing past a third of a parsec, and into a second system that woke ahead. | the owner's word, and the monitor's log: two crossings, no drop, no panic |

Slices 0 to 3 are the movement contract's own acceptance test and can be flown on the cluster with
the pilot watching the panel. Slices 4 and 5 make it survive a restart and keep the picture whole.

## 6. The gates that already exist and must stay green

- `world_from_inside` (the hull gate) and the window parity gates.
- The three-shard round trip of D-45 C-6c for an OCCUPANT: System 7 → Galaxy → System 8 and back.
  It is the template and must not regress.
- `just coverage-fast`: the new lanes are Tier-A and land at one hundred per cent, with the
  generic-code discipline (branchless shims, monomorphic helpers).
- The pre-existing reds (flight_table, ship_flight_e2e, dual_cluster_crossing_smoke) are measured
  before and after; none may get worse, and ship_flight_e2e is expected to go green at slice 3.

## 7. The asks (SL6 and the open decisions)

The owner decides these before slice 0. Default NO on each until answered.

1. **The `Ship(id)` key becomes live.** It exists in the key space and the design names its
   meaning. Making it live adds no wire arm and no field. It adds one directory row per hull,
   leased by the parent. — *Recommend YES.*
2. **NEW DATUM, parent → child, on change:** the child's LINEAGE (its `RealmCoord`), stated once at
   adoption by the new parent. Why the child cannot compute it: a lineage is the ancestors'
   identities, which a sealed child never holds. What doing without costs: every up-lane message
   the hull sends names the old parent and is refused as misrouted forever. It rides the existing
   parent-to-one-child arm (`RealmInterest`) as a new field, or a sibling arm beside it. —
   *Recommend the sibling arm: a coord is not an interest.* Under SL1 clause 4 this is lawful: a
   lineage is identity, not a placement, and lineage already flows down the tree at every spawn
   (`VD_OWN_COORD` is the parent's coord plus one level). The switch only moves that statement
   from launch time to adoption time.
3. **The exterior TLV blob on `StubCrossing.state`.** Spin and the berth fence only — the two
   parts of the parent's own authorship that the `StampedPose` does not carry. Bound and look are
   NOT on it: they are the hull's statements about itself, and the hull restates them to its new
   parent with its facts (SL1 clause 4, SL3). This is the first use of the field the transfer
   design reserved for a compound kind. No new arm. — *Recommend YES.*
4. **The wake-up constant.** `boot_ticks_p99` is six thousand ticks in the dev cluster because a
   shard still builds all 233,220 systems to find its own. The ruling forbids guessing it. —
   *Recommend: measure a planet's and a system's boot on the cluster in slice 6 and derive it
   from the p99 of that measurement, as the field's own name says.*
5. **An empty hull.** The machinery cannot tell a hull from a moon, so an empty drifting hull
   switches too. Its shard stays up while it is inside somebody's reach and sleeps otherwise. A
   SLEEPING child is still authored by its parent (the parent keeps its exterior). — *Recommend:
   accept; the exterior lives on the parent, so a sleeping hull needs no process to be carried.*
6. **The hull's boundary shape.** A hull is an `Aabb` in its OWN frame today, and the verdict
   tests a point after rotating it into the child's frame, so a turned hull is still a box, not
   an oriented box in the parent. Slice 1 pins that with a test on a hull turned by a quarter.
   If a built hull ever becomes an `Obb`, the sweep does not cover it until the `Obb` sweep lands
   (D-MOVE-3 piece 1). — *Recommend: keep `Aabb` for built hulls until then.*
7. **The ghost row across the band.** The registry gives the ship kind the `Always` ghost policy
   (a kinematic mirror across the whole overlap band, never integrating). The occupant path has it.
   The exterior crossing must have it too, or an outside observer's picture lacks the hull for the
   ticks between the freeze and the release, which is a seam (SL8) and a departure from the
   registry (HR2). The mirror rides the existing `Ghost` arm, old parent to new parent, keyed on
   `Ship(id)`. — *Recommend YES; it is the same machinery, not a new one.*

## 8. What the owner will see

Before: the hull leaves the shell, the planets and the star leave the picture, the coordinates
grow to a third of a parsec and freeze, and the sky stops.

After: the hull leaves the shell, System 7 hands it to the galaxy with the saga's ordinary log
lines, the panel's realm-in-galaxy coordinates keep growing without limit, the sky keeps sliding,
the next star system wakes ahead, the galaxy hands the hull down at its shell, and its planets
appear as the hull's reach meets theirs. The hull's crew feel nothing at either hand-over.
