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

# Signal authority, confidentiality, and relays as built infrastructure

*Adjudication of three owner points raised 2026-08-03 against `docs/investigation/block_system_design.md` §7 as
written that day (17,401 lines; §7 spans lines 12626–15072). Binding context: `CLAUDE.md` HR1–HR6,
`docs/design/PLAN.md` HR1, `docs/design/sealed_shards.md`, the Owner rulings table R1–R14, and the
landed code in `crates/sim/src/capability.rs`, `crates/sim/src/coupling.rs`,
`crates/wire/src/intershard.rs`, `crates/core/src/ids.rs`.*

*This document does not modify §7. It rules on three questions, names what §7 already closes, names
what it does not, and lists what must be planted before the first signal slice.*

---

## The owner's three points, quoted

**(1) Force goes up to the parent, and the parent's physics differs by realm kind.**

> *"the force in case of the Ship realm should go to the parent realm to calculate the position
> according to the Parent's realm physics: Planet will differ from calculation of Star System realm."*

**(2) Relays should be player-built functional blocks, and the cost should be time.**

> *"we will need to build relays (a functional blocks) that will pass the signal up and down to some
> of the Realms, and the cost should be time, as nobody expects it to be instant. So those Relays
> functional blocks can be queues, that pass signals to other relays and then sends to whoever is
> interested — it might be a complex task, but we will need to think how we will support it in the
> future — we don't need it now, but the foundation should be extensible."*

**(3) The takeover attack.**

> *"we need to have a mechanism, that will allow to cypher the signals, as in case if all signals are
> open and I didn't authorize to read certain signals, anybody can take over my ship or anything
> simply by sending signals with the correct names."*

---

## The answer in one page

**You are right, and you are right about a narrower thing than you feared — which is good news,
because the narrow version is cheap to close.**

Start with the ship. The way the design is written today, four different people could try to take
your ship, and three of them fail before they start.

Someone using a modified game program fails completely, and this is the strongest thing in the whole
design. The game program on a player's computer has no way to say "make this control do this". All it
can say is "the person in this chair moved control number three". Which control number three actually
does is a setting stored on the server, on your ship, that only you can change. So there is no
sentence a cheating program can send that names one of your controls, no matter what names it knows.

Someone flying past in their own ship fails too, and the reason is worth understanding because it
costs nothing: your ship's controls are private to your ship unless you deliberately publish them
outward, so a message from outside has nowhere to arrive. It is not blocked at your door; there is no
road to your door. Behind that there are two more locks — a stranger needs written permission from
your ship to write anything into it, and every message between two machines carries a tamper seal.

Someone who walks aboard and starts pressing your buttons fails, because pressing a button is a
different system with its own permission check.

**The fourth one succeeds, and it is the one you are asking about.** Someone who gets a part attached
to your ship — welded on, docked on, or, worst of all, included inside a free ship design you
downloaded and built — is on the *inside*, and the design says in its own words that anything on the
inside is trusted. It claims three separate times that a hostile part cannot drive your controls
because "the control says who may write it and this part's owner is not on the list". There is no such
list anywhere in the design. There is a field with three possible settings and nothing to hold the
names, no rule about who owns the setting, no stated default, and no place in the whole document where
the check is performed. It is asserted three times and specified zero times.

The attack that worries me most is the shared blueprint, because it defeats every obvious fix.
Somebody publishes a beautiful free ship design. Buried inside it is one small part wired to the
engines and set to listen to the open radio. You build it. Every permission check passes, because
*you* are the one who built it — you own the ship, you placed the part, your own build authored the
setting. Then the attacker simply talks on the radio, and your engines answer. Today the only thing
checked when you paste a blueprint is whether it is too expensive to run.

**So: the hole is real, but it is on the inside, and the rule that closes it is one sentence.**

> **Knowing the name of a control is never enough to use it.**

A name is a street address, not a key. Everyone can know where your ship's throttle lives — your crew
knows, your heads-up display shows it, anyone you share a design with can read it. What must be true
is that being able to *say* the name gets you nowhere unless the ship has separately, deliberately,
and revocably handed you a key. The key is not something anyone can compute or guess; it is *issued*
by your ship, to a named party, for a named list of controls, with an expiry. There is no name a
stranger can type that produces a key they were not given.

Now the part where I have to disagree with your wording, because getting it wrong would ship a real
hole.

You asked to encrypt the signals. Encryption is a curtain: it stops people *seeing*. What you
described — somebody sending commands and flying your ship — is somebody *doing*. A thief who wants to
fly your ship does not need to read one thing you send; he writes his own commands. Drawing a curtain
over your windows does not stop him opening the door. The tool for "nobody drives my ship" is
permission — a lock on the ignition, checked against a key you granted and can take back. There is a
third tool as well: a tamper seal, which proves *who* sent something, so that after the fact you can
say "the station's autopilot did this, at this moment". You need all three, for three different jobs,
and swapping them is how systems ship broken.

Using encryption as the lock would actually be worse than no lock. A shared secret code has no way to
be taken back from one person, no expiry, and no record of who used it — and a thief who never
decrypts anything can simply record one valid scrambled "full throttle" message and play it back later.

**Your instinct about encryption is not wasted, though — it lands on a real problem, just not that
one.** Radio is genuinely open. Once a message leaves your neighbourhood and travels through relay
infrastructure, the machines carrying it do not hold your ship's rules and cannot enforce them, so
anyone who guesses a channel name can listen. That is not a flaw to be fixed; on a broadcast medium it
cannot be fixed, because anyone you legitimately let listen can lawfully repeat what they heard — and
the design already ships a repeater block that does exactly that. So radio confidentiality has to come
from *you scrambling the contents yourself*, using a part you build and a code key you hold as a
physical item that can be captured with your ship. That is already the recommended answer in the
design and it is waiting on your decision. It should be sold to players honestly: it hides your
messages from other players, never from us — we hold every key, because our machines have to run your
parts.

One more thing about radio that is honest and worth turning into content rather than hiding: even
perfectly scrambled, a transmission tells a listener *how far away you are*, because the delay is
measurable and light travels at a known speed. Three listening posts locate a hidden base to within
about one Earth diameter. Silence is therefore a real tactical choice, a tight beam pointed at one
receiver is genuinely stealthy, and routing through a relay breaks the measurement. Declare this;
otherwise a player who scrambles his traffic will believe he is hidden and be wrong by one planet.

**On relays: take your model whole.** Player-built relay blocks, owned, powered, queued, destructible,
with time as the cost, is better than what the design currently has — and it is *less* engine work,
not more, because it deletes a whole machinery for automatically choosing which server holds a radio
network, plus the migration protocol that machinery needs. Your instinct that "the cost should be
time" turns out to be the cheapest possible thing to build: the design already computes an arrival
time from distance, and a relay simply adds to it based on how backed up its queue is. That makes a
well-built, well-fed relay network genuinely faster than a cheap one, which is a real reason to invest
in infrastructure. Same star system: instant. A few light years through two relays: a handful of
seconds, like a satellite call. Across the galaxy: minutes, which is mail. Never hours — that kills
the feature. And the delay must be *visible before you press send*, so a relay tells you "KEPLER-3 —
four minutes twelve" and you decide whether to write a letter or fly there.

Three relay rules that cost nothing now and are expensive later: a relay may only make a message
later, never earlier, and the receiver checks that against the laws of physics; a relay may never be
allowed to *authorise* anything, only to carry it, so permissions get their own seal that a relay
cannot forge; and a message stored in a relay cannot outlive the permission inside it, because the
expiry is on the universe clock and not on the relay's own.

**On force going up to the parent: you are restating a law you already gave, and the design already
obeys it word for word.** The section that traces your thruster example says the containing realm's
machine is the physics authority, that it integrates the hull, and that the ship never integrates its
own hull. Nothing needs to change to accommodate you. What nobody has designed is the *other* half you
named — that a planet computes differently from a star system, and that a ship changes parent while
flying.

The rule that makes that work: the ship always sends the *same* thing to every kind of parent — the
push and twist its engines produce, and the shape of its hull. The difference between a planet and a
star system is not two different calculations; it is a *list of ingredients* the parent has. A star
system has gravity from its bodies and nothing else. A planet adds air, wind, a gravity direction that
changes as you fly around it, and ground to hit. A spinning station adds the forces you feel from the
spin. Empty space is the same list with the air left out — an absence, not a special case. That is
what lets a part behave correctly everywhere with no rule anywhere asking what kind of place it is in.

Two things are missing from that handoff today, and both bite exactly when a ship crosses a boundary
at speed. First, the design says the ship sends its engine force, its mass and its balance point — and
nothing about its *shape*. Air resistance depends entirely on which way you are pointing: a long hull
presents five times more surface sideways than nose-first. Without the shape, a planet cannot compute
air resistance at all, so at the moment you cross into a planet the new machine holds your engine
thrust and applies no drag — a free push in exactly the direction you are already going, at the
precise moment a player is watching. Second, the same thing happens in reverse: your ship's air-density
reading briefly stops arriving during the handover, and the current rules would make it fall back to
zero — which cuts every air-breathing engine mid-manoeuvre. Both are fixed by carrying the missing
pieces in the same handover package that already carries your last engine setting.

And one number that is not a refinement: today, crossing into a planet's sphere of influence silently
drops the star's pull. That step is worth roughly four hundred and sixty metres per second of free
speed every time you fly a figure-eight across the boundary — a fuel-free engine, nineteen percent of
the starter planet's own gravity, flyable by an autopilot the design already ships. Twenty-four bytes
of "and here is the pull from further out" removes about ninety-nine percent of it. That is not
optional.

---

## The authorisation model, exact

### The one principle

> **SIG-NAME-IS-NOT-A-KEY.** A channel name is an ADDRESS, not a CREDENTIAL. Knowing where something
> lives is never sufficient to act on it. Every write capability is *issued* by the party that owns
> the channel; none is ever *derived* from a name, a hash, a pattern, a prefix, or a guess.

This sentence exists nowhere in §7 today and is load-bearing everywhere in it. §7.4 makes the key a
pure content hash of the name; §7.14 says out loud that the key is "computable offline by anyone —
that is the point of §7.4"; names leak legitimately and constantly through blueprints, heads-up
displays and crew. A design that ever leans on name secrecy is broken on arrival. Write the sentence
beside §7.4's loud collision check, and give it §7.18 item 12's anti-vacuity treatment.

### First: exactly what §7 as written does and does not close

**CLOSED, structurally — a modified client.**

- §7.10.3, and repeated at §7.13 step 1: *"The intent message names a `(seat_entity, slot, value,
  client_seq)`; it never names a port, a channel or a block."*
- §7.3.7: *"A client never holds a channel lease. It has no channel identity and no interest set."*
- §7.13 step 2: *"The client computed nothing and named no channel."*
- §7.5.4: *"Server-side validation must additionally confirm the claimed player is actually resident
  in the realm and actually seated — never trust a client's assertion about its own state."*

There is no verb in the client protocol that expresses "publish to channel X". The literal attack is
**unrepresentable**, and this is deliberate rather than lucky. Say this to the owner first.

**CLOSED, three deep — a stranger publishing from outside the realm.** The first refusal is free and
does all the work:

1. *Routing.* §7.8.5: *"The construct's channels are `Local` unless explicitly exported, export is
   capped at 64 keys, and `hops ≤ 2`. The bomb cannot leave the neighbourhood."* A non-exported key
   has no entry in the parent's inverted index, and §7.3.4: *"if nobody is interested the emission
   terminates at the first hop."*
2. *Grant.* §7.5.3: *"no grant, no cross-boundary write (nobody may write a channel in a realm they do
   not own without a durable, revocable grant held by that realm)."*
3. *Authenticity and freshness.* One truncated MAC per bundle at 345 ns, plus a per-peer monotone-max
   fence that kills mid-transfer frames.

**One weakness inside that half:** the `Local` default is doing most of the security work while being
stated only in an anti-abuse walkthrough. It is a convention, not a validated default on
`ChannelDecl` alongside SIG-HB. Promote it.

**OPEN — inside the realm.** §7.5.2 tier 1 states the exposure in its own words:

> *"Mutual auth once per shard pair; in-realm traffic is implicitly trusted because the realm's shard
> is the authority for everything in it."*

That is defensible as an *authenticity* claim (a shard need not authenticate itself to itself) and is
being relied on as an *authorisation* claim, which it is not. §7.5.4 asserts the fix:

> *"A hostile block welded onto your hull cannot write your flight channel, because the channel says
> who may write it and the welded block's owner is not on the list."*

§6.7 makes the mirror-image assertion for reads, and §7.12.1 rests `flight.abort` on the same
allowlist. **The entire specification behind those three assertions is one line** (§7.1.5):

```rust
pub access: AccessPolicy,   // OwnerOnly | AllowList | Public, separately read/write
```

Verified by grep: six occurrences of `AccessPolicy` in 17,401 lines, every one declarative or prose.
There is **no owner field** for `OwnerOnly` to refer to, **no list attached** to `AllowList`, **no
principal type** anywhere on the decl, **no stated default**, and **no enforcement site on any write
path** — not in §7.2.3's Pass A, not in §7.13 step 2 (the actual write of a seat input into `back[c]`),
not in §7.8.5's config-apply walkthrough, which recomputes the delivery budget at exactly the point
where the check belongs and does not perform one. Nothing anywhere records who placed or configured a
block.

**Verdict on the owner's question: the hole is REAL, it is on the INSIDE, and the outside is genuinely
well built.** Do not tell him it is fixed; do not tell him it is broken.

### THE THEFT — how bind-time checking is defeated by a blueprint

This is the most important finding in this ruling, because it defeats the obvious fix and every step
of it is a thing a player is supposed to be able to do.

1. I publish a free blueprint. Buried in it is one Signal Bus Hub whose `PortBinding` names
   `flight.thrust.fwd`, and whose channel declaration says `plane: Relay{net}`, write `Public`.
   Nothing checks that: §7.8.1's paste-time check counts `pcu` and only `pcu`.
2. You paste it. **Every authorisation check passes, because you are the authoriser** — you own the
   realm, you placed the block, your own config authored the declaration.
3. I lease the net. §7.14 already rules that the relay holds no `ChannelDecl` and enforces nothing.
4. I publish. The value arrives as an **in-realm** write from your own block, and §7.5.2 tier 1 says
   in-realm traffic is implicitly trusted. No grant is required, because no boundary is crossed.
5. Arbitration hands me the channel. `Exclusive { lease: bool, priority: u8 }` is declared by **the
   writer's `PortDef`**, and no channel owner anywhere in §7 sets a ceiling on it. Your seat is
   outranked.
6. Your HUD is supposed to say *"controls held by X"* (§7.12.1). It cannot: §7.1.1 deleted the sender
   entity id from the bus deliberately, so **X is not derivable**. In a 10,000-block hull the remedy is
   to find one block by grinding.
7. You unbind it. Unless the unbind is a **durable** write, it dies at the next reap: `RlmTuning::cloud`
   (verified, `crates/sim/src/rlm.rs`) makes an empty realm reapable in ≈2 s, and spin-up rebuilds
   `subs` from persisted `config_state`, which still says bind. Your revocation lasts as long as you
   stay logged in.

Note what §7 does *not* say, which is what makes step 1 possible: it never states where a channel's
`plane` and `access` come from. §7.10.2 says the player "adds one by typing a name and choosing a
value type"; the remaining declaration fields have no stated source. The first implementer decides,
and the natural implementation puts them in the player-authored config blob — which a blueprint is.

**Four rules close the theft, and none of them is a per-message check.**

| Step | The rule |
|---|---|
| 1–2 | **An imported config may never WIDEN a channel's policy beyond the realm default.** Non-`Local` planes, `Public` write, `grantable: true` and relay-origin acceptance each require an itemised per-channel confirmation at paste. One walk over the blueprint's bindings, O(bindings), bounded by `max_config_bytes`. |
| 4 | **`ChannelDecl::accepts_relay_origin: bool`, default false**, validated in `ChannelDecl::build()` beside the SIG-HB checks. Relay-delivered values are stamped with their origin realm (already in the bundle header), so a repeater cannot launder a foreign write into an owner-authorised one. |
| 5 | **Priority is GRANTED, never DECLARED.** An `Exclusive` channel's effective priority for a writer is `min(port's declared priority, the ceiling the channel OWNER attached to that authorisation edge)`, with a non-owner edge defaulting to band 0 — below the seat by construction. One `u8` on the compiled edge, one clamp at bind, zero on the hot path. |
| 6 | **The arbiter publishes its winner as a diagnostic.** A reserved `sys.signal.arbitration` channel carrying `(channel, winning EntityId, priority)` on change, readable by the realm owner only — the same treatment §7.8.4 gives `sys.signal.shed`. This does not reopen the sender-id decision: the arbiter already knows the winner at `resolve(pending[c])` time, and the value on the bus is untouched. |
| 7 | **Every authorisation mutation is a durable, fence-stamped write in the same transaction as the runtime recompile**, and spin-up RE-AUTHORISES from the durable grant/ACL rows rather than trusting the persisted binding. |

### Ownership

> **A channel declaration is owned by the REALM, never by whoever interns the name first.**

§7.4's `intern()` creates a decl on first sight of a name and checks only for a hash collision with a
different name. There is no check of *who* is interning or what policy the new decl gets. Combined
with §7.10.2's "the player's channel name lives in the binding", first-interner-wins is a live
privilege-escalation path: an attacker's block binds a name the owner has not used yet, the intern
creates the decl with the attacker's chosen policy, and the owner's seat later attaches to it.

`ChannelDecl.owner: PrincipalRef` defaults to the realm owner; only the realm owner may widen
`access`. `crates/core/src/ids.rs:66` already has the right primitive — `AccountId(pub u128)`,
documented as "The durable principal (random u128, non-PII)" — but 16 bytes per row is the wrong
storage, so principals are interned per realm to a `PrincipalRef(u16)`.

### Default scope

> **Realm-private in both senses: `plane: Local` (already true in practice, promoted to a validated
> default) plus write-scope-realm.**

The safe answer must cost the player nothing to get right, because §7.10.2's model is "an empty list
of signals, the player types a name" — which makes a permissive default the natural implementation
choice, and a permissive default *is* the owner's attack shipped as a default. This is a saved-world
one-way door in both directions: a world saved under a public default cannot be silently tightened
later, which is exactly the asymmetry §7.14 already names for radio.

### The write capability itself

> **A write capability is a compiled EDGE in the realm's own publisher/subscriber tables, admitted
> once at bind time by the realm that owns the channel, and it is ISSUED, never derived.**

A block writes by having a publisher edge into `pending[c]`; a block reads by being in `subs[c].local`.
Those lists are compiled at config apply (§7.2.1: *"built when a player wires a port and mutated only
when a player rewires one"*) and the hot path reads nothing else. So the enforcement point is
`bind()`: may THIS principal create THIS edge on THIS channel.

For a remote grantee, the grant response carries **the resolved key set** the grantee may write,
minted by the owner under the owner's fence. There is no name it can type and no hash it can compute
that produces a writable key it was not given. That single sentence is the direct answer to the
owner's fear, and it kills the tempting wrong design:

> **REFUSED: pattern-scoped grants (`flight.*`).** They are a privilege escalation — a grantee
> self-authorises by CREATING `flight.anything`. They are also *unevaluable*: §7.4 makes `ChannelKey`
> the first 8 bytes of a hash of the name, which destroys all prefix structure by construction, so
> §7.12.2's `channels: [flight.*]` cannot be checked against a key. Scope must be an explicit resolved
> key set closed at mint time, plus a declared `ChannelGroupId(u16)` on the decl for ergonomics.

The group is not optional. Without it a grant can only enumerate keys, so every new channel a player
adds to their flight bus silently falls outside every existing grant and a station's autopilot breaks
the day the pilot adds a channel.

### The grant object

§7.12.2 step 4 mints `{ grant_id, grantee, channels, priority, expiry_tick, revoked }`. That cannot
express the cases the design's own scenarios need. The corrected shape:

```rust
pub struct Grant {
    pub grant_id:        GrantId,
    pub grantee:         PrincipalRef,
    pub rights:          GrantRights,      // bitflags: WRITE | READ | BIND | DELEGATE
    pub scope:           GrantScope,       // explicit resolved key set + optional ChannelGroupId
    pub priority_ceiling: u8,              // the arbitration clamp above
    pub condition:       GrantCondition,   // Always | WhileDocked{peer} | WhileSeated{seat} | WhileInRealm{realm}
    pub not_after:       UniverseTick,     // MANDATORY, bounded by SignalTuning::max_grant_ticks
    pub parent:          Option<GrantId>,  // delegation; child ⊆ parent in rights, scope AND expiry
    pub key_epoch:       u8,
    pub token:           Option<TlvBlob>,  // already reserved by §7.5.3 for [7-F] Option B
    pub revoked:         bool,
}
```

- **`rights` is a bitflag** because reads need the same object and today only writes have one.
- **`not_after` is mandatory and bounded**, and a condition may only ever SHORTEN it — so a bugged
  docking port that never falsifies cannot grant forever.
- **Conditions are edge-triggered, never per-tick predicates.** A Docking Port's `connected` port
  going false is an ordinary in-realm edit, so a condition falsifying recompiles the affected edges
  exactly like any rewire. This is what makes undocking automatically end a station's authority over
  your ship, rather than the pilot having to remember.
- **Delegation falls out with no new library.** `parent: Option<GrantId>`, child rights/scope/expiry
  are subsets, revoking a parent revokes the subtree. This **re-takes [USER DECISION 7-F]**: the only
  thing `biscuit-auth` would still buy is *offline attenuation across trust domains*, and every
  scenario in §7 is online.

### Revocation

**Immediate, durable, no grace period, and evaluated at the destination at DELIVERY time.**

Three corrections to §7 as written:

1. **§7.12.2 step 6 revokes a grant by writing `false` to a port.** That is a durable authority change
   triggered by a bus value — and if that value rides the State lane it rides an unreliable datagram,
   so the pilot presses revoke, the datagram is lost, the grant survives, and the UI says revoked.
   That is the `DockState.clamped` bug class that `crates/sim/src/coupling.rs`'s sealed `EffectFree`
   marker exists to make uncompilable, smuggled in through the signal system instead. Worse, the
   natural channel name for it (`flight.autopilot_enable`) is *inside* the granted `flight.*` scope, so
   the station can hold it true. **Revocation is the same reliable discrete action as the approval
   (D-39.1's carrier), authenticated as the seated occupant against the realm ACL.**

   > **Invariant SIG-NO-AUTH-FROM-BUS.** No grant check, ACL check, lease decision or admission
   > decision may read `front[c]`.

2. **§7.12.2 step 6 says revocation takes effect because "the station's next bundle fails the grant
   check"** — i.e. the check runs at ingress on the next bundle. But §7.6 queues light-lagged Events at
   the *routing node* with `arrival_tick` frozen at emit, and [7-A] Option C's horizon is one light
   hour ≈ **72,000 ticks at 20 Hz**. Anything already in the wheel has passed a check that is now
   false. **Authorisation is evaluated at the DESTINATION at DELIVERY time, never at the routing node
   and never at emit.** The wheel then carries opaque bytes, which is consistent with §7.14's ruling
   that a forwarder holds no policy. This failure is untestable by accident — nobody writes a test
   that waits an hour — so it must be an asserted invariant.

3. **No grace period, and the safe state is correct.** §7.12.1's outranking rule already handles the
   main case well: *"the instant the higher-priority writer stops publishing the seat is the
   highest-priority live writer again and control returns on the next tick, with no re-lease handshake
   and no gap."* So revoking while you are seated costs 50 ms and no dropout. Revoking while
   **unseated** leaves an `Exclusive` port with no live writer, which reverts to `fallback` = 0 for a
   thruster — the engines cut mid-approach. That is the correct SAFE state (a runaway burn is the worse
   failure), the UI must warn before confirming, and a designer will be tempted to add a revocation
   grace period and must not: *a revocation that can be delayed is not a revocation.* Note the
   deliberate contrast with PLAN.md's hold-last-thrust guarantee — that is a COUPLING guarantee for a
   NON-authority event; a revocation is an authority event and must not inherit it.

### Consent names EFFECTS, not channels

§7.12.2 step 2 renders *"STATION KEPLER-3 REQUESTS AUTOPILOT CONTROL — APPROVE / DENY"*. A channel
name is not a capability a human can evaluate: what `flight.thrust.fwd` *does* is whatever your ship's
`subs[c]` currently contains, which is your config, not the station's. So a player approves a word and
grants an actuator set they never saw.

**At approval, walk `subs[c]` for each granted key** — O(1) per subscriber on the already-compiled
list, once per docking — **and render the resolved effect**: "MAIN ENGINES · RCS · DOCKING CLAMPS".
Then **re-resolve at mint and refuse if the set changed**, because the compiled list can change between
display and mint (a crew member with build rights re-points the granted channel at the reactor scram
while the dialog is open). The dialog must be composed by *your* realm, never by the station's offer
text — the same server-composes rule §7.9 already imposes.

Two further defects in the same handshake:

- **The identity check happens too late.** §7.12.2 verifies the station's Ed25519 signature at step 4
  (grant mint), *after* the player has clicked approve at step 2. Verify before display, not before
  mint. 18.86 µs, once per docking, either way.
- **`dock.autopilot.offer` is a player-typeable string.** §7.10.2 makes the namespace flat and
  player-authored; nothing reserves the game's own protocol names. A griefer with a Short-Range Antenna
  publishes a forged offer with an attractive `station_identity`, and every subscribing ship in the
  neighbourhood shows the dialog. **Reserve a `Protocol` key scope players cannot mint into** —
  `dock.*`, `sys.*`, `hull.*`, `chat.*` — compiled constants, `ALL`-loop asserted.

### Key scoping — the hard one-way door

§7.4 makes `ChannelKey` the hash of the **bare name bytes**, and treats cross-realm agreement as the
feature: *"any two realms in the galaxy derive the same key for the same name with zero coordination."*
Per plane that is:

- **`Local`** — irrelevant. My "thrust" and yours live in different `SignalGraph`s at different `Vec`
  indices; the key is never used. This is why the `Local` default makes squatting impossible by
  construction.
- **`Neighbourhood`** — a real defect. Two sibling ships both exporting "thrust" collide in the
  parent's inverted index, so your emission routes into my realm and only the ingress grant check
  stands. The takeover is stopped; the forced decode-and-refuse traffic is not. **Human-chosen names
  collide at close to 100%, not at §7.4's 2.7e-12** — everyone names their thrust channel `thrust`.
- **`Relay`** — open by design (§7.14).

> **`ChannelKey = H(scope_bytes || name_bytes)`, four scopes: `Realm` (default), `Construct` (for
> neighbourhood exports), `Net` (two strangers meet iff they chose the same net AND the same name — so
> a public distress frequency becomes deliberate rather than accidental), `Protocol` (compiled
> constants, mintable by nobody).**

Still a `u64` on the wire, still a `u32` index in-realm, still computed rather than allocated, and
§7.4's birthday bound becomes proven per scope rather than argued. **It changes what `ChannelKey::of()`
computes, so persisted player configs would re-resolve to different channels — it must ship in the
first slice**, the same argument §7.4 makes for its own loud collision check.

### The eight named scenarios, answered

| # | Scenario | Answer |
|---|---|---|
| 1 | A stranger presses a button on my ship | **Nothing happens.** The press is an INTERACT right checked against the realm ACL at the interaction seam (D-39.1's discrete-action carrier). The bus is not involved: the button's own output edge was authorised when *I* placed it. Making a public doorbell work is one deliberate per-block setting. |
| 2 | A stranger publishes my flight channel from outside | **Three refusals deep, and the free one does all the work** — no route (`plane: Local`), then no grant, then the bundle MAC and fence. |
| 3 | A stranger builds their own block on my ship | **Two gates, and the first is not §7's.** May they place at all — P6 edit admission / realm ACL, default no. Given a placed block, may its ports bind — bind-time check against the placer principal recorded in `config_state`. They may create their OWN private channels in my realm (which I own, can see and can delete) and may not bind a publisher edge to `flight.*`. |
| 4 | A friend flies it | A realm-ACL **ROLE** for a standing person, a **GRANT** for an episodic or foreign principal — both compiling to the same edges, so there is one enforcement path. Needs owner-authored CHANNEL GROUPS or the UI is a per-channel chore. |
| 5 | I revoke mid-flight | **50 ms and no gap if I am seated** (§7.12.1's outranking rule); **thrust cut to the safe state if I am not**; never a grace period; the revocation is a durable write so it survives the ≈2 s reap. |
| 6 | The station's authority ends at undock | `GrantCondition::WhileDocked{peer}`, edge-triggered off the Docking Port's `connected` port, plus a mandatory hard `not_after` as belt and braces. |
| 7 | Turrets but not engines | A role or grant scoped to an owner-authored `turret` group. The Turret Mount already carries an `Exclusive` authority port, and four gunners on four leases already coexist (§7.12.1). The only new rule is that group scope must be an explicit resolved key set, never a name pattern. |
| 8 | Someone stole my ship | **Two sub-cases, opposite answers, and both are deliberate.** A thief who takes the construct's ownership core SHOULD be able to rebind everything — owning a ship means owning its bus, and a stolen ship that is a brick means theft stops being content. The rule that makes it coherent: **an ownership transfer atomically revokes every grant and clears every non-owner role under the new owner's fence, and triggers one full recompile of the realm's edges** (O(bound ports), a rare event) — otherwise the previous owner keeps flying their stolen ship remotely. A thief who BOARDS without the core is a non-owner inside my realm and gets nothing: no placement, therefore no new edges, therefore no writes — which correctly makes the ownership core the objective. And the transfer itself is HR1's own case: PLAN.md says discrete authority-gating events *"are NOT couplings — they are Transfers through the saga, by construction"*, so ownership transfer rides the transfer saga and can never be a signal, however convenient a `takeover` channel would look. |

### The `PortGrowth::PlayerNamed` gap

§7.10.2 gives `port_growth: PortGrowth // Fixed | PlayerNamed { max_in, max_out }` and §7.11.4 states
*"All `Surface`, `port_growth = PlayerNamed`"* for the Interface/IO group — Seat/Console, Button,
Toggle, Lever, Keypad, Display Cover, Indicator, Sign. A player-grown port by definition has **no
static `PortDef` row**, therefore no declared `arbitration`, no `retention` and no `grantable`. §7
never states the defaults, and the block whose ports are most security-critical in the whole catalogue
— the Seat — is the one whose ports are player-grown.

Whatever the answer, it must be a validated `PlayerNamedPortDefaults` in `SignalTuning`, never an
inline literal, and **`grantable` must default to `false`** so a foreign writer can never be admitted
to a port the static table never reviewed.

---

## Where the check lives, and the proof that the hot path stays free

### Why a per-message check is not an option

§7.2.2 measures the delivery constant at **0.5 ns** and budgets
`deliveries_per_tick_budget = 1,000,000` per shard per tick = 0.5 ms = **1% of a 50 ms tick**. Against
that budget, a per-published-signal authorisation check costs:

| Check | Per-item cost | Per tick at budget | Share of a 50 ms tick | vs the whole subsystem |
|---|---|---|---|---|
| Permission bitmap bit-test | ~1.5 ns | 1.5 ms | **3.0%** | 3× the delivery constant |
| Hash-map ACL lookup (§7.2.2's measured SipHash) | 5.2 ns | 5.2 ms | **10.4%** | 10× |
| Keyed BLAKE3 MAC per signal | 51.6 ns | 51.6 ms | **103.2%** | more than one whole tick |
| HMAC-SHA256 per signal | 345 ns | 345 ms | **690%** | 6.9 ticks |
| Ed25519 verify per signal | 18.86 µs | 18.86 s | **37,720%** | 377 ticks |

The cheapest conceivable per-message check **triples the subsystem's headline number for zero security
benefit**. §7.5.3 already makes this argument at a smaller scale (5,000 signals/tick × 18.86 µs = 94 ms
for a 50 ms tick) and concludes per-bundle is the only affordable granularity. The same reasoning taken
one step further says: not per-bundle either — **per EDGE, at bind time**.

### The bind-time cost is zero on the hot path

The bind path is O(ports on the changed block), runs at roughly one rewire per player per second
rather than 20 Hz, and **already** does a sorted insert into `subs[c]` and recomputes the delivery
price (§7.2.1, §7.8.5). The authorisation decision is one more thing that happens at a rate five orders
of magnitude below the tick. An unauthorised edge never enters `subs[c]`, and Pass A never consults a
name.

Cross-realm ingress keeps §7.5.2's one MAC per bundle, amortised over up to 87 `Fx` items = **3.97
ns/item at HMAC-SHA256, 0.59 ns/item at keyed BLAKE3**. Per-item authorisation is free because an
unpermitted key resolves to no local id in the peer's compiled import table — **refusal by absence,
not by check**.

### The proof

> **Invariant SIG-EDGE.** Every entry in `subs[c].local`, every entry in `subs[c].remote`, and every
> publisher edge into `pending[c]` was admitted by an authorisation decision. `bind`/`unbind` are the
> ONLY mutators — private fields with accessors, the idiom `crates/sim/src/capability.rs` already uses.

**Soundness.** §7.2.3's Pass A delivers to exactly `subs[c]`, accumulates exactly the publisher edges,
and consults no name anywhere. So if every edge is authorised, every delivery is authorised. This is
the *same observation* as §7.2.3's own O(deliveries) proof, reused — which is why the security property
and the performance property have the same cause.

**Completeness of admission.** The only route to `bind()` is a config apply, which is already a
durable, fence-stamped, ACL-gated edit on the P6 pipeline. Anything that creates an edge another way is
a **structural** bug (an unreviewed insert into a private field) rather than a **logic** bug (a missed
check) — the kind review actually catches.

**Liveness.** Every authority reduction — grant revoked, `not_after` reached, condition falsified,
block destroyed, realm ACL changed, ownership transferred, fence bumped — must recompile the affected
edges before the next Pass A. This is exactly the edge-triggered discipline §7.1.5 already establishes
for in-realm liveness. The gate is one line: **revoke at tick N, zero deliveries at N+1.**

**Durability.** Because a realm is reapable in ≈2 s and spin-up rebuilds `subs` from `config_state`,
liveness alone is not enough: the mutation must be durable and spin-up must re-authorise. Assert that
too: **revoke, reap, respawn, still zero deliveries.**

### Make "forgot to check" uncompilable

`crates/sim/src/coupling.rs` demonstrates the project's own idiom in 60 lines: a private
`sealed::Sealed` trait means "only this module can name it, so only this crate can admit new
`EffectFree` payloads", and the file comment states the payoff — the dock-clamp bug class is made
UNCOMPILABLE.

> **Give authorisation identical treatment.** Writing `back[c]` requires a `WriteAuth` value that only
> the authorisation module can mint, from an in-realm owner check, a seat lease, or a verified grant.
> A future contributor adding a seventh write path cannot forget the check, because there is no way to
> obtain the token without passing one.

This matters specifically here because **§7.18 item 14 already schedules splitting
`crates/sim/src/stub.rs` (15,309 lines) along capability lines before signal systems land.** A large
mechanical refactor is exactly when a runtime `if !authorized { return }` gets moved, duplicated or
dropped, and exactly when a type-level token does not. Cost today: one sealed trait, one token type,
one constructor module, threading it through the write functions — roughly 150 lines.

### The Pass B defect — independent of authorisation, and it cuts the engines

§7.2.3's Pass B is written as:

```
for i in imported:  if now - i.last_arrival >  i.deadline:  front[i.ch] = i.fallback
```

It overwrites `front[c]` **unconditionally**, with no test for a live LOCAL writer of the same channel.
But the station-autopilot flow requires exactly that overlap: the seat and the granted station both
write `flight.thrust.fwd`, so the channel is simultaneously locally written and imported. When the
station stops — revoked, link lost, or simply finished — the import deadline (default 10 ticks =
**500 ms** at 20 Hz) expires and Pass B writes `fallback`, which §7.1.5 says *"for a thruster is 0"*.

§7.12.1 promises the opposite in bold: *"control returns on the next tick, with no re-lease handshake
and no gap."* **Both cannot be true.** This is SIG-HB's own engine-cut failure — the one §7.1.5 was
written to prevent — arriving through arbitration instead of through the epsilon gate.

> **Fix, one line: a deadline miss removes that import's contribution to `pending[c]` and marks the
> channel dirty, so `resolve` re-runs over the remaining live writers. It never touches `front[c]`.**
> Cost unchanged: still a ≤320-entry flat sweep, ≈0.16 µs.

---

## Confidentiality

### What is protected, and by what

| Property | Threat | Tool | Where §7 has it |
|---|---|---|---|
| Nobody makes my ship do something | write attack | **authorisation** — issued, revocable, edge-compiled | boundary yes, inside no (fixed above) |
| The message really came from that shard | forgery | **authentication** — one truncated MAC per bundle, 345 ns | §7.5.2 tier 3, sound |
| It is not an old message replayed | replay | fence monotone-max + the Event replay window | §7.5.4, sound but see the DoS below |
| Nobody overhears my fleet orders | read attack | **encryption** — player-built Cipher/Codec block | §7.14, decided as [7-G], correct |
| Who drove my ship | attribution | **the grant id in provenance** | §7.1.1 declares it; §7.1.2's struct sketch omits it |

### Where encryption is the wrong tool

**In-realm encryption between two blocks protects nothing from anyone who matters, and the bus
structurally cannot carry an encrypted control value anyway.** §7.5.2 tier 1: the realm's shard IS the
authority for everything in it. `front[c]` and `back[c]` are plain `Vec`s in that shard's own address
space, so a Codec block and its counterpart both run in one process with both keys in server-side
state. Another player gains nothing either: clients are not subscribers (§7.3.7), and §7.9 rule 1 is
"the server evaluates; the client draws", so a channel value never reaches a client at all.

The structural half is stronger than the argument from authority: a sealed value is by construction a
`Blob`; §7.1.4's ValueType × Lane table **forbids `Blob` on the State lane**; §7.12.1's ValueType ×
Arbitration table **forbids `Blob` under `Sum`, `Max` and `Min`**. So an encrypted value can never be a
State control channel, can never be epsilon-deadbanded (the deadband must be an integer comparison or
two shards disagree about whether a bundle exists), and can never be accumulated.

**And a cipher used AS a lock is a badly built MAC:** a shared key every receiver also holds, with no
revocation, no expiry and no audit row — plus a recorded ciphertext replays perfectly. Encryption on
the control path buys only obscurity (it raises the cost of *targeting* a channel) and must never be
counted as a control. §7.4's premise is that keys are computable offline; if anyone treats the Cipher
block as protecting integrity, the real hole reopens under a false sense of safety.

### Where encryption is the right tool — and §7.14's ruling is stronger than it states

§7.14's reasoning is correct: the relay does not hold the emitting realm's `ChannelDecl`, so
`AccessPolicy` is structurally unenforceable there; and Option B would need a distributed ACL store
outside the owning realm, which puts identity state on a node HR1 wants dumb.

**Two stronger reasons are missing.** First, HR1: an allowlist is another realm's identity state, so
replicating it onto a relay shard is a sealed-shard violation rather than an aesthetic preference — and
§7.3.6's rendezvous relay set changes membership continuously with a 16 s dual-read window, during
which a revocation must land on two relays or it has not landed. Second and decisively: §7.3.6 says
*"A Relay/Repeater block exists in the catalogue and rebroadcasts a channel onto a new net.
Player-built relay networks are content."* **A legitimate leaseholder can lawfully rebroadcast your net
onto one you do not control.** A read ACL cannot survive a legitimate reader who re-transmits. That is
true of any broadcast medium and it closes [7-G] Option B completely.

### Re-scope [7-G]: it currently answers only READ

§7.14 is written entirely about interception. The same argument that kills relay-enforced read ACLs
kills relay-enforced *write* ACLs, and §7 never writes the write half down. Meanwhile §7.10.2's
`PortBinding { port, channel: NameRef, source, curve }` lets a player type any channel name against any
port — including a relay-plane net against a thruster's `throttle`.

> **The missing sentence, and it is the single highest-value line in this ruling:
> SUBSCRIPTION IS CONSENT TO RECEIVE, NEVER CONSENT TO ACTUATE.** A Relay-plane frame that WRITES a
> channel in the destination realm is subject to the identical in-realm grant check as any other
> cross-boundary write. **The relay carries it and never authorises it.**

An implementer who reads §7.14 alone and infers "the relay does not check, therefore nobody checks"
would make the owner's takeover attack real on the relay plane specifically. It costs one line today;
discovered after players have built radio-driven autopilots, it is a behaviour change to shipped
content.

### SIG-OPAQUE-INERT — one 11-cell table that makes the invariant enforceable

An `Opaque` (sealed) channel may be bound only to:

- a `Pure{Codec}` port — the only way to decrypt;
- a `Surface` port — display it, as gibberish or as decrypted text;
- an `Emitter{SignalRoute}` port — an antenna must be able to relay ciphertext.

Typed refusal against `Actuator`, `Converter`, `Sampler`, and `Emitter{Force|Torque|Matter|Radiation}`.
Six kernel cells plus five medium cells = **11**, resolved once at `ChannelDecl::build()` exactly like
the existing 12-cell ValueType × Lane and 30-cell ValueType × Arbitration tables. Zero hot-path cost,
and the `assert_eq!`/`expect_err` shape HR5 wants.

It also promotes an existing *runtime* failure into a *bind-time* refusal: §7.1.4's `Blob` is
`TlvBlob` against a declared schema and decode-to-Default is banned, so ciphertext fed to a Nav
Computer's `waypoints` port already fails — today as a runtime error the player cannot debug, with this
table as a named refusal at wiring time.

### One more validity cell: forbid State on the Relay plane

§7.6.2 refuses State beyond the light budget — but §7.6.4 exempts the relay plane from light-lag
entirely, so the refusal never fires there. There is no Lane × Plane table anywhere. Meanwhile SIG-HB
*mandates* `heartbeat_ticks ≥ 1` on any exported State channel (default 4 ticks = 5 Hz) and it bypasses
epsilon by construction. Consequences: **65 kB/s from one channel that never changes** at 1,000 leased
subscribers, fanned out by a relay that cannot see the decl and cannot shed it intelligently; and an
unstoppable "I exist, I am running" beacon on an open net, which destroys silence discipline for anyone
who declares one. **Forbid `Lane::State` on `RoutingPlane::Relay` in `ChannelDecl::build()`** — one cell
in a 2×3 table.

### The envelope is always clear, and nothing on the enforcement path reads a value

Always clear and server-enforced: channel/net key, lane, plane, value-type tag, item size and count,
`source_tick` and `arrival_tick`, fence, `from_realm`/`to_realm`, grant id, retention class, and the
bundle MAC. That last one matters: §7.1.2's `BundleAuth` is one MAC over the *encoded body*, so it
covers ciphertext byte-for-byte unchanged and the whole authentication tier is untouched by sealing.
Everything that bounds abuse is envelope-only too — the token buckets, the 1132 B item budget, the
replay window, the timing wheel, and the derived `Telemetry`/`Control` bit computed from `subs[c]`.

**The routing plane never reads a value at any hop.** The only two things genuinely lost to sealing are
content moderation and content-based anomaly detection — a small, precisely located loss.

### Traffic analysis as gameplay — declare it

`arrival_tick − source_tick` **is a range measurement, in the clear, by design.** §7.6.1 makes the
relationship exact and integer; inverting it costs one multiply. One tick of light at 20 Hz is
**14,989,623 m**, so a single received message pins the emitter to a shell of **±15,000 km — about one
Earth diameter**. Three listening posts at known positions is trilateration. Under [7-A] Option C the
leak lives between the 2-light-second bubble and the one-light-hour horizon — exactly the interplanetary
range where hiding matters. Fractional precision improves with distance: 2.5% at the near edge, 0.0014%
at 1 AU.

This is honest physics and it is good content: silence becomes a real choice, the Directional Laser
Comm is genuinely stealthy off-axis, and routing through a relay breaks the inversion because the delay
becomes the relay's. **But it must be declared**, because §7.14 sells the Cipher block as
confidentiality and a player will reasonably assume it hides him.

**Two pricing defects that anti-traffic-analysis counters would turn into amplifiers.** §7.8.3's token
buckets count MESSAGES (200 burst / 100 per second), so padding a 40-byte fleet order to 1008 bytes is
a **25× bandwidth amplifier costing one token**. And §7.8.2 prices deliveries, not bytes, so a 1008-byte
relay fan-out to 100 leased subscribers (≈100 KB per emission) is priced identically to a 6-byte one —
on the order of **10 MB/s from a single session** at the current 100/s remote-publish bucket. If padding
and cover traffic ship, a byte-denominated bucket and a bytes × subscribers relay charge must ship in
the same slice.

### The interest-set leak, stated rather than fixed

§7.3.3 ships `InterestSet { child, keys: Vec<ChannelKey>, epoch, universe_tick }` one hop up, and
§7.3.6 grants `signal_relay` to `profiles::station()` — stations are player infrastructure and a docked
ship is a child. Keys are bare 8-byte hashes of player-typed names with no scope, and names are
conventional (`weapons.charge`, `cloak.enable`, `warp.charge`). A rainbow table over a 10⁶-name
wordlist costs 10⁶ × 51.6 ns = **51.6 ms once**. So docking at a hostile station leaks your fit.

Realm-scoping the key does not fix it — the attacker is the parent and already holds `child`. The
honest answers are all cheap and none is cryptographic: **state as a law that an exported key is public
to your parent**; keep the 64-key export cap as the real control (it is opt-in, and a ship's internal
bus is `Local` and never exported); and treat it as the sharp counterexample to any future feature that
treats a channel name as a credential.

**There is also a genuine read defect one hop up.** §7.5.4 promises read ACLs are enforceable "in-realm
and at the parent", but `InterestSet` gives the parent bare keys with no read rule, and §7.3.4's
inverted index knows nothing about privacy — so any sibling that puts a key in its own exported set
receives that channel. Fix: **one byte per exported key carrying a rule code** (`Owner | Group |
Public`), evaluated by the parent against the subscribing realm's owner. Interest set 536 → 600 B,
cluster state 14.47 → **16.2 MB (+11.9%)**, reconcile traffic 181 → 202 kB/s — still six players' worth
of proximity voice. Express it over **realm keys or opaque group ids the parent already holds, never
over `AccountId`s** — replicating account identity onto a parent shard is an HR1 leak, and §7.8.4 already
says account-tier policy "must never be readable by another realm's shard".

### The moderation tension is smaller than it looks

The Codec block runs server-side like every block, and its key lives in server-side durable state
(recommend a per-realm `key_state` redb table, a sibling of `port_state` and `config_state`, so §7.18
item 5's `on_block_removed` hook already deletes it on a break and §7.10.4's drain/spin-up path already
carries it across a reap). The operator can therefore always decrypt, and §7.14 already says exactly
this: *"The Cipher block's guarantee must be described to players as game confidentiality, not secrecy
against the operator."* So §7.8.4's `Reportable` moderation log keeps full content access; what changes
is only that other PLAYERS cannot read.

**The nonce must be derived deterministically** — from `(emitter RealmKey, channel key, seq,
source_tick)`, all four already on the bundle — because a random nonce breaks byte-identical replay.
The payoff is a property most games cannot have: the server can **re-seal a reported plaintext and
compare byte-for-byte against the retained ciphertext**, so a fabricated report fails and a genuine one
verifies.

The only thing that would genuinely break moderation is true end-to-end with client-held keys — which
also breaks "the server does all math", breaks the block model itself (a block whose function the
server cannot evaluate is not a block), and imports a legal exposure.

### One concrete crypto defect that must be fixed before anything reuses it

**§7.12.2 step 4 asks to encrypt to an Ed25519 key.** The step says the grant response *"carries the
grant id and its HMAC key material, encrypted to the station's public key from step 1"*, where step 1
carries `station_pubkey`. `ed25519-dalek 2` is a **signature** scheme with no encryption operation.

Verified dependency picture: `hmac 0.12`, `sha2 0.10` and `ed25519-dalek 2` are workspace dependencies
in `Cargo.toml:77-79`; `blake3`, `curve25519-dalek` appear in `Cargo.lock` transitively only;
**`x25519-dalek` and every AEAD crate (`chacha20poly1305`, `aes-gcm`) are absent entirely.** Converting
an Ed25519 key to X25519 via the birational map is possible and is a known key-reuse footgun. The clean
fix is a separate X25519 identity key alongside the signing key, or a signed ephemeral Diffie-Hellman —
either way a new workspace dependency, therefore a user decision, and it should be taken **once** for
the grant handshake and radio codebook exchange together.

### The replay window is a DoS on legitimate traffic

§7.5.4 declares `max_replay_entries = 16_384` per shard and — correctly — that a re-delivery past
eviction is REFUSED. But `seq` is chosen by the emitter, so every accepted message mints a resident
entry, and §7.8.3's remote-publish bucket is 100/s: **16,384 / 100 = 163.8 seconds** of one session at
exactly its legal rate fills the victim's whole window, after which eviction discards *legitimate*
peers' entries and their honest re-deliveries are refused by the design's own rule.

The mitigation is in §7.5.4's own wording but is not stated as a bound: promote *"LRU within a peer"*
to a hard per-peer quota, `max_replay_entries_per_peer = 16_384 / MAX_IMPORTED_INTEREST = 64`, so an
attacker can only evict himself. 64 is small enough that it needs a load test rather than a derivation.

Same shape one layer down: §7.8.3's `event_queue_depth = 256` is **per destination**, and refuse-newest
is right within one emitter's stream and wrong across emitters — a flooder who keeps 256 items resident
makes everyone else's message the newest. The ordering guarantee being protected is keyed per
`(emitter, channel)`, so **per-origin queues with deficit round-robin at the drain preserve it exactly**
while making the refusal fair. O(1) amortised, deterministic given a `(service_tick, origin, seq)`
service order.

---

## Relays as functional blocks

### The verdict

**Take the owner's model whole. Delete the engine relay service.** It is strictly cheaper in engine
code than §7.3.6, better fiction, and it turns the one part of the signal system that had no gameplay
into content players build, own, tax, defend and destroy.

What it deletes: rendezvous hashing (`relay = argmax_i H(net_key, relay_id_i)`), the fence-versioned
relay set, the 16-second dual-read window costing 2× relay traffic per membership change, and — most
valuable — the **migration protocol §7.3.6 itself says "must exist from the first slice, because adding
one after players have standing radio nets is a visible outage."** A player-built relay's identity is a
block at a place with an owner; it does not migrate, it is not remapped, and when it dies its routes
die visibly and locally rather than as a galaxy-wide re-subscription storm.

§7.3.6 already wanted this — *"the relay is player-visible infrastructure, not plumbing… Player-built
relay networks are content"* — but kept the routing as an engine service with the block as flavour on
top. The owner is asking for the inverse, and the inverse is the coherent version.

### The block, and why it adds ZERO runtime dispatch arms

§7.10.1's `MediumSpec` already has `SignalRoute` — *"inject something outside the block's own cell:
force, matter, radiation, or a routed signal"*. A relay is therefore an ordinary `FunctionDef` row
under the existing `Emitter` kernel, and §7.10.1's headline property holds unchanged: **adding a block
adds zero arms.**

The decomposition that keeps it that way: **relay = queue with a route cache; antenna = link.** A relay
with no antenna has no peers. That gives a real build decision (how many antennas, of which class,
aimed where) without a new concept, and it reuses the Short-Range Antenna, Long-Range Antenna and
Directional Laser Comm rows the catalogue already has.

Ports, in the existing `PortDef` shape:

| Dir | Port | Type | Meaning |
|---|---|---|---|
| in | `enable` | Bool | on/off |
| in | `power` | Fx (W) | gated by the power grid ([7-E] Option A's first deliverable) |
| in | `tx` | Blob, Event lane | what this realm wants injected |
| in | `aim` | Vec3 | directional links only |
| out | `rx` | Blob | received |
| out | `queue_depth` | Fx | how backed up |
| out | `queue_latency_ticks` | Fx | **the "time is the cost" gauge** a Display Cover renders |
| out | `route_cost_ticks` | Fx | the ETA to the currently selected destination |
| out | `peers` | Fx | link count |
| out | `link_quality` | Fx, milli-units | the field §7.6.3 already names for `sys.signal.uplink_lost` |
| out | `dropped` | Fx | shed count |
| out | `offline` | Bool | latched failure |

Line-of-sight belongs to the directional class only, tested against the closed-form celestial model
(Category A determinism), not the physics broad phase — so ~100 ns rather than §7.15's 1–10 µs probe,
re-checked at `TickPolicy::Divided(n)` because occultation changes on orbital timescales.

### One capability change, verified against the code

`crates/sim/src/capability.rs` line 87 already contains the correct idiom:

```rust
let signal_graph = req.signal_graph || req.functional_blocks;   // "Lattice derivation: functional blocks publish/consume signals."
```

but `signal_relay` is a bare grant (line 107: `signal_relay: req.signal_relay`). Verified consequences:
`profiles::galaxy()` has `signal_relay: true` with **no `functional_blocks`** — the one profile whose
whole purpose is relaying is a shard that cannot contain a block; `profiles::ship()` has
`functional_blocks: true` and **`!ship.signal_relay()`** — under the owner's model the realm kind where
players most build would be forbidden from hosting a relay.

> **`let signal_relay = req.signal_relay || req.functional_blocks;`** — one line, same shape as the
> line above it. It also matters for HR3: gating a relay block on a shard-kind-shaped capability is the
> G-NO-SHARD-FORK failure in capability clothing. The G-IDENTICAL fixture writes itself: one relay block
> forwarding one frame with identical hop count and identical `arrival_tick` on `profiles::planet()`
> (Spherical) and `profiles::ship()` (Cartesian).

### The queue, and the bandwidth that falls out of the frozen budget

A relay frame is `net` 8 + `channel` 8 + `origin` (RealmKey) 8 + `seq` 8 + `hops_left` 1 +
`emitted_tick` 8 + `service_tick` 8 + framing ≈ **52 B**, plus a typical ~100 B chat Blob ≈ **160 B**.
§7.1.3 already fixes the envelope: 1200 − 68 = 1132 B of items, so **7 frames per batch**.

The authored per-entity field is therefore `batches_per_tick: u8` on the antenna row (1..=8):

| Link class | frames/s | bytes/s | vs a 5.5 kB/s full-export realm link | vs a 30 kB/s voice player |
|---|---|---|---|---|
| Cheap (1 batch/tick) | 140 | 22.4 kB/s | 4× | 0.75× |
| Trunk (8 batches/tick) | 1,120 | 192 kB/s | 35× | 6.4× |

Queue wait is `depth / service_rate`:

| Depth | Service rate | Wait |
|---|---|---|
| 256 (§7.8.3's `event_queue_depth`) | 7 frames/tick | 36.6 ticks = **1.83 s** |
| 4,096 (relay cap) | 7 frames/tick | 585 ticks = **29.3 s** |
| 4,096 | 56 frames/tick (trunk) | 73 ticks = **3.66 s** |

**That is the design's centre of gravity: capacity investment buys an order of magnitude of latency, in
the seconds band where a player feels it and is not defeated by it.** Storage per relay is bounded and
small: a route table of `max_nets = 4096 × 24 B = 98 KB` plus a queue of `4096 × 160 B = 655 KB`,
against §7.3.4's already-accepted 1.3 MB at a 1,000-child system node.

### The hop model, the delay, and the ONE new number

> **Time as the cost needs exactly one new field in the whole design: a `propagation_c_multiple: u32`
> denominator on §7.6.1's existing formula, whose value is 1 for every physical link.**

```
delay_ticks = ceil( d_fine × UNIVERSE_HZ / (LIGHT_FINE_CELLS_PER_SECOND × propagation_c_multiple) )   // i128
```

`M = 1` reproduces today's arithmetic **bit for bit**, so planting it is byte-identical. `M > 1` is the
in-fiction quantum trunk §7.6.4 already promised. Crucially **there is no second delay mechanism**:

| M | 1 ly | Proxima (4.246 ly) | 100 ly | 1,000 ly |
|---|---|---|---|---|
| 1 (physical) | 1 year | 4.25 years | — | — |
| 10⁷ | 3.16 s | 13.4 s | 5 min 16 s | 52 min 40 s |
| **10⁸ (recommended)** | **0.316 s** | **1.34 s** | **31.6 s** | **5 min 16 s** |

At M = 1 the physical links keep §7.6's shape exactly: Earth–Moon (384,400 km) is 1.28 s = 26 ticks,
comfortably inside [7-A] Option C's 2-light-second bubble; 1 AU is 8.3 minutes. `u32` holds M up to
4.29×10⁹ with room, and the i128 numerator is untouched (1,000 ly is 1.94×10²³ fine-cell-ticks against
an i128 max of 1.7×10³⁸). **Storing M as a multiple of c rather than as an absolute speed is
deliberate:** an absolute fine-cells-per-second field at M = 10⁸ would be 3.07×10¹⁹ and overflow u64.

**What a player should experience**, composing propagation with queueing at M = 10⁸ on a moderately
loaded grid:

| Range | Path | Felt latency | Reads as |
|---|---|---|---|
| Same star system | no relay at all (Neighbourhood) | 50 ms + sub-bubble light lag | instant |
| Next system (~8 ly) | one relay each side | 2.6 s propagation + 2 queue hops ≈ **4–7 s** | a satellite call |
| Corner to corner (~1,000 ly) | ~6 hops | 5 min 16 s + ~6 queue hops ≈ **6–10 min** | mail (`RetentionClass::Mailbox` earns its keep) |

**Hours is the failure mode to design against explicitly.** A comms grid nobody uses is dead content,
and every minute past about ten converts a live medium into an inbox.

**Two invariants that make the delay honest:**

1. **A relay may only INCREASE `arrival_tick`, never decrease it.** §7.6.1 freezes the delay at emit
   deliberately to avoid *"causality inversions (a later message overtaking an earlier one)"*; a queued
   hop is the same problem one level up. The increment is `service_ticks = queue_depth /
   throughput_items_per_tick` from the relay block's own per-entity fields — no magic number.
2. **The DESTINATION recomputes the physical lower bound** from the server-authored absolute positions
   both endpoints already have, and refuses anything faster. One i128 multiply on a low-rate lane.

The second exists because a lying relay does not steal messages — **it steals the demand scheduler.**
§7.6.3 says a pending Event whose destination is dormant *"raises a demand on the destination through
the existing `RealmDemand` machinery"*, so an attacker with chosen arrival times chooses WHEN N realms
spin up. §7.6.3's mitigation charges the EMITTER, which a relay chain launders because the relay
re-emits — so the spin-up charge must follow an **end-to-end authenticated origin**, not the last hop.

### Bounding the storm

> **A hop limit does NOT bound a broadcast storm.** With `max_hops = 16` and average degree 4, one
> flooded frame still reaches 4¹⁶ ≈ 4.3 billion link traversals in the worst case. `hops_left` bounds
> depth and nothing else.

What bounds it: **each relay forwards a given frame at most once per outbound link, ever**, keyed on
`(origin: RealmKey, net: ChannelKey, seq: u64)` — which is exactly §7.1.2's `ChannelSeq`, the 24-byte
`Copy` shape chosen so `IdempotencyKey` and transitively `EffectClass` keep their `Copy` bound through
the G-SEALED conformance tests. With that plus **split horizon** (never forward back out the arriving
link), total copies of one emission are at most the number of relay links in the connected component,
once, for **any** topology including an adversarial one. Eviction reuses §7.5.4's existing
`max_replay_entries` with `Receipt::Refusal { ReplayWindowEvicted }`.

Four further layers: flooding is used only for **route discovery, never for data** (data is unicast
down cached routes); `hops_left` still caps depth; a per-link token bucket applies §7.8.3's shape at the
forwarding node; and `ChannelDecl::max_relay_hops = 0` by default means a channel is never relayed
unless it says so.

> **Plant-now sentence:** `ChannelSeq`'s `origin` field exists for relay duplicate suppression as well
> as for bundle dedup, so nobody narrows it to `(channel, seq)` to save eight bytes.

### Route discovery must be two-tier, and the arithmetic forces it

*Proactive interest flooding everywhere* is the obvious design and does not survive the numbers. Take
§7.3.4's 27,000 live realms, 10% subscribing to relay nets at ~3 nets each = 8,100 net-interests,
refreshed at §7.3.3's ≈80 s reconcile, each flood costing ~1,000 deduped link traversals on a 500-relay
grid of degree 4 at ~40 B per record: **4.05 MB/s cluster-wide** — 22× the 181 kB/s figure §7.3.4 uses
as its scaling headline, and 135× one player's voice.

*Reactive discovery* inverts it: when a frame arrives for a net with no cached route, the relay floods
one bounded route request and caches the answer for `relay_route_ttl`. At ~2,000 active conversations
refreshing every 300 s: **267 kB/s** — the same order as the reconcile traffic already accepted.

**The cost is that the first message of a new conversation pays a discovery round trip, and that is a
feature under "the cost should be time":** the HUD says "ESTABLISHING LINK" and the first message is
visibly slower than the rest. Near-field stays proactive because it is cheap and terminates most floods
before they cross the galaxy: a destination beacons `{net, destination_realm, epoch}` every ~600 s to
`max_presence_hops = 4`. Routing state is aggregated per `(net, link)` and **not** per subscriber —
`net → (link mask u32, expiry u64, cost_ticks u32)` at 24 B — which is §7.3.4's "aggregate per realm,
not per block" trick applied one level further out.

**Two named guards, because reactive routing is the classic amplifier:** a per-origin route-request
bucket much tighter than the data bucket (~1/s), and **negative caching** of "no route to X" so repeated
requests for a nonexistent destination do not re-flood.

### Ordering, durability and abuse

**Ordering: promise per-`(origin, net)` sequencing at the receiver and nothing more.** A cross-network
total order would need a root, and a root is the galaxy-wide message bus §7.3.6 already refuses as *"the
central bottleneck HR1 exists to prevent"*. Chat reorders at the receiving realm on `(origin, seq)` in a
bounded window — the Event lane's existing 64-bit sliding bitmap plus `high_seq` at 24 B per
`(peer, channel)`, reused. Distant *commands* are already designed away by §7.6.2 (a distant station
uploads a **program**, one atomic Event). The genuinely new case is a later frame overtaking an earlier
one because the route changed mid-flight; accept it, bound it with the reorder window and `LIFESPAN`,
and do not promise more.

> **Determinism requirement:** a relay's service order is `(service_tick, origin RealmKey, seq)`
> ascending and **never insertion order**, because insertion order depends on network arrival order and
> would diverge across replays — the same rule §7.15 imposes when it sorts probe results by `EntityId`.

**Durability has three cases and they deliberately get three different answers:**

| Case | Answer |
|---|---|
| Realm goes **dormant** | The queue MUST survive, and this is the normal path — an unattended relay satellite is exactly the realm §7.3.3 says is reapable in ≈2 s. Drain to a per-realm `relay_queue` redb table on drain, reload on spin-up, a sibling of `config_state` and `port_state`, moving with the realm's store on a re-home. Bounded at 4096 × 160 B = 655 KB. A queued frame stores its own `service_tick`, so the timing wheel is **rebuilt from the durable queue** on spin-up rather than being the source of truth. |
| The relay block is **destroyed** | The queue is lost, deliberately and visibly — that is the entire point of shooting a relay. §7.18 item 5's `on_block_removed` hook must therefore become a **registry** accepting a third registrant, not two hard-coded deletions. |
| The relay's host **crashes** | The queue survives to the last durable drain and no further. **Reliability is END-TO-END:** the origin re-drives on `Receipt` timeout; there are no per-hop acks and no per-hop durable outboxes. State this as a rule, because hop-by-hop reliable delivery is the intuitive design and it would put a distributed transaction on every hop — the "second distributed system with its own membership" §7.3.4 rejects. |

**Abuse is contained by existing machinery except in one place.** The mass-relay case is priced: at
§7.8.1's SE-calibrated PCU a trunk relay at ~1,600 pcu plus antennas means 62 trunk relays exhaust a
whole realm's 100,000 budget, so 10,000 relays is refused ~160× over. The gap is the *flooding* case:
§7.8.3's buckets are per session at the **source**, and `emit_rate_limit_hz` is per emitting port, so by
the time traffic reaches a third party's relay it has passed both and the relay sees one undifferentiated
stream in which one loud origin starves everyone else's queue. **Deficit round-robin over `origin` at
each outbound link** — O(1) amortised, deterministic given the service order above, no new state beyond
a per-origin deficit counter.

The remaining abuse — a hostile relay that advertises every net and then blackholes or amplifies — is
bounded by `max_nets`, by once-per-link dedup, and by accumulated `cost_ticks` being visible so a sender
can prefer another path. **Beyond that it should NOT be defended against:** a hostile relay in your path
is a game situation with two counters — build your own or destroy theirs — and that is the strongest
argument for the player-built model rather than an objection to it.

### Relays must never be principals

§7.5.2's tier-3 MAC is keyed **per shard pair**, so a relay in the middle TERMINATES it: emitter→relay
is one MAC, relay→destination is another. A player-built relay could therefore forge grant-bearing
items, which is unacceptable for infrastructure players own.

> **Keep the hop-by-hop bundle MAC for infrastructure we run (Local and Neighbourhood, where the LCA is
> our own shard), and add a separate END-TO-END authenticator on grant-bearing items that cross
> player-built relays, keyed on the grant's own key material rather than on the link.**

`grant_id` (8 B) + a 16 B MAC on write-bearing frames only, verified once at the destination (345 ns).
Cost is bounded by construction — relay traffic is the mail lane, not the control lane, and forbidding
State on the Relay plane keeps it that way — so at a bounded 1,120 frames/s that is **58 µs/s at keyed
BLAKE3 or 386 µs/s at HMAC-SHA256: 0.0004% and 0.0008% of one core.**

The property this buys is exactly what makes player-built relays safe to ship: **a hostile relay can
drop, delay, duplicate or reorder, and cannot forge.** The owner's "the cost should be time" instinct
composes for free: because `not_after` is on the universe clock rather than wall clock, **a relay that
holds a message cannot extend the authority inside it** — a delayed grant-bearing item arriving past
expiry is refused at the destination.

The honest asymmetry to state beside §7.14's read decision: an unauthenticated *broadcast* on an open
net can lie about who is speaking, but it can never make a receiving realm **act**, because acting
requires a write and a write requires the grant.

### What an engine fallback is still needed for

Three cases are usually cited and only one is real.

| Case | Answer |
|---|---|
| **Operator / system messages** | **Not a relay concern and never a signal at any plane.** They must reach a player with no ship, no antenna and no relay in range, so they ride the client protocol's control lane exactly as a shutdown notice does. Making them relay-plane traffic would mean a server notice that fails to arrive because nobody built a tower. |
| **Moderation** | **Already elsewhere.** §7.8.4 puts it on three egress chokepoints — the client Event/State egress, the SFU's per-listener consumer set, and the panel/sign draw-command composer — none of which is a relay; and §7.14 separately proves a relay structurally cannot enforce a policy it does not hold. |
| **Cold start** | **The only real one, and the honest answer is that it is a feature.** Interstellar comms should not exist until someone builds it, and building the first grid is a landmark. Where that is too harsh, seed the starter system with owned, destructible relay blocks as world **configuration**, which Addendum 1's "seed-plus-configuration worlds with a curated starter planet" already licenses. An operator wanting a guaranteed backbone builds operator-owned relay blocks in an operator-owned realm: same machinery, zero new code, and it can be attacked — better fiction than an invulnerable engine service. |

### The relay plane must be a supported ABSENCE

Every cross-realm case the owner named is `Neighbourhood`, not `Relay` (§7.3.2: a station offering
autopilot is one hop up and one down; a ship talking to its hull host is parent↔child at zero hops; a
fleet across a system is siblings at one hop). Only interstellar coordination, corp comms, radio nets,
text and voice are `Relay` — and §7.3.2 already says those *"should require infrastructure, be jammable,
and cost something"*.

> **So the entire relay subsystem can be absent and the game is complete — which makes it the second
> decoupled overlay in the design after the economy. Assert it the same way: a CI configuration with
> zero relay blocks and zero relay routes must build, run the full accumulated suite green, and produce
> byte-identical shard state.**

The value is that it forecloses the failure mode where the engine quietly grows a dependency on relay
delivery for something structural (a fleet order, a transfer saga step, a directory notification) —
exactly how the old project's cross-cutting couplings formed.

---

## The parent-physics handoff

### The owner's point is already the law, and §7 already states it

§7.13 step 6, verbatim:

> *"`ShipOutputs` leaves the realm as `InterShardFlow::Coupling(ShipThrustPort)` — an `EffectFree`,
> latest-wins sample to the hull-host shard. Per the standing law, **the containing realm's shard is the
> physics authority**: it integrates the hull rigid body and authors the ship realm's pose in its own
> frame. The ship shard never integrates its own hull."*

PLAN.md line 36 states the same dataflow at the architecture level. **Confirm it; do not re-litigate
it.** What is genuinely undesigned is the second half the owner named.

### What is missing today

§7 designs the whole child side and stops at the boundary. Step 7 returns
`HullFeedback { accel, g_load, ambient_density }` — **`ambient_density` is the only acknowledgement
anywhere in §7's 24,500 words that a parent might have a medium**, and there is no design for how the
parent computes drag from it, no statement of what changes when the parent changes, and no handoff rule
for the model transition.

Worse, the two binding documents specify **different payloads**. `docs/design/sealed_shards.md:99-101`
says `ShipOutputs { thrust_vector, torque, power_state, rcs_trim, mass_kg, com_offset }` with
`Feedback = HullPose { StampedPose, ang_vel, contact_flags }`; §7.13 step 5 says
`ShipOutputs { thrust, torque, mass, com, inertia }`. One has an inertia tensor and no power state; the
other has a power state and no inertia tensor.

**Neither exists in code.** `crates/sim/src/coupling.rs` holds only the sealed `EffectFree` marker and a
three-f64 `ContinuousSample` proof-of-shape, with the comment *"Real ports (ShipOutputs,
AtmosphereSample) land at P8 here."* That is the good news: the contract is genuinely unfrozen, so
widening it costs a design paragraph today and a wire break after P8.

### What differs by realm kind — four composable terms, not four models

> **The per-kind difference lives entirely in the parent's ambient TERM SET, which is composable DATA —
> never an enum of realm kinds, and never a per-kind CouplingPort. Vacuum is a VALUE, not a variant.**

```rust
pub struct AmbientPhysics {
    pub gravity: GravitySource,      // point masses the realm already authors closed-form from the seed
                                     //   + uniform_external: DVec3 (see below)
    pub frame:   FrameMotion,        // ω, dω/dt, host linear accel — ZEROED for a star system
    pub medium:  Option<MediumField>,// density, pressure, temperature, wind — ABSENT for a star system
    pub terrain: bool,               // are there colliders to hit
}
```

| Parent kind | gravity | frame | medium | terrain |
|---|---|---|---|---|
| Star system | point masses | zero | none | no |
| Planet realm | point mass + direction that varies with position | zero (inertial — see below) | exponential atmosphere, `H_scale` = 8,410 m | yes |
| Station / large ship interior | host's field | **ω ≠ 0** — centrifugal, Coriolis, Euler, host linear accel | uniform pressurised | yes |

Every term is derivable from data the realm already holds (`BodyPhysical { mass_kg, rotation_period,
atmosphere_scale_height }` in `crates/core/src/body.rs`). Zero terms present = free space. **Nothing
anywhere matches on a realm kind**, which is what HR3's G-NO-SHARD-FORK requires, and it is the same
shape `crates/sim/src/capability.rs` already uses.

**Critically: the CHILD's payload does not vary at all across the three.** That is the property that
makes "a ship changes parent as it flies" a non-event for the ship.

**On the planet body, "down" rotates one degree per 2.8 km of lateral travel** on the 161,671 m starter
body — which is why a flat-Earth approximation cannot be used and why gravity direction is a field
rather than a constant.

### The realm frame must be INERTIAL, and the arithmetic forces it

The tempting choice is a body-fixed rotating frame so terrain is static. It does not survive the SOI.
At the starter body (R = 161,671 m, GM = 3.544e9, 24 h rotation ⇒ ω = 7.2722e-5 rad/s), at the SOI at
~55 R = 8.892e6 m:

| Quantity | Value |
|---|---|
| Real gravity `GM/r²` | 4.482e-5 m/s² |
| Centrifugal `ω²r` | 4.703e-2 m/s² |
| **Ratio** | **1,049×** |
| Synchronous radius `(GM/ω²)^⅓` | 8.75e5 m = **5.4 R** — one tenth of the realm's radius |

> **Verdict: the realm frame is inertial and body-centred.** The planet's rotation appears (a) as a
> rotation applied when mapping an absolute position to a block address, and (b) as the atmosphere's
> co-rotating wind field — at the surface ωR = 11.76 m/s, so an unpowered balloon drifts with the ground
> exactly as it should, for free, with no fictitious forces anywhere. **A spinning station keeps its
> rotating frame**, because there its contents ARE rigidly attached to the rotation; that is the SAME
> `FrameMotion` term with a non-zero ω, not a second code path.

**Consequence, currently unmet:** `FlushSource` (`crates/wire/src/intershard.rs:696`) rebases only the
POSE, and its own doc comment records why that is safe today — *"At walk scale the dest frame is
identity ⇒ rebase is a no-op."* A station spinning for 1 g at 500 m has ω = 0.140 rad/s and a **rim
speed of 70.0 m/s**: a rebase that omits the ω×r term gives whoever crosses at the rim a free ±70 m/s,
on foot, through an airlock, repeatably. And `StubConfig::time_multiplier` (landed, default 1.0)
compounds: at k = 1.1 an undivided crossing makes a player 2× after 8 crossings and 1000× after 73.

> **The boundary rebase is a full TWIST — position, velocity, orientation, angular velocity, including
> the ω×r term and the realm time-multiplier ratio — or it is not a rebase.** Both failures are inert
> today and arm silently the first time a spinning station or a dilated realm ships.

### What the coupling must carry

**Up-flow (child → parent).** Flat core plus a closed-tag TLV tail:

| Field | Why |
|---|---|
| `thrust`, `torque` — **BODY frame**, `Fx` | A world-frame vector computed from a one-tick-stale orientation is misdirected by 2.9° at 1 rad/s. Body frame is a statement about the ship and not about the medium, which is what makes it model-independent. |
| `mass`, `com`, `inertia` | PLAN.md's set |
| **`HullShapeSummary`** — six projected areas, six area-weighted centroids, six drag coefficients, displaced volume, centre of buoyancy | ≈**248 B**. Drag is ½ρv²·C_d·A where A is the area *presented*: a 40×8×8 m hull presents **64 m² nose-on and 320 m² broadside** — a single scalar is wrong by **5×**. At 150 m/s at sea level that is 882 kN versus 4.41 MN, against a 500 t hull's 67.8 kN of weight on the starter body — drag is 13× to 65× the ship's own weight, so the axis is the dominant term, not a refinement. The **centroid** buys aerodynamic stability for free (a tail-heavy hull weathervanes), which is the difference between "drag exists" and "a player can build something that flies straight". |
| `bounding_radius_m` | §7.8.1's lead-time formula already needs it, so it pays for itself twice |

**The shape summary costs ZERO per tick.** It is computed in the SHIP shard (only the ship may see
blocks — HR1) as three 2D occupancy-bitmask reductions over the hull's own occupancy grid, recomputed
on block edit and on actuator state change, cached behind a monotone `design_epoch: u32`, shipped on
change plus a 1 Hz heartbeat, held at the sink, **and carried in the saga handoff so a fresh sink is
never without it.**

**Down-flow (parent → child).** The feedback type IS the abstraction over parent kinds, and its fields
are *derived* from §7.11's Sensors catalogue rather than invented: pose and angular velocity; **proper
acceleration** (what an IMU physically reads — zero in free fall, non-zero under thrust, non-zero under
spin gravity, correct in all three realms); the local gravity vector; ambient density, pressure,
temperature and wind; a medium composition tag; illuminance and star direction; contact flags.
≈**144 B at 20 Hz = 2.88 kB/s per ship**; the up-flow ≈128 B = 2.56 kB/s. A 200-ship hull host carries
~512 kB/s inbound — bounded and comparable to §7.1.5's 5.5 kB/s realm link.

> **The payoff sentence:** a fan-rotor block with the single data-driven transfer function *"thrust ∝
> density × throttle"* then produces full thrust in an atmosphere, nothing in vacuum, and something
> inside a pressurised hangar — from ONE table row, with no `match realm_kind` anywhere in the block
> layer. That is HR3 and HR4 satisfied by construction, and the exact analogue of §7.12.1's *"a thruster
> consuming `flight.thrust.fwd` cannot tell a human from an autopilot and must not be able to."*

**Quantise at the PRODUCER.** §7.13 step 7 puts the single float→`Fx` conversion inside the ship realm,
which keeps the conversion count at one but puts **f64 on the wire between hosts** — and the standing
rule is that every physics→control boundary is quantised to integer grids. Moving the conversion to the
parent keeps exactly one site while making the cross-host boundary integer, so every consumer sees
identical bits by construction. The parent's own rigid-body integration keeps full f64 inside its
private physics, which is where floats are allowed to live.

### The split between what the ship computes and what the parent computes is NUMERICAL

The obvious alternative — feed `ambient_density` back and let the ship fold drag into `thrust` — fails
three ways, and the first is quantitative.

Linearised drag is a damping term with coefficient `k = ρ·C_d·A·|v|`; an explicit update at
Δt = 0.05 s is non-oscillating only while `k·Δt/m < 1` and **DIVERGES above 2**:

| Craft | ρ·C_d·A·v | k·Δt/m | Outcome |
|---|---|---|---|
| 500 t hull, 200 m², 200 m/s | 49,000 | 0.005 | fine |
| 2,000 kg, 300 m², 100 m/s | 36,750 | 0.919 | loses 92% of velocity in one step |
| 1,000 kg, 300 m², 100 m/s | 36,750 | 1.84 | oscillates |
| **800 kg, 300 m² of plate, 100 m/s** | 36,750 | **2.30** | **blows up** |

A hang-glider, a parachute and a sail are all trivially buildable from 1 m blocks. A ship computing
drag from last tick's density and last tick's velocity **is** that explicit scheme, one tick late.

Second, it breaks authority: `sealed_shards.md` already requires the hull host to sanity-bound received
`ShipOutputs` against the ship's design envelope, and you cannot bound "thrust with drag folded in"
because it depends on a medium the parent owns — a buggy or hostile ship shard reporting negative drag
has invented free thrust. Third, it inverts the standing law's own wording: **drag IS ambient physics.**

> **So the parent integrates drag, buoyancy and ground contact semi-implicitly (`v' = v / (1 + kΔt/m)`,
> unconditionally stable, one divide), and the ship keeps thrust and LIFT.** Lift is not stiff along the
> velocity and, more importantly, is not derivable from a silhouette — so it belongs where the blocks
> are: a Wing or Control Surface block is an `Emitter{medium: Force}` reading `hull.density` and
> `hull.airspeed`, contributing through the existing `Sum` arbitration. The declared cost is a two-tick
> (100 ms at 20 Hz) lag on a control-surface loop, which every player-authored loop already has.

### What happens at the moment a ship changes parent

**hold-last-thrust survives the model change** — but only for the thrust term, and only because
`ShipOutputs` carries newtons and N·m in the BODY frame, which is model-independent. A destination with
a completely different term set can apply it without translation, and because it is a *force* rather
than an impulse it is also immune to the two shards running at different tick rates
(`crates/node/src/health.rs:55` explicitly contemplates "a 10 Hz cloud shard and a 50 Hz dev shard").

**What must be continuous is the acceleration attributable to the ship's OWN actuators.** Total
acceleration legitimately jumps when you enter a gravity well or a medium; demanding total-acceleration
continuity would be demanding that the physics not change, which defeats the point.

**Two holes at the flip, both fixed by the same handoff field:**

1. **Forward.** Shard B promotes, holds last thrust as designed, and has never received the hull's
   shape — so its drag term is zero or a guess. At 1,000 m/s with 200 m² presented in ρ = 1.225, the
   omitted force is **1.225e8 N = 122 MN**; on a 500 t hull for a 200 ms control-RTT window that is
   245 m/s² × 0.2 s ≈ **49 m/s of velocity error** and ~5 m of position error, injected exactly when a
   client is watching a boundary crossing. **Fix: the shape summary rides the SAME saga handoff state as
   `last_ship_outputs`, and the sink REFUSES to integrate a hull for which it has no summary rather than
   defaulting the term to zero** — refusing degrades to the already-declared `DegradedMode::CoastBallistic`,
   so no new degraded mode is invented. This is the decode-to-Default ban applied one level out.

2. **Reverse, and nobody has noticed it.** §7.13 step 7 injects the feedback onto reserved State-lane
   channels (`hull.accel`, `hull.gforce`, `hull.density`). §7.1.5 gives an IMPORTED State channel a
   `deadline` (default 10 ticks) and a `fallback`. During a hull-host transfer the feedback's SOURCE
   SHARD changes, so no fresh sample arrives — the deadline expires, `hull.density` adopts its fallback,
   which for a density is **0**, and every block gated on air cuts out: the Atmospheric Fan/Rotor whose
   *"thrust ∝ air density"* is in the catalogue, every wing, every scrubber. That is SIG-HB's own
   engine-cut failure reproduced by a transfer instead of by an epsilon gate. **Fix, symmetric and ~120 B:
   `last_hull_feedback: Option<HullFeedback>` in the same handoff state, held from tick 0 of the new host,
   and the `hull.*` channels declared hold-last across a fence change rather than deadline-expiring.**

### The SOI crossing is a free engine, and 24 bytes removes 99% of it

When the parent flips from the star system to the planet, the planet realm's gravity model naturally
contains only the planet. At the starter body's SOI:

| Quantity | Value |
|---|---|
| Planet's pull at r_soi | 4.482e-5 m/s² |
| Star's pull at 1 AU | 5.93e-3 m/s² |
| **Potential step across the boundary** `2·g·r_soi` | **105,447 J/kg** |
| **Δv per figure-eight cycle** `√(2ΔΦ)` | **459.2 m/s** |
| Cycle time traversing 2·r_soi at 1 km/s | 4.94 hours |
| **Sustained free acceleration** | **0.0258 m/s² — 19% of the starter planet's surface gravity, forever** |

That is 25–250× a real ion drive, it never runs out of fuel, and the catalogue already ships a Nav
Computer that can fly it.

> **`GravitySource` carries a `uniform_external: DVec3` supplied by the grandparent — 24 B, updated at
> realm-cascade rate, not per tick.** A uniform external field is exact to first order, so the residual
> is only the TIDAL variation: `2·GM_sun·r_soi/d³` = **7.05e-7 m/s²**, an **8,412× reduction** for 24
> bytes and one addition. The residual is still 12.5 J/kg = **5.0 m/s per cycle = 2.8e-4 m/s²**, so it
> needs a **measured gate**, not an assertion: 100 automated boundary crossings asserting |ΔE| against a
> declared `PhysicsTuning::max_boundary_energy_drift_j_per_kg`.

Saying no is legitimate (patched conics are what KSP ships) but then "your orbit changes when you cross
an SOI" must be declared as a game rule, and the free-engine number must be accepted.

### Three contract hazards to name before P8

1. **`contact_flags` is the DockClamp bug class one level down.** `HullPose` carries it, and it includes
   "landed" — which plausibly triggers a re-home into an Area realm, making authority-gating discrete
   state ride a lossy latest-wins datagram. That is exactly what the sealed `EffectFree` marker exists to
   make uncompilable. **Rule: contact flags are ADVISORY (a lamp, a HUD, a transfer function) and may
   never gate a saga; any authority consequence of touching down is a Transfer.** Likewise
   `power_state: PowerState` needs a one-line ruling — if it is an enum that gates anything, it cannot
   ride a coupling.

2. **`hull.*` is a takeover surface.** Those channels are the ship's only source of truth about the world
   it is flying in. A hostile station that can write them tells your ship it is in vacuum (every
   atmospheric fan cuts out) or inverts `hull.gravity` (flipping the sign of every player-built PID
   attitude loop). **Reserved namespace, written only by the one coupling→signal adapter, `OwnerOnly` for
   write, `grantable: false`** — the same treatment §7.12.2 step 7 gives `flight.abort`.

3. **`WeatherDragPort` is a pre-containment vestige.** PLAN.md:36 and `sealed_shards.md:127` specify it
   as a cross-shard planet→hull port. Under the landed containment model (deepest containing region is
   your realm; node-per-realm) a ship inside a planet's SOI **is contained by the planet realm, so the
   planet shard IS the hull host** — the atmosphere sample never crosses a boundary at all, and the port
   would put drag integration on the wrong side of the sealed line. The docked-station case resolves the
   same way: the station receives its own ambient context from its parent exactly as it receives its own
   pose, so the composition rule is `medium = own_medium.or(inherited_medium)`, one small field on the
   existing parent→child cascade. **Retire the port; keep `AtmosphereSample` as a TYPE; state the
   inheritance rule where the port used to be so a future reader does not rebuild it.**

### The biggest unowned item in this area is not aerodynamics

**Nothing anywhere says what SHAPE the hull host collides with.** PLAN.md P8 says the exterior hull body
is owned by the host; `sealed_shards.md` says the host integrates the hull rapier body; neither says
what collider it has, and `ShipOutputs` carries mass, COM and inertia — **no geometry at all**. Today a
ship would be a point mass to its own parent: it could not land, could not be shot, could not dock, and
could not collide with terrain. That blocks landing, docking, boarding-from-outside and all of P11
combat, and it is bigger than the aero summary.

Two rule-compliant candidates, and the choice is partly a **security** choice: a coarse voxel occupancy
shipped to the hull host hands your ship's shape, **including internal voids at that rung**, to whoever
hosts you — which at a player-owned station is a free hull scan of every docked craft. The six-face
silhouette does not have that property. See decision **D-16**.

---

## What must be planted now

Ranked by cost-if-retrofitted. **Tier 1 items touch a saved world or a positional postcard struct** —
§7.18 item 16 states the rule in the design's own words: *"adding a variant to a postcard enum later is
safe, but adding a field to a shipped struct is not (postcard is positional)."*

### Tier 1 — hard one-way doors. Answer before the first slice.

| # | Plant | Where | Cost today | Cost if retrofitted |
|---|---|---|---|---|
| 1 | **`ChannelKey = H(scope ‖ name)`, four scopes** (Realm / Construct / Net / Protocol) | `vd-core` §7.18 item 3 | Changes one function | **Every persisted player binding re-resolves to a different channel.** Hardest door in this ruling. |
| 2 | **`AccessPolicy` gains a subject; write defaults to `Owner`; the REALM owns the decl** | `ChannelDecl` | 2 B + one line in `intern()` | Saved-world migration, and player expectations cannot be tightened later (§7.14's own asymmetry) |
| 3 | **`ChannelDecl.group: ChannelGroupId(u16)`** | `ChannelDecl` | 2 B | Saved decls AND saved grants re-authored; without it grants can only enumerate and break whenever a player adds a channel |
| 4 | **`ChannelDecl.accepts_relay_origin: bool = false` and `max_relay_hops: u8 = 0`** | `ChannelDecl` | 2 B, inert until relays ship | Positional postcard — cannot be added later. **`accepts_relay_origin`'s default IS the security property.** |
| 5 | **`BundleAuth { mac: [u8;16], key_epoch: u8, grant: Option<GrantId> }`** | `crates/wire` | 9 B on a 68 B header; a State bundle drops 87 → 86 items (**1.1%**) | §7.1.1 and §7.5.3 both REQUIRE a grant id the §7.1.2 struct sketch does not have; `key_epoch` is what makes HMAC key rotation possible at all |
| 6 | **The four persisted tables** — realm owner on the metadata row §2 already writes; an interned per-realm `principals` map; an `acl` table; a `grants` table with `key_epoch` and §7.5.3's already-reserved `token: Option<TlvBlob>` | P6 storage | Four schemas | Saved-world migration with no way to determine authorship retroactively, because nothing recorded it |
| 7 | **`config_state.author: PrincipalRef(u16)`** | per-realm `config_state` | **2 B per CONFIGURED block** (a 10,000-block ship with 500 configured pays 1 KB); **ZERO on R1's 8-byte placement record**, which must not be asked to widen | Authorship becomes unknowable for every saved world |
| 8 | **One read-rule byte per exported key in `InterestSet`** | frozen wire | 536 → 600 B; cluster 14.5 → **16.2 MB (+11.9%)**; reconcile 181 → 202 kB/s | Positional postcard migration plus a galaxy-wide re-advertise |
| 9 | **`Option<RelayHeader>`** — origin `ChannelSeq` 24 B, `hops_remaining: u8`, `expires_at: UniverseTick`, `class: u8` | frozen wire | **1 B on local bundles, 34 B on relay bundles** (header 78 → 112 B; State items 86 → 83, −3.5%) | §7.3.6's own warning applies verbatim: adding this after players have standing radio nets is a visible outage |
| 10 | **`propagation_c_multiple: u32` = 1** in the link/tuning row | §7.6.1's formula | **Byte-identical arithmetic today** | A formula change over persisted `arrival_tick`s and replay fixtures |
| 11 | **The coupling contract**: flat core + closed-tag TLV tail, `Fx` at the boundary, BODY-frame thrust, `HullShapeSummary`, `AmbientPhysics` as a term set, a documented future `ReHomeState::Hull { pose, last_outputs, last_feedback, shape }` arm | `vd-core` + `crates/sim/src/coupling.rs` + `crates/wire` | A design paragraph and two type sketches — `CouplingPort` does not exist in code yet | **Frozen TWICE**: as wire AND as saga handoff state (PLAN.md's `last_ship_outputs`). `sealed_shards.md:342` schedules the freeze for P8; the reservation is free only until then. |
| 12 | **`GravitySource.uniform_external: DVec3`** | `vd-core` | 24 B + one addition | Otherwise ship a fuel-free 0.0258 m/s² interplanetary drive |

### Tier 2 — urgent by risk, cheap either way

| Plant | Why now |
|---|---|
| The sealed **`WriteAuth`** token | §7.18 item 14 already schedules the 15,309-line `stub.rs` split BEFORE signals land; a runtime check gets moved in a mechanical refactor, a type-level token does not |
| **Priority ceiling on the authorisation edge** | Today the shipped catalogue contains a block that out-ranks your seat with no grant, no boundary crossing and no check |
| **Docking BRIDGES a named channel group, never MERGES buses** | §7.11's Docking Port is specified as *"align, fuse, open resource + signal buses"*. Merging is one line cheaper and is the takeover. |
| **SIG-NO-AUTH-FROM-BUS** + revocation as a reliable discrete action | Otherwise a lost datagram means the pilot believes he revoked and did not |
| **Authorisation is durable; spin-up RE-AUTHORISES** | The ≈2 s reap otherwise undoes every revocation |
| **Pass B never writes `front[c]`** | Otherwise engines cut 500 ms after any autopilot disengages, on a channel the pilot is actively writing |
| **A relay may only increase `arrival_tick`; the destination recomputes the physical lower bound** | Otherwise a lying relay drives the demand scheduler |
| **`hull.*` reserved, `OwnerOnly`, `grantable: false`** | Before any block reads it |
| **`let signal_relay = req.signal_relay ‖ req.functional_blocks;`** | One line today; a capability-lattice migration plus a re-audit of every profile later |
| **`on_block_removed` becomes a REGISTRY** (config, port, grants, ACL, key, relay queue) | §7.18 item 5 currently names two hard-coded deletions; a destroyed Access Reader leaving a live grant is a security failure that presents as a storage leak |
| **Forbid `Lane::State` on `RoutingPlane::Relay`; SIG-OPAQUE-INERT (11 cells)** | One cell and one table at `ChannelDecl::build()`, zero hot-path cost |
| **`max_replay_entries_per_peer = 64`; per-origin Event queues** | The anti-replay control is currently a 164-second DoS on legitimate traffic |
| **`PlayerNamedPortDefaults` in `SignalTuning`, `grantable: false`** | The Seat's ports are player-grown and have no static row |
| **Reserve `RetentionClass::EnvelopeOnly`** | §7.18 item 16 already reserves the enum; taking a variant while it is unfrozen is free |

### Tier 3 — REFUSE, and record why so nobody re-proposes them

| Refused | Reason |
|---|---|
| A route or destination descriptor on the bundle | Breaks §7.1.1's *"the emitter is structurally incapable of naming a destination… the single property that makes the dormant galaxy free"* |
| Per-signal signatures | §7.5.1's measured 5,000 × 18.86 µs = **94 ms of verification for a 50 ms tick** — 1.9 cores doing nothing else |
| Any per-block endpoint announced across a shard boundary | §7.3.5's symmetric-discovery collapse: measured ROS 2 latency 5.5 ms at 5 nodes → **1,309 ms at 20**, and 94.98% packet loss at 25 |
| Pattern-scoped grants (`flight.*`) | Privilege escalation by creating a matching name — and unevaluable against a content hash anyway |
| A revocation grace period | A revocation that can be delayed is not a revocation |
| Hop-by-hop reliable relay delivery | A distributed transaction on every hop — the second distributed system §7.3.4 rejects |
| Client-held encryption keys | Breaks the moderation log, the "server does all math" law, and the block model itself |

---

## The permanent gates

| Gate | Asserts |
|---|---|
| **G-NO-NAME-AUTHORITY** *(the headline — the owner's sentence turned into a property test)* | For ANY name string a stranger can type and ANY realm he does not own, publishing produces **zero deliveries**. Driven by an exhaustive `WriteEntryPoint` enum in the `InterShardFlow::effect_class` idiom, so **adding a write path without a refusal case does not compile**. Entry points: a client input naming an unoccupied seat; a client input from a non-resident session; a foreign bundle with a valid MAC and no grant; a grant for a different channel; an expired grant; a REVOKED grant; a valid grant against a `grantable: false` channel; a replayed sequence inside the window and one past eviction; a stale-fence bundle after a re-home; a config apply by a non-owner; a docked or welded block publishing to an existing owner channel; a foreign sibling subscribing to a private key. **Anti-vacuity half: the SAME fixture with a proper grant DOES drive the ship, and the audit row naming `(grant_id, issuer, tick)` exists.** Run on `profiles::planet()` and `profiles::ship()` per HR4. |
| **G-REVOKE-ONE-TICK** | Revoke at tick N ⇒ zero deliveries at N+1. Then reap the realm, respawn it, and assert still zero — the durability half. |
| **G-BLUEPRINT-NO-WIDEN** | A pasted blueprint that widens any channel's plane, write policy, `grantable` or relay-origin acceptance beyond the realm default is refused, or requires an itemised confirmation. |
| **G-NO-AUTH-FROM-BUS** | No grant check, ACL check, lease decision or admission decision reads `front[c]`. |
| **G-ARBITRATION-CEILING** | A non-owner edge at declared priority 255 is clamped to band 0 and loses to a seated pilot. |
| **G-RELAY-STORM** | On a deliberately cyclic 500-relay topology, one emission produces at most one forward per link; `arrival_tick` is monotone non-decreasing across hops; a hop-limit exhaustion returns a typed `Receipt`. |
| **G-RELAY-ABSENT** | Zero relay blocks, zero relay routes: build green, full accumulated suite green, **byte-identical shard state** — the relay-absent twin of the standing economy-absent gate. |
| **G-COUPLING-MODEL-CHANGE** *(extends G-COUPLING-DEGRADED)* | Source = a system parent with no medium; dest = a planet parent with a medium. Assert thrust continuity; pose/velocity continuity within ε; **no drag applied before the shape summary is present**; **`hull.density` never adopts `fallback` during the flip**. |
| **G-COUPLING-ANYWHERE** | Identical ship, identical throttle, identical duration, hosted once by a system realm and once by a planet realm: the ship's OWN state (signal graph, `port_state`, aggregate outputs) is **byte-identical**; only the parent-authored pose differs. HR4's `assert_feature_anywhere` applied to the coupling, and the machine-checked form of "the physics model is a per-realm capability; the signal is generic". |
| **G-BOUNDARY-ENERGY** | 100 automated SOI crossings; \|ΔE\| against `PhysicsTuning::max_boundary_energy_drift_j_per_kg`. |
| **G-TWIST-REBASE** | Cross a boundary 100 times at a 1 g station's rim with a non-unit time multiplier; speed is conserved to a declared bound. |
| **Load assertions** (§7.18 item 13's list, extended) | Grant lookup ≈20 µs/tick at 200 bundles (**0.04%** of a tick) on top of the MAC's 69 µs (**0.14%**) — **total authorisation cost under 0.2% of a tick**; ACL and principal-table memory; per-peer replay quota; relay queue and route-table memory. |

---

## The decision register

Numbered D-1… for this ruling. Rows that re-open an existing §7 decision say so.

| # | Decision | Options | Recommendation | Cost of deferring |
|---|---|---|---|---|
| **D-1** | **Default write scope on a new channel** | (A) `write: Owner`, `read: Owner`, `plane: Local`, `Public` an explicit opt-in · (B) permissive, tighten later | **A** | **Ships the owner's attack as a default.** Saved-world door in both directions — a world saved permissive cannot be silently tightened. |
| **D-2** | **Key scoping — HARD ONE-WAY DOOR** | (A) `H(scope ‖ name)`, four scopes · (B) §7.4's `H(name)` | **A, in the first slice** | Every persisted player binding re-resolves to a different channel. Zero hot-path and zero wire cost. |
| **D-3** | **A reserved `Protocol` scope** (`dock.*`, `sys.*`, `hull.*`, `chat.*` unmintable by players) + **station identity verified BEFORE the approval dialog is displayed** | (A) yes · (B) no | **A** | Forged autopilot offers are spammable and phishable **today**, and the Ed25519 check currently happens after the player clicks approve. |
| **D-4** | **Grant scope shape** | (A) enumerated keys only — brittle · (B) hash per path segment — a hard door on `ChannelKey` · (C) `ChannelGroupId(u16)` + optional explicit keys | **C** | Must be settled before any grant row is written; otherwise every persisted grant and decl is re-authored. |
| **D-5** | **Grant conditions** | (A) `WhileDocked` / `WhileSeated` / `WhileInRealm` + a MANDATORY hard `not_after` · (B) §7.12.2's time-only `expiry_tick` | **A** | Decides whether undocking automatically ends a station's authority over your ship or you must remember to revoke — the class of thing a player gets wrong once and is furious about. |
| **D-6** | **Priority is granted, never declared** | (A) `min(declared, owner's ceiling)`, non-owner default band 0 · (B) today's `PortDef`-only model | **A** | Saved-world door: it changes what every persisted binding means. Under B the shipped catalogue contains a block that out-ranks your seat with no check. |
| **D-7** | **Revocation has no grace period; revoking an unseated flight grant cuts thrust to the safe state** | (A) confirm · (B) soften | **A, with a UI warning before confirming** | The temptation to soften will recur; a revocation that can be delayed is not a revocation. |
| **D-8** | **Ownership transfer revokes every grant and role atomically under the new owner's fence and rides the transfer saga** | (A) confirm · (B) partial | **A** | Decides whether a stolen ship is content or a brick. Under B the previous owner keeps flying his stolen ship remotely. |
| **D-9** | **Blueprint paste policy** | (A) an imported config may never WIDEN; widening is an itemised confirmation · (B) clamp silently to the realm default · (C) trust the blueprint | **A** | **C is how a shared "free fighter design" flies your ship.** A costs one walk over the bindings at paste. |
| **D-10** | **Docking: bridge or merge?** | (A) a revocable, grant-scoped BRIDGE of one named channel group · (B) §7.11's "open resource + signal buses" | **A** | B hands a docked stranger write access to your bus. |
| **D-11** | **The realm ACL's default role for an unlisted principal** | (A) Visitor = no interact, no build, no bind · (B) something more permissive | **A** | It is what makes scenarios 1 and 3 answer themselves. A gameplay-feel decision as much as a security one. |
| **D-12** | **Re-scope [7-G]** — it currently answers only READ | Restate as: radio is open for reading AND publishing; authorisation is evaluated at the DESTINATION realm's ingress, never at the relay; **subscription is consent to receive and never consent to actuate** | **Confirm the wording before the first radio slice** | §7.14's own asymmetry note applies: open→private later is additive; private→open later breaks everyone who built around secrecy. |
| **D-13** | **Game confidentiality vs true end-to-end** | (A) server holds every key; players told plainly it hides content from other PLAYERS and never from the operator · (B) client-held keys | **A, and the wording policy ships with it** | B costs the moderation log, the "server does all math" law, and the block model itself. |
| **D-14** | **Re-take [USER DECISION 7-F]** in light of the in-realm grant tree with `parent: Option<GrantId>` and the subset rule | (A) plain HMAC grant rows · (B) `biscuit-auth` | **A** — the tree gives delegation and subtree revocation without a new dependency; B's only remaining case is **offline attenuation across trust domains**, which no scenario in §7 needs | Only relevant if sub-letting control down a chain of principals becomes a mechanic. |
| **D-15** | **The key-agreement primitive** — §7.12.2 step 4 asks to encrypt to an Ed25519 key, which is not an operation Ed25519 has; `x25519-dalek` and every AEAD crate are absent from `Cargo.lock` | (A) a separate X25519 identity key alongside the signing key · (B) a signed ephemeral Diffie-Hellman | **Take it ONCE for the grant handshake and radio codebooks together** — a new workspace dependency either way, therefore a user decision | Currently the design specifies an impossible operation. |
| **D-16** | **The hull collider** — the biggest unowned item in the physics half | (A) coarse voxel occupancy at a declared rung (parry's sparse `Voxels` consumes it; ~1.9 KB for a 100 m ship at a 4 m rung) · (B) reuse the existing ghost-collider lane · (C) a convex hull | **B if the ghost lane can carry an owned body, else A.** C is wrong for any ship with a hole in it. **Note the confidentiality edge: A hands your ship's shape including internal voids to whoever hosts you — a free hull scan at a player-owned station.** | Blocks landing, docking, boarding-from-outside and all of P11 combat. |
| **D-17** | **The realm frame** | (A) inertial, body-centred; rotation carried as data on the terrain mapping and the wind field · (B) body-fixed rotating | **A** — at the SOI a body-fixed frame's centrifugal term is **1,049×** real gravity and the two balance only at 5.4 R | A spinning station keeps a rotating frame; that is the same `FrameMotion` term with ω ≠ 0, not a second code path. |
| **D-18** | **`uniform_external` gravity** | (A) 24 B, supplied by the grandparent · (B) patched conics as-is | **A** — cuts the boundary discontinuity from 5.93e-3 to 7.05e-7 m/s², **8,412×** | Under B, "your orbit changes when you cross an SOI" must be declared as a game rule, and the 459 m/s-per-cycle free engine accepted. |
| **D-19** | **Lift** | (A) per-block Wing / Control Surface reading `hull.density` and `hull.airspeed`, through `Sum` · (B) lift coefficients in the hull summary | **A** — lift is not derivable from a silhouette, and a generic lift fudge would be a magic number | Declared cost of A: a two-tick (100 ms) lag on a control-surface loop. |
| **D-20** | **Quantise the coupling at the producer or the consumer?** | (A) producer — `Fx` on the wire, integer boundary · (B) §7.13 step 7's current wording, f64 on the wire | **A** — keeps exactly ONE conversion site while making the cross-host boundary integer | Wire-shape decision on a reserved arm: free today, a protocol change later. |
| **D-21** | **Buoyancy in v1?** | (A) ship displaced volume + centre of buoyancy (32 B) with an interim occupied-cell approximation · (B) defer entirely | **A, ledgering the exact sealed-volume computation against §7.11.2's room graph** | The wire cost of a reserved field is bytes; adding one to a frozen postcard struct is a migration. |
| **D-22** | **The galaxy time budget** — `propagation_c_multiple` for the trunk relay class | (A) M = 10⁸: 1 ly in 0.32 s, 1,000 ly in 5 min 16 s · (B) M = 10⁷: 1,000 ly = 53 min · (C) M = 1: interstellar comms literally impossible | **A** | The field is config so the SHAPE is not a door, but the FEEL is: once fleets and markets are organised around a latency, changing M by an order of magnitude is a live-service incident. B converts the plane into an inbox. |
| **D-23** | **Is there any engine relay at all?** | (A) none — 100% player-built; operator notices on the client control lane; moderation on the existing three chokepoints · (B) keep §7.3.6's rendezvous relay beside player relays · (C) none, but the operator runs relay BLOCKS in an operator-owned realm | **A, with C available at any time** | B means two routing systems and brings back the migration protocol §7.3.6 says must ship in the first slice. |
| **D-24** | **Does a fresh galaxy ship with relays?** | (A) seed the starter system with owned, destructible relay blocks as world configuration · (B) nothing seeded — the first interstellar link is a player achievement | **A** (reversible: delete the rows). B is the purer design and the harsher first hour, and is not really reversible once players organise around the vacuum. | Addendum 1's seed-plus-configuration worlds already license A. |
| **D-25** | **Does destroying a relay destroy its queue?** | (A) yes — queued frames die with the block; emitters learn from `LIFESPAN` and `Receipt::Undeliverable` · (B) the queue is rescued to a neighbour | **A** | A gameplay decision about whether comms infrastructure is worth shooting at. |
| **D-26** | **Is the first message to a new destination visibly slower?** | (A) reactive route discovery — "ESTABLISHING LINK", then fast · (B) proactive interest flooding everywhere | **A** — 267 kB/s vs **4.05 MB/s** cluster-wide, so the cheap answer and the good-feeling answer coincide | Confirm the feel before it is built. |
| **D-27** | **Relay trust model** | (A) untrusted by construction — an end-to-end authenticator on grant-bearing items, distinct from the hop-by-hop bundle MAC · (B) trust the relay | **A** | Free to reserve as a wire field now; a protocol change once players have standing radio nets. |
| **D-28** | **Is the relay's cost purely TIME, or also a resource?** | (A) time only at P9, with a `power` input port declared on the row from day one · (B) power-gated immediately | **A** | Adding the gate later is data, not a redesign; B couples the comms plane to [7-E]'s power grid, which does not exist. |
| **D-29** | **Do padding and cover traffic ship as anti-traffic-analysis counters?** | (A) yes, **with a byte-denominated token bucket and a bytes × subscribers relay charge in the same slice** · (B) no | **A** | Without the pricing changes, padding is a **25× amplifier costing one token** and relay fan-out is ~10 MB/s from one session. |
| **D-30** | **Is the position leak acceptable content?** | (A) declare it — silence is a real choice; relays break the inversion; the laser comm is stealthy off-axis · (B) try to hide it | **A** | Any transmitter beyond the bubble is locatable to **±15,000 km** with three posts, regardless of encryption. Declare it before players believe the Cipher block hides them. |
| **D-31** | **May player-to-player social text ever be sealed?** | (A) no — a validated `sealable = false` on a reserved `chat.*` namespace; machine radio nets may be sealed · (B) yes | **A, with the residual stated honestly**: players can still build private chat from a Keypad, a Codec, an Antenna and a Display, and envelopes still log who/when/how much/from where | Content review is genuinely lost for that residual; state it rather than claiming coverage the design does not have. |
| **D-32** | **The cipher primitive** | (A) keystream from the existing `hmac 0.12`/`sha2 0.10` — zero new deps, ≈11 µs per 1 KB · (B) keyed BLAKE3 — ≈0.8 µs per 1 KB (**extrapolated, not measured**), one new direct dep · (C) a real AEAD (`chacha20poly1305`) — one new dep, the correct tool | **Defer to P9; reserve the sealed-blob SHAPE now; if it ships, prefer C over a hand-rolled keystream.** Never adopt unilaterally. | The 16-byte AEAD tag against `MAX_SIGNAL_BLOB_BYTES = 1024` means the sealed-plaintext cap is **1008 bytes**, free to set today and a format migration once player chat and fleet nets are persisted. |
| **D-33** | **Cut `Neighbourhood { hops: 2 }` or price it?** | (A) cut to hops = 1; cross-planet reach becomes a relay concern · (B) keep it and price the broadcast explicitly with a tight rate cap | **A** — it composes with the owner's relay ask | **As specified it cannot route**: §7.3.2 says nothing is re-advertised upward, so a grandparent has no index entry, and §7.3.4 says the emission terminates at the first hop. Either the Long-Range Antenna (a shipped catalogue block on this plane) does not work, or a 1,000-child system node does **1,000 lookups and ~13 kB per emission**, unpriced. |
| **D-34** | **Define `remote_expanded`** | (A) the parent reports fan-out width back to the child on the existing reconcile · (B) leave it | **A** | The symbol appears **exactly once** in 17,401 lines and carries §7.8.2's entire pricing claim. Left undefined, the first implementer writes `subs[c].remote.len()`, pricing a parent fan-out of K as 1 — and **human-chosen names collide at ~100%, not 2.7e-12**. |
| **D-35** | **Do the aero/geometry fields ride the per-tick sample or a change-triggered flow?** | (A) one struct, ~128 B, 2.56 kB/s per ship, no second flow · (B) change-triggered, ~75% cheaper, a second flow with its own liveness | **A unless a hull host is expected to hold well over 200 ships** | Simplest under HR3. |

**Still open and unaffected, listed so nothing is assumed:** [7-A] light-lag policy (Option C
recommended — **7-A Option A would delete both the trilateration leak and the State-refusal rule that
keeps the Relay plane honest**), [7-B] media transport, [7-C] scripting end-state, [7-D] the hot-path
MAC primitive (record it before `BundleAuth` freezes; the tag is 16 B either way so the wire does not
change), [7-E] which simulation subsystems ship (**Option C would remove the power gate that makes a
relay cost something to run**), [7-H] status-light range.

---

## Adjudicated objections

**"§7 already closes the takeover attack — §7.5.4 says so."** *Partly right, and the part it gets wrong
is the part that matters.* The BOUNDARY is genuinely well built: routing, grants, MACs and fences, three
deep, with the free one doing most of the work. The INSIDE is not built at all — §7.5.2 tier 1 says
in-realm traffic is implicitly trusted, and `AccessPolicy` is three subject-less arms with no owner, no
list, no default and no enforcement site in 17,401 lines. **The most dangerous outcome of this review
would be concluding "it's fixed" because the design contains the words "grant" and "ACL".**

**"Encrypt the signals."** *Wrong tool for the stated attack, right tool for a different one he will
also want.* The takeover is a WRITE attack; encryption is a READ control. Encrypting the bus would not
stop it by one inch. Encryption belongs on the relay plane, where §7.14 already puts it and where a read
ACL is not merely expensive but **semantically impossible**, because the catalogue ships a repeater and a
legitimate reader can lawfully rebroadcast.

**"Just check permission on every published signal."** *Refused, quantitatively.* The cheapest
conceivable check triples the subsystem's headline number (3.0% of a tick against a 1% budget); a MAC
costs **more than one whole tick**; Ed25519 costs **18.86 seconds per 50 ms tick**. Bind-time costs
exactly zero on the hot path, and the security property survives for the *same reason* the performance
property does: the hot path is edge-driven and consults no name.

**"Bind-time authorisation is enough."** *No — two independent things defeat it, and both are specific
to this project.* A pasted blueprint makes the VICTIM the authoriser, so every check returns yes; and a
realm is reapable in ≈2 s with `subs` rebuilt from persisted config, so an unbind that is not durable is
a session-local illusion. Fix both or the model is sound on paper and open in practice.

**"Keep §7.3.6's engine-chosen relay shard as a fallback beside player relays."** *Rejected.* Two routing
systems, and it brings back the migration protocol §7.3.6 itself says must ship in the first slice.
Everything a fallback is usually kept for resolves better elsewhere: operator notices must reach a player
with no antenna (client control lane), moderation already sits on three egress chokepoints that are not
relays, and cold start is answered by seeded, destructible, owned relay blocks.

**"A hop limit bounds the broadcast storm."** *False.* At `max_hops = 16` and degree 4 a single flooded
frame still reaches 4¹⁶ ≈ 4.3 billion link traversals. Duplicate suppression on `ChannelSeq` plus split
horizon bounds it to one copy per link per emission for **any** topology; the hop limit bounds depth
only.

**"Force going up to the parent needs a new design."** *No — §7.13 step 6 already says it verbatim and
the standing law says it more strongly.* What needs design is the parent side, and the answer is a
composable term set, not per-kind ports. **A per-parent-kind `CouplingPort` would be `match realm_kind`
in disguise** and a G-NO-SHARD-FORK violation.

**"hold-last-thrust already covers the model change."** *Half right.* It covers the THRUST term, because
`ShipOutputs` is newtons in the body frame and therefore model-independent. It does not cover the missing
SHAPE (the destination integrates thrust with no drag — 122 MN omitted at 1 km/s, ~49 m/s of error in
200 ms) or the missing FEEDBACK (`hull.density` adopts `fallback` = 0 and cuts every air-breathing
system). Both ride the same handoff field; both cost ~370 B.

**"The patched-conic discontinuity is a refinement."** *No.* It is worth 459 m/s of Δv per cycle =
0.0258 m/s² of free, fuel-free thrust — 19% of the starter planet's surface gravity, flyable by a
catalogue Nav Computer. 24 bytes removes 99.99%.

**"`Neighbourhood { hops: 2 }` works as specified."** *It does not.* §7.3.2 and §7.3.4 contradict each
other: interest is never re-advertised upward, so a grandparent holds no index entry for a grandchild's
key and, by §7.3.4's own rule, the emission terminates at the first hop. The Long-Range Antenna is a
shipped catalogue block on that plane.

**"§7.12.1's HUD will tell the victim who has their controls."** *It cannot.* §7.1.1 deleted the sender
entity id deliberately and gives a good reason. Both positions are individually right and together make
a hostile writer undiagnosable. The arbiter already knows the winner; publish it on an owner-only
diagnostic channel, which does not reopen the sender-id decision.

---

## Risks, stated plainly

1. **Believing the takeover attack is closed.** The mechanism is designed; the schema is not. If the
   first signal slice lands `access: AccessPolicy` with three subject-less arms and an implicit
   permissive default, the hole ships in a saved world and closing it later re-authors every player's
   wiring.
2. **SIG-EDGE rots silently.** If `subs` and `pending` are not private with `bind`/`unbind` as the sole
   mutators, one unreviewed insert anywhere reopens the whole hole with no test failing. This is a
   structural discipline, not a check — and `stub.rs`'s scheduled split is exactly when it would happen.
3. **Pass B's one line will be implemented exactly as written**, because it is short and looks obviously
   right. It presents as an intermittent 3 g dropout at the moment the headline feature disengages — a
   bug that reproduces only in the multi-writer case nobody unit-tests.
4. **Key scoping is a genuine one-way door.** Ship it in the first slice or accept that every persisted
   binding re-resolves when it lands.
5. **Postcard is positional.** `BundleAuth`, `InterestSet`, `RelayHeader` and the coupling structs are
   all bodies of the single reserved `Signal`/`Coupling` arms, so a forgotten field is a protocol version
   bump plus a fleet-wide outage, not an edit.
6. **Freezing `ShipOutputs` at P8 without the shape summary is a wire break AND a saga-state migration**,
   because PLAN.md carries it in the handoff. The reservation is free only until P8's first consumer.
7. **The Repeater laundering path is a hole a PLAYER builds**, so hardening the bus alone cannot fix it.
   `accepts_relay_origin: false` closes it by default, but a player who opens a channel to the grid and
   wires it to flight control has handed away their ship — **this is a build-time warning requirement,
   not just a field.**
8. **Explicit-Euler drag DIVERGES for buildable configurations** (ratio 2.30 for an 800 kg craft with
   300 m² of plate). Players will find it within a day of building their first glider, and it looks like
   a physics explosion rather than a design choice.
9. **The frame rebase is pose-only today and correct today.** The day realm frames genuinely move, a
   rebase omitting velocity and ω puts a ship at an SOI boundary 647 m/s off in one tick — the same
   teleport class this project has already paid for twice.
10. **`contact_flags` is a live re-entry point for the `DockState.clamped` bug class.** It will look
    harmless — "landed" is just a bool — right up to the moment somebody gates a re-home on it.
11. **Authorisation at ingress rather than at delivery** leaves up to 72,000 ticks (one hour) of queued
    authority alive after revocation. Untestable by accident; must be an asserted invariant.
12. **The relay plane's absence must stay a supported configuration** or the engine will quietly grow a
    dependency on relay delivery for something structural — exactly how the old project's cross-cutting
    couplings formed.
13. **Reactive route discovery is the classic amplifier.** Without the per-origin request bucket and
    negative caching, the relay plane becomes the easiest DoS surface in the game — and it is a surface
    players own rather than the operator.
14. **Telling players their radio is "encrypted"** invites a privacy expectation the design cannot
    honour. Treat the wording as a launch-blocking copy decision with legal edges, not a flavour note.
15. **HR5:** the bind-time authorisation path adds branching to config apply. Per CLAUDE.md's
    generic-code gotcha it must live in monomorphic helpers with generic surfaces as branchless shims,
    or the 100% region+branch target multiplies across monomorphisations.
16. **Four load-bearing §7 decisions are still open** ([7-A], [7-D], [7-E], [7-G]) and at least two move
    material parts of this ruling if decided the other way. Re-run the relay and confidentiality findings
    against whichever way they land.
