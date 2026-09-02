# Owner decisions — 2026-09-01 — VISIBILITY IS A RADIUS, NOT A TREE WALK

★ **LATER THAN EVERY DESIGN DOC AND EVERY EARLIER RULING.** Where a plan disagrees with this file,
this file wins. Read it before relaying any plan about what a player can see.

---

## V1. WHAT A PLAYER SEES IS DECIDED BY EACH REALM'S OWN RADIUS

**Owner, after a live session in which a player standing inside a ship saw a black sky:**
*"That's exactly why I wanted to use visibility radiuses of the realms, as they show exactly what
should be visible for the user, without even knowing the exact tree."*

**THE RULE.** Every realm states ONE radius: how far away it can still be seen. A realm is visible to
an observer when that observer is inside the radius. **Nothing else decides it.** Not who parents
whom, not who is inside whom, not how deep either party sits.

**What this replaces.** Until today the answer to *"can I see that star?"* required a chain of five
steps: the child reports that it holds somebody, the parent computes what that child can see, the
parent writes a vouch list, the list reaches the gateway, the gateway admits the star's drawing.
Every link can break and one did — MEASURED 2026-09-01: 96 000 drawings arrived at the gateway and
were parked behind an empty vouch list, while the composed-row count stayed frozen at 13 471.

**Under this rule the same question is one step.** No bit, no list, no vouch, no parent involved.

---

## V2. THE RADIUS IS THE DISTANCE AT WHICH THE REALM IS STILL A DOT

**Owner:** *"This radius should cover the visibility, when realm becomes a small dot (because of the
far distance). Then it can boot and starting draw itself."*

So the radius is NOT "a little way outside my surface". It reaches out to where the realm still
subtends a visible angle — which for a fixed smallest-visible angle makes the radius proportional to
the realm's own size. A star system reaches enormously far. A crate reaches metres.

**AND AT THAT RADIUS IT BOOTS.** Visible ⇒ running ⇒ it draws itself. That closes the loop the
draws-itself law opened: a realm that is not running cannot be drawn, which is WHY visibility is the
spin-up trigger.

**⇒ A DORMANT REALM IS NEVER DRAWN BY ANYBODY.** The parent-authored "one photometric marker per
dormant child" has no reason to exist under this rule: anything far enough to be dormant is outside
its own visibility radius, and anything inside that radius is running and draws itself. Removing the
markers removes the only reason a realm ever speaks about another realm — so the vouch list, the
roster admission and their whole failure surface go with them.

---

## V3. A REALM IN SOMEBODY'S AREA OF INTEREST IS NOT KILLED

**Owner:** *"we should not kill the realm when it's in somebody's AOI (occupant is inside the AOI
visibility radius)."*

Liveness follows visibility, both ways:
- inside the radius ⇒ **stay alive**, whoever the observer is and wherever they sit in the tree;
- outside it, with no occupants and no live child ⇒ shut down after the cooldown.

This does not relax the existing liveness law — a realm still decides its own shutdown by looking at
itself and one level down. It ADDS that being seen is itself a reason to live, and that reason is not
routed through a parent.

---

## V4. ONE MACHINERY, EVERY REALM

**Owner (emphatic):** *"Machinery should be applied to ALL REALMS GENERICALLY."*

No branch on realm kind anywhere on this path. A galaxy, a star system, a planet, a moon, a station,
a player-built ship and a crate all state a radius by the same rule and are made visible, booted and
kept alive by the same code. A test of "what kind of realm is this?" on the visibility path is a
defect, exactly as it is on the crossing path.

**A ship built by a player is not a special case and gets no special path.**

---

## V5. THE OCCUPANCY BIT SURVIVES, WITH A NARROWER JOB

**Owner:** *"also agree with occupancy"*.

The one bit a child reports upward — *"somebody is inside me"* — stays. It keeps realms alive and it
keeps the proxy law working for liveness.

**It no longer decides what anybody can see.** That was the load it was carrying today, and that load
is what made a single missing bit produce a black sky.

---

## WHAT THIS DOES NOT DECIDE

- **Who runs the radius test.** The composer inside the gateway is structurally forbidden from
  evaluating a position — it may only fold statements a realm attests about itself, and that guard is
  what keeps a picture honest. So the test must run where positions lawfully live. The demand loop
  already runs exactly this test to decide spin-up.
- **Cost.** The reversal that produced the placement law warned that a picture composed per observer
  costs contacts × observers, and the sky measured that fatal at 150 000 rows per player per tick.
  The radius test must remain a lookup and must never become a scan over every realm. A cost that
  grows with the number of realms is a defect, and it must be measured on a world with many.
- **What a client does with a realm that is visible but far.** Detail level is a separate question.

These are owed, and they are design work — not licence to keep the tree rule.
