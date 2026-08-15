# Owner decisions — 2026-08-15

The owner walked 13 decisions on 2026-08-15. This file is the in-repo source of truth for
those rulings: approval citations (wire ledger entries, DEFERRED rows, test doc comments)
point HERE. Items are numbered 1–11 as walked; item 7 carries three rulings (the two
retroactive wire approvals, the approval-citation gate, and the ask-first re-emphasis),
so the register holds 13 rulings in total.

## 1. SL1 self-placement filter stays; the window lane / observer chain is next

The filter that keeps a realm from ever hearing its own position STAYS. The next
prioritized slice is the WINDOW LANE (observer chain): the parent authors ONE row for its
own body per occupied child, plus per-hop transform rows; the gateway stacks those rows
per observer; the client stays a passthrough. In that slice the `--realm-boxes` boot file
is REMOVED. Acceptance: warp is visible in pixels, with no flicker at the
star-map↔live handover.

## 2. Directory head-reads confirmed; mTLS at cloud

Reading the directory head where the code reads it today is CONFIRMED as correct.
Authenticating those reads is handled by the cluster's mTLS at cloud deployment — no new
in-process auth machinery.

## 3. Fail-closed drop confirmed, with two protected couplings

Dropping undeliverable data instead of guessing is CONFIRMED. Two couplings are protected
as law: (a) retention is FOREVER DERIVED — at least two cadences plus one — never a free
literal; (b) drop-lanes carry FULL-STATE signals only. A delta riding a drop-lane is a
defect, not a tuning choice.

## 4. One-way demand gate confirmed

The demand gate stays one-way and GENERIC: demand flows toward the orchestrator, spin-up
flows out of it; no realm-kind special case may enter it.

## 5. Two-level bound + generator visibility check approved

The two-level bound (a realm draws itself; its parent draws only placements one level
down) is approved, and the world generator gains a VISIBILITY CHECK that turns the bound
into a proof: no body two or more levels deep may ever be visible (angular size at or
above the interest band's threshold) from outside its ancestor, at worst-instant
positions. Same worst-instant machinery as the existing reach guard; the threshold is
read from the ONE source the interest band uses — never a second literal. Implemented
2026-08-15 in `vd-physics` worldgen validation.

MEASURED 2026-08-15 ("see what will happen"): THE world FAILS — every planet of both
ring-placed systems stays visible from just outside the galaxy shell (a 3.954 m planet
is visible out to 302.06 m; the galaxy surface passes within 161–281 m of their
worst-instant positions). Per the ruling, no world number was changed; the check stays
test-only (the pinned measurement test in `vd-physics` worldgen) and is NOT wired into
the boot fence until the owner rules on the galaxy shell / threshold / planet extent.

**Owner addendum (2026-08-15) — the re-solve ruling.** Re-solve THE world's numbers so the
check passes, under three binding constraints:

- **(a) The generator SOLVES the margin — a general constraint, never a hand-tuned
  number:** for every ancestor `p` and every descendant `g` two-or-more levels below,
  `R_p ≥ worst_instant_dist(g in p) + r_g + visibility_range(r_g)`. Written as a
  derivation the generator applies when placing children, so ANY future world re-solve —
  including the owed near-real-scale world — satisfies it automatically. Direction:
  prefer pulling the ring placement inward (galaxy unchanged); if that collides with an
  existing derived constraint (inter-system spacing / warp-gap compression / the ±Z
  corridor laws), grow the galaxy shell instead — whichever falls out of the constraint
  solve, NEVER a hand-picked value. The constraint algebra is SCALE-INDEPENDENT: at
  near-real scale it must be trivially satisfied, not accidentally binding.
- **(b) Today's world is interim IN SCALE ONLY** — nothing may over-fit to today's
  magnitudes; every new bound is expressed in terms of extents/thresholds, never
  literals.
- **(c) World generation is ALGORITHMS FROM SEED ONLY** — any file-read or hand-authored
  table found on the generation path is REPORTED to the owner, never fixed unasked.

Implemented 2026-08-15 in `vd-physics` worldgen: `two_level_clearance_m` /
`galaxy_shell_r_m` solve the shell as the ring plus the LARGER of the containment
headroom (two system SOIs) and the worst descendant's two-level visibility clearance.
The solve GREW THE SHELL (12 331.40 m → 12 483.46 m): the ring could not move inward
because it already sits at the wake law's lower bound (`system_soi · cot(θ/2) · slack` —
a star must be asleep at departure), which is exactly the ruling's stated fallback. The
guard is now WIRED into the shard boot fence beside the nest guard (a violating world
refuses to boot), and the failing measurement test is flipped into the green pin
(`the_two_level_bound_re_solved_on_the_world_…`; the pre-solve offence numbers survive
as history in its doc comment and as a live measurement in
`the_guard_refuses_a_shell_that_hugs_its_ring`).

## 6. Heartbeat with fast-alive / lazy-empty asymmetry confirmed

The liveness heartbeat stays, with its deliberate asymmetry CONFIRMED: going alive is
FAST (a realm must be kept warm the moment anything needs it), going empty is LAZY
(cooldown-gated) — hysteresis, so demand flapping never tears a realm down under a
returning occupant.

## 7. Retroactive wire approvals; the approval-citation gate; ask-first (three rulings)

- `InterShardFlow::RealmShapeObservation` (wire minor 11) and `InterShardFlow::ShardRoster`
  (wire minor 9) are RETROACTIVELY OWNER-APPROVED, 2026-08-15. The shape lane is INTERIM:
  its content evolves to self-authored looks once the observer chain (item 1) lands.
- The approval-citation build gate is APPROVED: from minor 9 upward, every wire version
  ledger entry must carry an owner citation, enforced by a test that fails the build
  (see item 5 of the 2026-08-15 work order; the gate lives beside the version pins in
  `vd-wire`).
- ASK FIRST is re-emphasized: SL6 stands — before any new data crosses a realm boundary,
  and before any new wire arm, ask the owner. Default NO.

## 8. Coverage gate: option 3

The coverage gate takes OPTION 3: the pass/fail decision drops report rows that have no
source-line mapping (the duplicate-instantiation tool artifact class), by an OBJECTIVE
rule — never a blessed location list — printing the dropped count every run; every
remaining real miss still fails. Locatable real misses are covered or lawfully exempted;
no assert is ever weakened. See the DEFERRED.md row and `scripts/coverage_gate.py`.

## 9. Scene origin = explicit gateway-stamped origin marker

The scene's origin becomes an EXPLICIT gateway-stamped origin marker. `RealmShape.center`
leaves the wire in ONE CUT together with the observer chain (item 1) — there are no old
clients, so no shims and no transition period.

## 10. THE DRAW LAW

Every pixel has a lawful author: a RUNNING realm draws itself; a NOT-RUNNING realm
appears ONLY as its parent's placement marker; there is never a third source. Enforced
STRUCTURALLY, not by care. The star map is the galaxy's own 3D parallax map of real
placements — dormant stars get no servers.

**Owner addendum (2026-08-15):** the item-5 re-solve addenda (b) and (c) apply here with
the same force — the draw law's geometric guarantee (the two-level visibility bound) is
solved from extents and thresholds, never from today's magnitudes, and never from an
authored table: today's world is interim in scale only, and generation stays algorithms
from seed.

## 11. Demand provenance: measure now, enforce at cloud

When a demand names realm X, the lawful senders are the process the directory head shows
holding X's parent, or the process holding X itself (own-coord keep-alive). Today the
orchestrator MEASURES: mismatched or unresolvable senders are counted and warned, and the
demand is processed UNCHANGED — the warm-ahead path must never eat a blind window.
Refusal is deferred to cloud deployment (where mTLS names the sender). Implemented
2026-08-15 as the orchestrator's `demand_sender_mismatch` counter.
