# MONEY AND MARKETS — the design (2026-07-27, REVISED 2026-07-27: accounts-only)

> ## ⚑ REVISION 2026-07-27 — ACCOUNTS ONLY, NO COINS
>
> **The user asked:** *"Would that be easier if we have only bank accounts associated with the player? No
> coins."* **The answer is yes, with one condition and one bill, and it is recorded in the new §0.**
> §0 is the current word; everything from §1 onward was written for the two-form design and has been
> amended in place. **Every amendment carries the marker `[REV 07-27]`.**
>
> **The two-form design is NOT deleted.** Physical coins are demoted from "the other half of the design"
> to **a later, optional, additive extension** (§0.6 lists exactly what must be shaped now to keep it
> additive). Every passage that survives as the later option says so explicitly.
>
> **The revision was checked by three adversaries against the real code.** Their verdict, what was
> sustained and what was rejected on the merits, is the review record at §0.7. **One of their findings
> overturns a load-bearing claim of §3 that has nothing to do with coins** — the money supply never
> self-corrected, and the death drain was masking it. That is §0.4 and it is a negative, stated as one.

> **What this is.** One document covering both the money layer and the consignment markets that use it.
> It **supersedes the abandoned consignment-market design** (started, deliberately abandoned when the
> user reversed its premise) and is the current word on both.
>
> **Why it exists.** On 2026-07-26 the user chose *"the currency is a material"* — goods as money. On
> 2026-07-27 they **reversed it**: *"having money as items is not a very scalable idea — we will not be
> able to transfer money anywhere, we will not be able to have banks or similar. We need to have currency
> and some ways to deal with the money locally (stations/cities/star systems) and globally (galaxy and
> universe levels)."* This document answers that.
>
> **What is unchanged.** The economy is optional. It is purely player-driven — no computer traders, no
> formula prices, no background market evolution. Nothing is charged by elapsed time; a time-based rule
> is legal only if evaluated on touch. An economy object is never the container of record for goods.
> The market shape is physical consignment with a per-sale venue fee. Only *how a buyer pays* changes.
>
> **Status.** Design-grade, vetted by three adversarial reviewers against the real code. Every sustained
> finding is folded into the substance below; the four I did not sustain are named in §12.3.
> **Where this document contradicts the code, the code wins** — and every such case is called out.
> **Nothing here is built.** `grep -rn "Wallet\|CurrencyId\|Ledger\|AccountStore" crates/` returns zero
> hits outside `AccountId`. That is the good news: every decision below is free today.
>
> **Dated reconciliation notes have been added to the three prior economy documents.** They are not
> rewritten; each carries a note saying exactly which of its parts this reverses, weakens, or leaves
> untouched.

---

## 0. THE ACCOUNTS-ONLY REVISION (added 2026-07-27)

### 0.1 The answer, in plain language

**Yes — but not in the way it sounds, and it comes with one condition and one bill.**

**What gets easier is the order of the work, not the amount of code.** The money layer itself barely
shrinks. The book, the proof that money is never created or destroyed, the market, the fee arithmetic and
the minting were always the bulk of it, and all of them stay. What shrinks — by more than half — is the
pile of *other* work that had to land before any money could exist at all. Carried money needed four
separate defects fixed in machinery that has nothing to do with money, and it needed every place in the
world to be given a disk it does not have. With no money in anyone's pockets, none of that stands between
us and a working economy. Money stops being the last thing the roadmap can reach and becomes something
that can be built on a real login service, two new messages and one deployment.

**What it deletes.** Bank branches lose their entire reason to exist — walking somewhere to make your
money weightless is meaningless when it was never heavy — and bank notes go with them, because they were
solving a problem that only physical money created. The support answer "your money died with the server,
and the answer is no" disappears. And for the first time the game can add up every unit of money in
existence while it is running, instead of only inside a test, which turns "money is never lost" from a
hope into something we check.

**What it costs.** Three real things.

*Paying stops working when you are out of contact.* Today's design lets a station cut off from the whole
universe run a full cash market. That capability goes. What survives is barter — you leave goods, you say
what you want for them, a buyer turns up with it — and barter only survives if we keep one decision: **a
price is a short list of things wanted, and an amount of money is merely one entry on that list.** Keep
that and a cut-off station still trades. Drop it and a cut-off station trades nothing at all.

*Money stops being loot.* Kill someone and you take their ship and their hold; you never take their
savings.

*And the sale gets more complicated everywhere.* The design deliberately made the simple sale the normal
one — one write, one instant, cannot half-happen, works with the economy switched off entirely — and the
complicated one an exception for big-ticket items. Removing cash deletes the simple one. The complicated
one, which this document itself calls the only place where a crash leaves a player looking at a
half-finished purchase, becomes the only sale there is. It now has to be built properly rather than
offered as an option.

**And one thing the question did not ask about, which matters more than any of the above.** Losing money
when you die was one of only two ways money ever left the world. Removing it exposed something that was
already wrong: **the money supply in this design does not look after itself, and it never did.** The whole
claim rested on people being able to hand money back and get the metal out whenever there was too much
money about. But the same document also decided that the metal must be useless for absolutely everything
else — and handing back ten units of money gets you enough metal to make nine. So handing money back is a
guaranteed loss no matter how much money exists, nobody will ever do it at any price, and the safety valve
was welded shut by a different decision written a page later. Losing money on death was quietly covering
for that. Take it away and there is essentially nothing removing money at all.

That is fixable, the fix is cheap, and it is cheap *because* there are no coins: with everything in one
book, we can take a small slice out of every payment and destroy it, measure exactly what that does while
the game is running, and turn one number. Physical money could never be counted while the game was
running, so it could never be managed.

**The recommendation.** Adopt accounts only. Keep physical money written down as a later addition, and
shape a small number of things now so that adding it stays cheap. Make every payment a round trip to the
money service, with no local spending allowance anywhere — that is the condition, and it is the thing that
keeps the economy from ever touching the machinery that switches regions on and off. Drop the claim that
the supply looks after itself, and replace what death was taking with a small destroyed fee on every
payment. Settle what the money is made of before terrain is generated, and give it one genuine use, so
that handing money back is worth doing.

---

### 0.2 THE CONDITION, stated as a rule

> **RULE (binding). No shard, region, venue or client may ever hold a spending allowance, float, till,
> credit line or cached balance. Every payment is a synchronous round trip to the account service. A world
> server never holds money; it only forwards a refusable request and applies the answer.**

**Why the rule exists.** The reviewers who rejected the earlier accounts-only sketch rejected it for a
specific reason: it delegated a local allowance to each shard so that purchases would not need a round
trip. An allowance is money held on a diskless machine that is killed as routine, so a region could not be
switched off until it settled — which puts economic state on the region-lifecycle path (forbidden by the
standing decision), and under a network split the wait is **unbounded**. Drop the allowance and that
coupling disappears entirely. This is the same reasoning that rejected the shard-local Empty-suppression
stopgap at §6.3 and §12.3, applied one level up.

**Verified: with no allowance, the lifecycle genuinely never learns the economy exists.** Walked in the
real code:

| Step | Where | What it reads |
|---|---|---|
| `empty_confirmed` | `crates/sim/src/rlm.rs:437-448` | the shard's Empty self-report and nothing else |
| the Empty self-report | `crates/sim/src/stub.rs:4494-4521` | occupancy only — owned simulating dots plus held transients |
| `desired_alive` | `crates/sim/src/rlm.rs:451-465` | `demanded_recently \| (running_live & !empty_confirmed)` |
| `teardown_ready` | `crates/sim/src/rlm.rs:465-489` | the five `TeardownFacts` at `:377-388` — running_live, desired_in_closure, has_desired_descendant, in_transfer, quiesced. **None economic.** |
| the drain veto | `crates/sim/src/rlm.rs:555-572` | `teardown_drain_ticks = tick_hz/2` = 25 ticks = **500 ms** at 50 Hz |
| the kill | `crates/node/src/rlm_runtime.rs:281-294` | `kill_realm` then revoke |

An in-flight payment **cannot** enter that decision, because the Empty report has no way to see one. Under
partition it is identical: the reconciler is level-triggered from the orchestrator's own ledger and
directory heads, so a shard that cannot reach the account service still spins down normally. **The
coupling is closed by construction, not by discipline.**

#### The two residues that survive even without an allowance

**Residue 1 — the money moves and the goods do not (bounded, must be designed in from slice one).**
If the venue's region dies between the service applying the debit and the goods changing hands, the buyer
has paid for nothing. Conservation is not violated — the money reached the seller or a hold — but it is a
support case, and it is one purchase per in-flight payment.

> **Fix, and it is a data-model decision rather than a later addition: the reserve is a HOLD at the
> service, never a payment straight out.** The buyer's balance moves into a held position keyed by the
> payment correlation id; the seller is credited only by the settlement that also moves the goods; the
> hold is released by **the buyer's own next login** touching their outstanding reservations (§6.4). The
> account row must therefore carry a held field and a monotonic stamp **from the first version** —
> retrofitting an escrow shape onto live balances is a money migration.

**Residue 2 — money leaves a killable place, but GOODS DO NOT, and neither does the backing vault.**
Accounts-only removes value from a region that reports Empty and is killed with no flush. It does **not**
remove consignment cargo, a venue's stock, want-ad escrow, loan collateral or **the physical vault of
backing material** from that region. So the abuse at §6.3 — mint at a station about to be reaped and the
backing evaporates — **survives accounts-only, and is arguably sharper**: the account credit is durable
while the vault behind it dies, leaving currency that is permanently unbacked.

> **Consequence: "no value in a killable place" drops as a MONEY blocker and stays as a MARKET and MINT
> blocker.** Per-place durable storage keyed by realm identity (P5 / D-87) is unchanged in scope and still
> gates markets and mints. Dropping coins does not buy a schedule win there; it only changes what dies
> from cash to cargo.

**Residue 3 — the payment request now originates on a diskless machine 100% of the time, not 2%.**
A shard-originated payment is side-effecting and classifies as `ProducerLessReliable`
(`crates/wire/src/intershard.rs:313-317`), which trips a **debug-only** assertion unless pushed `Retained`
(`crates/sim/src/runtime.rs:86-97`), and `Retained` needs a durable outbox that no manifest enables
(`VD_OUTBOX_PATH` appears in zero deploy files; `spawn_anchor_keys`,
`crates/bins/src/lib.rs:2203-2226`, carries **no storage key at all**). §2.4's escape still holds — the
shard forwards a **re-assertable player intent** whose loss is a failed purchase the player can retry, and
the durable re-driven half lives at the service. But under accounts-only that escape covers **every**
payment rather than one in fifty, so it must be **decided explicitly and written into the arm's contract**,
not noted in passing.

#### The round trip, costed honestly

Deployed tick rate is **50 Hz / 20 ms** (`deploy/k3d/10-configmap.yaml:15-16`, cross-checked at shard boot
by `validate_tick_pair`). Each node drains inbound at the top of its tick and flushes outbound at the end
of the same tick (`crates/node/src/app.rs:163-166`, drain at `:206`), so a message arriving at a uniform
point in a tick waits U(0, 20) ms: **mean 10 ms, max 20 ms per hop.**

| Leg | Mean | Worst |
|---|---|---|
| shard emits at its own tick end | 0 | 0 |
| wire | ~1 ms | ~1 ms |
| service hop + apply inside the existing shared per-tick fsync | 10 ms | 20 ms |
| wire | ~1 ms | ~1 ms |
| shard hop, then the goods move | 10 ms | 20 ms |
| **server-side total** | **≈22 ms** | **≈42 ms** |
| + client snapshot wait at 20 Hz (`crates/io-prod/tests/mesh_snapshot_latency.rs:91`) | 25 ms | 50 ms |
| **player-visible** | **≈47 ms** | **≈92 ms** |

The standing 100–150 ms interpolation buffer applies to every action equally and is not additional to the
purchase specifically. **For contrast, a durable crossing's floor is nine acked phases at 20 ms = 180 ms,
and a player pays that on every dock, undock and region change. A purchase costs 42/180 = 23% of one
region crossing.** This supersedes §9.2's "9× faster on 10% of the messages": it is now 4× faster on
100% of them. **Acceptable, and not close.**

**Volume, at the planning population.** Baseline from §6.4: ~9,600 shop interactions/day at 1,000
concurrent = 9.6 per concurrent player per day. Under accounts-only **all** of them are round trips
(vs 10% before):

| Concurrent | Payments/day | Per second | Per tick | 10× prime-time peak |
|---|---|---|---|---|
| 400 | 400 × 9.6 = **3,840** | 3,840/86,400 = 0.044 | 0.00089 | 0.44/s |
| 1,000 (doc planning) | **9,600** | 0.111 | 0.0022 | 1.11/s |
| 2,000 | 2,000 × 9.6 = **19,200** | 0.222 | 0.0044 | **2.22/s** |

Against §9.3's sizing — 1,000 payments in one tick = 2,000 rows × ~113 B = 226 KB in one fsync, i.e. a
ceiling of 1,000/tick × 50 = **50,000 payments/second** — the headroom at the top of the planning range
during a peak is 50,000/2.22 = **≈22,500×**. Volume is a non-issue and must be presented as one.

**The only two shapes that break this, in order of likelihood:**

1. **One round trip per ITEM instead of per BASKET.** A 100-item purchase becomes 100 × 42 ms = **4.2 s**.
   *Rule: one payment per basket, never per line.*
2. **Routing a payment through the directory transfer lock.** `lock_transfer`
   (`crates/sim/src/directory.rs:530-541`) permits one in-flight saga per key and a second attempt returns
   `false` and is **silently skipped, not queued**; at the 180 ms crossing floor a hot venue account caps
   at 1000/180 = **5.5 payments/second with the excess dropped**. §2.3's prohibition is unchanged and now
   applies to every payment in the game.

---

### 0.3 The deletion ledger

Counted against this document's own inventory. **Nothing is deleted silently — every row below is applied
in place downstream with a `[REV 07-27]` marker.**

| Inventory | Total | Dropped | Simplified / narrowed | Unchanged | Newly needed |
|---|---|---|---|---|---|
| Shape-now items (§9.6) | 14 | **0** | 3 reduced to contract text | 11 | +3 (§0.6) |
| Prerequisites (§9.5) | 14 | **5** | 3 narrowed | 6 (1 newly critical) | 0 |
| Gates (§6.5) | 21 | **6** | 2 reclassified, 2 simplified, 1 replaced | 10 | +2 |
| Ledger entries (§10.3) | 17 | **5** | 1 reclassified, 3 narrowed | 8 | +2 |
| Build slices (§10.1) | 8 | **2** | 3 | 3 | 0 |

**Prerequisites dropped as economy blockers (5).** P1 durable applied-once (rescoped into the service's own
store, ~150 lines, no longer a shard problem); P2 the promote/crossing-producer fixes; P3 the rescue
payload's state arm (dies entirely — there is no carried value to rescue); P4 the per-kind carried-state
seam; P13 decide coin mass (dies entirely). **P2 and P4 are still owed as GAME work and keep their
severity** — see the review record at §0.7, which is emphatic about this.

**Prerequisites narrowed (3).** P5 per-place durable storage (money half gone; goods-and-vault half
unchanged and still blocks M-3/M-4 — residue 2 above); P7 items and containers (goods only); P11 the
applied-once retention bound (service-side only, and much smaller for it).

**Prerequisites unchanged (6), one of them newly critical.** P6 a real login service, P8 a recipe system,
P9 the two reviewed arms, P10 the separate deployment unit, P12 decide the backing material, P14 one or
several currencies. **P6 becomes the single most critical prerequisite in the document**: with no coins an
account *is* a player's entire net worth, and the account principal is the only thing protecting it.

**Gates dropped as money gates (6).** G-3 the double-apply cell, G-4 the bank-then-rollback cell, G-7
rollback-loses-never-credits, G-11 mixed-epoch three-hop movement, G-12 unknown-tag round trip, G-20 note
conservation. *(G-11 and G-12 survive as carried-state gates owed by the seam work, not by the economy.)*

**Gates reclassified out of the economy but still owed as game gates (2).** G-5 promote-defers-then-journals
and G-8 epoch-mismatch-zero.

**Gate replaced (1). G-13 becomes unachievable and inverts.** "Complete a full consignment sale with the
service down and the station partitioned, zero bytes leaving the shard" cannot pass without cash. It is
replaced by: **a partitioned venue refuses currency payment cleanly, falls back to a barter bundle, and
stalls nothing — no timeout, no stuck saga, no failure to spin down.**

**Gates simplified and strengthened (2).** G-1 conservation becomes a **production** property rather than
a harness-only one; G-19 load drops the coin terms.

**Ledger entries dropped (5).** D-85 value-safe rollback; D-86 epoch-mismatch-money-unsafe; D-91
production money-supply telemetry (**a clean structural win — this was the entry that risked putting
economic data on a lifecycle message, and all money is now locally summable, so it evaporates**); D-95
bounded purse; D-96 coins as world objects (**demoted to the door-open list, §0.6**).

**Ledger entry reclassified (1).** D-84 promote-journals-before-effect: **stays 🟥 and stays BLOCKING**, but
as a game-correctness defect, not an economy blocker. See §0.7.

**Ledger entries narrowed (3).** D-87 no-value-in-a-killable-place (goods and vault half stands); D-94 the
carried-state contract addendum (contract text with no current subject); D-99 the applied-once retention
bound.

**Build slices deleted (2).** M-2 carried purse and branches; M-5 notes.
**Build slices simplified (3).** M-0, M-3, M-4. **Unchanged (3).** M-1, M-6, M-7.

#### Revised cost, against this document's own estimate

| | §9.1 as written | Accounts-only | Δ |
|---|---|---|---|
| Economy product | 2,580 | **≈2,250** | **−330 (−12.8%)** |
| Economy test | 5,550 | **≈4,500** | **−1,050 (−18.9%)** |
| **Economy all-in** | **8,130** | **≈6,750** | **−1,380 (−17.0%)** |
| **Blocking prerequisites** | **6,900** | **≈3,000** | **−3,900 (−56.5%)** |
| **Total** | **≈15,030** | **≈9,750** | **−5,280 (−35.1%)** |

Product derivation: amount type 250 unchanged; wire arms 220 unchanged; account operations 600 → **650**
(hold/release becomes load-bearing on every sale); ledger family 450 → **600** (the service's own durable
applied-once table, ~150, moves in from the prerequisite bill); venue 420 → **390** (coin sale, strongbox
and proceeds box out ≈ −150; universal two-sided lock and service-held want-ads in ≈ +120); mint/melt 180
→ **140** (credits an account, produces no objects); branch affordance 180, carried purse 120, notes 160
**deleted**. Sum 250 + 650 + 600 + 220 + 390 + 140 = **2,250**.
Test derivation: branches 300, purse 300, notes 300 deleted; harness 1,200 → 800 (six gates dropped, one
added); account ops 900 → 950; ledger 700 → 950. Sum = **4,500**.
Prerequisite derivation: 6,900 − 900 (applied-once rescoped) − 600 (rescue arm) − 2,000 (carried-state
seam) − 400 (promote fixes) = **3,000**, which is P5 per-place durable storage, unchanged, because goods
and the backing vault still die on a routine teardown.

> **Verdict on "accounts-only deletes the expensive half": TRUE for the schedule, FALSE for the code.**
> The blocking prerequisite bill more than halves and the money service can ship on a real login service,
> two appended arms and one deployment unit — none of which depend on the four defects. The economy proper
> shrinks by about a sixth. **These are this document's own unvalidated estimates and nothing is built; the
> deltas are exactly as trustworthy as the baseline and no more. The reliable claim is the ordering one.**

---

### 0.4 THE SUPPLY ANSWER — mint-and-melt does NOT self-regulate, and never did

**This is a negative and it is not softened. The claim at §3.1 — "it is the only option whose supply is
self-correcting, so nobody ever has to tune a faucet" — is FALSE, and it was false before coins were
dropped.** Removing the death drain did not break the stabiliser; it removed the flat leak that was hiding
the absence of one.

#### The structural finding

§3.1's correction argument is: *if too much currency exists, its purchasing power falls below the
redemption return, players cash in, and currency is destroyed.* §3.4 separately decides: **the backing
material must be USELESS for anything else.** These two decisions destroy each other, and nowhere in 1,560
lines does either reference the other.

> Melt returns **0.9 units per 10 coins** (§3.3 launch settings). Those 0.9 units re-mint to **9 coins**.
> The material has no demand except from would-be minters, who gain nothing by buying it, so its price is
> pinned at exactly the mint yield. **Melting therefore exchanges 10 coins for a good whose only realisable
> value is 9 coins — a guaranteed 10% loss, at every possible supply level.**

Melting is a **strictly dominated action**. A perfectly attentive, perfectly informed, profit-maximising
player melts **zero** at every price. **The redemption drain is identically zero, and the corrective
feedback term does not exist.** This cannot be fixed by better tooling, price boards, or bots — only by
giving the material a genuine non-monetary use, which reverses §3.4 and Decision 3. Note also that §3.4's
supporting analogy is wrong on the facts: gold's monetary role rested on real ornamental demand — a
non-monetary value floor — which is precisely the property §3.4 removes by decision.

#### The arithmetic, at this document's own planning scale

All inputs are §3.3's own, so the result cannot be blamed on new assumptions. 3,000 monthly-active,
10 played hours each, 20 units/hour extraction (flagged as a guess at §12.1), 10 coins/unit, 5% of effort
on the backing material, 6,000,000 demanded supply, venue fee 4%, burn share 5%.

**Gross issuance** = 3,000 × 10 × 20 × 10 × 0.05 = **300,000 coins/month = 5.0% of the demanded supply.**
(6,000,000 / 300,000 = 20 months to first fill — §3.3's own figure, and it is population-invariant: at 400
concurrent and at 2,000 concurrent every percentage below is identical and only the absolute counts move.)

**The three candidate drains, enumerated and quantified:**

| Drain | Rate | Note |
|---|---|---|
| Melting | **0.000%/month** | Structurally zero — see above |
| The mint owner's cut | **0.000%/month** | It is 0.1 unit of *material* retained per melt; not a currency drain, and zero twice over |
| Material consumed | **0.000%/month** | Zero by construction — a material useless for everything is consumed by nothing |
| Coins lost on death / host kill | **1.200%/month** = 13.5%/yr | §3.3: 0.20 carried × 0.002/day × 30. **REMOVED by accounts-only** |
| Venue-fee burn at launch (5%), at §3.3's assumed 1.5× turnover | **0.300%/month** = 3.5%/yr | 0.05 × 0.04 × 1.5 |
| Venue-fee burn at **measured** turnover | **0.060%/month** = 0.72%/yr | see below |

**The turnover assumption is unsupported and the only measured comparable is five times slower.** §3.3
assumes 1.5× supply turned over per month. The reference economy's audited figures
(`scripts/economy_research_20260726.md:316`, `:322`, marked verified) give monthly trade 870.89 T against a
money supply of 2,908.38 T = **0.2994× per month**. At that turnover the burn is 0.05 × 0.04 × 0.2994 =
**0.0599%/month**. Every destruction figure in §3.3 is proportional to this assumption.

**Death loss was 80% of the entire drain at §3.3's turnover (1.2 / 1.5) and 95% at measured turnover
(1.2 / 1.26).**

**The number that decides it — how long an overshoot takes to work off.** Half-life = ln(0.5) / ln(1−d) / 12:

| Drains | d per month | Correction half-life |
|---|---|---|
| Death + burn (as designed) | 0.01500 | **3.8 years** |
| Burn only, §3.3 turnover | 0.00300 | **19.2 years** |
| Burn only, measured turnover | 0.00060 | **96 years** |

**Worked example — six extra months of minting past the fill point.** 6 × (300,000 issued − 18,000 burned)
= 1,692,000 excess on 6,000,000 = **+28.2%**, supply 7,692,000. Time to return =
ln(6,000,000/7,692,000) / ln(1−d) = −0.248398 / ln(1−d):

- d = 0.015 → −0.248398 / −0.0151136 = 16.4 months = **1.4 years**
- d = 0.003 → −0.248398 / −0.0030045 = 82.7 months = **6.9 years**
- d = 0.00060 → 414 months = **34.5 years**

**The trajectory, and why this passes review for two years.** S(n) = (I/d)(1−(1−d)ⁿ), I = 300,000:

| | Burn only (d = 0.003) | Both drains (d = 0.015) |
|---|---|---|
| Fixed point I/d | 100,000,000 = **16.7× demand** | 20,000,000 = 3.33× demand |
| Year 2 (n=24) | 100M × (1 − 0.997²⁴ = 0.069569) = 6.96M = **1.16× — looks fine** | 1.16× — also looks fine |
| Year 5 (n=60) | 100M × 0.165154 = 16.5M = **2.75×, +22.4%/yr** | 20M × 0.59622 = 11.9M = 1.99×, +14.7%/yr |

**The failure is invisible for the first two years in both cases — which is exactly the window in which it
would be signed off.**

**The decisive comparison.** The reference economy destroys **4.609% of its money supply per month**
(134.06 / 2,908.38 T = 43.2%/yr) against creation of **6.773%/month**, and *still* runs +2.16%/month
persistent inflation. We would destroy **0.060–0.300%/month** against creation of **5.0%/month** — a
shortfall of **15× to 77×** on a creation rate in the same league. It is not close.

**A second internal inconsistency, exposed by the same arithmetic.** §3.3 assumes 5% of community effort
goes to the backing material (300,000 coins/month) while computing that only 72,000 coins/month are needed
to hold the level, and calls the margin "trivial". Those differ by **4.2× with the death drain and 17×
without it.** A tap running 17× the drain with no feedback term does not hold a level — it converges on the
fixed point above. The only brake is that effort share is endogenous (players stop mining money when it
stops paying), but that brake engages only after the currency has lost most of its value, and §3.3 nowhere
models it.

#### The options, ranked, with a recommendation

| # | Option | Replaces the drain? | Cost | Verdict |
|---|---|---|---|---|
| **1** | **A destroyed fee on every account-to-account movement.** Only possible with no cash, because only then does 100% of value movement pass through one writer. Required rate = 0.012 / 0.2994 = **4.01%** at measured turnover → 1.20%/month, replacing the death drain exactly. | **Yes, exactly** | One number in the settings block. Measurable in production every tick | **RECOMMENDED** |
| **2** | **Give the backing material one genuine, high-value, low-volume use**, restoring a real reason to redeem. Reverses Decision 3's absolutism without reverting it — the volume stays low enough that sterilising stock is still affordable. | Partly — it restores the *feedback term*, which is the thing actually missing | A terrain/content decision, free before terrain generation, impossible after | **RECOMMENDED alongside 1**; the two fix different halves |
| **3** | **Raise the burn share.** Needs **20%** at §3.3's turnover, or **100% of the fee** at measured turnover — which leaves the venue owner nothing and forces them to charge separately on top. | Yes at 20%, structurally broken at 100% | A number, but the base is only venue sales at player-built venues | **Adopt as the interim** until option 1 lands; **20%, not 5%** |
| **4** | **Cap how much may be minted per period.** A cap is not a computer handing anybody money, so it stays within the standing rules. | It caps the tap; it removes nothing | Content and comprehension cost; feels arbitrary | Fallback |
| **5** | **Keep physical coins so that dying still destroys money.** | Yes | The entire bill this revision exists to delete | **REJECT** — the drain is worth ~3,900 lines of blocking prerequisites |
| **6** | **Do nothing.** | No | — | **REJECT.** On the measured turnover figure the supply ratchets up, plateaus at the redemption floor, and never comes back down within the lifetime of the game |

> **RECOMMENDATION.** Adopt **1 + 2**, with **3 at 20%** as the interim from the first version. Delete the
> claim that the supply looks after itself (Decision 2 and §3.1 are amended accordingly). **Reopen the
> backing-material question before terrain generation** — that deadline was already binding for a different
> reason and is now binding for two.

**Three consequences to absorb even after the fix.** (a) Supply becomes a **ratchet**: it climbs while
trade grows, plateaus at what trade demands, and has no reason to fall absent the burn. (b) **Minting stops
being a business at the plateau** — §3.3's premise that mint ownership is contested content depends on a
continuing need to replace destroyed money, and without the death drain the burn is the only such need.
(c) **Stranded reservations rise tenfold** (below), and their permanent fraction immobilises money at
roughly the rate death was destroying it, while looking perfectly healthy to the conservation gate.

#### Stranded reservations, recomputed — the new largest exposure

§6.4 sized this assuming 10% of sales are account-paid. **Under accounts-only it is 100%.** At this
document's planning population: 9,600 shop interactions/day, 1% strand rate, 500 coins each.

| | §6.4 as written | Accounts-only |
|---|---|---|
| Account-paid/day | 960 | **9,600** |
| Stranded/day | 9.6 → 4,800 coins | 96 → **48,000 coins** |
| Per month | 144,000 = **2.4% of supply** | 1,440,000 = **24.0% of supply** |

**With the release-on-next-login rule (§6.4) actually landed**, the standing pool settles to about one
day of strands = 48,000 / 6,000,000 = **0.80%**, and only the never-returning fraction immobilises
permanently: at 5%/month churn, 0.05 × 1,440,000 = 72,000/month = 1.2%/month → (1−0.012)¹² = 0.86513 =
**13.5%/year — the same order as the drain that was lost.** This is immobilisation, not destruction:
conservation holds and the gate reads healthy while circulating supply falls. **The release-on-login rule
stops being a safety net and becomes load-bearing; it must land in the same slice as the account-paid
purchase path, never after it, and the immobilised total must be published from the first version.**

---

### 0.5 The feel cost, evidenced

**What is genuinely lost:** paying while out of contact; money as loot; bank notes and large denominations;
and coins trading at a discount far from a vault (§8 abuse 19's "frontier arbitrage" content), which a
balance with no location cannot express. Also lost is §3.2(c)'s anti-compression purpose — coin mass ≥ 1
existed to stop the game's highest-value cargo teleporting, and a balance teleports by definition.

**Does shipped evidence say this matters for a game with our premise? No — and the evidence is direct.**

- **The largest space sandbox ever made has a wallet that is a pure number no player can ever take by
  force.** Every documented way its currency changes hands is voluntary — trade window, contract, chat
  transfer, paid ransom — and its own guidance is *"once you transfer it, it's gone"*: advice about not
  being persuaded, never about being robbed. Physical loot is defined as the *contents of a destroyed
  ship*. It has run over two decades this way and it is the benchmark for danger and irrecoverable loss.
  (`wiki.eveuniversity.org/Scams`, `/Piracy`, `/Ransom`)
- **Its piracy maths contains no term for the victim's savings.** Its ganking guide states the governing
  rule outright: *"the aim is to destroy a higher value target than the value of the ship or ships being
  used to gank"* — hull value and hold contents, nothing else. Its most famous grand thefts are treasury
  heists executed through granted permissions and infiltration, which proves money-scale robbery stories
  still exist in an accounts-only world; they just run on trust and access rather than violence.
  (`wiki.eveuniversity.org/Suicide_ganking`)
- **THE DECISIVE ONE: the closest shipped analogue to our premise has no currency at all.** A persistent
  logistics war game built on physical hauling, convoys, ambushes and geography that hurts — trucks at
  1,500–2,000 crates, flatbeds at 5,000, freighters at 25,000; travel guidance to stay on watchtower-covered
  roads because enemies wait *"to steal or destroy your shipment"*; capturing a base hands the attacker 25%
  of its contents. **Every element our premise names is fully delivered with zero money in the game.** That
  is a direct experimental result: **the cargo carries the risk, not the cash.**
  (`foxhole.wiki.gg/wiki/Logistics`)
- **Where lootable coins do ship, they are person-scale full-loot games and the money is not what makes
  robbery work.** In the classic fantasy sandbox coins are an ordinary stackable item and the lawless-zone
  rule keeps only your three most valuable items (four with a protection prayer) — coins compete for those
  slots against gear worth far more, so the well-played answer is to bank cash before entering, meaning
  even there well-played money is not actually at risk. The harshest shipped extraction shooter loses
  everything on death **except a deliberately guaranteed safe container**. **Our game is ship-scale: the
  ship and its hold are already the largest thing a player can lose.**
  (`oldschool.runescape.wiki/w/Items_Kept_on_Death`)
- **"There is no market where you are" is a shipped and loved constraint; "your money itself stops working
  here" has no shipped precedent I could verify.** Deep uncharted space in the reference sandbox has no
  fixed infrastructure, no NPC market, no station trading — residents build their own or fly home. But the
  thing geography removes there is the **market**, never the money. **This cuts in favour of dropping
  coins**: §1's blunt rule *"no communications, no banking; you have what is in your pockets"* is the coins
  design's headline benefit and it would be defending a novel restriction, whose failure mode is being
  wealthy and unable to buy a repair. Marked as an **absence of evidence, not evidence of absence.**
  (`wiki.eveuniversity.org/Wormhole_space`)
- **Notes and large denominations are the clearest case of all.** In the old sandbox that had them, gold
  had weight — 150 coins to the stone, a full stack at 400 stone — and *"gold has no weight when located in
  a bank box"*; bank checks stored up to a million coins *"in check form, which can save space"*. They
  existed **solely** to defeat weight and stack limits, i.e. a problem physical coins created, and the game
  later deleted them. §3.2(c) reaches the same conclusion from first principles. **Dropping notes also
  deletes a laundering channel and a documented scam surface at zero gameplay cost** (§8 abuses 12 and 15,
  Decision 10). (`uoguide.com/Gold`)
- **Policing real-money selling gets EASIER, not harder — on the axis that matters.** Accounts-only makes
  the act one click, but makes it a permanent row naming both parties and a timestamp. A coin handover is
  *"one shard-local write; identical to handing over a rock"* (§2.4) and produces **no central record at
  all**; notes are *"untraceable in the middle"* (§5.5). The historical pattern is unambiguous: when the
  large fantasy sandbox attacked the auditable channels, sellers immediately migrated to hand-to-hand
  meetings **precisely because those leave no ledger row**. §8 abuse 11's caseload arithmetic
  (0.144 bans/concurrent/month) gives **≈2/day at 400 concurrent and ≈10/day at 2,000** — a human review
  workload, *and only if the transfers are auditable.*

> **What genuinely and permanently dies is the small human story of taking the purse off someone you just
> beat.** It is a good story and it will not exist. Everything larger — taking their ship, their hold,
> their station stock, an organisation's treasury through betrayal — survives untouched. **In a game about
> ships and freight, the thick version is already fully served and the thin one is not worth ~3,900 lines
> of blocking prerequisites.**
>
> **Weakest link, stated plainly:** the player-*sentiment* half of this is inferred from shipped mechanics
> and from what those games chose never to change over long lifetimes, **not measured**. Every discussion
> venue the reviewer tried was blocked or paywalled. If the decision needs measured opinion, gather it
> separately.

---

### 0.6 The keep-the-door-open list — what must be shaped NOW so coins stay additive

**Adding physical coins later is structurally additive and this is verified.** The closed cross-server list
is append-only, and every recent arm carries the note *"APPENDED (preserves every existing postcard
discriminant)"* (`crates/wire/src/intershard.rs`, ReleaseComplete → ShardPresence). The entity-kind band
documents deliberate gaps and **tag 3 is free** in the Durable band (`crates/core/src/entity_kind.rs:26-41`
— 0/1/2 used, 10–13 transient). The carried-state blob already retains unknown tags as opaque bytes
(`crates/core/src/tlv.rs:160-172`). **And the account service does not change at all.** Coins are: a new
object kind, deposit and withdraw arms at a branch, and a branch capability flag.

**All eight of the following are free today. Each would otherwise be a live migration of real player
money.**

1. **A price is a list of entries where currency is one entry kind among materials.** Already §9.6 item 1
   and already the highest-value item; **under accounts-only it is also the ONLY thing that keeps a market
   functioning with the economy switched off**, because coins previously provided that fallback
   independently. **Its importance goes up, not down. Non-negotiable.**
2. **One monomorphic amount type**: integer minor units, currency a runtime `u16` tag never a type
   parameter, no conversion from a floating-point value. (§9.6 items 6 and 9, unchanged.)
3. **Reserve durable entity-kind tag 3 for a future coin kind, and record that it is unconditionally
   compiled and never feature-gated.** The kind decoder errors on an unknown tag and never defaults
   (`crates/core/src/entity_kind.rs:55-68`), so a later feature-gated variant would turn every existing
   coin into a hard decode error (§7.4).
4. **The totals-not-deltas law and its object-creation exception, written now.** Any money-bearing message
   aimed at a machine without a disk states an absolute value at a stamp; where an object must be created,
   **its identity is minted by the disk-holding side and derived from the request id**, so a redelivered
   withdrawal names the same object and creates nothing (§6.2b and its exception).
5. **Write into the not-yet-built carried-state seam contract that money tags are always OPTIONAL, that
   the required floor is unchanged by the presence of money, and that reconstruct-then-reserialise
   round-trips unknown tags verbatim** (Laws A and B, §7.3). Free now; a live migration later. **Note the
   enforcement point still does not exist** — `floor_ok` (`crates/core/src/tlv.rs:250-258`) has no caller
   outside tests — so this is contract text either way.
6. **Never key money by realm identity.** The top two realm levels collapse to fixed singletons today and a
   planned slice will change the identifier (`crates/core/src/realm_path.rs:25-27`, `:74-92`). Unchanged
   from §9.6 item 12.
7. **Write the conservation identity from day one with the world-held-coin terms present and summing to
   zero**, so that adding coins later does not change the assertion's shape — only which terms are
   non-zero.
8. **The account row carries a held/escrow field and a monotonic stamp, and errors rather than decoding to
   a default.** Required by residue 1 above regardless of coins; it is also what a later deposit/withdraw
   pair needs.

> **Explicitly kept as the later option:** §2.1's two-form table, §5.1's one-write coin sale, §5.3's
> locally-funded want-ads, §4.1's branch affordance and §1's out-of-contact rule all remain **valid
> descriptions of the coins extension**. They are not deleted; they are conditional on a decision to add
> physical money later, and each carries a `[REV 07-27]` marker saying so.

---

### 0.7 Review record

**Three adversaries checked this revision against the real code and the shipped record. Verdict: the
revision stands, with three corrections and one thing the original analysis had backwards.**

**Sustained.**

- **All four named blocking defects are real, and three stop blocking the economy under accounts-only.**
  Verified in code: promote-journals-before-effect (`crates/sim/src/stub.rs:2274-2298`, `:2327-2340`, sole
  re-driver at `crates/sim/src/saga.rs:1069-1080`); rollback-restores-an-older-snapshot
  (`crates/node/src/saga_runtime.rs:862-883`, `:834-841`, with `ReHomeState::PoseOnly` the only arm at
  `crates/wire/src/intershard.rs:658-661`); epoch-mismatch discard (`crates/sim/src/stub.rs:2765-2772`,
  `:2469-2474`); no-value-in-a-killable-place (`crates/sim/src/rlm.rs:377-388`,
  `crates/node/src/rlm_runtime.rs:281-294`).
- **Conservation genuinely becomes provable in production.** The existing oracle takes a vector of every
  node's private world (`crates/harness/src/oracle.rs:396-424`) and is only assemblable because the harness
  runs the cluster in one binary; production has no cross-server query (the only request/reply pair in the
  24-arm closed set is Directory → DirectoryReply, orchestrator-only). One book, one writer, one local read
  per tick. Its per-kind loss tolerance (`:440-455`) also stops being a tension: the money loss budget is
  zero **by construction**, not by a stricter-than-anything-else contract.
- **The optional-economy rule gets cleaner, conditionally.** With no money in core state, §7.3's
  highest-risk cornering item becomes vacuous — there is no money tag to mis-mark REQUIRED, so the
  movement-outage failure mode cannot exist — and §7.5's dependency-direction check becomes a compile-time
  truth. **Conditional on price-as-a-bundle surviving** (§0.6 item 1).
- **The round trip is affordable and the arithmetic is not close** (§0.2).
- **Additive-later is structurally true** (§0.6).

**Rejected on the merits, with reasons.**

- **"The four defects remain real bugs for other carried state" — REJECTED as an undersell of one of
  them.** The promote defect does not merely lose *carried state*; it strands **the player entity itself**,
  with the saga reporting success and no alarm firing. Dropping coins buys no permission to leave it.
  **D-84 keeps its 🟥 BLOCKING status on its own merits as a game-correctness defect**, and this document
  says so wherever it reclassifies it.
- **"Accounts-only deletes the expensive half" — REJECTED as stated, sustained as re-scoped.** True for the
  schedule (blocking prerequisites −56.5%); **false for the code** (the economy proper −17%). §0.3 states
  both.
- **"No value in a killable place stops blocking" — REJECTED as written.** True for money only. Consignment
  goods and the backing vault still die on a routine teardown (residue 2, §0.2), so P5 and D-87 are
  narrowed, not dropped.
- **"The melt arbitrage fails because it needs attention nobody pays" — REJECTED as the wrong diagnosis.**
  The arbitrage is never profitable, so attention is irrelevant. This is a *stronger* negative than the
  attention hypothesis and cannot be fixed by tooling (§0.4).
- **"The reference economy destroys about 40% of production value per month" — REJECTED; the document's
  only 40% is a burn-share table row, not a measurement.** The measured comparables are 4.609% of money
  supply destroyed per month against 6.773% created. **THE DOCUMENT WINS over the framing**, and §0.4 uses
  the measured figures.

**Corrections to this document's own grounded record — THE CODE WINS on both.**

- **§6.3's "teardown is an outright kill with no save step" is inaccurate in mechanism.** The spawner sends
  SIGTERM to the process group, polls for a configurable drain grace, and only then escalates to SIGKILL
  (`crates/bins/src/proc_launch.rs:214-244`), with a 2,000 ms shutdown linger deployed
  (`deploy/k3d/10-configmap.yaml`). **A graceful window EXISTS; what is missing is any flush CODE** — the
  shard binary's drain comment states plainly that *"the shard holds no un-fsynced durable state"*
  (`crates/bins/src/bin/shard.rs:314`). **Consequence: adding save-on-shutdown later is a change inside an
  existing lifecycle window, not a new lifecycle phase — which makes P5 cheaper than §6.3 implies.**
- **§2.5's "hardcoded 16-key allow-list" is 13 keys** (`spawn_anchor_keys`,
  `crates/bins/src/lib.rs:2203-2226`), plus the orchestrator id and peer book appended by the caller = 15
  (`:2228-2233`). The substantive claim is CORRECT and unchanged: **not one of them is a storage key**, so
  a demand-started region cannot be given a disk by configuration.

**Not raised by the original analysis, and now folded in.**

- **Accounts-only deletes the SIMPLE sale and makes this document's own worst-case path universal**
  (§0.1, applied at §5.1 and §5.2).
- **The escrow must live at the service from slice one** (residue 1, §0.2).
- **Stranded reservations rise tenfold and become load-bearing** (§0.4).
- **Two intended gameplay properties die quietly** — anti-compression and frontier arbitrage (§0.5).

**Unverifiable, and marked as such.** Player *sentiment* about unrobbable savings (every discussion venue
blocked or paywalled); shipped precedent for a game blocking spending on a *communications* ground (none
found in either direction); and the wider survey of player-run financial services (§12.1 item 3 stands).

---

## 1. The answer, in plain language

> **[REV 07-27] This section describes the TWO-FORM design and is now the description of the LATER,
> OPTIONAL coins extension, not of what ships. Read §0.1 for what ships.** The passages below about coins
> in your pockets, bank branches, notes and the out-of-contact rule are retained verbatim because they are
> the specification of that extension and §0.6 keeps it additive. Under accounts-only: money is a number in
> a book, every payment is a round trip, a cut-off station trades by barter rather than by cash, and money
> is never lost and never looted.

**Money is a number in a book, and coins are things in your pockets, and the difference between them is
the answer to every hard question in this design.**

There is one currency. It is real, it is transferable, and it is not cargo.

**In your pockets.** You carry coins. They are objects in the world with weight, like ore or ammunition.
You hand them to another player standing in front of you and the payment is done — instantly, with no
server talking to any other server, at the far edge of the galaxy, with the entire economy switched off.
Nobody has to look anything up, because the thing in your hand *is* the value. Coins can be dropped,
looted from your wreck, and lost forever if the server holding you dies. That is deliberate: it is the
only way money leaves the world without a computer deciding to delete it, and it is what makes robbing
someone worth doing.

**In an account.** You also have an account. It lives at one service that is always running and has a
disk, and it belongs to no place in the world, so it cannot be destroyed by a region being switched off,
cannot be lost when a server dies, and is exactly the same whether you are standing next to it or four
galaxies away. Paying a wage, funding a company, or sending money to someone you cannot reach is two
numbers changing in one book. It takes about a fiftieth of a second and two messages.

**What a player actually does.**

*To get paid:* you sell something. If a buyer is standing in front of you, they hand you coins and you
walk away with them. If you left goods at a station on consignment, the sale happens when a buyer
arrives — you were not there and did not need to be — and your money waits for you in your takings box
at that station, or goes to your account if you asked for that when you listed.

*To pay:* you hand over coins, or you pay from your account. Coins always work. Your account works when
your ship can reach the outside world.

*To save:* you walk into a bank branch — a building another player put up — and hand over your coins.
They are counted into your account and the coins physically cease to exist. Now your money weighs
nothing, cannot be robbed, and survives anything. Walking out with them again is the reverse. **This is
the entire point of a bank: it is the only way to make value weightless, and it requires you to
physically go somewhere.**

*To send money far away:* you tell your bank to move it. It arrives immediately, because both accounts
live in the same book and neither of them is anywhere. If you cannot reach the outside world, you can
instead have your branch print you a **note** — a piece of paper worth a fixed amount, an object like a
coin. You carry it across the galaxy with no communications at all, hand it to a stranger, and they cash
it at any branch. Cashing it destroys it, so it cannot be cashed twice.

*When they are out of contact:* coins work, notes work, hand-to-hand payment works, buying and selling at
a market works, minting works. Your account does not — you cannot see it, cannot pay from it, cannot
deposit into it. The rule is exactly as blunt as it sounds: **no communications, no banking; you have
what is in your pockets.** A station cut off from the whole universe still runs a full cash market.

**Where money comes from.** Somebody mines a rare, heavy, otherwise-useless material, hauls it to a mint
another player built, and feeds it in. Coins come out. Bring coins back and you get the material out
again, minus a small cut for the mint's owner. That is the only source and one of the two drains — the
other is losing coins when you die. No computer ever hands anybody money. If too many coins exist they
are worth less than the metal behind them, so people cash them in and coins vanish. If too few exist,
minting pays and people mint. The supply follows demand instead of following a schedule, so nobody ever
has to sit and tune it.

**Banks.** Anyone can build one. What the game provides is boring on purpose: it holds the numbers, it
never lends, never invests, never promises a return, and cannot go bust because it holds nothing it was
not given. Everything that makes banking a business is a player's: where the branches are, what they
charge, company treasuries, shares, and lending against real collateral. One rule makes all of it safe —
**there is no way for one player to hold another player's balance.** A run on a bank is not something the
game can express. Somebody will still stand in a station saying "give me your coins and I will pay you
five percent", and some of them will fly away with it, and that will be a story rather than a systems
failure, because the game never wrote the number down as though it were yours.

**What this costs, honestly.** Three things. **[REV 07-27] Under accounts-only the second and third are
both gone**: money is provably never lost, in production, so the refusal policy disappears; and the pocket
tier that made banks-first "the reverse of the intuition" is deleted, so the build order stops being a
compromise. **The first — a permanent real-money-trading and botting cost — stands, and is the one thing
having a currency at all buys you regardless of its form.** It does get materially easier to police: every
transfer is a permanent row naming both parties, where a coin handover left no record anywhere (§0.5).
Adding a real currency permanently adds a real-money-trading
and botting problem that trading cargo largely did not have — moving value stopped being a hauling job
and became a line in a database. Money is provably never duplicated, but it is **not** provably never
lost: if a server dies with coins on it, they are gone, and the only safe answer to "give them back" is
no, because giving them back is the one action that creates money from nothing. And the thing everyone
assumes is easy — coins in your pocket — is the expensive half, because today a world server has no disk
and is killed and forgotten as a matter of routine. Accounts are the cheap half. So banks get built
first and pockets second, which is the reverse of the intuition, and until pockets exist the markets
trade by barter.

---

## 2. The model

### 2.1 Two forms, split by what each survives

> **[REV 07-27] Only the right-hand column ships. The Coin column is the specification of the later,
> optional extension.** Read the table as: *what accounts-only gives up is every **Yes** in the left
> column* — works with no communications, and can be robbed. Everything else in the left column was a
> liability, not a feature. **The bottom row is the point of the revision: "provably never lost" is now
> Yes for all money in the game, in production, not only in the harness.**

| | **Coin** (and note) | **Account balance** |
|---|---|---|
| What it is | A thing in the world with weight | A row in one book |
| Authoritative home | The shard owning the holder | The account service |
| Works with no communications | **Yes** | No |
| Survives its region being switched off | **No** | **Yes** |
| Survives the holder's server dying | No — it is lost | **Yes** |
| Can be robbed / looted | **Yes** | No |
| Moves between servers | Rides the crossing the holder was already making | Never moves — it is nowhere |
| Provably never duplicated | Yes (single-owner authority) | Yes (single writer) |
| Provably never lost | **No** | Yes |

The rule that decides which: **if it must work with no communications it is carried; if it must survive
your host being killed it is held.** Nothing is both, and that is the honest cost of the split.

### 2.2 Where a balance authoritatively lives

**Account balances live in one account service.** Concretely: a new `NodeKind` variant built from the
same `build_app(cfg, transport)` entry point (`crates/node/src/app.rs:87`), deployed as its own workload
with its own data root and volume, exactly the way the gateway already is. There is a precedent for a
non-shard, non-place node kind and the vetters found it: `NodeKind::GalaxyRelay` already exists in the
closed enum at `crates/sim/src/capability.rs:24-31`. Adding a ledger variant is an additive arm to a
closed enum with a live sibling — the exhaustive-match churn is the intended cost of the sealed design,
not an obstacle.

**It is NOT a realm, and this is load-bearing.** Three code facts force it:

1. **There is no always-alive root.** `desired_alive` is exactly
   `demanded_recently | (running_live & !empty_confirmed)` (`crates/sim/src/rlm.rs:451-465`), and
   `demanded_recently` deliberately excludes the never-demanded sentinel (`:417-429`). The ancestor rule
   is real and recomputed every tick (`ancestor_close`, `:492-517`) but it only preserves ancestors of
   something *wanted*. **With nobody logged in anywhere, the root is torn down like anything else.** The
   brief's claim that the top of the tree is always alive is FALSE as written. A galaxy-tier institution
   built as a place would be available only while the game is populated.
2. **The spawner can only be asked for a place.** `spawn_realm(&RealmCoord, at_tick)`
   (`crates/sim/src/io/mod.rs:468`), with an entirely realm-derived child environment
   (`crates/bins/src/proc_launch.rs:123-137`). A bank that is not a place cannot be demand-spawned.
3. **Realm identity is not usable as a money key.** `to_realm_id` collapses the two top levels to fixed
   singletons — `UNIVERSE_STANDIN = RealmId::System(0)` and `GALAXY_STANDIN = RealmId::System(1)`
   (`crates/core/src/realm_path.rs:25-27`, `:74-92`) — so **every galaxy is currently the same realm**,
   and both stand-ins collide with any real star system whose seed is 0 or 1. The comment at `:77-78`
   says a later slice flips them to dedicated arms, which will **change the identifier**, orphaning
   anything keyed by it. Worse, two identity schemes are in play and disagree under a re-parent: the
   directory keys by the path-independent seed identifier (`crates/node/src/rlm_runtime.rs:322-325`)
   while the demand ledger keys by the full lineage path (`crates/sim/src/rlm.rs:492-517`).

**Therefore: money is never keyed by realm identity.** It is keyed by `AccountId`
(`crates/core/src/ids.rs:63-66`) — a 128-bit random non-personal principal that embeds nothing, already
minted at account creation and already riding the login ticket (`crates/wire/src/seams/tickets.rs:17-24`).
Where a *place* must be named (a vault, a venue, a branch), it is keyed by the full lineage path plus a
never-reused local identity, never by the collapsed seed identifier, and the coming top-level rewrite is
treated as a planned migration event.

**Do NOT co-locate the service in the coordinator.** It reuses the same program, so someone will propose
it to save a deployment object. That single shortcut would put money on the barrier that every saga,
every ownership change and the clock already share (`group_commit`,
`crates/node/src/saga_runtime.rs:2485-2511`), on the one node running at guaranteed quality of service
with a 512Mi ceiling (`deploy/k3d/30-orch.yaml:65-67`). Forbid it structurally — separate instance,
separate data root, separate volume — and record why in the manifest.

**Storage shape is read-through with a bounded hot cache — mandatory, not an optimisation.** The
directory holds every record resident and rebuilds it row-by-row on every restart
(`crates/sim/src/directory.rs:380-394`; rehydrate at `crates/node/src/saga_runtime.rs:1390-1400`).
Copying that would make **restart time grow with accounts ever created rather than players currently
online** — a strictly worse curve that no early test would surface. Account rehydrate touches zero rows.
This deliberate inconsistency with the directory must be written into the code comment, or someone will
"fix" it.

### 2.3 Why double-spend is structurally impossible

**Account side.** One process, one writer, one system in the schedule chain. Two rows changing in one
store transaction, sharing the existing one-barrier-per-tick discipline. There is no second writer of an
account row anywhere in the cluster — strictly stronger than the directory's own advertised
"writes are linearizable (single writer)" (`crates/wire/src/seams/directory.rs:5`). Every operation is a
vector of signed deltas and there is **exactly one assertion**, in one monomorphic helper: the vector
sums to zero, except for three named operations that each require a matching proof (§6.1).

**Coin side.** A coin stack sits in a holder's carried state. That holder has exactly one `Owned`
authority at a time, flipped only by the single fence compare-and-set at
`crates/sim/src/directory.rs:552-580`, with demote strictly preceding promote in the pipeline
(`crates/sim/src/saga.rs:15-17`). **A purse can be lost but it can never be forked, for exactly the same
reason a player cannot be in two places at once.** It is not a new guarantee; it is the same guarantee.

**Payments must never take a directory transfer lock.** `lock_transfer`
(`crates/sim/src/directory.rs:530-541`) permits at most one in-flight saga per key and a second attempt
returns `false` and is **silently skipped, not queued**. A durable crossing's floor is nine acked phases
at the deployed 50 Hz (`deploy/k3d/10-configmap.yaml:15`) = 9 × 20 ms = **180 ms**, so a balance
expressed as a directory key would serialise at 1000/180 = **5.5 payments per second** on a hot venue
treasury or exchange account, with the excess dropped rather than backpressured. This is the decisive
argument against reusing the transfer machinery for value.

**And the multi-key commit does not exist.** `crates/sim/src/directory.rs:543-551` documents the bundle
as deferred (ledgered `docs/design/DEFERRED.md:96`, scheduled as ship-phase work). A two-sided debit and
credit expressed as two directory keys would be two independent commits with a money-creating window
between them. **Keeping both sides inside one local store transaction at one writer avoids this
entirely** and is the second decisive argument for the central account service.

### 2.4 Local versus global, and exactly which machinery carries what

> **[REV 07-27] Under accounts-only, every row whose machinery is "shard-local" and whose subject is money
> becomes a round trip** — buying at a market, and the seller collecting proceeds. Rows about listing,
> withdrawal and container moves are unaffected because they are **goods**, not money. The rows about
> coins, notes and branches describe the later extension. **The two new arms are unchanged**, and so is
> the rule that a payment never takes a directory transfer lock (§2.3) — which now applies to 100% of
> payments rather than 10%, making it the single most load-bearing prohibition in the document.

| Operation | Crosses a server boundary? | Machinery |
|---|---|---|
| Hand coins to a player in front of you | **No** | One shard-local write; identical to handing over a rock |
| Buy at a market with coins | **No** | One shard-local write, one tick |
| List / withdraw consignment goods | **No** | Game container move |
| Mint coins, melt coins | **No** | A recipe; the vault is a game container |
| Deposit / withdraw at a branch | Yes — one request, one reply | New request/reply arm pair |
| Pay from your account at a market | Yes — reserve then settle | Same arm pair, two round trips |
| Pay someone in another galaxy | Yes — one request, one reply | Same arm pair; **neither account moves** |
| Wages, company treasury, share payout | Yes — one request | A fan-out of deltas in one tick, one barrier |
| Issue / cash a note | Yes at the branch; **no** in between | Note is an object; carrying it crosses nothing |
| Carry coins across a boundary | Rides the crossing already happening | Carried-state blob |

**The two new arms, appended to the closed 24-arm list** (`crates/wire/src/intershard.rs:118-271` — it is
**24 arms, not 26**; Design B's count was wrong, and the message bus really is a prose reservation only,
at `:29`, with no signal or relay arm anywhere):

- **A payment request** (`SideEffecting`), keyed on the shared `IdempotencyKey::TransferStep` shape with
  a payment correlation id in the transfer slot — a reuse the contract already sanctions
  (`crates/wire/src/intershard.rs:286-296`). Its confirmation follows the shipped durable-pending-reply
  template `CrossingAborted` → `CrossingAbortedAck` (`:443-451`) with its own durable key family, re-driven
  until acked.
- **A balance read and its reply** (both `FireAndForget`), following `DirectoryOp`/`DirectoryReply`
  (`crates/wire/src/seams/directory.rs:119-120`, `:161-165`, arm at `crates/wire/src/intershard.rs:135`).

**Do not fold the read and the payment into one arm.** The effect classifier is per-arm and exhaustive
(`crates/wire/src/intershard.rs:322-324`), so one arm forces one classification for both, and the only
classification safe for a payment makes every balance read carry needless durability. And the reply must
be its own arm rather than opaque bytes, for the reason already recorded at `:352-358`: the encoding is
not self-describing, so a co-routed bare reply **mis-decodes into the neighbouring arm rather than
failing** — silent corruption, not an error.

**A read is answered only by the account service; there is no cross-server query.** The only
question-and-answer path in the codebase is to the coordinator. Two world servers can never ask each
other anything. So a branch can resolve a balance at the service; a branch can never ask another branch
anything, and none of this design ever needs it to.

**Payments must originate at the service tier, not at a shard.** **[REV 07-27] Under accounts-only this
applies to EVERY payment in the game rather than one in fifty, so it must be settled explicitly in the
arm's contract rather than noted in passing** — see residue 3 at §0.2. A payment is an *edge* event — the player
clicked buy once — so a shard-originated payment arm classifies as `FlowDurabilityClass::ProducerLessReliable`
(`crates/wire/src/intershard.rs:313-317`), which carries an enforced obligation: a debug assertion fires
on every push that is not `Durability::Retained` (`crates/sim/src/runtime.rs:86-97`). Retained requires a
durable outbox, and **there is none in the deployed cluster** — `open_node_outbox` returns `Ok(None)`
unless `VD_OUTBOX_PATH` is set (`crates/bins/src/lib.rs:674-694`), and that variable appears in no
manifest. So the shard's role is to forward a player intent that is itself re-assertable (the player can
click again); the durable, re-driven step belongs to the service. Note the assertion is **debug-only** — a
payment arm added without `Retained` passes a release build silently.

### 2.5 Correction to the grounded record: shards *can* open a disk

Both the brief and the prior research say world servers have no durable storage at all. That is wrong in
a way that changes cost estimates. The shard binary **does** open a redb-backed durable outbox through
the shared boot sequence (`crates/bins/src/bin/shard.rs:36-40` → `crates/bins/src/lib.rs:781-829` →
`:674-696`). What is actually true is narrower and sharper:

- The outbox opens only when `VD_OUTBOX_PATH` is set, and it is set in **no** manifest, so it is inert
  cluster-wide.
- The environment a demand-started shard inherits is a hardcoded allow-list
  (`spawn_anchor_keys`, `crates/bins/src/lib.rs:2203-2226`) containing **not one storage key** — no
  outbox path, no store path, not even the durability root. So a demand-started region cannot be given a
  disk by configuration at all.
  **[REV 07-27] Correction, re-counted against the code: the list is 13 keys, not 16** (plus the
  orchestrator id and the peer-book closure appended by the caller at `:2228-2233` = 15). The line
  reference above was also wrong. The substantive claim is unchanged and correct.
- The volumes that do exist bind to numbered pod ordinals (`deploy/k3d/50-shard.yaml:90-96`, a
  StatefulSet claim template with a hardcoded per-slot realm seed at `:53`), so a place cannot come up
  elsewhere with its own data.

**Consequence:** a durable applied-once table on a shard is a second key family in a file the binary
already knows how to open — not a from-scratch storage tier. The two genuinely owed pieces are a storage
path derived from the realm coordinate rather than a pod ordinal, and adding it to the allow-list. Note
the allow-list is currently preventing several demand-started shards on one host from opening the same
database file — by accident, not by design.

---

## 3. Where money comes from and goes

### 3.1 The recommendation

**Coins are minted from a designated mined material at a player-built mint, and anyone can bring coins
back and get the material out again.** All three candidate designs reached this independently and I
agree with them. Formally this is *representative money* — a token with a redemption promise against a
stored commodity — and naming it correctly matters, because it carries a known failure mode addressed in
§3.4.

> **[REV 07-27] THE PARAGRAPH BELOW IS WITHDRAWN. It is the single most load-bearing sentence in the money
> design and it is FALSE — and it was false before coins were dropped.** The redemption drain is
> structurally **zero**, because §3.4 separately decides the backing material is useless for everything
> else, so melting 10 coins returns metal whose only realisable value is 9 coins: a guaranteed 10% loss at
> every supply level, which no attentive player ever takes. **The corrective feedback term does not exist.**
> Losing coins on death was a flat leak masking its absence, and accounts-only removes it. The full
> arithmetic, the options and the recommendation are at **§0.4**. **Minting remains the right issuance
> mechanism** — the alternatives at §3.5 still all lose — but it must be adopted with a managed drain
> (a destroyed fee on every payment) and a measured monthly report, **not** on a promise of
> self-correction.

~~Why this and not the alternatives, in one sentence: **it is the only option whose supply is
self-correcting, so nobody ever has to tune a faucet.** If too much currency exists, its purchasing power
falls below the redemption return, players cash in, and currency is destroyed. If too little exists,
minting is profitable and players mint. Both ends are player actions — hauling material to a mint is the
tap, cashing in and dying are the drains — so nothing computer-controlled ever creates or destroys money.~~

**Replacement statement.** Minting is chosen because it is the only issuance mechanism that is a *player
action* and therefore compatible with the standing ban on computer-controlled economic actors. Its supply
is **not** self-correcting and must be managed by one measured, retunable drain (§0.4 option 1). The
redemption promise still does real work — it sets a **floor** the currency cannot fall below, which is why
prices can be quoted against it from month three — but a floor is not a feedback loop.

### 3.2 The three rules that make it safe, each of which an adversary found

**(a) The mint and melt ratios are ONE globally fixed pair, identical everywhere, forever.** Never
per-venue, never adaptive, never seed-derived. Two of the three candidate designs proposed a per-venue
yield ("a content number that can be retuned", "a seed-and-content number per venue") and **that creates
an infinite money loop**:

> Mint at a venue paying 12 coins per unit; melt at a venue returning 0.09 units per coin.
> 12 × 0.09 = **1.08 units back for 1 unit in — an 8% gain per lap, with nothing consumed.**
> A hundred laps multiplies your material by 1.08¹⁰⁰ = e^(100 × 0.0769610) = e^7.6961 = **2,199×**.
> If both venues sit in the same station there is no hauling cost to slow it down.

This is not a subtle interaction; it follows directly from letting the ratio vary. Venue-to-venue
competition is expressed as a **fee the owner charges on top**, which is a transfer between players and
cannot loop.

**[REV 07-27] Rule (a) is unchanged and is now the ONLY part of §3.2 that still does what it claimed.**
The fixed pair still forbids the infinite loop. What it does **not** do — and §3.1 wrongly claimed it did —
is regulate the supply, because the lower end of its "band" is a price nobody will ever transact at
(§0.4).

Say plainly what the fixed pair buys and costs: the currency becomes a fixed-band claim on the backing
material — it cannot rise above minting cost or fall below melting return, a band of the spread — so the
currency cannot have an inflation or deflation crisis measured against the material. **The reversal has
escaped the logistics of material money, not its price discipline.** That is a good outcome and it
should not surprise anyone in a year.

**(b) The backing must be a physical vault standing in the world, never an institution's promise.**
Anyone can walk up to it, look at it, and in a dangerous system rob it. Redemption pays out of physical
stock or refuses. A currency's credibility becomes an observable, attackable fact rather than an
assertion — which is stronger than any rule we could write, and it is content. The known failure mode of
representative money is that if minting and redemption can happen where the reserve cannot be checked,
the token silently becomes unbacked; a physical vault closes it by construction.

**(c) Coin mass per redeemable unit of material must be at least 1.** **[REV 07-27] MOOT under
accounts-only, and its purpose is a stated LOSS, not a solved problem.** There is no coin to weigh, and a
balance teleports by definition — so the compression the rule existed to prevent happens unconditionally
for money. It does **not** happen for goods, which is what actually matters: the compression hazard was
that the *single highest-value material in the game* could be moved as light coins, and with no coins that
material simply never leaves a vault. **The rule becomes binding again the day physical coins are added**
(§0.6), which is why Decision 4 is retained rather than deleted. Ten coins must weigh at least what
the 0.9 units they redeem for weigh. Otherwise minting is a **compression trick**: buy material where it
is cheap, mint, haul the light coins across the galaxy, melt at the far end, sell the material — moving
the value of ninety kilograms by carrying ten. That grants goods global reach through the back door, for
the single highest-value material in the game, in a project whose entire architecture exists to make
geography matter. **Compression is what a bank provides** — deposit and your money becomes weightless —
and a bank requires a branch, which requires a place, which is the gameplay. If coins are weightless
nobody ever needs a bank and the local/global split the user asked for collapses into everyone carrying
everything.

### 3.3 Arithmetic, and the two-year picture

**Planning population** (the realistic one, not a hypothetical): 1,000 concurrent, ~10,000 registered,
3,000 monthly active. Every figure below is shown at that scale.

**Launch settings** — all named fields in one settings block, none written into the code:

| Setting | Launch value | Why |
|---|---|---|
| Mint yield | 10 coins per unit of refined backing material | Fixed globally (§3.2a) |
| Melt return | 0.9 units per 10 coins | A 10% spread, paid to the mint owner |
| Venue fee | 4% of sale price | Calibrated below |
| Burn share of the fee | **5%**, never 0 | See §8 — 0% makes laundering free |
| Minimum absolute fee | 1 minor unit per sale | See §8 — kills the split-the-sale dodge |

**Why 4%.** The only measured game economy at scale audits to 34.45 T of transaction tax on 870.89 T of
monthly trade = **3.96%** realised take, and that tax alone is 34.45 of 134.06 T = **25.7% of all money
destruction** there (`scripts/economy_research_20260726.md` §2.1, from the operator's published data).
So ~4% is the shipped calibration point.

**Critical point two of the three candidate designs missed: a venue fee is NOT a drain.** It moves money
from a buyer to a venue owner, both players. In the reference economy the equivalent tax is a drain only
because the operator deletes it. Here the only real drains are: **melting, coins lost on death or host
kill, and whatever share of the fee we choose to destroy.**

> **[REV 07-27] Of those three, one is structurally zero and one is deleted by accounts-only.** Melting is
> zero because the backing material is useless, so redemption is a guaranteed 10% loss at every supply
> level (§0.4). Death loss is gone with the coins. **The destroyed share of the fee is the only drain left,
> and at the launch setting it is 15× to 77× short of the only measured comparable.** §0.4 recomputes all
> of the below, adds the measured-turnover figures, and ranks the options. The sizing that follows is
> retained because §0.4's recommendation is calibrated against it.

**Sizing the loss drain — [REV 07-27] this is what accounts-only REMOVES, and it was 80% of the total drain
at the turnover assumed here and 95% at the measured turnover.** Assume 20% of supply is carried at any
moment and 0.2% of carried coin is lost per day: 0.20 × 0.002 = **0.04% of supply per day** = 1.2% per
month. Annualised: ln(0.988) = −0.0120727; × 12 = −0.1448724; e^−0.1448724 = 0.865129 → **13.5% of supply
destroyed per year** before anybody melts anything. **Nobody melts anything.**

**Sizing the burn lever.** At 6,000,000 supply and turnover of 1.5× supply per month = 9,000,000:

| Burn share | Destroyed / month | % of supply / month | Annualised |
|---|---|---|---|
| 5% (launch) | 0.2% × 9M = 18,000 | 0.30% | 3.5% |
| 20% | 0.8% × 9M = 72,000 | 1.20% | 13.5% |
| 40% | 1.6% × 9M = 144,000 | 2.40% | **25.3%** (ln 0.976 × 12 = −0.29151; e^ = 0.74712) |

Combined with the loss drain at 40%: 3.6%/month → ln(0.964) × 12 = −0.44001; e^ = 0.64404 → **35.6% a
year**, which brackets the reference economy's persistent **+29.3%/year** money creation
(196.98 − 134.06 = 62.92 T net on 2,908.38 T = 2.163%/month; 1.02163¹² = 1.293). **The lever has enough
range to cover anything realistic.** ~~Ship it at 5% and measure monthly from the first version~~ — the
evidence is unambiguous that a currency does not need to be balanced to be playable, but it absolutely
needs to be measured.

> **[REV 07-27] "Ship it at 5%" is WITHDRAWN. Ship it at 20%.** The 5% launch setting was chosen alongside
> a death drain four times larger; with the death drain gone, 20% is the row that reproduces the designed
> behaviour exactly (0.06 × 0.20 = 1.20%/month = 13.5%/year), and that is only true at **this table's
> assumed 1.5× turnover**. At the only measured turnover (0.2994×) the whole table divides by five and 100%
> of the fee would be needed — which leaves the venue owner nothing and is a structural change, not a knob
> turn. **That is why §0.4's primary recommendation is a destroyed fee on EVERY account movement rather
> than on venue sales alone: it is the only base wide enough, and it is available only because there is no
> cash.** Decision 8 is amended accordingly.

**Supply sizing and its affordability — the question nobody had asked.** Every coin locks a matching
amount of material in a vault where nobody can use it. At 3,000 monthly-active players and a 2,000-coin
working float each: **supply 6,000,000 coins, locking 600,000 units of material.** If a player extracts
20 units/hour (a guess — flagged in §12) and plays 10 hours a month, the whole active community mining
*nothing but this one material* produces 3,000 × 10 × 20 = 600,000 units/month, i.e. **one full month of
total effort**. At a realistic 5% of effort it is 30,000/month → 600,000 / 30,000 = **20 months**. So the
currency is barely established at the end of year two. That is fine, and it is the single strongest
argument for §3.4's material choice.

**Steady-state balance at end of year two.** Loss drain 1.2% of 6,000,000 = 72,000 coins/month. To hold
the supply level, minting must run at 72,000 coins = 7,200 units of material = **2.4 units per active
player per month.** ~~Trivial.~~ That is the right way round: the tap is easy to meet and the drain is the
binding constraint, because a tap that cannot keep up makes prices fall, which makes hoarding better than
spending, which stops the economy dead.

> **[REV 07-27] "Trivial" is exactly backwards, and this is the second internal inconsistency §0.4
> exposes.** The paragraph above computes a *required* 72,000 coins/month while the paragraph before it
> assumes the community actually mints **300,000/month** at 5% of effort. That is **4.2× the drain with the
> death loss and 17× without it.** A tap running 17× the drain with no feedback term does not hold a level;
> it converges on issuance ÷ drain = **16.7× the supply the game demands** (§0.4). The comfort margin read
> here as reassuring is the failure condition.

**The two-year trajectory.**

- **Months 0–3, no money exists at all.** Nobody gets a starter grant. New players get material and
  tools, never coins — the one game that restarts its whole economy roughly 135 times has never once
  seeded currency, only physical stock. The first coin in the universe exists when the first player
  finishes building a mint.
- **Months 3–12, monetisation.** Supply grows faster than the drain because trade volume is growing.
  Prices in coin are volatile and everyone quotes against the melt floor, because it is the only anchor
  there is.
- **Months 12–24, approach.** Supply approaches whatever level trade actually demands; the drains start
  to bite; the burn share gets its first adjustment based on the monthly report.

**A second-order effect that must be named because it is not obvious.** Destroying fees removes coins
without releasing the material behind them, so the vaults become progressively **over-backed** and the
surplus accrues to whoever owns the mints. That is a slow transfer of wealth to mint owners. Either make
the destroyed portion release its matching material too, or accept it and make mint ownership contested
content. **It is a decision, not an accident** (Decision 8).

### 3.4 Which material — the reversal of the prior recommendation

> **[REV 07-27] REOPENED, and it is now the highest-stakes open decision in the document.** The
> uselessness rule is what makes redemption a guaranteed loss and therefore what welds the supply's only
> safety valve shut (§0.4). **Revised position: the material must be LOW-VOLUME and CHEAP TO STERILISE —
> which is what the affordability argument below actually requires — but it must have ONE genuine,
> high-value, low-volume use, so that redeeming money is worth doing at some price.** A single use with
> small absolute demand costs almost nothing in locked-away stock and restores the feedback term. This is
> §0.4 option 2, it is free until terrain generation and impossible after, and **the deadline below is now
> binding for two reasons rather than one.** Decision 3 is amended accordingly.

~~**The backing material must be USELESS for anything else: dense, rare, no industrial application.**~~ This
directly reverses the earlier material-money design, which recommended backing with the universal fuel
and repair input.

The reason is the affordability arithmetic above. The more useful the material, the more painful it is
to sterilise twenty months of the community's mining output in vaults — and players will simply refuse
to. A material with no other use has zero opportunity cost when locked away. ~~**That is precisely what
gold was and precisely why gold was money.**~~

> **[REV 07-27] The gold analogy is wrong on the facts and it was doing real persuasive work.** Gold's
> monetary role rested on **real ornamental demand** — a non-monetary value floor beneath the monetary
> one — which is exactly the property this section removes by decision. The correct lesson from gold is the
> opposite of the one drawn here: **a commodity currency needs a small, genuine, non-monetary demand to
> anchor redemption**, and that is precisely §0.4 option 2. The affordability argument survives intact; it
> requires *low volume*, not *zero use*.

Second rule: **the backing material's scarcity must be gated on something that does not improve with
technology** — deposit rarity, distance, danger — rather than on extraction rate. If better tools double
extraction each year then prices in coin double each year too, and that is the honest inflation forecast
nobody had stated.

Both are **terrain decisions** and both must be made **before terrain generation lands**. That is already
the roadmap's stated deadline for the economy's terrain dependency, and this makes it binding rather than
advisory.

### 3.5 The alternatives, and why each loses

- **Fixed supply issued at character creation — REJECT, dead on arithmetic already computed for this
  project.** 5,000 players × 1,000 each = 5,000,000; 50 coins destroyed per player per month × 5,000 =
  250,000/month; 5,000,000 ÷ 250,000 = **20 months to zero**. And with no issuance the measured 61.0%
  fall in circulation velocity over nine years (−9.43%/year compounding) implies prices falling ~9.4%/year
  at flat output, **halving every ln(0.5)/ln(0.9057) = 7.0 years**, which makes holding money strictly
  better than using it. Do not reopen this.
- **Currency issued by player banks against reserves — REJECT.** This is fractional reserve, which is a
  *background simulation of the money supply* driven by a credit multiplier nobody controls. Banned by
  the standing decision, not by timidity. Its empirical record is in §4.3.
- **A computer payout for bounties, missions or insurance — REJECT.** Exactly the computer-controlled
  economic actor the standing decision forbids, and it is the largest faucet in the reference economy
  (58.47 T/month). **Any compensation, insurance or death-protection mechanism is issuance and must be
  counted as such**: in the reference economy ship insurance pays out 4.495 T/month against 1.698 T of
  premiums — **+2.797 T/month created by a mechanism nobody thinks of as a money printer.**
- **A real-money exchange rate — REFUSE OUTRIGHT.** One shipped game pegs at 10 units to the dollar with
  genuine withdrawal, and consequently had to auction virtual banking licences (six two-year exclusives,
  US$404,000 total, each winner posting US$100,000 working capital) rather than let players build banks.
  That converts a game-design decision into a legal and compliance programme.
- **No shipped game has ever run a fixed-supply tradeable currency**, and every one verified has a
  computer-controlled source. **Minting from a mined material would be the first.** It is defensible —
  the tap is a player action — but present it to the user as an experiment with a stated escape hatch,
  and the escape hatch is that the single global mint yield is a retunable setting.

---

## 4. Banks

### 4.1 What an institution is

**A row with a licence flag and a signer list.** That is the whole definition. It holds an identifier, a
display name, an owning account, and a per-transaction branch fee. Its accounts are rows referencing it.
It has **no realm, no shard, no memory of its own, and cannot be switched off**, because it is not a
place. A *branch* is a place; an institution is not.

> **[REV 07-27] BRANCHES ARE DELETED (slice M-2 goes with them). A branch's entire purpose was making heavy
> money weightless by requiring you to physically go somewhere — meaningless when money was never heavy.**
> Of its five operations, deposit and withdraw have no subject, note issue and cash are deleted with notes
> (Decision 10), and **reading your balance needs no building at all**. Everything above the branch survives
> untouched and is where the player banking business actually was: institutions, company treasuries with
> signer lists, shares, escrow and custody, and fully-collateralised lending. **The branch text below is
> the specification of the later coins extension** and is retained for that purpose only.

**A branch is a player-built structure plus a capability flag, and it stores nothing.** It turns on a
`ShardProfile` capability in the same shape as the existing flags (`crates/sim/src/capability.rs:53`,
`:107`, `:137` — note the actual relay flag is named `signal_relay`, not `long_range_relay`; it is set
for the galaxy and station profiles and **read by nothing** outside boot logging and tests, so its
presence is a reservation and must not be treated as evidence of a partly-built relay). Capability flags
already carry derived coherence rules that fail loud at config load on incoherent combinations
(`crates/sim/src/capability.rs:5`, `:86-87`), so the branch flag gets one.

A branch offers exactly five operations, all forwarding one refusable request: deposit coins, withdraw
coins, issue a note, cash a note, read your balance. The branch owner sets a fee on each, credited to
their own account. **That is the real player banking business: coverage and price competition on where
branches exist and what they charge, with zero custody risk.**

### 4.2 Who may run one — both, split at custody

**The service is plumbing and is deliberately boring.** It never lends, never invests, never pays a
return, never takes a position, never sets a price, never has an opinion. It cannot become insolvent
because it holds nothing it was not given: the sum of all balances is exactly the sum of everything ever
deposited plus everything ever transferred in, by construction and by a checked invariant.

**Everything that makes banking a business is a player's:** branches, company treasuries (an account
whose signers are an organisation, with a small fixed number of named divisions — the shipped answer
elsewhere is exactly seven, and a small fixed count keeps it a bounded, fully coverable structure),
venues taking per-sale fees, share issuance, escrow and custody services, and fully-collateralised
lending.

### 4.3 The rule: no promised yield, and no player holds another player's balance

**Structural form (the strong one):** there is no operation anywhere that lets an institution spend a
depositor's balance. A deposit creates a holding the institution's owners cannot touch. **A run on a bank
is not something the data model can express.**

**Regulatory form (the transplant):** an institution may never promise a return on a custodial balance.
This is the only rule an operator has ever found that actually ended the problem, and it worked
immediately: on 2008-01-08 the operator of the largest social virtual world prohibited paying fixed
interest on deposits without a real-world banking charter, and by 2008-01-22 **every unchartered bank had
closed or converted to a share-issuing company.**

**The evidence, stated accurately.** Two verified collapses with two *different* causes, and they must be
separated because only one is fixable by rules:

1. **A run against a fractional reserve.** L$55 million of withdrawals demanded against L$180 million of
   deposits — 55/180 = **30.6% of deposits called was enough** — with the reserve drained within hours and
   about US$750,000 lost. The trigger was an unrelated in-world gambling ban announced 2007-07-26: **an
   operator policy lever acting as a systemic-risk instrument.** Fixed by requiring a full reserve.
2. **The operator simply took the money.** The largest player-run bank in the most successful player
   economy ever shipped: its chief executive withdrew 200 billion of the currency (≈US$4,590) for a house
   deposit and medical costs, in a game whose operator explicitly does not treat theft or Ponzi schemes
   as petitionable. **Nothing fixes that except not letting one player hold another player's balance.**

**Honest caveat:** this rests on two verified cases, not an exhaustive survey. The research budget was
exhausted before a wider sweep was possible. The rule is sound on structural grounds independently.

**What is lost, and it is real.** Finance-minded players will find banking flat: no leverage, no yield
curve, no crises, no fractional reserve. That is a genuine loss of an entire genre of player activity,
and the user should confirm it rather than have it assumed (Decision 7).

**What is given back, cheaply.** An **engine-enforced escrow**: a player places coins against a stated
condition with a named beneficiary, and the engine holds them. That is "trust me with your money" in a
form the simulation can actually keep. My working hypothesis — unverified — is that pure custody and
escrow services are the ones that historically survived where deposit banks did not.

**And one hard interface rule that arrives with the block system: never render a balance the engine does
not hold.** A convincing player-built sign reading "your balance: 12,400" is the modern form of the
note-hiding scam. Player-built displays may show engine-held values only.

### 4.4 Lending — decided explicitly: credit may NEVER create money

**Fully collateralised only.** A loan is the lender's balance moving to the borrower, and at the same
moment a game container of the borrower's collateral moving into a holding the borrower cannot open. Two
matched movements of things that already existed. Repayment moves the money back and releases the
container. Default moves the container to the lender. **The total amount of money in the world is
unchanged at every step.**

**Uncollateralised lending is not offered and is not expressible**, because there is no operation that
credits an account without an equal debit — apart from the mint, which requires material to have
physically entered a vault. This is enforced by the single conservation assertion (§6.1), so fractional
reserve is **unconstructible, not merely forbidden**.

**Uncollateralised lending exists socially and costs zero lines:** one player transfers coins to another
and trusts them. The game records nothing and enforces nothing. That is what happens in every game
anyway and it creates no money.

**Default is checked only when somebody opens the loan, never by a timer.** The lender is the one who
wants their money back, so the lender is the one who looks. A loan neither party touches never defaults;
a region asleep for five years wakes owing nothing and having forfeited nothing.

**Lending is NOT in the first version, and this must be stated rather than implied.** The collateral is a
container of goods, containers live in regions, and until regions have durable storage tied to place
identity a borrower can default for free by parking the collateral somewhere about to be switched off.
Going dark does not help the borrower — the engine holds the collateral, not them — but going *dormant*
does. Do not ship a lending feature whose collateral a routine shutdown can eat.

---

## 5. Markets, paid in currency

The market shape is unchanged: physical consignment, seller flies in and offloads, sale happens locally
when a buyer arrives, venue owner takes a per-sale share, seller may withdraw unsold goods.

### 5.1 The consignment sale end to end

**Step 1 — list.** The seller flies to the venue and moves cargo from their own container into the
venue's consignment container. This is a **game** container move: item positions genuinely change
custody. The listing is a *reference* to those positions and is never the container of record — so
"economy absent" means "the claim is unenforceable", never "the goods are nowhere". The listing records:
the positions, the asking price, **the venue's per-sale cut as it stands right now**, and the proceeds
destination.

**The cut is fixed at listing time. This is a design requirement, not an implementation detail.** The
settlement uses the rate stored on the listing, never the venue's current rate. Consequences, all good:
the seller knows the fee when they commit the goods; a venue owner cannot raise the rate on goods already
in their hold; a rate change affects only new listings; and no notification, no migration and no
awakening is needed, because the rate travels with the listing. It is on-touch semantics by construction.

**Proceeds destination is a player choice at listing time, and it is exactly the local/global split the
user asked for, surfaced as gameplay:** *"pay me in coin here"* (you must fly back to collect) or
*"credit my account"* (arrives wherever you are, may cost a routing fee, needs the venue to be able to
reach the outside world).

**Step 2 — buy.** A buyer arrives **physically**. Goods never travel. The governing law, copied verbatim
from the only market model compatible with a private per-server world: **orders and information may
travel; goods never do.** The shipped precedent states it explicitly — remote trading skills "do not
magically move goods around", and purchased items remain where the sell order was fulfilled. Copy it into
the design in exactly those terms. The counter-example is stark: the one game that bolted a frictionless
global goods market onto a game about acquiring goods shut it down two years later and its director said
he would have switched it off on day one.

**Step 3 — pay.** Three cases, and the interface offers them in this order:

- **Coins in your inventory — always accepted, always works**, including with the entire economy off and
  the station cut off from the universe.
- **A note you are carrying** — same, with the branch cashing it later.
- **From your account** — accepted only when the venue can reach the account service. Otherwise the venue
  says "your bank is not here". That is the honest sealed-world answer, it is good gameplay, and it is
  why coins have a permanent role.

**Step 4 — settle (the coin path: one write, one tick, zero cross-server bytes).** In a single shard-local
mutation, in this order:

1. Check the listing is still live and the price and quantity match what the buyer accepted (a plain
   equality check — the venue is the sole writer of its own listings, so no fence is needed).
2. Check the buyer's purse covers the price. If not, nothing is written.
3. Compute the split in integer minor units:
   `fee = max(floor(rate_at_listing × price), minimum_fee)`;
   `burn = ceil(burn_share × fee)`; `owner_cut = fee − burn`; `seller_proceeds = price − fee`.
   The three always sum exactly to the price. **The remainder is assigned to the burn, deterministically,
   never to the owner** (§8). No floating point anywhere.
4. Apply: buyer's purse down by the price; the burn counter up; the owner's strongbox up by the cut; the
   seller's proceeds box (or a remote-credit advice) up by the proceeds; the item positions move to the
   buyer's hold.
5. Append one row to the venue's local journal: buyer, seller, item identity, price, split, tick.

**Atomicity is trivial and total: it is one system in one tick in one process with no await points and no
I/O.** Either every mutation in step 4 lands or the precondition in step 2 failed and none of them did.
There is no saga, no idempotency key, no compensating action. It is atomic *because it is one write*, not
because anyone made it atomic. **This is the entire payoff of keeping cash physical.**

> **[REV 07-27] THIS PATH DOES NOT SHIP, and that is the largest single cost of the revision.** The design
> deliberately made this the default and §5.2 the exception above a size threshold, precisely because this
> one cannot half-happen and §5.2 can. **Accounts-only deletes the default and promotes the exception to
> universal.** Consequences, all applied below: §5.2's two-sided lock must be **hardened rather than
> offered**; the "make coin the default" recommendation at the end of §5.2 is **struck**; and gate G-13 is
> unachievable and inverts (§0.3). **What survives with the economy off is the same steps 1–5 with a
> BARTER bundle in place of the currency entry** — one shard-local write, zero cross-server bytes, works
> under total partition — which is why price-as-a-bundle is now structurally required rather than merely
> the highest-value shape-now item. **Steps 3–4 above are the specification of the later coins extension**
> and are retained verbatim for that purpose.

**Step 5 — collect.** The seller returns and moves proceeds from their box into their purse, or banks
them at a branch. Unsold goods are withdrawn the same way — a game container move, working with the
economy entirely off.

### 5.2 The account-paid path, and the two-sided lock

This is the only place in the design where a crash leaves a visibly half-finished state a player can see.
It exists because money and goods are under different authorities.

> **[REV 07-27] Under accounts-only this is THE sale path, not a path. Three amendments:**
> **(1)** The two-sided lock is **mandatory and hardened**, not offered — there is no simpler alternative
> to fall back to.
> **(2)** Step 1's "credit a venue escrow position" is sharpened into a **HOLD on the buyer's balance at
> the service**, released by the buyer's own next login (§0.2 residue 1). Paying the seller straight out
> leaves a buyer who paid for nothing with no way to be made whole if the venue's region dies between
> steps 2 and 3. **The account row therefore carries a held field and a monotonic stamp from the first
> version.**
> **(3)** Stranded holds are now created by **every** purchase rather than one in ten — a tenfold rise in
> the largest accidental removal from circulation in the economy (§0.4). The release rule and the published
> immobilised total are load-bearing, not hygiene.

1. **Reserve, locking BOTH sides in the same step.** The venue marks the listing *reserved* (the seller's
   withdraw is now refused) and sends one payment request: debit the buyer, credit a venue escrow
   position, keyed by a payment correlation id.
2. The service performs it in one local store transaction and replies **after** its barrier.
3. The venue moves the goods **and** settles, conditional on the goods still being present. **Settlement
   is the thing that moves them.**
4. If the venue dies between 2 and 3, the reservation refunds **on touch** (§6.4).

**Why both sides must lock at reservation time, not at settlement.** A vetter found the hole: with only
the money locked, a seller can accept a reservation, withdraw the goods while the payment is travelling,
and let settlement fire against an empty listing — **the buyer pays for nothing.** This cannot happen on
the coin path, because a coin sale is one write with no window at all; the hazard is created entirely by
money no longer being physically present, which is exactly what the reversal introduced.

**The seller may cancel a reservation at any time, returning the money.** It is their shop. That also
answers the griefing case where somebody reserves many listings to lock them out — a reservation now
costs its full price up front, so mass-locking a market requires actually having the money.

~~**Recommendation: make coin the default and offer the account path only above a size threshold.**~~ Every
sale that goes through the account path is a sale that stops working when communications drop.

> **[REV 07-27] The recommendation is STRUCK — there is no coin default to fall back to.** The residual
> sentence is retained because it is still true and is now the honest statement of the cost: **every
> currency sale in the game stops working when communications drop.** The fallback is barter, not cash.

### 5.3 Want-ads (the buy side)

> **[REV 07-27] With no coins there is nothing local to fund a want-ad with, so the "never funded from an
> account" rule inverts: a currency want-ad is funded by a HOLD at the service, placed when the ad is
> posted and released when it is withdrawn.** That satisfies the escrow invariant below by relocating the
> escrow rather than weakening it — the funds are removed from the buyer's spendable balance at post time,
> exactly as the shipped precedent requires. **The cost is that a currency want-ad cannot be FILLED while
> the venue is out of contact**; a **barter** want-ad (a bundle of materials wanted for a bundle offered)
> still fills in one shard-local write under total partition, which is the accounts-only form of this
> section's guarantee. The locally-funded coin form below is the later extension.

A want-ad is a listing in the other direction: *"I will pay N for M units of X."* Under the escrow
discipline below it must be **locally funded**: the buyer deposits coins into the venue's want-ad escrow
container when posting. Consequences: a want-ad settles in one shard-local write exactly like a sale;
it works under total partition; it cannot be backed by money the settling server cannot see; and an
unfilled want-ad is withdrawn by taking the coins back out. **Want-ads are never funded from an account
in the first version**, because that would require the account service to be reachable at fill time by a
party who is not present, which is the two-sided lock problem with nobody standing there to resolve it.

**The escrow invariant, adopted from the shipped precedent and stated as law: no offer, order, want-ad or
contract may ever be backed by value the settling server cannot see.** Buy orders in the reference game
hold 100% of funds from the moment they are placed (partial escrow was removed in 2020), and contracts
remove and hold items and cash on creation. Full escrow converts a distributed-credit problem into a
local-funds problem, and that is what lets a market settle asynchronously with neither party present or
reachable.

### 5.4 What stays local, what now crosses

**Entirely local — zero cross-server bytes:** hand-to-hand payment; listing; withdrawal; the coin sale
and its fee split; the want-ad post and fill; note handover; minting; melting; the seller's collection of
proceeds in coin; currency exchange between two players standing in the same place (which is just a
listing whose good happens to be money — no computer counterparty, no formula, no oracle, rates emerge
from listings).

**Now crosses, and did not under cargo payment:** deposit; withdrawal; a sale paid from an account (two
round trips); remote proceeds credit; note issue and cash; balance read; wages and treasury movements.

**Traffic budget.** A busy station running 1,000 sales between settlements generates: 0 bytes for coin
sales, and for remote-proceeds credits one advice per distinct seller. At 200 distinct sellers × ~40
bytes = 8,000 bytes over 1,000 sales = **8 bytes of network per sale**, amortised.

> **[REV 07-27] Recomputed: ~80 bytes per sale, and still nothing.** Every currency sale is now a round
> trip, so 1,000 sales generate 1,000 request/reply pairs at ~40 bytes each = ~80,000 bytes, plus the same
> 8,000 bytes of proceeds advice. **80 bytes of network per sale against a per-sale latency of 42 ms
> worst-case and a service ceiling of 50,000 payments a second (§0.2).** The split at the top of this
> section moves accordingly: only the **goods** operations stay entirely local — listing, withdrawal,
> container moves and a barter-bundle sale. Every currency operation crosses.

### 5.5 The abuse currency opens that cargo payment did not

Cargo payment was a barter of two physical things and both sides were locally visible and heavy. Six
things change:

1. **Wash trading is now free at zero burn.** Sell to your own second account at a venue you own: the fee
   is a transfer to yourself and the round trip costs exactly nothing. Cargo payment had a hauling cost.
2. **The fee can be dodged by splitting.** Cargo could not be split below one item.
3. **Reservation griefing** — locking listings you never intend to buy. Cargo payment had no reservation
   step at all.
4. **Value transfer became invisible.** Cargo money forced the seller to haul cargo to the buyer:
   expensive, visible, and it had a location. A transfer is a row with no location. **The reversal
   converted a logistics problem into a spreadsheet entry**, and that is the real-money-trading bill.
5. **Notes are untraceable in the middle.** The service sees an issue and, months later and to a different
   account, a redemption. Everything between is invisible.
6. **Price-tag confusion.** Cargo prices were self-evidently physical. Coin magnitudes are not, and the
   historical scam is hiding low-value notes among high-value ones — only fixed when the display added
   separators.

All six are handled in §8 with a prevented / detected / accepted verdict each.

---

## 6. Correctness

### 6.1 The conservation identity

> **At every commit boundary:**
> sum(all account balances) + sum(all held/escrow positions) + sum(all coin in every purse, container,
> strongbox, proceeds box and want-ad escrow) + sum(all outstanding note face values)
> **= total minted − total melted − total burned − total lost.**

Every operation is a vector of signed deltas and there is **exactly one assertion, in one monomorphic
helper**: the vector sums to zero, with **exactly three named exceptions**, each requiring a proof:

| Exception | Proof required |
|---|---|
| **Mint** (positive) | Matching material destruction in the game's item store, in the same tick, at the same server |
| **Melt** (negative) | Matching material creation from a vault that physically held it |
| **Confiscation / restitution** (either sign) | An operator identity and a case number, ledgered (§8) |

Enumerate the operations and **assert the set size**, so a fourth exception cannot be added silently.

**The honest limit on what is provable in production, and no candidate design stated it.** The existing
conservation oracle works by reading every node's private world in one process
(`crates/harness/src/oracle.rs:396-424`, `crates/harness/src/topology.rs:570-604`) — only possible because
the harness runs the whole cluster inside one binary. **In production there is no cross-server query at
all.** Therefore:

- **Account money: conservation is provable in production**, because the sum is a local read inside one
  process. This is a decisive argument for the central shape.
- **Coin: conservation is provable only as a harness property.** The production guarantee for pockets is
  **"never duplicated"** — which the single-owner compare-and-set genuinely gives — and **NOT "never
  lost"**. Say this plainly to the user (Decision 14).

> **[REV 07-27] Under accounts-only the second bullet has no subject, and this is the cleanest single win
> of the revision.** All value sits in one book at one writer, so the identity above is a **local read
> inside one process, assertable live every tick** — not only in the harness. **Decision 14 is fully
> retired**: money is provably never duplicated *and* provably never lost, in production. The
> "your coins died with the server, and the answer is no" support policy disappears with it, and so does
> **D-91** (production supply telemetry), whose only purpose was counting world-held coin — the entry that
> risked putting economic data on a lifecycle message. **The identity must nevertheless be written from
> day one with the world-held-coin terms present and summing to zero** (§0.6 item 7), so that adding coins
> later changes which terms are non-zero and never the assertion's shape.
>
> **The one residual support case is not money loss:** a buyer can be debited for goods they never receive
> if the venue's region dies mid-purchase. Conservation still holds — the money is in a hold — and it is
> closed by the service-side hold released on the buyer's next login (§0.2 residue 1).

Note also that the existing oracle **tolerates loss within a per-kind budget**
(`crates/harness/src/oracle.rs:440-455`) and forbids only duplication. **For money the loss budget must be
exactly zero in the harness**, which is a stricter contract than anything currently proven anywhere, and
that tension should be resolved explicitly rather than assumed away. **[REV 07-27] The tension dissolves:
with no world-held value there is no loss term to budget, so the money loss budget is zero BY
CONSTRUCTION rather than by a stricter-than-anything-else contract. §12.2 item 13 is closed for money and
remains open for everything else.**

### 6.2 The crash interleavings, each closed

> **[REV 07-27] Re-verified against the code, and the status of all six changes.** I1, I3 and I4 could only
> ever move money because money rode **inside a moving player**; with no carried value they are
> **structurally unreachable as money bugs** and drop as economy blockers. I2 is different and the
> reviewers were emphatic about it: **it is not a money bug at all — it strands the PLAYER ENTITY, with
> the saga reporting success and no alarm firing — so it keeps its blocking status on its own merits as a
> game-correctness defect. Dropping coins buys no permission to leave it.** I5, I6 and I7 are unchanged or
> narrowed. Each is marked below.

**(I1) The double-apply — CONFIRMED, and it is the blocking defect.** **[REV 07-27] Still confirmed in
code; DROPS as an economy blocker.** With no value in the blob a redelivered crossing re-applies a pose,
not a balance. The durable applied-once record is still owed — **rescoped to the account service's own
store, where it is ~150 lines in a file the service already opens, not a new shard storage tier.**

The record that stops an incoming step being applied twice is a plain in-memory set with a
`Default`-derived constructor, re-inserted empty at every boot:
`crates/sim/src/stub.rs:1120-1121`, installed at `:1218`. The re-driver **is** durable: the saga snapshot
is persisted (`crates/node/src/saga_runtime.rs:192-201`) and the post-commit timeout re-issues both the
route swap and the value-bearing crossing (`crates/sim/src/saga.rs:1064-1080`), with an inline comment
reading *"the dest journal dedups a redelivery"* — **which is false across a receiver restart.**

Exact sequence with money in the blob: the coordinator wins the compare-and-set and persists in the same
tick barrier → sends the crossing carrying the purse → the destination applies it and records the key in
RAM only → the destination is killed (SIGKILL *or a routine reap*) → it restarts with an empty set → the
re-drive deadline of 8 ticks (`crates/sim/src/saga.rs:86`, = 160 ms at 50 Hz) fires → the identical
crossing is re-sent → **the purse is credited a second time.**

Ledgered still-open at `docs/design/DEFERRED.md:2766-2789`. It is invisible today only because the
destination has no disk, so the credit is lost on the crash anyway and no test can observe the
duplication. **It becomes a live money printer the same week any balance survives a restart** — the most
expensive possible moment.

**Closed by three things:**
- **(a)** The durable applied-once table lands **strictly before** the first durable balance, never in the
  same slice. Cheaper than believed (§2.5): a second key family in a file the binary already opens.
- **(b)** The **totals-not-deltas wire law** — the best engineering idea in any of the three candidate
  designs, adopted verbatim: *any money-bearing message aimed at a machine without a disk states an
  absolute value at a stamp, never a change; only a machine with a disk may apply a change.* Applying a
  total twice is a no-op, so the defect becomes **structurally unable to duplicate money**.
- **(c)** Correct the false comment at `crates/sim/src/saga.rs:1078` in the same commit, so nobody else
  reads it as a guarantee.

**The exception to (b), which a vetter caught and which nobody else noticed: object creation cannot be
expressed as a total.** "Your coins are now 500" is a total; "make a stack of 500 coins" is inherently a
change, and applied twice it makes 1,000. **Fix: the coin stack's own identity is minted by the
disk-holding side and derived from the withdrawal's identifier.** A redelivered withdrawal then names the
same object, the destination sees it already has it, and nothing is created. This reuses the property the
game already needs — item identities are never reused — and it means this particular path does not need
the durable table at all. **But it must be designed deliberately: if the destination mints the identity,
redelivery doubles the money.**

**And the exploit form, which is repeatable and has a watchable trigger.** Fly into a station carrying
coins, walk straight to the branch and deposit them (a durable credit at the service), then have the
station die before your arrival has finally settled. The arrival is re-sent, replaying the state you
arrived with — including the coins you already banked. **You now hold the money twice.** Regions are
switched off on demand all the time, so the attacker need only notice one coming. A note makes it worse:
one high-value object designed to be carried and cashed elsewhere. **Second closure: no operation may
move value out of a newly arrived player until that arrival is finally settled** — a few seconds of
"your funds are still arriving", enforced on the destination.

**(I2) Silent money destruction with the transfer reporting success — the worst finding, and it was in
none of the candidate designs.** **[REV 07-27] RE-VERIFIED IN THE CODE AND RETAINED AT FULL SEVERITY.**
`on_saga_promote` calls `journal_step(cmd.transfer, PROMOTE_STEP)` **before** `promote_apply`
(`crates/sim/src/stub.rs:2274-2298`), `promote_apply` returns without flipping when the crossing has not
landed (`:2327-2340`, bumping `promote_before_crossing`), and the ack at `:2289-2297` is unconditional and
outside the FirstApply gate — so the step is permanently consumed, a redelivery takes the AlreadyApplied
branch, and the saga reaches Done with the subject owned by nobody. The only crossing re-driver is still
the Swapping timeout arm (`crates/sim/src/saga.rs:1069-1080`). **Retitle it: this destroys the PLAYER, not
the money.** Accounts-only removes the money from the loss; it removes nothing from the defect.
**D-84 stays 🟥 BLOCKING as a game-correctness defect and must not be downgraded because the economy
stopped needing it.**

The promote handler records the step **before** it knows whether it did anything:
`crates/sim/src/stub.rs:2274` journals the promote step, then `promote_apply` at `:2327-2338` checks
`if !applied.is_applied(cmd.transfer, STUB_CROSSING_STEP)` and **returns without flipping** when the
crossing has not landed. The ack is then sent **unconditionally** at `:2289-2297` ("never wedge the saga
in Promoting"). So the step is permanently marked applied, a redelivered promote takes the
already-applied branch at `:2285` and never retries the flip, and the saga advances
Promoting → Releasing → Done **believing it succeeded.**

Meanwhile the **only** re-drive of the value-bearing crossing lives in the Swapping timeout arm
(`crates/sim/src/saga.rs:1069-1080`), a phase that exits on the gateway's route-swap ack, typically within
a tick or two. **Past Swapping there is no crossing producer at all** — the code says so itself at
`crates/sim/src/stub.rs:2329-2337`. The flip was also deliberately relocated out of the crossing handler
into the promote handler (`:2250-2260`), so a crossing that arrives *late* lands the pose and still never
triggers a promote.

Net effect with money in the blob: the source is already demoted to Ghost, the destination never becomes
owner, **the entity and everything on it is held nowhere, the money is gone, and no alarm fires.**

**Closed by three small changes, all required before value rides the blob:**
- Move the promote journal to **after** the effect, so the deferred branch does not consume the step —
  record-after-effect is already the stated discipline everywhere else (`crates/sim/src/stub.rs:1108-1112`).
- Give the crossing a producer that survives past the route-swap phase: re-emit it on the Demoting and
  Promoting timeouts too, exactly as the forward re-home already does with its own dedicated re-drive
  egress (`crates/sim/src/saga.rs:447-453`).
- Make the loss **loud**: the counter `promote_before_crossing` already exists
  (`crates/sim/src/stub.rs:2339`) — assert it is zero at scenario quiescence, and treat a Done saga whose
  subject is owned by nobody as a hard oracle failure rather than a statistic.

**(I3) Money created by an automatic rollback.** **[REV 07-27] DROPS ENTIRELY — structurally unreachable.**
`build_rehome` reconstructs from the pose stashed at the source cut
(`crates/node/src/saga_runtime.rs:862-883`, `:834-841`) and `ReHomeState` has exactly one arm today,
`PoseOnly(StampedPose)` (`crates/wire/src/intershard.rs:658-661`), so nothing but a pose is restorable.
With the value in a row at the service that no rehome touches, the duplication has no mechanism.
**D-85 and prerequisite P3 die with it; the purse version and the zero-on-stale rule are unneeded.**

The forward re-home used when a destination dies reconstructs the entity from the pose the **source**
flushed before the transfer: `build_rehome` (`crates/node/src/saga_runtime.rs:862-883`) takes
`flush_pose`, stashed at the source's cut (`:836-840`) and durably persisted with the saga (`:192-201`),
so the rollback survives an orchestrator restart too.

Sequence: cross carrying 1,000 → deposit 800 at a branch (durable, at the service) → the destination is
permanently killed → the confirmed-dead detector fires (`crates/sim/src/saga.rs:141-160`) → the subject is
re-homed onto a fresh shard from the **source's** snapshot → the player has 1,000 while the 800 also
exists. **800 created from nothing.** This is not a defect of the rescue path so much as an unavoidable
property of restoring from a checkpoint older than the last externalised effect.

**Closed by law plus a version:** a rollback of carried money is **always a loss and never a re-credit** —
refunding is the only path that creates money. Concretely, the carried blob carries a **monotonic purse
version**, and any restore whose version is older than the highest the receiving side has seen for that
account **zeroes the purse rather than restoring it.** Note the destruction case lands first: the rescue
payload has exactly one shape today, `ReHomeState::PoseOnly(StampedPose)`
(`crates/wire/src/intershard.rs:658-661`), and until the state arm lands a routine rescue silently zeroes
a purse anyway — so the creation case only becomes reachable once the arm is built.

**(I4) A deterministic silent loss during a universe re-genesis.** **[REV 07-27] DROPS as a money blocker;
survives as a game bug.** Both refusals are confirmed in code (`crates/sim/src/stub.rs:2765-2772` for the
crossing, `:2469-2474` for the rescue adopt — return, no ack, no journal). With no value in the payload
they refuse a **pose**; combined with I2 they still silently drop an **entity** during an epoch change,
which remains a real defect. **D-86 drops; the substance folds into D-84's re-drive fix.**

Both pose-placing entry points refuse a message whose universe epoch does not match, **without
acknowledging it and without applying anything**: `crates/sim/src/stub.rs:2765-2772` for the crossing and
`:2469-2472` for the rescue adopt. Combined with (I2) this turns a probabilistic loss into a certain one:
during an epoch change the crossing is refused, the promote arrives, journals itself, defers, acks, and
the saga completes with the entity held nowhere. The refusal is deliberate and correct for a pose (never
place an entity at a stale celestial position) and **silently catastrophic for value.**

**Closed by:** value must not be refusable on a pose-validity ground. Either split the message so the
value half and the pose half are separately acceptable, or make an epoch-mismatched crossing carrying
value a **loud terminal fault** that aborts the transfer and returns authority to the source. The counter
`crossings_epoch_mismatch` already exists (`:2770`) — gate on it being zero in any scenario where value
crosses.

**(I5) Value on a dead server's entity is frozen indefinitely.** **[REV 07-27] NARROWED to an entity
problem, not a money problem.** A parked re-home no longer strands a balance — the balance is at the
service and is reachable from any shard the player logs in on. The operational alarm is still owed, and
**Decision 15 (frozen vs lost) becomes moot for money and remains open for the entity**, which resolves
§12.2 item 12's dependency on it.

When a server holding a durable entity dies with no transfer in flight, the recovery arms a saga that
**parks and never adopts**: `process_rehome_starts` (`crates/node/src/saga_runtime.rs:1978-2010`) always
parks with no recoverable pose, documented at `:1929-1932` and `:1976-1980`. It also takes and holds the
transfer lock on the entity key (`crates/sim/src/directory.rs:532-541`) — deliberately, to keep the reaper
off — and the parked saga is durably persisted so it survives an orchestrator restart. Money on that
entity is neither lost nor duplicated but is **unreachable for an unbounded time**, with no operator path
to release it short of the deferred checkpoint work.

**Closed by:** acceptable as a temporary state, unacceptable as a *silent* one. Report parked re-homes as
a first-class operational alarm with age (the staleness field already exists,
`crates/node/src/saga_runtime.rs:2237-2244`). And **decide now** whether a parked account is displayed to
the player as frozen or as lost — the two produce very different support loads and the choice cannot be
made after the fact.

**(I6) Ordinary crash cells, all closed by existing machinery.**

- *Service crashes after applying, before replying.* Rehydrate reads the durable applied-once record and
  the balance from disk; the re-send hits already-applied. Balance unchanged. Rehydrate touches **zero**
  account rows.
- *Service crashes after committing, before sending.* Persist-before-effect is already the barrier's rule
  (`crates/node/src/saga_runtime.rs:2513-2517`). On rehydrate it re-sends; the message states a total, so
  applying it twice is identical.
- *Reply lost, requester re-sends.* Durable dedup at the service; re-ack without re-effect. Note this is
  the **correct asymmetry** — the re-sender has no disk and the de-duplicator does.
- *Shard dies before sending a deposit.* The coins are lost with everything else on that shard. A loss,
  not a duplication.
- *The buyer disconnects mid-payment.* The gateway's applied-steps record is per-session and in memory by
  design (`docs/design/DEFERRED.md:2765-2777`) and it holds at most **one** in-flight transfer per session
  (`:2790-2801`). A player who buys and immediately closes the client is the ordinary case. **Closed by
  rule: a payment is not a transfer and must never occupy the per-session transfer slot**; a payment
  confirmation is recoverable by re-asking the durable authority after reconnect, never by replaying a
  gateway's memory.

**(I7) The fence trap any new money saga family would inherit.** There are two abort paths with opposite
fence behaviour: `abort_cas` is fence-**moving** (`crates/sim/src/directory.rs:589-606`) and `abort_clear`
is fence-**neutral** (`:608-640`), and the doc warns that routing a terminal abort through the moving arm
"strands the source one fence behind the directory and wedges a post-abort logout". The no-bump
correctness is also stated to be contingent on value-bearing messages staying strictly post-commit
(`:634-637`). **If any money saga family is ever built, pin both preconditions as tests rather than
comments** — cheap now, nearly impossible to retrofit once a second family exists.

### 6.3 The region-switched-off cases

**An account is a non-event, and that is the entire reason for the design.** Accounts are not hosted by
regions. A station going dormant, a system torn down, the whole galaxy branch reaped — none of it touches
a balance. `kill_realm` (`crates/node/src/rlm_runtime.rs:278-310`) has no reach here.

> **[REV 07-27] The single most important thing accounts-only does NOT fix.** It removes **money** from a
> killable place. It removes **nothing else**: consignment cargo, a venue's stock, want-ad escrow, loan
> collateral and — decisively — **the physical vault of backing material** all still sit in a region that
> reports Empty and is killed. **The exploit two paragraphs down therefore SURVIVES accounts-only and is
> arguably sharper**: you cannot carry coins out any more, but your account credit is durable while the
> vault behind it dies, which leaves currency that is **permanently unbacked**. So P5 / D-87 are
> **narrowed, not dropped** — they stop blocking money and keep blocking markets (M-3) and mints (M-4)
> exactly as before. **Correction to the mechanism below, verified in code:** teardown is *not* an outright
> kill — the spawner SIGTERMs the process group, polls for a drain grace and only then SIGKILLs
> (`crates/bins/src/proc_launch.rs:214-244`), with a 2,000 ms linger deployed. **A graceful window exists;
> what is missing is any flush CODE** (`crates/bins/src/bin/shard.rs:314`: *"the shard holds no un-fsynced
> durable state"*). That makes save-on-shutdown a change **inside an existing lifecycle window** rather
> than a new lifecycle phase, and P5 correspondingly cheaper than stated.

**Coin and every game container are destroyed by a routine teardown, and this is normal operation, not a
crash.** The signal that lets a region be reaped is emitted purely on occupancy and has **no notion of
pending value**: `crates/sim/src/stub.rs:4509-4521` reports Empty the moment the occupant set is empty,
where occupants are owned simulating dots plus held transients (`:4494-4506`) and nothing else. That
report drives `empty_confirmed` → `desired_alive` → `teardown_ready` (`crates/sim/src/rlm.rs:437-490`), and
the kill is an outright process kill with **no save step** (`crates/node/src/rlm_runtime.rs:279-293`). The
two-phase drain (`crates/sim/src/rlm.rs:552-573`) is a **veto window, not a flush window** — there is no
persistence step anywhere in it.

So a station holding a venue's takings, a seller's uncollected proceeds, a consignment hold, a want-ad
escrow, loan collateral or a mint's backing vault reports itself Empty the moment the last player walks
out, and is killed shortly after with all of it in RAM.

**The exploit form:** mint at a station you know is about to be reaped, carry the coins out, and the
obligation behind them evaporates. Repeat and **the currency quietly becomes unbacked** — worse than money
creation, because it is invisible until the first refused melt, at which point the coin is unbacked
everywhere at once.

**The answer is not a stopgap; it is the build order.** **[REV 07-27] And the same reasoning, applied one
level up, is what forbids a local spending allowance — see the binding rule at §0.2. An allowance is money
held on a machine that is killed as routine; it could not be switched off until settled, which is this
exact coupling with a different name, and unbounded under partition. With no allowance, the reconciler
never learns the economy exists.** One vetter proposed making the Empty self-report
conditional on the shard holding no unsettled value, arguing it is cheaper than adding a boolean to the
teardown facts. It *is* cheaper, and I reject it anyway: it still puts economic state on the keep-alive
path, just shard-locally instead of orchestrator-locally, and under partition the effect is identical and
unbounded — **the region never spins down.** That is exactly the coupling the standing rule forbids, and
the design that proposed it admits it does not bound the wait.

**Therefore: no value may live in a place that can be switched off until per-place durable storage keyed
by realm identity exists.** That is what makes the account tier the cheap tier and the pocket tier the
expensive one, and it is why **mints and vaults are gated hardest of all** (§10). ~~Coins, hand-to-hand
payment and consignment can ship earlier;~~ a mint without durable backing makes the currency's central
claim false. **[REV 07-27] Read "value" as "value OR GOODS". Accounts-only satisfies this rule for money
and leaves it entirely unsatisfied for cargo, stock, escrowed goods, collateral and the backing vault, so
the gating of M-3 and M-4 on P5 is unchanged.**

### 6.4 Deadlines evaluated on touch

**There is not one accrual, interest calculation, upkeep charge, storage fee or decay anywhere in this
design.** Everything time-shaped is a comparison performed when someone touches the record:

| Time-shaped thing | Who touches it | Never |
|---|---|---|
| Loan maturity | The lender, who wants the money | A timer |
| Reservation refund | **The buyer's next login** (see below) | A sweep |
| Note expiry | Recommended: none at all | — |
| Listing age | Nobody — a listing simply persists | A decay |
| Applied-once journal pruning | Pruned by age **on touch** | A sweep, which would reintroduce clock work |

**The stranded-reservation problem, which a vetter found and which would otherwise have become the largest
removal from circulation in the economy by accident.** Nobody said who touches a reservation whose sale
never completed. Arithmetic at the planning population: ~9,600 shop interactions/day; 10% paid from an
account = 960; a 1% strand rate ≈ 10/day; at 500 coins each = 5,000/day = **150,000/month = 2.5% of a
6,000,000 supply**, compounding to **26% a year — roughly twice the designed loss drain.**

> **[REV 07-27] Recomputed for accounts-only: TENFOLD, because every purchase now creates a hold.** 100%
> account-paid → 9,600/day × 1% = **96 strands/day × 500 = 48,000 coins/day = 1,440,000/month = 24.0% of
> supply per month.** With the release-on-next-login rule actually landed the standing pool settles to
> about one day of strands = **0.80% of supply**, and only the never-returning fraction immobilises
> permanently: at 5%/month churn, 0.05 × 1,440,000 = 72,000/month = 1.2%/month → (1−0.012)¹² = 0.86513 =
> **13.5%/year, the same order as the drain accounts-only removed.** **The fix below stops being a safety
> net and becomes load-bearing: it must land in the same slice as the account-paid purchase path, never
> after it, and the immobilised total must be published from the first version** — an undetected
> immobilisation is indistinguishable from a leak *and* reads as perfectly healthy to the conservation
> gate.

**Correction to the vetter's framing, and it matters for §6.1:** stranded money is **immobilised, not
destroyed.** The conservation identity is not violated — the coins are still in a held position and still
counted. But the *circulating* supply falls, which produces the same deflationary effect while looking
fine to the conservation gate. **Fix: the buyer's own next login touches their outstanding reservations.**
That is a bounded per-account list checked when a session opens — an on-touch rule with no timer and
nothing awake. **And publish the total sitting in reservations as a first-class dashboard number from the
first version**, because an undetected immobilisation is indistinguishable from a leak.

**The proof obligation:** a region asleep five simulated years and an identical region awake but untouched
five years must produce identical state.

### 6.5 The gates and chaos cells

**A correction to the five-year gate every candidate design proposed, and it would have gone red on a
correct system.** `group_commit` writes the clock record **unconditionally every tick** and calls
`store.commit()` regardless of whether anything changed (`crates/node/src/saga_runtime.rs:2507-2511`). At
50 ticks/second an idle coordinator writes and fsyncs fifty times a second forever. A whole-store
byte-identity assertion would fail on tick one, get "fixed" by weakening it, and stop testing the thing it
exists to test.

**Scoped correctly:** after five simulated years with zero economic requests, assert (a) the set of keys
under the money family tag and their bytes are identical, **and** (b) the count of money-family deltas
staged into the barrier is exactly zero. The second assertion is the one that actually proves nothing is
clock-driven, and it is stronger than a whole-store comparison because it is specific.

> **[REV 07-27] Six gates drop, two are reclassified out of the economy, one is replaced by its inverse,
> two are simplified, ten are unchanged, and two are new.** The status column below is added; the original
> text of every row is untouched.
>
> | Gate | Status under accounts-only |
> |---|---|
> | G-3, G-4, G-7, G-11, G-12, G-20 | **DROPPED as money gates.** G-11 and G-12 survive as carried-state gates owed by the seam work, not by the economy |
> | G-5, G-8 | **RECLASSIFIED** — still owed, as game-correctness gates (D-84) |
> | G-13 | **REPLACED by its inverse** — see below |
> | G-1, G-19 | **SIMPLIFIED and strengthened** — G-1 becomes a production property, not harness-only; G-19 drops the coin terms |
> | G-2, G-6, G-9, G-10, G-14, G-15, G-16, G-17, G-18, G-21 | **UNCHANGED** |
> | **G-22 (new)** | **Partitioned venue degrades cleanly** — with the service unreachable, a currency purchase is refused with a normal outcome, a barter bundle still settles in one shard-local write, and nothing times out, wedges or fails to spin down. **This replaces G-13, which is unachievable without cash.** |
> | **G-23 (new)** | **Hold release on next login** — strand a hold by killing a venue mid-purchase, log the buyer back in, assert the hold releases exactly once and the published immobilised total returns to zero |

| # | Gate | What it proves |
|---|---|---|
| G-1 | **Conservation under chaos** — a property test over random operation streams, plus a run under the full kill-9 matrix, asserting the identity in §6.1 at every commit boundary with a **zero** loss budget | Money is never created or destroyed |
| G-2 | **Zero-sum assertion** — every operation's delta vector sums to zero, with exactly three named exceptions, and the exception set size is asserted | A fourth exception cannot be added silently |
| G-3 | **THE double-apply cell (RED-LISTED — it fails today)** — kill a receiver immediately after it applies an inbound value-bearing step, restart, let the sender's re-drive fire, assert the balance moved once | Turning this green **is the definition** of "balances may become durable" |
| G-4 | **The bank-then-rollback cell** — arrive, bank the coins, kill the destination, let the re-drive fire, assert the money moved once | The repeatable exploit is closed |
| G-5 | **Promote-defers-then-journals** — kill the crossing, deliver the promote, assert the subject is **not** left owned by nobody and the saga does **not** report success | Silent money destruction (I2) |
| G-6 | **Same-barrier proof** — tear the process between the credit and its applied-once journal entry; assert they are never separable across a durability boundary | The two-barrier duplication window |
| G-7 | **Rollback loses, never credits** — kill an owner and rescue from an older checkpoint after local spending; assert a loss and that no path re-credits | (I3) |
| G-8 | **Epoch-mismatch** — assert `crossings_epoch_mismatch` is zero in any scenario where value crosses | (I4) |
| G-9 | **Five-year dormancy, money-family-scoped** — as corrected above, plus the differential form (asleep vs awake-and-untouched) | Nothing is clock-driven |
| G-10 | **Economy-absent build** — compile with the economy removed and run the entire accumulated suite, asserting byte-identical results | The economy is optional |
| G-11 | **Mixed-epoch movement (three hops)** — economy-on → economy-off → economy-on, asserting the balance survives and no transfer is refused | §7.2 and §7.3 |
| G-12 | **Unknown-tag round trip** — a blob with a synthetic unknown optional tag survives reconstruct-and-reserialise **byte-identically** | The silent destruction in §7.3 |
| G-13 | **Local sale under total partition** — complete a full consignment sale with the account service not running and the station partitioned; assert **zero bytes** leave the shard | The market is economy-optional |
| G-14 | **Identical fixture on two shard kinds** — the same branch and payment fixture on a station profile and a system profile, written as one parameterised scenario | HR4, enforced by construction rather than by memory |
| G-15 | **Fee arithmetic exhaustiveness** — for every price in a bounded range, owner cut + burn + seller proceeds = price exactly, remainder to the burn, minimum fee applied, no floats | Conservation at the sale |
| G-16 | **No uncollateralised credit** — no operation credits without an equal debit outside the three exceptions | Fractional reserve is unconstructible |
| G-17 | **Custody enforcement** — an institution's signers cannot debit a depositor's balance by any path | A bank run is unrepresentable |
| G-18 | **Latency isolation** — flood payments at the budget ceiling and assert the crossing-saga latency percentile does not move | The game never waits on the economy |
| G-19 | **Load** — 10,000 accounts, 1,000 sales/second at one venue, 200 sellers per settlement batch; assert restart time is independent of accounts ever created, and assert a hot single account does not silently drop operations | The per-key serialisation trap |
| G-20 | **Note conservation** — issue, carry across a crossing, carry through a host kill and rescue, cash. Exactly once or lost entirely, never twice, never cashable after its issuing debit was rolled back | Bearer instruments |
| G-21 | **Dependency direction** — nothing in the core, sim, node or realm-lifecycle crates names an economy type | One-way, mechanically |

---

## 7. The optional-economy answer

### 7.1 Two distinct "off" states, and both must be tested separately

> **[REV 07-27] Accounts-only makes this rule CLEANER, not weaker — but the two off-states change places.**
> With no coins there is **zero money anywhere in the game's core state**: no money tag in a carried blob,
> no money entity kind, no money field in the pure crates. §7.3's highest-risk cornering item becomes
> **vacuous** — there is no money tag that could be mis-marked REQUIRED, so the movement-outage failure
> mode cannot exist — and §7.5's dependency-direction check becomes a compile-time truth rather than a
> discipline. **What a player experiences with the economy off: no money exists at all, the balance panel
> is absent, and markets run on BARTER** — fly in, leave cargo, state what you want for it, a buyer arrives
> with it, one shard-local write, zero cross-server bytes, works under total partition. **That degradation
> is clean and declared, and it holds only if price-as-a-bundle survives (§7.2).** Under the coins design
> cash provided that fallback independently; under accounts-only the bundle is the **only** thing keeping a
> market alive with the economy off, so its importance goes UP.

**(1) The account service is not deployed.** Everything physical works: movement, transfers, realm
lifecycle, combat, building. ~~**And the entire coin economy works**~~ **[REV 07-27] And the entire GOODS
economy works** — consignment listing, buying and selling against a barter bundle, withdrawal, and every
container move. What no longer works in this state is **every currency payment**, which is the honest cost
of the revision. The original text listed — purses, hand-to-hand payment,
consignment listing and buying, venue fees into strongboxes, want-ads, minting at a mint, melting at a
vault. All of it is shard-local writes that never address the service. **Of those, only the goods half
survives without coins; minting, melting and currency fees all become round trips.**

What stops: deposits, withdrawals, account-to-account transfers, wages, treasuries, note issue and cash,
balance display, remote proceeds. Branches render **closed**. Every one is a request the world was free
to refuse, and the refusal path is the same code as "the branch is out of range" — a normal outcome, not
an error path.

What must never happen: no scenario times out, no saga stalls, no realm fails to spin down, no tick budget
is exceeded, and the coordinator's store is byte-identical to a run where the service was never built.

**(2) The economy is compiled out entirely.** Deeper, and it is why the price-bundle shape matters.

### 7.2 Price as a bundle — the single highest-value thing to shape now

**A listing's asking price is a small list of entries, where an entry is either a quantity of a material
or an amount of currency.** Currency becomes one entry kind among several.

> **[REV 07-27] Under accounts-only this is promoted from "the highest-value thing to shape now" to
> STRUCTURALLY REQUIRED.** It is the only mechanism by which a cut-off station, a partitioned region or an
> economy-absent build has a working market at all — cash used to provide that fallback and no longer
> exists. **Non-negotiable, and it is the first item on the keep-the-door-open list (§0.6).**

Consequences, all of which are free today and expensive later: the market machinery is
currency-agnostic and **does not branch on whether currency exists** — it iterates a bundle; barter still
works; a build with currency compiled out still has a functioning market; and **nothing designed under
yesterday's "currency is a material" premise is thrown away.** Build the bundle first and the currency
entry second.

### 7.3 The two carried-state laws, without which "optional" is false

> **[REV 07-27] Both laws become CONTRACT TEXT WITH NO CURRENT SUBJECT, and must still be written.** There
> is no money tag in a carried blob under accounts-only, so neither law has anything to govern today —
> which is exactly why writing them now is free and why leaving them unwritten is a live migration of real
> player money the day coins are added (§0.6 item 5). **Note the enforcement point still does not exist
> either way:** `floor_ok` (`crates/core/src/tlv.rs:250-258`) has no caller outside tests. **Gate G-12
> drops as a money gate and is still owed by the seam work.**

**Law A — every money-bearing tag in a carried blob is OPTIONAL, never REQUIRED.**

This is one bit and it decides whether money is on the movement path. The version-floor handshake is
explicit: *"a dest whose `max_known_tag < required_max_tag` cannot represent the state and the transfer is
REFUSED at PREPARE"* (`crates/core/src/tlv.rs:20-22`, `:250-255`), and `required` is documented as *"a
reader that does not know this tag cannot represent the state, and the version-floor handshake will refuse
routing to it"* (`:90-92`).

**An economy-absent build is a reader with a lower `max_known_tag`.** So a purse written as a required tag
makes the economy-absent gate fail **by design**. And because re-home is the containment and docking
primitive — every dock, undock and region change is a crossing — **a refused transfer is a movement
outage.** An accounting version bump would become a movement outage. The identical bug fired before:
`crates/core/src/tlv.rs:5` records that decode-to-Default-on-any-error *"silently zeroed players on rolling
deploys"*.

Note the enforcement point **does not exist yet**: `floor_ok` is implemented (`:253`) but has no caller
outside tests — it lands with the unbuilt per-kind seam. **So the rule must be written into that seam's
contract now, while it is free.** The machine-checkable form is one line: assert `required_max_tag` is
unchanged by the presence of money.

Corollary: **the crossing precondition may never consult a balance.** A precondition that refuses a
crossing on economic grounds is the same outage by another route.

**Law B — reconstruct-then-reserialise must round-trip unknown tags verbatim.**

An optional tag prevents refusal but **not destruction.** The reader retains unknown tags as opaque bytes
(`crates/core/src/tlv.rs:165-166`, `:172`), but the write side is the unbuilt per-kind pair, and nothing
obliges a reconstructed entity to re-emit tags the reconstructing server did not understand.

Concrete failure: a player crosses from an economy-enabled shard, **through** an economy-absent or older
shard, to an economy-enabled shard. The middle server parses the blob, skips the unknown money tag,
reconstructs the player, and later re-serialises from the fields it knows. **The purse is gone. No error,
no counter, no test catches it** — and it occurs only in exactly the configuration the economy-absent gate
is supposed to run.

**This is the highest-value roadmap-cornering item in the whole review: free to state now, and a live
migration of real player money if discovered later.** Gate it with G-12, which is meaningful today and
costs nothing.

### 7.4 What happens to money that already exists when the economy goes off mid-life

> **[REV 07-27] Under accounts-only only the second paragraph applies, and it is the whole answer:**
> balances become an unenforceable claim, nothing accrues, turn the service back on and every balance is
> exactly where it was. **Nothing is destroyed by absence and nothing is created by return.** The first
> paragraph describes the later coins extension. The third paragraph — the entity-kind tag that cannot be
> feature-gated — **still applies and is item 3 on the keep-the-door-open list**, because reserving tag 3
> now is what keeps coins additive later.

**Coins are unaffected**, because they are things. They still exist, still sit in inventories, still have
weight, are still carried between realms by the ordinary machinery, still drop on death, still conserved
by the game's own no-duplication property. **The game does not know they are money; it knows they are
small dense objects.** So barter and hand-to-hand payment continue perfectly, and "economy off" degrades
to "the game plus barter", not "the game minus trade".

**Account balances become an unenforceable claim.** The rows are on disk in a family nobody reads.
Nothing accrues against them. Turn the service back on and every balance is exactly where it was. Nothing
is destroyed by absence and nothing is created by return.

**One thing that cannot be feature-gated, and it must be written down before the flag exists.** The entity
kind tag decoder errors on an unknown tag and never defaults (`crates/core/src/entity_kind.rs:54-68`). So
if coins ever become their own entity kind rather than blob content, an economy-absent build that dropped
the variant would turn **every existing coin into a hard decode error.** Reserve the tag now in the
Durable band (`:26-41` documents 0..10 as Durable with deliberate gaps; **tag 3 is free**) and record that
it is unconditionally compiled and never feature-gated.

### 7.5 The one-way dependency, made mechanical

- **Ordering and budget.** The ledger system is ordered *after* saga driving and realm-lifecycle
  reconciliation, and it has a **named per-tick operation budget** in the one economy settings block. Over
  budget, the remainder queues to the next tick: **payments get slower, transfers do not.** Proven by G-18.
  This closes the low-severity finding that a bot submitting payments at maximum rate could make region
  handovers wait behind them.
- **No economic read on any hot path.** Realm admission, movement, physics and lifecycle contain no read
  of any balance. A player with a million coins and one with none move, collide, cross and re-home
  identically.
- **No economic state in the reap decision.** Not in the orchestrator's teardown facts, and not
  shard-locally either (§6.3).
- **The crate check.** Nothing in the core, sim, node or realm-lifecycle crates may name an economy type
  (G-21). Adding it now, while the answer is trivially yes, means it can never quietly become false.

---

## 8. Abuse, ranked

> **[REV 07-27] Re-ranked for accounts-only. Five abuses die with the coins; one gets SHARPER; one flips
> from harder-to-police to easier; and one new one arrives.**
>
> | # | Change |
> |---|---|
> | 1, 4, 15, 18 | **MOOT** — bank-then-rollback, withdrawal redelivery, note theft and the compression trick all require a physical object that no longer exists |
> | 3, 12 | **MOOT** — notes are deleted (Decision 10 amended) |
> | **6** | **STILL LIVE, and SHARPER.** Minting at a doomed station now leaves a **durable** credit behind a vault that dies — permanently unbacked currency, where before at least the coins died too. Still PREVENTED by schedule (P5), and the schedule is unchanged |
> | 2, 5, 8, 9, 13, 14, 16, 17 | **UNCHANGED** |
> | **10** | **UNCHANGED in mechanism, but the burn share that makes it costly moves from 5% to 20%** (§0.4), so the deterrent gets stronger, not weaker |
> | **11** | **EASIER TO PERFORM, MUCH EASIER TO CATCH — and the second effect dominates.** Every transfer is one permanent row naming both parties and a timestamp. A coin handover was *"identical to handing over a rock"* and produced no central record at all; notes were untraceable in the middle. The shipped pattern is that sellers migrate to whatever leaves no record, which is precisely what coins and notes were. Caseload at the planning range: **≈2/day at 400 concurrent, ≈10/day at 2,000** — human review, *and only if the transfers are auditable* |
> | **19** | **DELETED as content.** Coins trading at a discount far from a vault cannot be expressed by a balance with no location. A stated loss (§0.5), not a solved problem |
> | **20 (new)** | **Stuck-purchase farming / accidental hoarding.** Every purchase creates a hold; unreleased holds immobilise up to 24% of supply per month (§0.4). **PREVENTED** by release-on-next-login plus the published immobilised total (D-92, promoted from hygiene to load-bearing, new gate G-23) |

| # | Abuse | Verdict | How |
|---|---|---|---|
| 1 | **Bank-then-rollback money printer** — deposit coins, get rolled back, hold them twice | **PREVENTED** | Durable applied-once before any durable balance; totals-not-deltas; the settle-before-spend window on arrival; G-3 and G-4 |
| 2 | **Infinite mint/melt loop** across venues with different ratios (8%/lap → 2,199× in 100 laps) | **PREVENTED** | One globally fixed ratio pair; competition is a fee, which is a transfer and cannot loop |
| 3 | **Double-cash a note** | **PREVENTED** | Cashing destroys the object; the note is the value, not a receipt |
| 4 | **Withdrawal redelivery doubles a coin stack** | **PREVENTED** | The stack's identity is minted by the service, derived from the withdrawal id |
| 5 | **Tradeable starter kit → account-creation printer** (50 coins × 1,000 accounts/day = 50,000/day; 6M supply in 120 days; at 10,000/day, **12 days**) | **PREVENTED** | Nothing in the starter kit is tradeable — bind it to the character. Also: gate account creation behind the real login service, and make a zero-balance no-history account removable on next touch so a bot farm cannot grow the store |
| 6 | **Mint at a station about to be reaped, carry the coins out, the backing evaporates** | **PREVENTED, by schedule** | Mints and vaults do not exist until per-place durable storage does (§10) |
| 7 | **Buyer pays for nothing** — seller withdraws goods while an account payment is in flight | **PREVENTED** | Both sides lock at reservation; settlement is conditional on the goods and is the thing that moves them |
| 8 | **Reservation griefing** — lock a market by reserving everything | **PREVENTED** | A reservation costs its full price up front; the seller may cancel at any time |
| 9 | **Fee dodge by splitting** — 4% floored on ≤24 units pays zero, so 41,667 small sales beat one large one | **PREVENTED** | Minimum absolute fee of 1 per sale: 41,667 × 1 = 41,667 > 40,000, so splitting is strictly more expensive; and with a floor the effective rate never falls below the headline |
| 10 | **Wash trading through your own venue** to launder or to fake volume | **DETECTED, and made costly** | Never set the burn share to zero — even 5% makes a round trip cost 0.2% of turnover (2,000 coins on a 1,000,000 wash). Round the remainder toward the burn, never away. Watch the sale-size distribution for a spike just under the fee threshold |
| 11 | **Real-money trading**, made structurally easier by the reversal — a transfer is a row with no location, where cargo money was a hauling job | **DETECTED, accepted as an operating cost** | Notes are the first transaction type reviewed; issue-to-redemption pairing retained and joinable; new-account inflow watched. **Scale is reassuring**: the reference game banned 3,165 accounts in one month at ~22,000 concurrent = 0.144/concurrent/month → **~145/month, ~5/day at 1,000 concurrent**. A spreadsheet, not a machine-learning problem, for years |
| 12 | **Note-magnitude confusion** — hiding low-value notes among high-value ones (documented historical scam, only fixed by adding separators) | **PREVENTED, as a correctness requirement** | Separators plus a distinct visual weight per order of magnitude. Not polish |
| 13 | **Engineered-loss refund farming** — withdraw at a branch about to be reaped, then claim compensation | **DETECTED, and refused by design** | The refusal is written into the design document, not just the support policy, so it survives a change of staff. Track repeated withdraw-then-loss events per account — trivially cheap, and the pattern is unmistakable |
| 14 | **A social deposit bank absconding with savings** (a 10% share of a 6M supply ≈ 600,000 coins across ~100 players, ~3× each of their working floats) | **ACCEPTED — but the engine never endorses it** | It happened in every verified precedent and cannot be prevented socially. What is prevented: the engine never renders a balance it does not hold, and player-built signage may display engine-held values only. The honest capability is given back as engine-enforced escrow |
| 15 | **Note theft** | **ACCEPTED, deliberately** | A bearer instrument's upside (no lookup, works with no communications) and its downside (loss is unrecoverable) come from the same fact. You cannot keep one and remove the other |
| 16 | **Moderator balance edits punching a hole in the conservation proof** | **PREVENTED** | Confiscation and restitution are **ordinary ledgered operations** with an operator identity and a case number, sitting alongside mint and melt as named exceptions. **Never a balance edit.** Almost free now; genuinely painful to retrofit, because retrofitting means the conservation gate was wrong for however long it took to notice |
| 17 | **Payment flood starving the region-handover machinery** | **PREVENTED** | The named per-tick operation budget with overflow queued to the next tick (§7.5), proven by G-18 |
| 18 | **Coins used as a compression trick** to teleport the value of heavy cargo | **PREVENTED** | Coin mass per redeemable unit ≥ 1 (§3.2c) |
| 19 | **Melt-arbitrage at the frontier** — coins worth less far from a vault | **ACCEPTED, and it is good** | Melting works only where a vault physically has stock, so coins trade at a discount at the frontier. That is hauling and arbitrage gameplay and it makes geography matter. But it means "one perfectly interchangeable galaxy-wide currency" is a simplification the user should confirm (Decision 5) |

---

## 9. Cost and prerequisites

### 9.1 Lines of code

> **[REV 07-27] Revised totals are at §0.3. In one line: the economy proper goes 8,130 → ≈6,750 (−17%),
> the blocking prerequisite bill goes 6,900 → ≈3,000 (−56.5%), and all-in goes ≈15,030 → ≈9,750 (−35%).
> The table below is the two-form baseline; the accounts-only column and its derivation are in §0.3.**
> **Verdict on "accounts-only deletes the expensive half": TRUE for the schedule, FALSE for the code.**

| Piece | Product | Test | Tier |
|---|---|---|---|
| Amount type (integer minor units, runtime currency tag, explicit overflow), delta vector, the one conservation assertion | 250 | 500 | A |
| Account operations: open, credit, debit, transfer, hold, release, custody sub-account, share register | 600 | 900 | A |
| Ledger key family + read-through cache + bounded per-tick budget + rehydrate | 450 | 700 | A/B |
| Two appended wire arms + effect classification + durable pending-reply family + golden pins | 220 | 350 | A |
| Branch affordance (forward, refuse, render closed) + branch capability flag | 180 | 300 | B |
| Venue: consignment, listing with rate-at-listing, coin sale, fee split, want-ad, strongbox, proceeds box | 420 | 700 | A |
| Carried purse (bounded, fixed-width) + blob tags + version | 120 | 300 | A |
| Mint / melt recipes + vault | 180 | 300 | A |
| Notes: issue, carry, cash | 160 | 300 | A |
| Harness: conservation property, the kill-9 cells, two-kinds fixture, economy-off gate, latency gate, load | — | 1,200 | tests |
| **Total** | **≈2,580** | **≈5,550** | **≈8,130** |

Tier-A share of product code ≈ 2,220 / 2,580 = **86%**, at 100% region and branch.

**Prerequisites owed anyway but blocking (not chargeable to the economy):** durable applied-once ≈900;
the per-kind carried-state seam ≈2,000; the rescue payload's state arm ≈600; per-place durable storage
keyed by realm identity ≈3,000; the promote/crossing-producer fixes ≈400. **≈6,900.**

**All-in ≈15,000 lines**, of which ~8,100 is the economy proper.

**What the reversal actually cost, honestly.** The material-currency answer needed none of the account
service, the conservation ledger, the institution tier, or cross-server value movement. So the reversal's
marginal price is roughly **2,000–2,600 product lines plus their tests**, **one new always-on deployment
unit**, **one already-owed prerequisite promoted from someday to blocking**, and **a permanent
real-money-trading and botting cost.** No lines-of-code baseline exists for either prior design, so an
exact before/after is not available — this is a bounded estimate from the candidate designs' own figures,
re-added and internally consistent.

### 9.2 Per-payment and per-sale cost

| | Coin sale | Account payment | Durable crossing (for contrast) |
|---|---|---|---|
| Cross-server messages | **0** | 2 | ~20 |
| Ticks | 1 | 1 | 9 |
| Latency floor | 20 ms | 20 ms + round trip | **180 ms** |
| Player-visible | ~45 ms (incl. mean 25 ms snapshot wait at 20 Hz) | ~55 ms mean, ~110 ms worst | — |
| Disk barriers | shared, 1/tick | shared, 1/tick | up to 9 |

~~**A payment is 9× faster on 10% of the messages.**~~ **[REV 07-27] A payment is ~4× faster than a region
crossing, on 100% of the messages.** Re-derived from the deployed 50 Hz tick and the drain-at-top /
flush-at-end discipline (`crates/node/src/app.rs:163-166`, `:206`): **server-side mean ≈22 ms, worst
≈42 ms; player-visible mean ≈47 ms, worst ≈92 ms** (full table at §0.2). Against a 180 ms durable-crossing
floor a purchase costs **23% of one region crossing**, and players pay the crossing on every dock, undock
and region change. That is the whole justification for not reusing the
transfer machinery, and the per-key serialisation limit (§2.3) is the second — **and the second is now the
more important of the two, because it applies to every payment in the game (5.5/second per key with the
excess silently dropped).**

### 9.3 Storage and throughput — sized for the real game, not a hypothetical one

**Account row.** Key = 1 (family tag) + 16 (account id) = 17 B; value = balance 16 + held 16 + stamp 8 +
owner 16 + institution 8 + flags 4 = 68 B; tree overhead ≈ 28 B; → **≈113 B, budget 128 B.**

| Accounts | On disk | Note |
|---|---|---|
| 10,000 (planning) | 1.28 MB | Trivial |
| 1,000,000 | 128 MB | Comfortable |
| 10,000,000 | 1.28 GB | Needs a larger volume; still fine |

**Contrast, and it is the reason for read-through:** the fully-resident ownership table at ~104 B/row
would put 1,000,000 rows at 104 MB = **20.3% of the coordinator's entire 512 Mi ceiling**, and 10,000,000
would exceed it outright. Read-through with a 100,000-entry hot cache is 100,000 × 128 = **12.8 MB
resident**, and rehydrate touches **zero** rows.

**Volume size — a correction to the judgement's figure.** The 8 GB recommendation was sized at 200
payments *per second*. The real rate is ~200 payments per **day** (§9.4). Audit trail at 64 B/row:
200/day × 64 = **12.8 KB/day = 4.7 MB/year.** Even at 100× the planning rate it is 470 MB/year.
**Recommend 1 GB**, with the arithmetic recorded so growth is a calculation rather than a guess: grow when
`payments/day × 64 B × retention_days` approaches half the volume.

**Throughput.** One disk barrier per tick = 50/s, shared. A thousand payments in one tick share one
fsync. At 1,000 payments/tick the ledger writes 2,000 rows × ~113 B = **226 KB per fsync** — trivial. The
binding constraint is memory, not disk.

### 9.4 Volume, honestly stated

~~All consignment sales are local by construction.~~ **[REV 07-27] REVISED UPWARD BY ~48×, AND IT IS STILL
A NON-ISSUE.** Under accounts-only every currency payment is a round trip, so the rate is the full shop-
interaction rate from §6.4: **9,600 payments/day at 1,000 concurrent = 0.111/second = 0.0022 per tick.**
At the planning range: **3,840/day (0.044/s) at 400 concurrent; 19,200/day (0.222/s) at 2,000**, and
**2.22/s during a 10× prime-time peak.** Against §9.3's ceiling of 1,000 payments per tick × 50 =
**50,000/second**, the headroom at the top of the range during a peak is **≈22,500×**. The two shapes that
would break it are one round trip per *item* instead of per *basket*, and routing a payment through the
directory transfer lock — both prohibited at §0.2.

The original text: the only things needing the account service are wages,
treasuries, remote proceeds, deposits, withdrawals and paying someone you cannot reach — **roughly 2% of
economic activity, about 200 payments a day at 1,000 concurrent players**, i.e. 0.0023/second.
**That figure is superseded; the 1 GB volume recommendation at §9.3 is not, since 9,600 payments/day ×
64 B = 614 KB/day = 224 MB/year, still comfortably inside it.**

**This is a modest-volume feature and it should be presented as one.** It needs no scaling work, no
sharding, and no performance engineering. What it needs is correctness. **[REV 07-27] Unchanged, and the
48× revision does not dent it.** A vetter argued this is a reason
to defer the service entirely; I disagree and explain why in §10.1, but the volume figure is right and
the user should know they are buying a small, cheap, correct thing rather than a large one.

### 9.5 Prerequisites, with roadmap phase

> **[REV 07-27] Five drop as economy blockers, three narrow, six stand, and one of the six becomes the most
> critical in the document. This is where more than half the schedule win lives.**
>
> | # | Status under accounts-only |
> |---|---|
> | **P1** | **DROPPED as a shard problem, RESCOPED into the service** — ~150 lines in a file the service already opens |
> | **P2** | **DROPPED as an economy blocker, RETAINED AT FULL SEVERITY as a game defect.** It strands the player entity, not the money (§6.2 I2). Do not downgrade it |
> | **P3** | **DROPPED ENTIRELY** — there is no carried value to rescue |
> | **P4** | **DROPPED as an economy blocker**, still owed for the game; its contract must still carry Laws A and B (§0.6 item 5) |
> | **P13** | **DROPPED ENTIRELY** — no coin, no mass. Reinstated the day coins are added |
> | **P5** | **NARROWED, not dropped.** Money half gone; goods, stock, escrow, collateral and the backing vault still die on a routine teardown, so it still gates M-3 and M-4. Cheaper than stated — a graceful shutdown window already exists (§6.3) |
> | **P7** | **NARROWED to goods** |
> | **P11** | **NARROWED to the service's own journal** |
> | **P6** | **UNCHANGED and NEWLY CRITICAL.** With no coins an account *is* a player's entire net worth, and the account principal is the only thing protecting it |
> | **P8, P9, P10, P12, P14** | **UNCHANGED.** P12's deadline is now binding for two reasons (§0.4 option 2) |

| # | Prerequisite | Phase | Why it blocks |
|---|---|---|---|
| P1 | **Durable applied-once record on any machine value can reach** | **Before the first durable balance — never alongside it** | §6.2 (I1). Cheaper than believed (§2.5) |
| P2 | **The promote/crossing-producer fixes** (journal-after-effect; re-emit on Demoting and Promoting timeouts; loud loss) | With P1 | §6.2 (I2) — silent destruction with the transfer reporting success |
| P3 | **The rescue payload's saved-state arm** + the purse version and zero-on-stale rule | With P1 | §6.2 (I3). Additive by design, no compatibility cost |
| P4 | **The per-kind carried-state seam** (serialise, reconstruct, precondition, rebind) **with Laws A and B in its contract** | Checkpoint phase | Nothing is carried today — `state: vec![]` at the only build site |
| P5 | **Per-place durable storage keyed by realm identity** (a coordinate-derived store path + the spawn allow-list entry + a guard against several shards sharing one file) | Checkpoint phase | §6.3 — strongbox, proceeds, consignment, want-ad escrow, collateral, backing vault |
| P6 | **A real login service** with a store behind the account principal and revocation | **Before any balance exists — currently unowned** | The identifier shape exists and is right; there is no accounts table among the seven durable families, no password handling, and the test client signs its own pass with a shared key |
| P7 | **Items as positions with never-reused identities, and containers with owners and holds** | Block/inventory phase | Coins are objects; this is what makes no-duplication provable with the economy out of the loop |
| P8 | **A recipe system** | Block phase | A mint must be content a player builds, never code |
| P9 | **Two reviewed appended arms** + the durable pending-reply family (tag 9) | With the service | §2.4, including the decode-collision trap |
| P10 | **A separate always-on deployment unit** with its own data root and volume | With the service | §2.2 — must not be shortcut into the coordinator |
| P11 | **Retention bound on the applied-once journal**, by age on touch | With P1, not after | The set is explicitly unbounded (`crates/sim/src/stub.rs:1113-1118`) and a payment-per-correlation-id design grows it **forever, in RAM, on a 384 Mi node** |
| P12 | **DECIDE the backing material and its distribution** | **Before terrain generation — the hard deadline** | §3.4. Terrain sets rarity, distance and danger, which set the whole money supply |
| P13 | **DECIDE coin mass** | Before block/inventory | §3.2c. Weightless coins make banks decorative |
| P14 | **DECIDE one currency or several** | Before block/inventory | The tag costs nothing now and a great deal to retrofit |

### 9.6 The SHORT list — shape now, at near-zero cost

> **[REV 07-27] All 14 survive: eleven unchanged, three reduced to contract text with no current subject,
> NONE dropped — and three are added. The consolidated accounts-only version is the keep-the-door-open
> list at §0.6.**
> **Reduced to contract text (3):** item 2 (there is no money tag to mark optional), item 4
> (no money-bearing message reaches a diskless machine yet), item 14 (the double-apply crash cell stops
> being a money gate; the other two red-listed tests stand).
> **Elevated (1):** item 1 is no longer merely the highest-value item — it is **structurally required**,
> the only thing that keeps a market working with the economy off or a station cut off.
> **Added (3):** the conservation identity is written with the world-held-coin terms present and summing
> to zero (§0.6 item 7); the account row carries a held/escrow field and a monotonic stamp from the first
> version (§0.6 item 8, forced by §0.2 residue 1); and the destroyed-fee-on-every-movement rate is a named
> field in the settings block from the first version (§0.4 option 1).

Each of these would otherwise be a **live migration of real player money**.

1. **Price is a bundle of entries**, where a currency amount is one entry kind among materials. Highest
   value of everything on this list.
2. **Money tags in carried state are always OPTIONAL, never required** — plus the one-line assertion that
   `required_max_tag` is unchanged by the presence of money (Law A).
3. **Unknown tags round-trip verbatim through reconstruct-and-reserialise**, written into the per-kind
   seam's contract, with the round-trip test now (Law B).
4. **Totals-not-deltas** as a wire law for money aimed at a diskless machine — and the object-creation
   exception: the object's identity is minted by the disk-holding side, derived from the request id.
5. **A credit and its applied-once journal entry land in the same durable barrier.** Free under the
   existing one-barrier-per-tick batching; a money printer if they ever split.
6. **Integer minor units, no floats anywhere authoritative, remainder assigned to the burn, minimum
   absolute fee per sale.** The amount type is a newtype with no conversion from a floating-point value,
   so the ban is enforced by the type rather than by review (the clippy configs ban wall clocks, default
   hashers and raw sockets — **not floats**).
7. **One economy settings block**, created before any knob exists, so no number is ever inline.
8. **Reserve the next durable key family tag (8) for accounts**, and tag 9 for the pending-reply family.
   The tag space is already append-only by design.
9. **Currency is a runtime `u16` tag on ONE monomorphic amount type, never a type parameter.** Coverage
   counts regions per monomorphization *and* per test binary — three currency types across four binaries
   turns one branch into twelve regions.
10. **Account rows never decode to a default** (error on an unrecognised version, matching the kind-tag
    precedent) and carry their own monotonic stamp, which buys optimistic concurrency for free.
11. **The dependency-direction check** (G-21), added while the answer is trivially yes.
12. **Never key money by realm identity.** Use the account principal; where a place must be named, use the
    full lineage path plus a never-reused local identity (§2.2).
13. **Add the economy crate to the coverage package list in the same commit that creates it** — the list
    is an explicit enumeration, not a property, so a new crate is silently unmeasured. The exemptions
    registry is currently empty; the economy must not be what opens it.
14. **Land the three red-listed tests now, before there is any money**: the money-family-scoped five-year
    no-write gate, the economy-absent build gate, and the double-apply crash cell (which fails today —
    turning it green *is* the definition of "balances may become durable").

---

## 10. What lands when

### 10.1 Slice sketch

**Ordering principle: build what is *possible* first, not what is *cheap* first.** A vetter argued for
shipping coins and local markets before the account service, on the grounds that the service serves only
~200 payments/day. I sustain the volume figure and reject the ordering, for one reason: **coins are not
buildable yet and the service is.** Coins need the carried-state seam, shard durability, the durable
applied-once record and three crash fixes. The service needs a node-kind variant, two arms and a store
that already exists. Building the service first also means the conservation machinery is proven under the
full chaos matrix **before any value exists** — which is exactly this project's structural discipline.

> **[REV 07-27] Accounts-only is this ordering principle taken to its conclusion.** The argument above says
> the buildable thing is the service and the unbuildable thing is coins. Dropping coins does not reverse
> the order — **it deletes the unbuildable half**, so the ordering stops being a compromise and becomes the
> whole plan. **The build order is: M-0 → M-1 → M-3 → M-4 → M-6 → M-7.**
>
> | Slice | Status | Note |
> |---|---|---|
> | **M-0** | **SIMPLIFIED** | 14 shape-now items → 11 with a current subject, 3 as contract text, +3 new (§9.6). Now also carries the destroyed-fee field and the held/escrow row shape |
> | **M-1** | **UNCHANGED, and unblocked far earlier** | Its blockers are P6, P9, P10 — **none of which were among the four defects.** This is the schedule win |
> | **M-2** | **DELETED** | A branch's only purpose was making heavy money weightless. Meaningless when money was never heavy. Reinstated only if coins are added later |
> | **M-3** | **SIMPLIFIED and re-scoped** | The one-write coin sale is gone; the two-sided lock becomes universal and hardened; want-ads become service-funded holds; a **barter** bundle is the partition fallback. Still blocked on P5 and P7 — **goods still die in a killable place** |
> | **M-4** | **SIMPLIFIED** | Minting credits an account rather than producing objects. **Still gated hardest**, and for the unchanged reason: a vault that a routine teardown can eat makes the currency's central claim false |
> | **M-5** | **DELETED** | Notes solved a problem only physical coins created (§0.5, Decision 10) |
> | **M-6, M-7** | **UNCHANGED** | M-6's escrow is now load-bearing on every sale rather than an added service |

| Slice | What | Blocked on | Delivers |
|---|---|---|---|
| **M-0** | The 14 shape-now items (§9.6). No money. | Nothing | Every later slice becomes additive |
| **M-1** | **The account service**: node kind, deployment unit, store family, accounts, transfer, hold/release, the conservation assertion, the two arms, the operator issue/confiscate operation, the monthly report | P6, P9, P10 | **Send money to anybody, anywhere** — the thing only this can do. Zero balances at first; the operator issue operation exercises the machinery and is needed anyway for confiscation |
| **M-2** | **Carried purse + branches**: bounded purse in the blob, deposit, withdraw, balance read, branch capability | P1, P2, P3, P4, P11 | Coins exist. Money becomes weightless by going somewhere |
| **M-3** | **Local markets in currency**: consignment with rate-at-listing, coin sale, fee split, want-ads, strongbox, proceeds box, the account-paid path with the two-sided lock | P5, P7 | The market the user described, paid in currency, fully local |
| **M-4** | **Mints and vaults** | P5, P7, P8, P12, P13 | The money supply. **Gated hardest** — a mint without durable backing makes the currency's central claim false |
| **M-5** | **Notes** | M-2 | Large-value transfer with no communications |
| **M-6** | **Escrow and custody**, company treasuries, shares | M-1, M-3 | The honest replacement for deposit banks |
| **M-7** | **Collateralised lending** | P5 | Only when collateral cannot be eaten by a routine shutdown |

**Everything above lands after terrain, physics, block editing, save-and-restore and ships**, as already
decided. ~~**The only item that should be pulled earlier is P1 (and P2, P3 with it)**~~ **[REV 07-27] The
only item that should be pulled earlier is P2 — and NOT for the economy's sake.** It strands the player
entity with the saga reporting success (§6.2 I2), which is a game-correctness defect that outranks
anything in this document. P1 is rescoped into the service and P3 dies, so neither needs pulling.

### 10.2 What must NOT be built yet

- **Any per-place account book.** Until P5 exists, a book inside a station is destroyed by a routine,
  correct teardown.
- **The chain-of-institutions tier** (value passing up to a common ancestor and back down). Elegant,
  historically well grounded, ~5× the cost, needs the block system before an institution can exist, and
  structurally builds a distributed hierarchy on top of what is really one central store. It is also
  **not currently expressible**, because realm identity collapses every galaxy to one (§2.2). **Write the
  netting design down and leave it unbuilt.** It becomes the right answer the day the single service needs
  splitting, and that arithmetic must not be invented under load.
- **Any second saga family for money.** Value movement between accounts is not a transfer, because nothing
  changes authority — an account has no position, no realm, no containment, so there is nothing to hand
  over. Record that reasoning so a future reviewer does not reopen it.
- **Fractional reserve, promised yield, uncollateralised credit, any real-money peg.**
- **Any scaling, sharding or performance work on the service** (§9.4).
- **Interest, upkeep, account fees, storage charges, listing decay** — all clock-driven, all banned.
- **[REV 07-27] Any local spending allowance, float, till, credit line or cached balance on a shard,
  region, venue or client.** This is the binding condition of the accounts-only decision (§0.2): an
  allowance is money held on a machine that is killed as routine, so it puts economic state on the
  region-lifecycle path and is unbounded under a network split. **Every payment is a synchronous round
  trip.** Someone will propose an allowance to save 42 milliseconds; the answer is no, and the reason is
  recorded here so it is not reopened.
- **[REV 07-27] Physical coins, bank branches, notes and large denominations.** Not banned — **deferred as
  an additive extension** whose specification is retained throughout this document. §0.6 lists the eight
  things that must be shaped now to keep it additive; build none of them beyond those eight.

### 10.3 Ledger entries

Continuing from **D-83**, the highest identifier across the three prior documents (in the dormant-world
design; the workspace registry itself reaches D-46).

> **[REV 07-27] Status of all 17 under accounts-only: 5 drop, 1 reclassifies, 3 narrow, 8 stand, 2 are
> added.**
>
> | Id | Status |
> |---|---|
> | **D-84** | **RECLASSIFIED, NOT DOWNGRADED.** Stays 🟥 BLOCKING as a **game-correctness** defect: it strands the player entity, not the money. Re-verified in code (§6.2 I2) |
> | **D-85** | **DROPPED** — no carried value to duplicate; `ReHomeState` has only a pose arm |
> | **D-86** | **DROPPED as a money entry**; the substance folds into D-84's re-drive fix |
> | **D-87** | **NARROWED.** Money half gone; goods, stock, escrow, collateral and the backing vault half stands and still blocks M-3/M-4. Mechanism corrected: a graceful SIGTERM window exists, flush code does not (§6.3) |
> | **D-91** | **DROPPED — a clean structural win.** All money is now locally summable, so the one-way coin-telemetry report evaporates, and with it the entry that came closest to putting economic data on a lifecycle message |
> | **D-92** | **STANDS, and is promoted from hygiene to LOAD-BEARING.** Ten times the strand volume; release-on-next-login must ship in the same slice as the account-paid purchase path (new gate G-23) |
> | **D-94** | **NARROWED to contract text with no current subject.** Still written now; free now, a money migration later |
> | **D-95** | **DROPPED** — no purse to bound |
> | **D-96** | **DEMOTED to the keep-the-door-open list (§0.6 item 3).** Reserve durable entity-kind tag 3 and record that it is never feature-gated; build nothing |
> | **D-97** | **STANDS, AMENDED.** The fixed ratio pair is unchanged and still forbids the loop. The **useless-material** half is REOPENED: the material must have one genuine, high-value, low-volume use, or redemption never happens and the supply has no feedback term (§0.4) |
> | **D-99** | **NARROWED** to the service's own journal |
> | **D-88, D-89, D-90, D-93, D-98, D-100** | **UNCHANGED.** D-88 additionally carries the destroyed-fee-on-every-movement rate; D-98's escrow becomes load-bearing on every sale |
> | **D-101 (new)** 🟥 | **Service-side HOLD on every purchase, released on the buyer's next login.** The reserve moves the buyer's balance into a held position at the service; the seller is credited only by the settlement that also moves the goods. Without it, a venue region dying mid-purchase leaves a buyer who paid for nothing. **The account row's held field and monotonic stamp are a slice-one data-model decision, not a later addition.** — economy crate + service — **M-0 (row shape) / M-1 (behaviour)** |
> | **D-102 (new)** 🟥 | **A destroyed fee on every account-to-account movement, replacing the death drain.** ~4% at measured turnover reproduces 1.2%/month exactly. The claim that mint-and-melt self-regulates is withdrawn (§0.4); the supply is managed by one measured, retunable number and a monthly report from the first version. — economy crate settings block — **M-0 (field) / M-1 (levied)** |

| Id | Entry | Where | When |
|---|---|---|---|
| **D-84** 🟥 | **Promote journals before effect + no crossing producer past the route-swap phase.** A deferred promote consumes its step, acks unconditionally, and the saga reports success with the subject owned by nobody. Three fixes: journal-after-effect; re-emit the crossing on Demoting and Promoting timeouts; assert `promote_before_crossing` is zero at quiescence and treat a Done saga with an unowned subject as an oracle failure. | `vd-sim` stub + saga | **BLOCKING for any carried value** |
| **D-85** 🟥 | **Value-safe rollback.** The forward re-home restores the source's pre-transfer snapshot, so money externalised at the destination is duplicated. Purse version + zero-on-stale, plus the rescue payload's state arm. Law: a rollback of carried money is a loss, never a re-credit. | `vd-wire` + `vd-node` | With D-84 |
| **D-86** 🟥 | **Epoch-mismatch discard is money-unsafe.** Value must not be refusable on a pose-validity ground: split the value half from the pose half, or make a value-carrying epoch mismatch a loud terminal abort returning authority to the source. | `vd-sim` stub | With D-84 |
| **D-87** 🟥 | **No value in a killable place.** The Empty self-report knows only about occupants; teardown is an outright kill with no flush. Requires per-place durable storage keyed by realm identity: a coordinate-derived store path, the spawn allow-list entry, and a guard against several demand-spawned shards opening one file. **Explicitly rejects** the shard-local Empty-suppression stopgap. | `vd-bins` + deploy | Checkpoint phase; blocks M-3/M-4 |
| **D-88** 🟩→ | **Economy settings block + fee arithmetic.** Integer minor units, remainder to the burn, minimum absolute fee, burn share never zero, one global mint/melt ratio pair. No inline literals. | economy crate | M-0 |
| **D-89** 🟥 | **The account service as its own node kind and deployment unit**, with its own data root and volume. Coordinator co-location forbidden structurally, with the reason recorded in the manifest. | `vd-sim` capability + deploy | M-1 |
| **D-90** 🟥 | **Two appended cross-server arms**: a side-effecting payment request with the shared once-only key and a durable pending-reply family (tag 9), and a fire-and-forget balance read with its own reply arm. Payments originate at the service tier, never at a diskless shard. | `vd-wire` | M-1 |
| **D-91** 🟥 | **Money-supply measurement in production.** Account balances are locally summable; world-held coin is not. Needs a one-way telemetry report of coin held per live region. **Must NOT ride the realm-lifecycle demand message** — that would put economic data on the lifecycle path. Until it exists, the monthly report covers accounts only and says so. | `vd-wire` + `vd-sim` | With M-3 |
| **D-92** 🟩→ | **Reservation strand handling.** The buyer's next login touches their outstanding reservations (on-touch, no timer, no sweep). Publish total immobilised value as a first-class dashboard number. | economy crate | M-3 |
| **D-93** 🟩→ | **Confiscation and restitution as ledgered operations** with an operator identity and case number, as the third named exception to the conservation identity. Never a balance edit. Plus an immutable audit trail on every economy setting change, with staged rollout. | economy crate | M-1 |
| **D-94** 🟥 | **Carried-state contract addendum** (extends D-31): money tags are always optional; `required_max_tag` is unchanged by money; reconstruct-then-reserialise round-trips unknown tags verbatim; the crossing precondition may never consult a balance. | `vd-core` tlv + D-31 seam | **M-0 (contract), enforced at D-31** |
| **D-95** 🟥 | **Bounded purse + enforce `max_state_bytes` at the write site.** The cap is declared and read by no code path; an unbounded purse would make a wealthy player's transfer refusable, which is a movement outage *and* a money migration. | `vd-core` | Before any money field |
| **D-96** 🟥 | **Coins as world objects** — weight, drop, loot, never-reused identity — gated on the inventory subsystem, which does not exist. The carried form is a bounded fixed-width purse; the world form lands with inventory. Reserve entity-kind tag 3 in the Durable band; never feature-gate it. | `vd-core` + block phase | M-2 / P7 |
| **D-97** 🟥 | **Mint and melt as one globally fixed ratio pair**, plus the physical backing vault. Backing material must be otherwise useless and its scarcity gated on something that does not improve with tooling. **Reverses the prior material recommendation.** | content + `vd-core` | **Decided before terrain** |
| **D-98** 🟩→ | **Escrow and custody service** as the honest replacement for deposit banks, plus the interface law: never render a balance the engine does not hold (binding once player-built signage exists). | economy crate + client | M-6 |
| **D-99** 🟥 | **Retention bound on the applied-once journal**, by age on touch, landing *with* the durable table rather than after it. A payment-per-correlation-id design otherwise grows it forever in RAM on a 384 Mi node. Also: the stranded early-arrival buffer for an entity that never adopts. | `vd-sim` stub | With D-84/P1 |
| **D-100** 🟦 | **The correspondent-netting design for splitting the account service** — mirrored opposite-sign positions, bulk settlement, no distributed transaction on the player-facing path. **Written down, deliberately unbuilt.** | design doc | Written at M-1; built never, until needed |

---

## 11. Decisions for the user

**1. Two forms of money — coins you carry and an account you cannot lose. Adopt?**
*Options:* (a) both, as designed; (b) accounts only, no pocket cash; (c) pocket cash only, no accounts.
*Consequences:* (b) means nobody can rob your money, which deletes a real payoff in a game built on
danger, and it means you cannot buy anything when you are out of contact — you can be rich and unable to
spend it. (c) means no banks, no wages, no treasuries and no paying anyone you cannot see, which is the
thing you asked for. ~~**Recommend (a).**~~

> ### **[REV 07-27] DECISION 1, REVISED: RECOMMEND (b) — ACCOUNTS ONLY.**
>
> The recommendation changed because the evidence went the other way on both halves of the old one.
>
> **On "it deletes a real payoff in a game built on danger" — the shipped record says no.** The largest
> space sandbox ever made has a wallet no player can take by force, and it is the benchmark for danger and
> irrecoverable loss; its piracy maths contains no term for the victim's savings. The closest analogue to
> our premise — physical hauling, convoys, ambushes, geography that hurts — has **no currency at all** and
> delivers every element of that premise on cargo alone. Where lootable money does ship it is person-scale
> full-loot, and even the harshest of those gives everyone a guaranteed safe pocket. **Ours is ship-scale:
> the ship and its hold are already the largest thing a player can lose.** What actually dies is the small
> story of taking the purse off a body; everything larger survives (§0.5).
>
> **On "you cannot buy anything when you are out of contact" — true, and it is the real cost.** A cut-off
> station cannot run a currency market and two players in deep space cannot trade in money. **What survives
> is barter, and it survives only if a price stays a short list of things wanted with money as one entry.**
> That decision is now structurally required rather than merely valuable.
>
> **Two things settled it.** First, **the schedule**: carried money required four separate defects fixed in
> machinery that has nothing to do with money, plus a disk in every place in the world — about 3,900 lines
> of blocking prerequisites, more than half the bill, and the reason money was the last thing the roadmap
> could reach. Accounts-only deletes it. Second, **provability**: with all value in one book, money is
> provably never lost **in production**, not merely in a test, and the *"your money died with the server,
> and the answer is no"* policy disappears (Decision 14 is retired).
>
> **What it does NOT buy, and the old recommendation was right about this:** it does not remove goods,
> stock, escrow, collateral or the backing vault from a region that gets switched off. Killing coins
> changes what dies from cash to cargo; it does not stop the dying.
>
> **THE CONDITION, and it is binding: no local spending allowance anywhere. Every payment is a round trip
> to the money service.** Delegating an allowance is what puts economic state on the region-lifecycle path,
> and it is unbounded under a network split. Verified in the code: with no allowance, nothing economic can
> enter the shutdown decision, because the signal that lets a region be reaped can only see occupancy. The
> cost is about a twentieth of a second per purchase — a quarter of what a player already pays walking
> through a region boundary — at a few payments a second at the top of the planning population.
>
> **Not a one-way door.** Adding physical coins later is additive: a new object kind plus deposit and
> withdraw at a branch, with the money service unchanged. Eight things must be shaped now to keep it that
> way and all eight are free today (§0.6). **The two-form design is retained in this document as the
> specification of that later extension.**

**2. Money comes from minting a mined material, and can be turned back into it. Adopt?**
*Options:* (a) mint and melt; (b) a fixed amount at character creation; (c) player banks issue it against
reserves; (d) the computer pays it out for missions and bounties.
*Consequences:* (b) is arithmetically dead — five thousand players with a thousand each are out of money
in twenty months, and with no issuance prices halve about every seven years, which makes hoarding better
than spending. (c) is the one mechanism that collapsed every time it shipped, and it is a background
simulation of the money supply, which you banned. (d) is the computer-controlled economic actor you
banned. (a) has never shipped at this scale and is therefore an experiment — ~~the escape hatch is that the
mint yield is one retunable number~~. **Recommend (a).**

> **[REV 07-27] AMENDED — the recommendation stands, the reasoning behind it does not.** Minting is still
> the right answer, because the alternatives all still lose and it is the only issuance mechanism that is a
> player action. **But it does not self-regulate, and it never did.** Handing money back returns metal
> worth less than the money handed in, at every price, because the metal is useless for everything else —
> so nobody ever does it and there is no corrective feedback at all. **The escape hatch is not the mint
> yield; it is a small destroyed fee on every payment, sized against how fast money actually changes
> hands, plus giving the backing material one genuine use.** Full arithmetic, options and recommendation
> at §0.4. **Adopt (a) with a managed drain and a monthly measurement, not on a promise of
> self-correction.**

**3. The backing material must be USELESS for everything else. Confirm?**
This reverses the earlier recommendation that money be the universal fuel and repair input.
*Consequence:* every coin locks material in a vault where nobody can use it. At the planning population
that is twenty months of community mining effort. If the material is useful, players will simply refuse
to sterilise it and the currency never establishes. A useless material has no opportunity cost — which is
exactly what gold was and why it was money. **This must be settled before terrain generation, because
terrain decides rarity, distance and danger, and those three facts set the entire money supply.**
~~**Recommend: confirm the reversal.**~~

> **[REV 07-27] REVISED: recommend LOW-VOLUME BUT NOT USELESS.** The uselessness rule is what makes handing
> money back a guaranteed loss, which is what leaves the money supply with no way of correcting itself
> (§0.4). The affordability argument that produced the rule actually needs the material to be **cheap to
> lock away**, which means *low volume*, not *zero use*. **Give it one genuine, high-value, low-volume
> use.** A single use with small absolute demand costs almost nothing in locked-away stock and restores the
> only reason anyone would ever redeem. The gold analogy above is also wrong on the facts and was doing
> real persuasive work: gold's monetary role rested on real ornamental demand, which is the very property
> the rule removes. **The deadline is unchanged and is now binding for two reasons rather than one.**

**4. Coins have real weight — heavy enough that ten coins weigh what the metal behind them weighs.
Confirm?**
*Consequence:* if coins are lighter than their backing, minting becomes a compression trick and the
highest-value cargo in the game teleports — through the back door, in a game whose whole point is that
geography matters. And if coins are weightless nobody ever needs a bank and the local/global split you
asked for collapses into everyone carrying everything. ~~**Recommend: confirm, and make weightlessness the
thing a bank sells.**~~

> **[REV 07-27] MOOT under accounts-only, and RETAINED for the later extension.** There is no coin to
> weigh, and a balance teleports by definition — so the thing this rule prevented happens unconditionally
> for money. It does not happen for goods, which is what mattered: with no coins, the game's
> highest-value material simply never leaves a vault. **The rule becomes binding again the day physical
> coins are added**, which is why it is kept rather than deleted. **Recommend: note as a stated, accepted
> loss now; confirm the rule if coins are ever added.**

**5. One currency at launch, or several?**
*Consequences:* several backed by different materials cost almost nothing structurally — the label is one
small field — and cost a great deal in player comprehension and interface work. Historical free banking
produced notes trading at a discount by distance and published discount tables: genuine content for some
players, pure friction for most. Note also that with one currency, coins are still worth slightly less far
from a vault, so "perfectly interchangeable everywhere" is a simplification either way.
**Recommend: one at launch, with the label present from the first line so more remain possible.**

**6. Lending may never create money — fully collateralised only. Confirm?**
*Consequence:* a loan is a matched pair of movements of things that already existed, so the total in the
world never changes. Uncollateralised lending still happens socially — one player hands another coins and
trusts them — and the game records nothing. **Recommend: confirm. It is what your ban on background
simulation means when applied to banking.**

**7. No institution may promise a return on a deposit, and no player may hold another player's balance.
Confirm?**
*Consequence:* a run on a bank becomes unrepresentable in the data. But finance-minded players will find
banking flat — no leverage, no yield, no crises — and that is a genuine loss of a whole genre of activity.
The evidence is two verified collapses with two different causes: one a run at about a third of deposits
called, one the operator simply taking the money. The one rule an operator ever found that ended the
problem was exactly this, and it worked in two weeks. **Recommend: confirm, and give the capability back
as an engine-enforced escrow.**

**8. What share of the venue fee is destroyed at launch, and does destroying it release the material
behind it?**
*Options:* 0% / **5%** / more.
*Consequences:* at 0% a player who owns the venue and sells to their own second account launders for
free and the fee provides no drain at all. At 5% a wash round trip costs 0.2% of the amount — 2,000 coins
on a million — which is a real cost at volume and a signal for investigators. The lever reaches 25% a year
of destruction at 40% if inflation ever demands it. **Separately:** destroying coins does not release
their backing, so vaults become over-backed and the surplus accrues to mint owners — a slow transfer of
wealth. Either release the matching material too, or accept it and make mint ownership contested content.
~~**Recommend: 5%, never zero; and decide the material question deliberately rather than discovering it.**~~

> **[REV 07-27] REVISED: 20% at launch, and a destroyed fee on EVERY payment as the primary lever.** The 5%
> setting was chosen alongside a drain — money lost when you die — that was four times larger and is now
> gone. Twenty percent is the row that reproduces the designed behaviour, and only at this document's
> assumed rate of money changing hands; at the only measured comparable rate the whole table divides by
> five and 100% of the fee would be needed, which leaves the venue owner nothing. **The venue fee is too
> narrow a base.** The primary recommendation is therefore a small destroyed slice of **every** payment
> — about four percent, sized against measured turnover — which is available only because there is no cash
> for money to move through invisibly (§0.4). **Recommend: destroyed fee on every movement as the primary
> lever, burn share at 20% as the interim, never zero, and a published monthly measurement from the first
> version.**
>
> **A note the old text implied and should say outright:** without the death drain the supply becomes a
> **ratchet** — it climbs while trade grows, plateaus at what trade demands, and has no reason to fall.
> At the plateau **minting stops being a business**, which undercuts the premise that mint ownership is
> contested content. The burn is what keeps it a business.

**9. Build order: the bank first, pockets second, markets in currency third — and markets trade by barter
until pockets exist. Accept?**
*Consequence:* this is the reverse of the intuition, and it is forced by the code: a world server has no
disk today and is killed and forgotten as routine, so a coin in your pocket is a coin in a machine
designed to forget. The account service, by contrast, needs almost nothing that does not already exist.
~~**Recommend: accept.**~~

> **[REV 07-27] REVISED: the money service first, markets in currency second, and THERE IS NO POCKETS
> STEP.** The reasoning above is unchanged and is exactly what makes accounts-only the right answer — it
> deletes the step that was forced to come last. Markets still trade by barter with the economy off or a
> station cut off, but that is now a permanent property rather than an interim. **Recommend: accept the
> revised order.** **One item should be pulled earlier and it is not for the economy's sake:** the handover
> defect that reports success while leaving a player owned by nobody is a game-correctness problem that
> outranks anything in this document, and dropping coins is not permission to leave it.

**10. Bank notes — yes, later, or never?**
*Consequence:* notes are the only way to hand a large sum to somebody far away with no communications and
no lookup, because possession is the proof. They are also the perfect instrument for selling game money
for real money, because a note changing hands leaves no record anywhere. And they carry a documented
scam — hiding low-value notes among high ones — which makes the display a correctness requirement rather
than polish. ~~**Recommend: yes, but after coins and branches, with the three safeguards attached from the
first commit.**~~ ~~A cheaper interim exists: large-denomination coins solve most of the carrying problem
with no notes at all.~~

> **[REV 07-27] REVISED: NEVER — notes and large denominations are both deleted, and nothing of value goes
> with them.** Both existed solely to defeat the weight and stacking of physical money, i.e. a problem that
> only physical money created. The historical precedent is unambiguous: in the sandbox that had them, money
> had weight, it weighed nothing once banked, and bank cheques existed *"to save space"* — and that game
> later removed them entirely. Their second function, handing a large sum to somebody unreachable, is
> definitionally empty once the account is always reachable. **And deleting them is a net safety gain:**
> this document itself calls a note the perfect instrument for selling game money for real money, because a
> note changing hands leaves no record anywhere, and it carries a documented scam that made the display a
> correctness requirement. **Both go. Recommend: never — reinstate only alongside physical coins, if ever.**

**11. Lending is not in the first version. Confirm?**
*Consequence:* until regions have durable storage tied to place identity, a borrower can default for free
by parking the collateral somewhere about to be switched off. **Recommend: confirm, and say so explicitly
rather than leaving it as an implied capability.**

**12. Refuse any exchange rate to real money, outright?**
*Consequence:* the one shipped game that does this had to auction virtual banking licences for four
hundred thousand dollars rather than let players build banks. It converts a game-design decision into a
legal compliance programme. **Recommend: refuse.**

**13. Taking money off a cheater is a recorded operation with a case number, never an edit. Confirm?**
*Consequence:* almost free to design in now, genuinely painful later — retrofitting means the proof that
money is never created or destroyed was wrong for however long it took to notice. **Recommend: confirm.**

**14. Accept that money is provably never duplicated, but NOT provably never lost?**
*Consequence:* the conservation proof is only possible where the value is. Account money is fully
provable in production. Coins are provable only in the test harness, because no server can ask another
server anything. So if a server dies with coins on it, they are gone, and the only safe answer to "give
them back" is no — because giving them back is the one action that creates money from nothing. That
policy will produce a steady stream of angry, entirely correct complaints. ~~**Recommend: accept, and write
the refusal into this document rather than into a support policy, so it survives a change of staff.**~~

> **[REV 07-27] RETIRED — there is nothing left to accept.** With all value in one book at one writer,
> adding it all up is a local read the live system performs every tick, so money is provably never
> duplicated **and** provably never lost, in production rather than only in a test. **The refusal policy
> disappears entirely, and it was the ugliest thing in this document.** The one residual support case is
> not money loss: a buyer can be debited for goods they never receive if a venue's region dies
> mid-purchase. That is closed by holding the money at the service rather than paying it straight out, and
> releasing the hold on the buyer's next login. **Decision 14 becomes live again only if physical coins are
> ever added.**

**15. A parked account — one whose owner's server died with a rescue that never completed — is displayed
to the player as frozen, or as lost?**
*Consequence:* the two produce very different support loads and the choice cannot be made after the fact.
**Recommend: frozen, with a visible age, and an operational alarm on the operator side.**

---

## 12. Open questions and what could not be verified

### 12.1 Decision-relevant unknowns

1. **A real extraction rate for the backing material.** Every supply figure here is anchored on a guessed
   20 units/hour. The whole affordability argument is sensitive to it and it cannot be settled until
   terrain exists — another reason the material decision belongs at the terrain deadline.
2. **Whether prices in coin will roughly double each year if extraction technology improves.** That is my
   estimate from first principles, unverified against any shipped case. It is why §3.4 recommends gating
   scarcity on something tooling cannot improve.
3. **Whether any player-run financial service ever survived long-term.** My hypothesis is that pure
   custody and escrow — never lending, never promising a return — survived where deposit banks did not,
   which would strengthen §4.3 considerably. The web budget was exhausted before it could be checked, and
   the sweeping claim that every player bank collapsed rests on **two verified cases, not a survey.**
4. **Wealth concentration for a comparable currency economy.** Verified concentration data exists for
   *trade* (a Gini of 0.916, one region holding 68.8% of value) and for *destruction* (0.761), but nothing
   for wealth *held*. A working assumption of 0.8–0.95 is unverified.
5. **Whether large-denomination coins can substitute for notes in the first release.** If so, note
   untraceability arrives later, with better tooling and a smaller population to police.
6. **Whether the account service is meant to stay a single process.** Its key family is documented as
   independently splittable without a cross-file atomic transaction, which suggests a split is
   anticipated, but no design exists. §10.2 keeps the netting design written and unbuilt for exactly this.

### 12.2 Code-level gaps nobody owns

7. **What happens when the carried-state size cap is exceeded** is undefined, because nothing enforces it
   — the cap is declared and read by no code path. The enforcement itself becomes a money migration if it
   lands after money does.
8. **The version-floor check has no caller.** `floor_ok` is implemented and used only in tests; the
   enforcement point for Law A does not exist and lands with the unbuilt per-kind seam. It should be
   named as part of that work rather than assumed.
9. **Whether the durable applied-once record is meant to be per-place or per-server** is still circular:
   the deferred entry describes a per-place table, per-place storage does not exist, and no design says
   which lands first. §2.5 makes this cheaper than believed but does not resolve the intent.
10. **Whether the production transport can shed a saga-class frame under load.** The shed notice is a
    first-class inbound event with a closed reason taxonomy. If a reliable frame can be locally refused,
    a value-bearing message can be dropped by the *sender* — a different and possibly worse case than the
    losses confirmed here. Not traced.
11. **The real duration of the route-swap phase.** It bounds the window in which a lost value-carrying
    crossing is still re-driven, and it is the difference between "rare" and "routine" for §6.2 (I2). It
    deserves a measurement rather than an estimate.
12. **Whether a fresh login mints a new entity identity or reuses a stored one.** That decides whether a
    permanently parked re-home strands a player's money or merely strands an abandoned entity.
13. **Whether the existing conservation oracle's tolerance for loss is deliberate policy or interim.** A
    money design demanding zero loss asks for a stricter contract than the codebase currently intends to
    provide anywhere; the tension should be resolved explicitly.
14. **The cost of adding a variant to the closed node-kind enum** — every exhaustive match must be
    updated. That is the intended cost of the sealed design, and it is unbudgeted in §9.1.
15. **No load or perf test for the balance-read path's effect on the tick budget.** Payments have a
    latency-isolation gate; reads are the higher-volume path and are unmeasured.

> **[REV 07-27] Status of the fifteen unknowns.** **CLOSED by accounts-only (5):** item 5 (large
> denominations — deleted with notes, Decision 10); items 7 and 8 (the carried-state size cap and the
> version floor stop being money questions and stay open as seam questions); item 12 (a parked re-home no
> longer strands money, only an entity); item 13 (the oracle's loss tolerance — the money loss budget is
> zero by construction, §6.1). **SHARPENED (3):** item 1, the extraction rate, now also governs how much
> the destroyed fee must remove; item 2, technology-driven inflation, is worse without a second drain;
> item 6, whether the service stays one process, matters more now that 100% of payments cross it — though
> the ≈22,500× headroom at the top of the planning range says not yet. **UNCHANGED (5):** items 3, 4, 9,
> 14, 15. **NEWLY OPEN (2):** whether the shipped record supports blocking spending on a *communications*
> ground (no precedent found in either direction, §0.5); and whether player sentiment about unrobbable
> savings can be measured rather than inferred — every discussion venue the reviewers tried was blocked or
> paywalled, and that is the weakest link in the feel argument.

### 12.3 Findings I did not sustain, and why

> **[REV 07-27] Five findings from the accounts-only review were rejected on the merits. They are recorded
> in full in the review record at §0.7 and named here so this list stays the single index:**
> **(1)** *"The four defects remain real bugs for other carried state"* — rejected as an undersell: one of
> them strands the player entity itself and keeps its blocking status on its own merits.
> **(2)** *"Accounts-only deletes the expensive half"* — rejected as stated, sustained re-scoped: true for
> the schedule, false for the code.
> **(3)** *"No value in a killable place stops blocking"* — rejected as written: true for money only; goods
> and the backing vault still die.
> **(4)** *"The melt arbitrage fails because it needs attention nobody pays"* — rejected as the wrong
> diagnosis: it is never profitable, so attention is irrelevant. A stronger negative.
> **(5)** *"The reference economy destroys ~40% of production value per month"* — rejected as a
> misattribution: the only 40% in this document is a burn-share table row, not a measurement. The measured
> figures are used instead.

- **The shard-local Empty-suppression stopgap** (proposed as a cheap way to stop a routine teardown
  destroying value). Rejected: it still couples economic state to the keep-alive path, and under partition
  the region never spins down — unbounded. The substance (value must not be destroyed by a correct
  teardown) is fully kept as D-87; only the mechanism is refused.
- **Piggy-backing a per-region coin total onto the realm-lifecycle demand message** (proposed as the fix
  for production supply measurement). Rejected as a mechanism: it puts economic data on the lifecycle
  message, which is exactly the coupling the standing rule forbids. The requirement is kept as D-91, which
  demands a separate one-way telemetry report.
- **Stranded reservations described as a "drain" that destroys money.** Refined: they are **immobilised,
  not destroyed** — the conservation identity holds and the coins are still counted. The economic effect
  (circulating supply falls, deflationary, twice the designed loss drain) and both fixes are kept in full.
- **"Ship coins and local markets first, defer the account service until there is measured demand."** The
  volume figure behind it is sustained and published (§9.4); the ordering is rejected, because coins are
  not buildable yet and the service is (§10.1).

### 12.4 Corrections to the record

Two claims circulating in the candidate designs do not survive checking. Neither changes the substance,
but reasoning from them would reach wrong conclusions about what is already built.

- **The closed cross-server list has 24 arms, not 26** (`crates/wire/src/intershard.rs:118-271`, counted).
  Worth making a live assertion in the existing closed-set conformance test, one line, so future documents
  stay honest.
- **There is no `long_range_relay` capability flag.** The actual flag is `signal_relay`
  (`crates/sim/src/capability.rs:53`, `:107`, `:137`), set for the galaxy and station profiles and read by
  nothing outside boot logging and tests. Treating its presence as evidence of a partly-built relay is
  exactly the mistake to avoid — the message bus is a prose reservation at `crates/wire/src/intershard.rs:29`
  and nothing more.
- **World servers are not incapable of durable storage.** They open a redb-backed outbox through the
  shared boot path; it is inert because no manifest sets its path and the demand-spawn allow-list carries
  no storage key at all (§2.5). This makes several prerequisites materially cheaper than the prior record
  suggested.

**[REV 07-27] Two further corrections, from re-checking this document against the code. THE CODE WINS on
both.**

- **The demand-spawn allow-list is 13 keys, not 16**, at `crates/bins/src/lib.rs:2203-2226` (the line
  reference at §2.5 was also wrong), plus the orchestrator id and the peer-book closure appended by the
  caller at `:2228-2233` = 15. **The substantive claim is correct and unchanged: not one of them is a
  storage key.**
- **Teardown is NOT an outright kill with no save step.** §6.3 says so and it is inaccurate in mechanism.
  The spawner SIGTERMs the process group, polls to a configurable drain grace, and only then escalates to
  SIGKILL (`crates/bins/src/proc_launch.rs:214-244`), with a 2,000 ms shutdown linger deployed. **A
  graceful window exists; what is missing is any flush CODE** — the shard binary's own drain comment reads
  *"the shard holds no un-fsynced durable state"* (`crates/bins/src/bin/shard.rs:314`). **This matters for
  costing: adding save-on-shutdown later is a change inside an existing lifecycle window, not a new
  lifecycle phase, so per-place durable storage is cheaper than §6.3's framing implies.**
- **The closed cross-server list was re-counted at 24 and is confirmed** (`crates/wire/src/intershard.rs`,
  Ghost through ShardPresence). Accounts-only needs the same two appended arms; deposit and withdraw
  remain a later append, and every recent arm carries the note that appending preserves existing wire
  discriminants — which is what makes coins additive (§0.6).
